#!/usr/bin/env python3
"""
fbvi_native.py — Flow-based Variational Inference for Deep GPs, from scratch.

No gpytorch. q(U^(l)) is either:
- variant='dsvi': free Gaussian N(m^(l), L^(l) L^(l)T) (Salimbeni-Deisenroth DSVI baseline)
- variant='fbvi': implicit, sampled by integrating an ODE flow with velocity
                  field v_phi from U_0 ~ N(0, K_ZZ) (prior) to U_1 ~ q.

Training: -ELBO_data + KL (analytical for DSVI; zero for FBVI in v1 — flow's
zero-init + prior-sampled initial condition provides implicit regularisation,
and ELBO gradients reach the flow naturally because backprop runs through the
ODE integration). No auxiliary FM regression loss in v1; can be added later.

Evaluation: MC-RMSE and MC-NLL via num_samples samples of U through the DGP.
"""
import argparse
import csv
import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm

from torch.utils.data import TensorDataset, DataLoader


# =====================================================================
# Data utils
# =====================================================================
def _norm_xy_last(data):
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0].clamp_min(1e-12)) - 1
    y = data[:, -1]
    y = (y - y.mean()) / y.std().clamp_min(1e-12)
    return X, y


def _norm_xy_xcols(data, x_cols):
    X = data[:, :x_cols]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0].clamp_min(1e-12)) - 1
    y = data[:, x_cols]
    y = (y - y.mean()) / y.std().clamp_min(1e-12)
    return X, y


def load_concrete(p, dev): return _norm_xy_last(torch.tensor(pd.read_excel(p).values, dtype=torch.float32, device=dev))
def load_power(p, dev):    return _norm_xy_last(torch.tensor(pd.read_excel(p).values, dtype=torch.float32, device=dev))
def load_energy(p, dev):   return _norm_xy_xcols(torch.tensor(pd.read_csv(p).values, dtype=torch.float32, device=dev), 8)
def load_yacht(p, dev):    return _norm_xy_last(torch.tensor(pd.read_csv(p, sep=r'\s+', header=None, engine='python').dropna().values, dtype=torch.float32, device=dev))
def load_qsar(p, dev):     return _norm_xy_last(torch.tensor(pd.read_csv(p, sep=';', header=None).values, dtype=torch.float32, device=dev))
def load_boston(p, dev):   return _norm_xy_last(torch.tensor(pd.read_csv(p, sep=',', header=None, skiprows=45, engine='python').values, dtype=torch.float32, device=dev))
def load_protein(p, dev):
    df = pd.read_csv(p); cols = list(df.columns); df = df[cols[1:] + cols[:1]]
    return _norm_xy_last(torch.tensor(df.values, dtype=torch.float32, device=dev))


def split_and_loaders(X, y, batch_size, split=0.8, test_batch_size=16384):
    N = X.shape[0]
    perm = torch.randperm(N, device=X.device)
    ntr = int(split * N)
    tr, te = perm[:ntr], perm[ntr:]
    train_x, train_y = X[tr].contiguous(), y[tr].contiguous()
    test_x,  test_y  = X[te].contiguous(), y[te].contiguous()
    return (train_x, train_y, test_x, test_y,
            DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True),
            DataLoader(TensorDataset(test_x, test_y),  batch_size=test_batch_size, shuffle=False))


# =====================================================================
# Sparse GP layer
# =====================================================================
def rbf_kernel(X1, X2, log_ls, log_scale):
    """RBF kernel with ARD lengthscales. X1: [N1, d], X2: [N2, d]. Returns [N1, N2]."""
    ls = log_ls.exp().clamp_min(1e-3)            # [d]
    scale = log_scale.exp().clamp_min(1e-3)       # scalar
    d = (X1.unsqueeze(-2) - X2.unsqueeze(-3)) / ls
    return scale * torch.exp(-0.5 * (d * d).sum(-1))


class SparseGPLayer(nn.Module):
    """One sparse GP layer with RBF-ARD kernel. Per-output-dim independent (shared kernel)."""
    def __init__(self, M, d_in, d_out, device, jitter=1e-4):
        super().__init__()
        self.M, self.d_in, self.d_out, self.jitter = M, d_in, d_out, jitter
        # Init inducing points spread out in [-1, 1]
        self.Z = nn.Parameter(torch.empty(M, d_in, device=device).uniform_(-1, 1))
        self.log_ls = nn.Parameter(torch.zeros(d_in, device=device))
        self.log_scale = nn.Parameter(torch.zeros((), device=device))
        # Linear mean function (Salimbeni-style residual init): identity if d_in == d_out, else zero
        if d_in == d_out:
            W = torch.eye(d_in, device=device)
        else:
            W = torch.zeros(d_in, d_out, device=device)
        self.mean_W = nn.Parameter(W)

    def Kzz(self):
        return rbf_kernel(self.Z, self.Z, self.log_ls, self.log_scale) \
            + self.jitter * torch.eye(self.M, device=self.Z.device)

    def conditional(self, X, U):
        """X: [N, d_in], U: [M, d_out] -> (mean: [N, d_out], var: [N])."""
        Kzz = self.Kzz()
        Kxz = rbf_kernel(X, self.Z, self.log_ls, self.log_scale)
        scale = self.log_scale.exp().clamp_min(1e-3)
        L = torch.linalg.cholesky(Kzz)
        alpha = torch.cholesky_solve(U, L)            # [M, d_out]
        v = torch.cholesky_solve(Kxz.T, L)            # [M, N]
        mean = Kxz @ alpha + X @ self.mean_W          # [N, d_out]
        var = scale - (Kxz * v.T).sum(-1)             # [N]
        return mean, var.clamp_min(1e-6)

    def prior_sample(self):
        """Draw U ~ N(0, K_ZZ ⊗ I_{d_out})."""
        L = torch.linalg.cholesky(self.Kzz())
        eps = torch.randn(self.M, self.d_out, device=self.Z.device)
        return L @ eps

    def kl_to_prior(self, m, L_q):
        """Analytical KL(N(m, L_q L_q^T) || N(0, K_ZZ)), m: [M, d_out], L_q: [d_out, M, M]."""
        Kzz = self.Kzz()
        Lp = torch.linalg.cholesky(Kzz)
        kl = 0.0
        logdet_p = 2 * torch.log(torch.diag(Lp)).sum()
        for j in range(self.d_out):
            mj = m[:, j:j+1]
            Lqj = torch.tril(L_q[j])
            Sqj = Lqj @ Lqj.T
            Kinv_S = torch.cholesky_solve(Sqj, Lp)
            Kinv_m = torch.cholesky_solve(mj, Lp)
            tr = Kinv_S.diagonal().sum()
            quad = (mj * Kinv_m).sum()
            logdet_q = 2 * torch.log(torch.abs(torch.diag(Lqj)).clamp_min(1e-8)).sum()
            kl = kl + 0.5 * (tr + quad - self.M + logdet_p - logdet_q)
        return kl


# =====================================================================
# Velocity field (for FBVI variant)
# =====================================================================
class VelocityField(nn.Module):
    """v_phi(U, t, d) on R^{M x d_out}. d = shortcut step size (Frans 2024).
    When d ~ 0 the field acts as instantaneous velocity (standard ODE flow);
    when d > 0 it acts as the average velocity over [t, t+d] (shortcut)."""
    def __init__(self, M, d_out, hidden=128):
        super().__init__()
        in_dim = M * d_out + 2   # flatten U, append (t, d)
        out_dim = M * d_out
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.M, self.d_out = M, d_out

    def _scalar_to_tensor(self, x, like):
        if not torch.is_tensor(x):
            return like.new_full((*like.shape[:-1], 1), float(x))
        if x.dim() == 0:
            return x.expand(*like.shape[:-1], 1)
        return x.reshape(*like.shape[:-1], 1)

    def forward(self, U, t, d=0.0):
        # U: [..., M, d_out], t & d scalar in [0,1] -> v same shape as U
        shape = U.shape
        U_flat = U.reshape(*shape[:-2], -1)              # [..., M*d_out]
        t_in = self._scalar_to_tensor(t, U_flat)
        d_in = self._scalar_to_tensor(d, U_flat)
        inp = torch.cat([U_flat, t_in, d_in], dim=-1)
        return self.net(inp).reshape(shape)


class ScoreField(nn.Module):
    """s_phi(U, t) ≈ ∇log p_t(U) under VP noising. No shortcut input."""
    def __init__(self, M, d_out, hidden=128):
        super().__init__()
        in_dim = M * d_out + 1
        out_dim = M * d_out
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.M, self.d_out = M, d_out

    def forward(self, U, t):
        shape = U.shape
        U_flat = U.reshape(*shape[:-2], -1)
        if not torch.is_tensor(t):
            t_in = U_flat.new_full((*U_flat.shape[:-1], 1), float(t))
        elif t.dim() == 0:
            t_in = t.expand(*U_flat.shape[:-1], 1)
        else:
            t_in = t.reshape(*U_flat.shape[:-1], 1)
        inp = torch.cat([U_flat, t_in], dim=-1)
        return self.net(inp).reshape(shape)


def _alpha_sigma(t, beta_min, beta_max):
    """VP marginal: U_t | U_0 ~ N(alpha_t U_0, sigma_t^2 I)."""
    int_beta = beta_min * t + 0.5 * t * t * (beta_max - beta_min)
    if torch.is_tensor(t):
        alpha = torch.exp(-0.5 * int_beta)
    else:
        alpha = math.exp(-0.5 * int_beta)
    sigma2 = 1.0 - alpha ** 2 if torch.is_tensor(t) else max(1.0 - alpha ** 2, 1e-8)
    sigma = sigma2 ** 0.5 if torch.is_tensor(t) else math.sqrt(max(sigma2, 1e-8))
    return alpha, sigma


def _precompute_doob(n_grid, lambda0, g0, sigma0):
    """
    Numerically integrate Prop 2 ODE (DBVI 2509.19078) for the bridge marginal
    coefficients under affine forward drift f(U,t)=-lambda*U, g(t)=g0 const,
    and isotropic initial p_0^theta(U_0|x) = N(mu_theta(x), sigma0^2 I).
    Returns: ts grid, phi(t) such that m_t = phi(t) * mu_theta, and kappa(t).
    """
    ts = torch.linspace(0.0, 1.0, n_grid + 1)
    dt = 1.0 / n_grid
    phi = torch.zeros(n_grid + 1)
    kappa = torch.zeros(n_grid + 1)
    phi[0] = 1.0                                  # m_0/mu_theta = 1
    kappa[0] = sigma0 ** 2
    for i in range(n_grid):
        t = float(ts[i])
        a_t = math.exp(-lambda0 * t)
        if t < 1e-6:
            c_t = 0.0                              # h-transform vanishes at t=0
        else:
            q_t = g0 ** 2 * (1.0 - math.exp(-2 * lambda0 * t)) / (2 * lambda0 + 1e-12)
            denom = max((a_t ** 2 * sigma0 ** 2 + q_t) * q_t, 1e-12)
            c_t = g0 ** 2 * sigma0 ** 2 * (a_t ** 2) / denom
        dphi = -(lambda0 + c_t) * phi[i].item() + c_t * a_t
        dkappa = -2 * (lambda0 + c_t) * kappa[i].item() + g0 ** 2 + 2 * c_t * a_t * sigma0 ** 2
        phi[i + 1] = phi[i] + dphi * dt
        kappa[i + 1] = kappa[i] + dkappa * dt
    kappa = kappa.clamp_min(1e-6)
    return ts, phi, kappa


def _interp_grid(values, ts, t_query):
    """Linear interpolation on a uniform grid ts -> values at t_query (scalar)."""
    n = ts.shape[0] - 1
    t_query = max(0.0, min(1.0, float(t_query)))
    idx_f = t_query * n
    i0 = int(math.floor(idx_f))
    i1 = min(i0 + 1, n)
    w = idx_f - i0
    return (1 - w) * values[i0] + w * values[i1]


class Amortizer(nn.Module):
    """Z (inducing locs) -> mu_theta of shape [M, d_out]. Per-layer."""
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, d_out),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, Z):
        return self.net(Z)  # [M, d_out]


class ConditionalScoreField(nn.Module):
    """s_phi(U_t, t, ctx) for Doob-bridged DBVI. ctx has same shape as U (M x d_out)."""
    def __init__(self, M, d_out, hidden=128):
        super().__init__()
        in_dim = 2 * M * d_out + 1     # U_flat + t + ctx_flat
        out_dim = M * d_out
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.M, self.d_out = M, d_out

    def forward(self, U, t, ctx):
        shape = U.shape
        U_flat = U.reshape(*shape[:-2], -1)
        ctx_flat = ctx.reshape(*shape[:-2], -1)
        if not torch.is_tensor(t):
            t_in = U_flat.new_full((*U_flat.shape[:-1], 1), float(t))
        elif t.dim() == 0:
            t_in = t.expand(*U_flat.shape[:-1], 1)
        else:
            t_in = t.reshape(*U_flat.shape[:-1], 1)
        inp = torch.cat([U_flat, t_in, ctx_flat], dim=-1)
        return self.net(inp).reshape(shape)


class Generator(nn.Module):
    """g_phi: noise eps in R^{M*d_out} -> U in R^{M x d_out}. For IPVI."""
    def __init__(self, M, d_out, hidden=128):
        super().__init__()
        in_dim = M * d_out
        out_dim = M * d_out
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.M, self.d_out = M, d_out

    def forward(self, eps):
        # eps: [M, d_out] or [B, M, d_out]
        shape = eps.shape
        flat = eps.reshape(*shape[:-2], -1)
        return self.net(flat).reshape(shape)


class Discriminator(nn.Module):
    """T_psi: U in R^{M x d_out} -> scalar. For IPVI."""
    def __init__(self, M, d_out, hidden=128):
        super().__init__()
        in_dim = M * d_out
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, U):
        shape = U.shape
        flat = U.reshape(*shape[:-2], -1)
        return self.net(flat).squeeze(-1)


class ConditionalVelocityField(nn.Module):
    """v_phi(U_t, t, ctx) — flow-matching analog of conditional score."""
    def __init__(self, M, d_out, hidden=128):
        super().__init__()
        in_dim = 2 * M * d_out + 1
        out_dim = M * d_out
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.M, self.d_out = M, d_out

    def forward(self, U, t, ctx):
        shape = U.shape
        U_flat = U.reshape(*shape[:-2], -1)
        ctx_flat = ctx.reshape(*shape[:-2], -1)
        if not torch.is_tensor(t):
            t_in = U_flat.new_full((*U_flat.shape[:-1], 1), float(t))
        elif t.dim() == 0:
            t_in = t.expand(*U_flat.shape[:-1], 1)
        else:
            t_in = t.reshape(*U_flat.shape[:-1], 1)
        inp = torch.cat([U_flat, t_in, ctx_flat], dim=-1)
        return self.net(inp).reshape(shape)


# =====================================================================
# Flow Deep GP
# =====================================================================
class FlowDGP(nn.Module):
    def __init__(self, dims, M, variant='fbvi', n_steps=10, flow_hidden=128,
                 sde_sigma=0.1, beta_min=0.1, beta_max=20.0,
                 doob_lambda=1.0, doob_g=1.0, doob_sigma0=1.0,
                 device='cuda'):
        super().__init__()
        self.dims = dims
        self.M = M
        self.variant = variant
        self.n_steps = int(n_steps)
        self.sde_sigma = float(sde_sigma)         # noise scale for DBVI variant
        self.beta_min = float(beta_min)            # VP noise schedule (score variant)
        self.beta_max = float(beta_max)
        self.doob_lambda = float(doob_lambda)      # for dbvi-s
        self.doob_g = float(doob_g)
        self.doob_sigma0 = float(doob_sigma0)
        L = len(dims) - 1

        self.layers = nn.ModuleList([
            SparseGPLayer(M, dims[i], dims[i+1], device=device) for i in range(L)
        ])

        if variant in ('fbvi', 'dbvi'):
            self.flows = nn.ModuleList([
                VelocityField(M, dims[i+1], hidden=flow_hidden) for i in range(L)
            ])
        elif variant == 'score':
            # DDVI-style: unconditional VP DSM with score network s_phi(U, t).
            self.scores = nn.ModuleList([
                ScoreField(M, dims[i+1], hidden=flow_hidden) for i in range(L)
            ])
        elif variant == 'dbvi-s':
            # Proper DBVI: amortizer mu_theta(Z) gives data-anchored initial mean,
            # Doob bridge forward, conditional score s_phi(U_t, t, ctx) where ctx
            # is the amortizer output. Bridge marginal m_t = phi(t)*mu_theta,
            # variance kappa(t), precomputed on a 100-pt grid.
            self.amortizers = nn.ModuleList([
                Amortizer(d_in=dims[i], d_out=dims[i+1]) for i in range(L)
            ])
            self.scores = nn.ModuleList([
                ConditionalScoreField(M, dims[i+1], hidden=flow_hidden) for i in range(L)
            ])
            ts_grid, phi_grid, kappa_grid = _precompute_doob(
                100, self.doob_lambda, self.doob_g, self.doob_sigma0
            )
            self.register_buffer('doob_ts', ts_grid.to(device))
            self.register_buffer('doob_phi', phi_grid.to(device))
            self.register_buffer('doob_kappa', kappa_grid.to(device))
        elif variant == 'ipvi':
            # Implicit Posterior VI (Yu et al. 2019): GAN-style. Generator g_phi
            # maps noise -> U (implicit q); discriminator T_psi estimates
            # log q(U) - log p(U) ≈ density ratio, replacing analytic KL.
            self.generators = nn.ModuleList([
                Generator(M, dims[i+1], hidden=flow_hidden) for i in range(L)
            ])
            self.discriminators = nn.ModuleList([
                Discriminator(M, dims[i+1], hidden=flow_hidden) for i in range(L)
            ])
        elif variant == 'fbvi-bridge':
            # Velocity counterpart of DBVI-s: amortizer + bridge-anchored start +
            # conditional velocity NN. No DSM; trained end-to-end via ELBO backprop
            # through ODE. The bridge structure just gives a data-anchored starting
            # distribution for U_T.
            self.amortizers = nn.ModuleList([
                Amortizer(d_in=dims[i], d_out=dims[i+1]) for i in range(L)
            ])
            self.flows = nn.ModuleList([
                ConditionalVelocityField(M, dims[i+1], hidden=flow_hidden) for i in range(L)
            ])
            ts_grid, phi_grid, kappa_grid = _precompute_doob(
                100, self.doob_lambda, self.doob_g, self.doob_sigma0
            )
            self.register_buffer('doob_ts', ts_grid.to(device))
            self.register_buffer('doob_phi', phi_grid.to(device))
            self.register_buffer('doob_kappa', kappa_grid.to(device))
        elif variant == 'dsvi':
            # Free q(U^(l)) = N(m^(l), L^(l) L^(l)T) per output dim
            self.q_means = nn.ParameterList([
                nn.Parameter(torch.zeros(M, dims[i+1], device=device)) for i in range(L)
            ])
            # chol per output dim, init as I
            self.q_chols = nn.ParameterList([
                nn.Parameter(torch.eye(M, device=device).unsqueeze(0).repeat(dims[i+1], 1, 1).contiguous())
                for i in range(L)
            ])
        else:
            raise ValueError(variant)

        self.log_noise = nn.Parameter(torch.tensor(-2.0, device=device))    # log std of likelihood

    def sample_U(self, layer_idx, n_steps=None):
        """Draw one U^(l) according to current q. n_steps overrides default
        (used for few-step inference)."""
        layer = self.layers[layer_idx]
        if n_steps is None:
            n_steps = self.n_steps
        if self.variant == 'fbvi':
            # Pure ODE: dU/dt = v_phi(U,t,d) with d = dt (shortcut step)
            U = layer.prior_sample()                                     # [M, d_out]
            dt = 1.0 / n_steps
            for k in range(n_steps):
                t = k * dt                                                # left endpoint
                v = self.flows[layer_idx](U, t, dt)
                U = U + v * dt
            return U
        if self.variant == 'dbvi':
            U = layer.prior_sample()
            dt = 1.0 / n_steps
            sqrt_dt = math.sqrt(dt)
            for k in range(n_steps):
                t = k * dt
                v = self.flows[layer_idx](U, t, dt)
                eps = torch.randn_like(U)
                U = U + v * dt + self.sde_sigma * sqrt_dt * eps
            return U
        if self.variant == 'score':
            # Reverse VP SDE: start from N(0, I) at t=1, integrate backward to t=0
            # using score s_phi.
            score = self.scores[layer_idx]
            dt = 1.0 / n_steps
            U = torch.randn(self.M, layer.d_out, device=self.log_noise.device)
            for k in range(n_steps):
                t = 1.0 - k * dt                # decreasing from 1 to ~0
                beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
                # Reverse SDE drift in t-direction (going backward):
                #   dU_back = (0.5*β*U + β*s_phi) * dt + sqrt(β*dt) * eps
                reverse_drift = 0.5 * beta_t * U + beta_t * score(U, t)
                noise = torch.randn_like(U) * math.sqrt(beta_t * dt)
                U = U + reverse_drift * dt + noise
            return U
        if self.variant == 'ipvi':
            eps = torch.randn(self.M, layer.d_out, device=self.log_noise.device)
            return self.generators[layer_idx](eps)
        if self.variant == 'dbvi-s':
            # Reverse Doob-bridged SDE (per Prop 1).
            amortizer = self.amortizers[layer_idx]
            score = self.scores[layer_idx]
            ctx = amortizer(layer.Z)                         # [M, d_out]
            dt = 1.0 / n_steps
            kappa_T = _interp_grid(self.doob_kappa, self.doob_ts, 1.0).item()
            phi_T = _interp_grid(self.doob_phi, self.doob_ts, 1.0).item()
            U = phi_T * ctx + math.sqrt(max(kappa_T, 1e-6)) * torch.randn_like(ctx)
            for k in range(n_steps):
                t = 1.0 - k * dt
                f_t = -self.doob_lambda * U
                s_cond = score(U, t, ctx)
                reverse_drift = -f_t + self.doob_g ** 2 * s_cond
                noise = torch.randn_like(U) * (self.doob_g * math.sqrt(dt))
                U = U + reverse_drift * dt + noise
            return U
        if self.variant == 'fbvi-bridge':
            # Velocity counterpart: bridge-anchored start, deterministic ODE.
            amortizer = self.amortizers[layer_idx]
            flow = self.flows[layer_idx]
            ctx = amortizer(layer.Z)                         # [M, d_out]
            dt = 1.0 / n_steps
            kappa_T = _interp_grid(self.doob_kappa, self.doob_ts, 1.0).item()
            phi_T = _interp_grid(self.doob_phi, self.doob_ts, 1.0).item()
            U = phi_T * ctx + math.sqrt(max(kappa_T, 1e-6)) * torch.randn_like(ctx)
            # Integrate ODE backward in t from 1 to 0 (mirrors dbvi-s but no noise).
            for k in range(n_steps):
                t = 1.0 - k * dt
                v = flow(U, t, ctx)
                U = U + v * dt
            return U
        # DSVI: reparam from N(m, L L^T)
        m = self.q_means[layer_idx]                                      # [M, d_out]
        chols = self.q_chols[layer_idx]                                  # [d_out, M, M]
        z = torch.randn_like(m)
        # Apply per-output-dim Cholesky
        U_cols = []
        for j in range(m.shape[1]):
            Lj = torch.tril(chols[j])
            U_cols.append(m[:, j] + Lj @ z[:, j])
        return torch.stack(U_cols, dim=1)

    def sample_all_U(self, n_steps=None):
        return [self.sample_U(i, n_steps=n_steps) for i in range(len(self.layers))]

    def dgp_forward(self, X, Us):
        """Forward DGP layer-by-layer with reparameterised sample of each F."""
        h = X
        for layer, U in zip(self.layers, Us):
            mean, var = layer.conditional(h, U)
            std = var.sqrt().unsqueeze(-1)
            h = mean + std * torch.randn_like(mean)
        return h

    def dsm_loss(self, n_samples=1):
        """Score-matching auxiliary loss.
        - variant=score: vanilla VP DSM (DDVI-style).
        - variant=dbvi-s: conditional DSM against Doob-bridge marginal N(m_t, kappa_t I)
                          with m_t = phi(t)*mu_theta, conditional score s_phi(U_t, t, ctx).
        """
        if self.variant == 'score':
            total = 0.0
            for layer_idx, (layer, score) in enumerate(zip(self.layers, self.scores)):
                for _ in range(n_samples):
                    with torch.no_grad():
                        U_0 = self.sample_U(layer_idx).detach()
                    t = float(torch.rand((), device=U_0.device).item()); t = max(t, 1e-3)
                    alpha_t, sigma_t = _alpha_sigma(t, self.beta_min, self.beta_max)
                    eps = torch.randn_like(U_0)
                    U_t = alpha_t * U_0 + sigma_t * eps
                    target = -eps / sigma_t
                    pred = score(U_t.detach(), t)
                    total = total + (sigma_t ** 2 * (pred - target) ** 2).mean()
            return total / n_samples
        if self.variant == 'dbvi-s':
            total = 0.0
            for layer_idx, (layer, amortizer, score) in enumerate(
                zip(self.layers, self.amortizers, self.scores)
            ):
                for _ in range(n_samples):
                    ctx = amortizer(layer.Z)                       # [M, d_out]
                    t = float(torch.rand((), device=ctx.device).item()); t = max(t, 1e-3)
                    phi_t = _interp_grid(self.doob_phi, self.doob_ts, t).item()
                    kappa_t = _interp_grid(self.doob_kappa, self.doob_ts, t).item()
                    kappa_t = max(kappa_t, 1e-6)
                    sigma_t = math.sqrt(kappa_t)
                    m_t = phi_t * ctx.detach()
                    eps = torch.randn_like(ctx)
                    U_t = m_t + sigma_t * eps
                    target = -(U_t - m_t) / kappa_t                # -(U_t - phi*mu)/kappa
                    pred = score(U_t, t, ctx)
                    total = total + (kappa_t * (pred - target) ** 2).mean()
            return total / n_samples
        return torch.zeros((), device=self.log_noise.device)

    def shortcut_loss(self, n_samples=2):
        """Self-consistency loss (Frans 2024):
              v_phi(U, t, d) ≈ 0.5 * ( v_phi(U, t, d/2) + v_phi(U', t + d/2, d/2) )
        where U' = U + v_phi(U, t, d/2) * d/2.
        Trains the velocity to be a valid average over varying step sizes.
        """
        if self.variant not in ('fbvi', 'dbvi'):
            return torch.zeros((), device=self.log_noise.device)
        total = 0.0
        # Step sizes d as fractions of 1 — covers both fine (small d) and
        # coarse (1-step) inference regimes.
        d_choices = [0.125, 0.25, 0.5, 1.0]
        for layer, flow in zip(self.layers, self.flows):
            for _ in range(n_samples):
                d = d_choices[int(torch.randint(len(d_choices), (1,)).item())]
                t = float(torch.rand((), device=self.log_noise.device).item()) * (1.0 - d)
                # Sample U by partial ODE integration from prior up to time t
                with torch.no_grad():
                    U = layer.prior_sample()
                    n_pre = max(1, int(round(self.n_steps * t)))
                    if t > 1e-6:
                        dt_pre = t / n_pre
                        for k in range(n_pre):
                            U = U + flow(U, k * dt_pre, dt_pre) * dt_pre
                    U = U.detach()
                # Target: 2-step composition with d/2
                with torch.no_grad():
                    v1 = flow(U, t, d / 2)
                    U_mid = U + v1 * (d / 2)
                    v2 = flow(U_mid, t + d / 2, d / 2)
                    target = 0.5 * (v1 + v2)
                # Prediction: single-step with d
                v_pred = flow(U, t, d)
                total = total + ((v_pred - target.detach()) ** 2).mean()
        return total / n_samples

    def fm_loss(self, x_batch, y_batch, N_total, fm_samples=1):
        """
        Conditional Flow-Matching regression toward the annealed-posterior
        Langevin velocity v_target(U_t, t) = ∇_U log π_t(U)
                          = -K_ZZ^-1 U + t * ∇_U log p(y|U, x).
        Implemented per layer; U_t obtained by partial ODE integration from prior.
        Likelihood gradient computed via autograd, scaled by N/|B|.
        """
        if self.variant not in ('fbvi', 'dbvi'):
            return torch.zeros((), device=x_batch.device)

        device = x_batch.device
        B = x_batch.shape[0]
        lik_scale = float(N_total) / float(B)
        total_fm = 0.0

        for layer_idx, (layer, flow) in enumerate(zip(self.layers, self.flows)):
            for _ in range(fm_samples):
                t_val = float(torch.rand((), device=device).item())
                t_val = max(t_val, 1e-3)   # avoid degenerate t=0

                # ---- Sample U_t ~ p_t via partial ODE integration (no grad) ----
                with torch.no_grad():
                    U_0 = layer.prior_sample()
                    U = U_0
                    n_partial = max(1, int(round(self.n_steps * t_val)))
                    dt_partial = t_val / n_partial
                    for k in range(n_partial):
                        U = U + flow(U, k * dt_partial, dt_partial) * dt_partial
                    U_t = U.detach()

                # ---- Compute v_target ----
                # (a) prior gradient -K_ZZ^-1 U_t (analytical)
                Kzz = layer.Kzz()
                L_prior = torch.linalg.cholesky(Kzz)
                prior_grad = -torch.cholesky_solve(U_t, L_prior).detach()    # [M, d_out]

                # (b) likelihood gradient via autograd on DGP forward
                U_t_lead = U_t.clone().requires_grad_(True)
                Us = [self.sample_U(i) if i != layer_idx else U_t_lead
                      for i in range(len(self.layers))]
                h = x_batch
                for L_l, Ul in zip(self.layers, Us):
                    mean, var = L_l.conditional(h, Ul)
                    std = var.sqrt().unsqueeze(-1)
                    h = mean + std * torch.randn_like(mean)
                f = h
                noise_var = (self.log_noise.exp() ** 2).clamp_min(1e-6)
                log_lik = -0.5 * ((y_batch - f.squeeze(-1)) ** 2 / noise_var
                                  + torch.log(2 * math.pi * noise_var)).sum()
                log_lik_scaled = lik_scale * log_lik
                lik_grad = torch.autograd.grad(log_lik_scaled, U_t_lead,
                                                create_graph=False,
                                                retain_graph=False)[0].detach()

                v_target = prior_grad + t_val * lik_grad

                # ---- v_pred at U_t (re-do forward through flow, with grad) ----
                v_pred = flow(U_t.detach(), t_val, 1.0 / self.n_steps)
                total_fm = total_fm + ((v_pred - v_target) ** 2).mean()

        return total_fm / fm_samples

    def kl_total(self):
        if self.variant in ('fbvi', 'dbvi', 'score', 'dbvi-s', 'fbvi-bridge', 'ipvi'):
            # No exact KL; implicit regularisation only. For score variant,
            # the DSM regulariser indirectly enforces score-of-q ≈ score-of-noised-q,
            # which keeps the reverse-SDE samples Bayesian.
            return torch.zeros((), device=self.log_noise.device)
        total = 0.0
        for i, layer in enumerate(self.layers):
            total = total + layer.kl_to_prior(self.q_means[i], self.q_chols[i])
        return total

    def loss_and_metrics(self, X_batch, y_batch, N_total, mc_samples=2, kl_weight=1.0):
        """-ELBO/N (returns mean over batch, scaled), plus diagnostics."""
        log_lik_sum = 0.0
        for _ in range(mc_samples):
            Us = self.sample_all_U()
            f = self.dgp_forward(X_batch, Us)                           # [B, d_L]
            noise_var = (self.log_noise.exp() ** 2).clamp_min(1e-6)
            ll = -0.5 * ((y_batch - f.squeeze(-1)) ** 2 / noise_var
                         + torch.log(2 * math.pi * noise_var))
            log_lik_sum = log_lik_sum + ll.sum()
        log_lik = log_lik_sum / mc_samples
        elbo_data = (N_total / X_batch.shape[0]) * log_lik              # scale to full dataset
        kl = self.kl_total()
        elbo = elbo_data - kl_weight * kl
        loss = -elbo / N_total
        return loss, log_lik.detach() / X_batch.shape[0], kl.detach()


# =====================================================================
# Evaluation
# =====================================================================
@torch.no_grad()
def eval_metrics(model, test_loader, test_y, device, num_samples=32, infer_steps=None):
    sq_err_sum = 0.0
    n_total = 0
    ll_sum = 0.0
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device); y_batch = y_batch.to(device)
        # Collect num_samples MC samples of F_L
        fs = []
        for _ in range(num_samples):
            Us = model.sample_all_U(n_steps=infer_steps)
            fs.append(model.dgp_forward(x_batch, Us).squeeze(-1))
        fs = torch.stack(fs, dim=0)                                       # [S, B]
        pred_mean = fs.mean(dim=0)
        sq_err_sum += ((pred_mean - y_batch) ** 2).sum().item()
        n_total += y_batch.shape[0]
        # log-marginal via Gaussian likelihood mixture
        noise_var = (model.log_noise.exp() ** 2).item()
        # log p(y|f_s) per sample
        log_p_s = -0.5 * ((y_batch.unsqueeze(0) - fs) ** 2 / noise_var
                          + math.log(2 * math.pi * noise_var))            # [S, B]
        log_p = torch.logsumexp(log_p_s, dim=0) - math.log(num_samples)   # [B]
        ll_sum += log_p.sum().item()
    rmse = math.sqrt(sq_err_sum / max(n_total, 1))
    nll = -ll_sum / max(n_total, 1)
    return rmse, nll


# =====================================================================
# Main training loop
# =====================================================================
def make_data(args, device):
    if args.dataset == 'concrete': return load_concrete(args.data_path, device)
    if args.dataset == 'power':    return load_power(args.data_path, device)
    if args.dataset == 'energy':   return load_energy(args.data_path, device)
    if args.dataset == 'yacht':    return load_yacht(args.data_path, device)
    if args.dataset == 'qsar':     return load_qsar(args.data_path, device)
    if args.dataset == 'boston':   return load_boston(args.data_path, device)
    if args.dataset == 'protein':  return load_protein(args.data_path, device)
    raise ValueError(args.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', type=str, default='fbvi',
                   choices=['fbvi', 'dsvi', 'dbvi', 'score', 'dbvi-s', 'fbvi-bridge', 'ipvi'])
    p.add_argument('--ipvi_disc_steps', type=int, default=1,
                   help='Discriminator updates per generator update for IPVI BRD')
    p.add_argument('--doob_lambda', type=float, default=1.0)
    p.add_argument('--doob_g', type=float, default=1.0)
    p.add_argument('--doob_sigma0', type=float, default=1.0)
    p.add_argument('--sde_sigma', type=float, default=0.1,
                   help='Noise scale for velocity-based DBVI variant; 0.0 reduces it to FBVI')
    p.add_argument('--dsm_weight', type=float, default=0.0,
                   help='Denoising-Score-Matching loss coefficient (for variant=score)')
    p.add_argument('--dsm_samples', type=int, default=1,
                   help='Number of (t, eps) MC samples per layer for DSM loss')
    p.add_argument('--beta_min', type=float, default=0.1)
    p.add_argument('--beta_max', type=float, default=20.0)
    p.add_argument('--fm_weight', type=float, default=0.0,
                   help='Coefficient for the flow-matching regression loss (annealed-Langevin target)')
    p.add_argument('--fm_samples', type=int, default=1,
                   help='Number of (t, U_t) MC samples per layer per minibatch for FM loss')
    p.add_argument('--shortcut_weight', type=float, default=0.0,
                   help='Coefficient for Frans 2024 shortcut self-consistency loss')
    p.add_argument('--shortcut_samples', type=int, default=2,
                   help='Number of (t, d) MC samples per layer for shortcut loss')
    p.add_argument('--shortcut_warmup_epochs', type=int, default=20,
                   help='Disable shortcut loss for the first N epochs (let ELBO converge first)')
    p.add_argument('--eval_steps_list', type=str, default='',
                   help='Comma-separated list of inference n_steps to evaluate at end (e.g. "1,2,4,10")')
    p.add_argument('--dataset', type=str, default='energy',
                   choices=['energy','concrete','power','yacht','qsar','boston','protein'])
    p.add_argument('--data_path', type=str, default='energy.csv')
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--num_inducing', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=None,
                   help='hidden DGP dim; default = input dim')
    p.add_argument('--flow_steps', type=int, default=10)
    p.add_argument('--flow_hidden', type=int, default=128)
    p.add_argument('--mc_samples', type=int, default=2)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--test_batch_size', type=int, default=16384)
    p.add_argument('--eval_samples', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split', type=float, default=0.8)
    p.add_argument('--log_csv', type=str, default='native_log.csv')
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--grad_clip', type=float, default=10.0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    X, y = make_data(args, device)
    train_x, train_y, test_x, test_y, train_loader, test_loader = split_and_loaders(
        X, y, batch_size=args.batch_size, split=args.split, test_batch_size=args.test_batch_size
    )

    d_in = train_x.shape[-1]
    d_h = args.hidden_dim if args.hidden_dim else d_in
    dims = [d_in] + [d_h] * (args.layers - 1) + [1]
    print(f"[Info] variant={args.variant} dims={dims} M={args.num_inducing} "
          f"device={device} seed={args.seed} N_train={train_x.shape[0]}")

    model = FlowDGP(dims=dims, M=args.num_inducing, variant=args.variant,
                    n_steps=args.flow_steps, flow_hidden=args.flow_hidden,
                    sde_sigma=args.sde_sigma,
                    beta_min=args.beta_min, beta_max=args.beta_max,
                    doob_lambda=args.doob_lambda, doob_g=args.doob_g,
                    doob_sigma0=args.doob_sigma0,
                    device=device).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] trainable params: {n_params}")

    N_total = train_x.shape[0]
    rows = []; t0 = time.time()
    global_step = 0

    if args.variant == 'ipvi':
        # IPVI: two-player BRD. Discriminator vs (generator + GP layers + likelihood).
        import torch.nn.functional as F
        disc_params = [p for n, p in model.named_parameters() if 'discriminators' in n]
        gen_params  = [p for n, p in model.named_parameters() if 'discriminators' not in n]
        opt_disc = torch.optim.Adam(disc_params, lr=args.lr)
        opt_gen  = torch.optim.Adam(gen_params, lr=args.lr)
        for epoch in tqdm.tqdm(range(args.epochs), desc="Epoch"):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                # ---- (A) Discriminator update ----
                for _ in range(args.ipvi_disc_steps):
                    opt_disc.zero_grad()
                    disc_loss = 0.0
                    for li, layer in enumerate(model.layers):
                        gen = model.generators[li]; disc = model.discriminators[li]
                        with torch.no_grad():
                            V = layer.prior_sample().detach()
                            eps = torch.randn(model.M, layer.d_out, device=V.device)
                            U = gen(eps).detach()
                        T_real = disc(V)
                        T_fake = disc(U)
                        disc_loss = disc_loss + \
                            F.binary_cross_entropy_with_logits(T_real, torch.zeros_like(T_real)) + \
                            F.binary_cross_entropy_with_logits(T_fake, torch.ones_like(T_fake))
                    if torch.isfinite(disc_loss):
                        disc_loss.backward()
                        torch.nn.utils.clip_grad_norm_(disc_params, args.grad_clip)
                        opt_disc.step()
                # ---- (B) Generator + model update ----
                opt_gen.zero_grad()
                Us = []
                kl_surrogate = 0.0
                for li, layer in enumerate(model.layers):
                    gen = model.generators[li]; disc = model.discriminators[li]
                    eps = torch.randn(model.M, layer.d_out, device=x_b.device)
                    U = gen(eps)
                    Us.append(U)
                    kl_surrogate = kl_surrogate + disc(U).mean()
                f = model.dgp_forward(x_b, Us)
                noise_var = (model.log_noise.exp() ** 2).clamp_min(1e-6)
                ll = -0.5 * ((y_b - f.squeeze(-1)) ** 2 / noise_var +
                             torch.log(2 * math.pi * noise_var))
                log_lik = ll.sum()
                elbo_data = (N_total / x_b.shape[0]) * log_lik
                gen_loss = -elbo_data / N_total + kl_surrogate
                if not torch.isfinite(gen_loss):
                    global_step += 1; continue
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_params, args.grad_clip)
                opt_gen.step()
                if global_step % args.eval_every == 0:
                    rmse, nll = eval_metrics(model, test_loader, test_y, device,
                                             num_samples=args.eval_samples)
                    rows.append([global_step, epoch, float(gen_loss.detach().cpu()),
                                 float(disc_loss.detach().cpu()) if torch.is_tensor(disc_loss) else 0.0,
                                 rmse, nll])
                global_step += 1
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in tqdm.tqdm(range(args.epochs), desc="Epoch"):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss, log_lik_mean, kl = model.loss_and_metrics(
                    x_b, y_b, N_total=N_total, mc_samples=args.mc_samples
                )
                if args.fm_weight > 0:
                    fm = model.fm_loss(x_b, y_b, N_total=N_total, fm_samples=args.fm_samples)
                    loss = loss + args.fm_weight * fm
                if args.shortcut_weight > 0 and epoch >= args.shortcut_warmup_epochs:
                    sc = model.shortcut_loss(n_samples=args.shortcut_samples)
                    loss = loss + args.shortcut_weight * sc
                if args.dsm_weight > 0 and args.variant in ('score', 'dbvi-s'):
                    dsm = model.dsm_loss(n_samples=args.dsm_samples)
                    loss = loss + args.dsm_weight * dsm
                if not torch.isfinite(loss):
                    print(f"[Warn] non-finite loss at step {global_step}; skipping")
                    optimizer.zero_grad(); global_step += 1; continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                if global_step % args.eval_every == 0:
                    rmse, nll = eval_metrics(model, test_loader, test_y, device,
                                             num_samples=args.eval_samples)
                    rows.append([global_step, epoch, float(loss.detach().cpu()),
                                 float(kl.detach().cpu()), rmse, nll])
                global_step += 1

    rmse, nll = eval_metrics(model, test_loader, test_y, device,
                             num_samples=args.eval_samples)
    rows.append([global_step, args.epochs, float('nan'), float('nan'), rmse, nll])
    print(f"FINAL RMSE: {rmse:.6f}  NLL: {nll:.6f}  wall: {time.time()-t0:.1f}s")
    # Few-step inference sweep (if requested) — only meaningful for fbvi/dbvi
    if args.eval_steps_list and args.variant in ('fbvi', 'dbvi', 'score', 'dbvi-s', 'fbvi-bridge'):
        step_list = [int(s) for s in args.eval_steps_list.split(',') if s.strip()]
        print(f"[Info] few-step inference sweep:")
        for ns in step_list:
            r, n = eval_metrics(model, test_loader, test_y, device,
                                num_samples=args.eval_samples, infer_steps=ns)
            print(f"  STEPS={ns:3d}  RMSE={r:.6f}  NLL={n:.6f}")

    if os.path.dirname(args.log_csv):
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    with open(args.log_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step','epoch','loss','kl','rmse','nll'])
        w.writerows(rows)
    print(f"[Info] log saved: {args.log_csv}")


if __name__ == '__main__':
    main()
