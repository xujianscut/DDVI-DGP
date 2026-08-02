"""
SOLVE-GP (Shi, Titsias & Mnih, AISTATS 2020) vs DSVI vs DDVI on UCI regression,
under a matched per-iteration compute budget.

Reviewer 2 asked for "a compute-controlled comparison with a decoupled inducing
points approach such as SOLVE-GP where you can also enhance the posterior
approximation by spending more flops". The point of the comparison is that both
methods can buy a better posterior with more computation, so the honest
question is which one buys more per FLOP.

SOLVE-GP places a second set of inducing points a (size M2) in the ORTHOGONAL
complement of span(Z):
    C_vv = K_aa - K_za^T K_zz^{-1} K_za
and gives u and v independent Gaussian variational factors. Predictions add the
in-span and perpendicular contributions:
    mean = K_xz K_zz^{-1} m_u  +  (K_xa - K_xz K_zz^{-1} K_za) C_vv^{-1} m_v
This is the whitened form, matching the reference implementation.

DDVI instead keeps a single set of M inducing points and spends its extra
compute on the diffusion chain that parameterizes q(u).
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
def load_uci(name, root):
    f = {'boston': 'boston.csv', 'energy': 'energy.csv',
         'concrete': 'Concrete_Data.xls', 'yacht': 'yacht.data',
         'qsar': 'qsar_fish.csv', 'power': 'power.xlsx'}[name]
    path = os.path.join(root, f)
    if name == 'boston':
        d = pd.read_csv(path, sep=',', header=None, skiprows=45,
                        engine='python').values
    elif name == 'energy':
        d = pd.read_csv(path).values
        return np.asarray(d[:, :8], np.float64), np.asarray(d[:, 8:9], np.float64)
    elif name in ('concrete', 'power'):
        d = pd.read_excel(path).values
    elif name == 'yacht':
        d = pd.read_csv(path, sep=r'\s+', header=None,
                        engine='python').dropna().values
    else:
        d = pd.read_csv(path, sep=';', header=None).values
    d = np.asarray(d, dtype=np.float64)
    d = d[~np.isnan(d).any(axis=1)]
    return d[:, :-1], d[:, -1:]


class RBF(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.log_ls = nn.Parameter(torch.zeros(d_in))
        self.log_var = nn.Parameter(torch.zeros(()))

    def forward(self, A, B):
        a = A / torch.exp(self.log_ls)
        b = B / torch.exp(self.log_ls)
        d = (a * a).sum(-1, keepdim=True) + (b * b).sum(-1) - 2 * a @ b.t()
        return torch.exp(self.log_var) * torch.exp(-0.5 * d.clamp_min(0))


def jitter(K, eps=1e-5):
    return K + eps * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)


# --------------------------------------------------------------------------
class SparseGP(nn.Module):
    """method: dsvi | solvegp | ddvi   (all whitened)"""

    def __init__(self, d_in, Z_init, method='dsvi', A_init=None,
                 n_steps=10, hidden=128):
        super().__init__()
        self.method = method
        self.kern = RBF(d_in)
        self.Z = nn.Parameter(Z_init.clone())
        M = Z_init.shape[0]
        self.M = M
        self.log_noise = nn.Parameter(torch.tensor(-1.0))

        if method == 'ddvi':
            self.score = nn.Sequential(
                nn.Linear(M + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, M))
            nn.init.zeros_(self.score[-1].weight)
            nn.init.zeros_(self.score[-1].bias)
            self.u0 = nn.Parameter(torch.zeros(M))
            self.log_scale = nn.Parameter(torch.tensor(-1.0))
            self.n_steps = n_steps
            self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        else:
            self.qu_mu = nn.Parameter(torch.zeros(M))
            self.qu_L = nn.Parameter(torch.eye(M))
            if method == 'solvegp':
                self.A = nn.Parameter(A_init.clone())
                M2 = A_init.shape[0]
                self.M2 = M2
                self.qv_mu = nn.Parameter(torch.zeros(M2))
                self.qv_L = nn.Parameter(torch.eye(M2))

    # ---- q(u) --------------------------------------------------------------
    def sample_u(self):
        if self.method == 'ddvi':
            dev = self.betas.device
            u = torch.randn(1, self.M, device=dev)
            l1 = u.new_zeros(())
            for k in range(self.n_steps - 1, -1, -1):
                b = self.betas[k]
                t = (k + 1) / self.n_steps
                inp = torch.cat([u, u.new_full((1, 1), float(t))], -1)
                s = self.score(inp)
                l1 = l1 + b / (2 * (1 - b)) * ((u + s) ** 2).sum()
                mean = (u + b * s) / torch.sqrt(1 - b)
                u = mean + torch.sqrt(b) * torch.randn_like(mean) if k > 0 else mean
            return self.u0 + torch.exp(self.log_scale) * u.squeeze(0), l1
        L = torch.tril(self.qu_L)
        u = self.qu_mu + L @ torch.randn(self.M, device=self.qu_mu.device)
        kl = 0.5 * ((L ** 2).sum() + (self.qu_mu ** 2).sum() - self.M
                    - 2 * torch.log(torch.diagonal(L).abs() + 1e-8).sum())
        return u, kl

    def forward(self, x):
        Kzz = jitter(self.kern(self.Z, self.Z))
        Lz = torch.linalg.cholesky(Kzz)
        Kxz = self.kern(x, self.Z)
        Az = torch.linalg.solve_triangular(Lz, Kxz.t(), upper=False).t()  # whitened
        u, reg = self.sample_u()
        mean = Az @ u
        kxx = torch.exp(self.kern.log_var)
        var = (kxx - (Az ** 2).sum(-1)).clamp_min(1e-8)

        if self.method == 'solvegp':
            Kaa = jitter(self.kern(self.A, self.A))
            Kza = self.kern(self.Z, self.A)
            Lz_inv_Kza = torch.linalg.solve_triangular(Lz, Kza, upper=False)
            Cvv = jitter(Kaa - Lz_inv_Kza.t() @ Lz_inv_Kza)
            Lv = torch.linalg.cholesky(Cvv)
            Kxa = self.kern(x, self.A)
            Cxa = Kxa - Az @ Lz_inv_Kza                       # perp cross-cov
            Av = torch.linalg.solve_triangular(Lv, Cxa.t(), upper=False).t()
            Lq = torch.tril(self.qv_L)
            v = self.qv_mu + Lq @ torch.randn(self.M2, device=x.device)
            mean = mean + Av @ v
            var = (var - (Av ** 2).sum(-1)).clamp_min(1e-8)
            reg = reg + 0.5 * ((Lq ** 2).sum() + (self.qv_mu ** 2).sum()
                               - self.M2
                               - 2 * torch.log(torch.diagonal(Lq).abs() + 1e-8).sum())
        return mean, var, reg


class DeepGP(nn.Module):
    """Doubly-stochastic DGP (Salimbeni & Deisenroth, 2017) whose every layer
    carries the chosen variational family. Matches the manuscript's regression
    setup, where L ranges from 2 to 5."""

    def __init__(self, d_in, Z_init, method, A_init, n_steps, hidden, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [d_in] + [d_in] * (n_layers - 1)
        for l in range(n_layers):
            Zl = Z_init.clone() if l == 0 else Z_init[:, :dims[l]].clone()
            Al = None
            if method == 'solvegp':
                Al = A_init.clone() if l == 0 else A_init[:, :dims[l]].clone()
            self.layers.append(SparseGP(dims[l], Zl, method, Al, n_steps, hidden))
        self.log_noise = nn.Parameter(torch.tensor(-1.0))
        self.n_layers = n_layers

    def forward(self, x):
        reg_tot = x.new_zeros(())
        h = x
        for i, layer in enumerate(self.layers):
            mean, var, reg = layer(h)
            reg_tot = reg_tot + reg
            if i < self.n_layers - 1:
                # sample the intermediate layer, keep the input dimension
                f = mean.unsqueeze(-1) + var.unsqueeze(-1).sqrt() * torch.randn_like(
                    mean.unsqueeze(-1))
                h = h + f                      # skip connection (DSVI default)
        return mean, var, reg_tot


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, Y = load_uci(args.dataset, args.data)
    n = len(X)
    idx = np.random.RandomState(args.seed).permutation(n)
    ntr = int(n * 0.9)
    tr, te = idx[:ntr], idx[ntr:]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    ym, ys = Y[tr].mean(), Y[tr].std() + 1e-8
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32, device=dev)
    Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32, device=dev).squeeze(-1)
    xtr, ytr, xte, yte = Xt[tr], Yt[tr], Xt[te], Yt[te]

    perm = torch.randperm(ntr)[:args.M]
    Z0 = xtr[perm].clone()
    A0 = xtr[torch.randperm(ntr)[:args.M2]].clone() if args.method == 'solvegp' else None
    if args.layers > 1:
        model = DeepGP(X.shape[1], Z0, args.method, A0, args.n_steps,
                       args.hidden, args.layers).to(dev)
    else:
        model = SparseGP(X.shape[1], Z0, args.method, A0, args.n_steps,
                         args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for it in range(args.iters):
        opt.zero_grad()
        b = torch.randperm(ntr, device=dev)[:min(args.batch, ntr)]
        mean, var, reg = model(xtr[b])
        noise = torch.exp(model.log_noise) ** 2
        ll = (-0.5 * ((ytr[b] - mean) ** 2 + var) / noise
              - 0.5 * torch.log(2 * math.pi * noise)).sum()
        loss = -(ll * (ntr / len(b)) - reg) / ntr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    wall = time.time() - t0

    with torch.no_grad():
        ms, vs = [], []
        for _ in range(args.eval_samples):
            m, v, _ = model(xte)
            ms.append(m); vs.append(v)
        m = torch.stack(ms); v = torch.stack(vs)
        noise = torch.exp(model.log_noise) ** 2
        pm = m.mean(0)
        rmse = (((pm - yte) * ys) ** 2).mean().sqrt().item()
        lp = -0.5 * ((yte - m) ** 2) / (v + noise) \
             - 0.5 * torch.log(2 * math.pi * (v + noise)) - math.log(ys)
        nll = -(torch.logsumexp(lp, 0) - math.log(m.shape[0])).mean().item()
    tag = args.method + (f'-M{args.M}+{args.M2}' if args.method == 'solvegp'
                         else f'-M{args.M}') + f'-L{args.layers}'
    print(f'FINAL[{tag}] {args.dataset} seed={args.seed} rmse={rmse:.4f} '
          f'nll={nll:.4f} wall={wall:.1f}s', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='dsvi',
                   choices=['dsvi', 'solvegp', 'ddvi'])
    p.add_argument('--dataset', default='boston')
    p.add_argument('--data', default='../data')
    p.add_argument('--M', type=int, default=64)
    p.add_argument('--M2', type=int, default=64)
    p.add_argument('--n_steps', type=int, default=10)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--eval_samples', type=int, default=32)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    main_args = p.parse_args()
    run(main_args)


if __name__ == '__main__':
    main()
