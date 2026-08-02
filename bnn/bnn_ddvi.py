"""
DDVI for Bayesian neural networks with inducing weights.

Reimplementation from the paper (BNN half of the TPAMI submission), since the
original BNN code is no longer available. Structure follows the manuscript:

  * Inducing-weight prior  (Ritter et al., 2021):
        [ W   U_c ]
        [ U_r  U  ]  ~  MN(0, Sigma_r, Sigma_c),
    with Sigma_r = L_r L_r^T,  Sigma_c = L_c L_c^T and block-Cholesky factors
        L_r = [[sigma_r I_{d_out}, 0], [Z_r, D_r]],
        L_c = [[sigma_c I_{d_in},  0], [Z_c, D_c]].
    Marginal prior  p(U) = MN(0, Psi_r, Psi_c),
        Psi_r = Z_r Z_r^T + D_r^2,   Psi_c = Z_c Z_c^T + D_c^2.

  * Conditional  p(vec W | vec U) = N(mu, Sigma) with
        mu    = sigma_r sigma_c  Z_r^T Psi_r^{-1} U Psi_c^{-1} Z_c
        Sigma = [sigma_c^2 (I - Z_c^T Psi_c^{-1} Z_c)]
                     kron [sigma_r^2 (I - Z_r^T Psi_r^{-1} Z_r)]

    NOTE ON TWO CORRECTIONS relative to the submitted manuscript. Both were
    found while reimplementing and both are needed for the shapes to work out:
      (a) mu carries transposes on Z_r and Z_c. As printed in the manuscript
          (Z_r Psi_r^{-1} U Psi_c^{-1} Z_c^T) the product is not even defined,
          since Z_r is M_out x d_out. Reviewer 2 flagged this.
      (b) Sigma is a Kronecker product of two (I - .) factors, NOT
          "I - (A kron B)" as printed. This follows from the standard Gaussian
          conditioning identity applied blockwise to Sigma_r and Sigma_c and is
          not something a reviewer raised; the manuscript should be corrected.

  * q(U) is defined implicitly as the terminal state of a reverse-time VP SDE
    driven by a learned score network (DDVI), exactly as in the DGP code.

  * Objective (manuscript Eq. 17), per layer l:
        l(phi) = E_q[log p(U)] + (N/B) E[log p(y_B | W)]
                 - l_1(phi) - E_q[log p_fix(U_T)]
    Under the variance-preserving schedule with p_fix = N(0, I) we have
    kappa_t == 1 and p_T^Bri = p_fix, so KL(p_fix || p_T^Bri) = 0 and
        l_1(phi) = 1/2 \int_0^1 beta(t) || U_t + s_phi(U_t, t) ||^2 dt,
    accumulated along the sampled reverse trajectory.

Baselines implemented in the same harness (so that the only thing that varies
is the family used for q(U)):
    ffg-u       fully-factorised Gaussian in inducing space
    fcg-u       per-layer full-covariance Gaussian in inducing space
    ensemble-u  K delta measures in inducing space
    ddvi-u      ours
    map         no inducing structure, point estimate (sanity reference)
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# score network for q(U)
# --------------------------------------------------------------------------
class ScoreField(nn.Module):
    """s_phi(U, t) ~= grad log p_t(U) under VP noising."""

    def __init__(self, m_out, m_in, hidden=128):
        super().__init__()
        dim = m_out * m_in
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.m_out, self.m_in = m_out, m_in

    def forward(self, U, t):
        flat = U.reshape(-1)
        t_in = flat.new_full((1,), float(t))
        out = self.net(torch.cat([flat, t_in], dim=-1))
        return out.reshape(self.m_out, self.m_in)


# --------------------------------------------------------------------------
# one layer with inducing weights
# --------------------------------------------------------------------------
class InducingWeightLayer(nn.Module):
    def __init__(self, d_in, d_out, m_in, m_out, variant='ddvi-u',
                 hidden=128, n_ensemble=5):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.m_in = min(m_in, d_in)
        self.m_out = min(m_out, d_out)
        self.variant = variant

        # ---- prior parameters (all learnable, per reviewer 1 comment 2) ----
        self.Z_r = nn.Parameter(torch.randn(self.m_out, d_out) / math.sqrt(d_out))
        self.Z_c = nn.Parameter(torch.randn(self.m_in, d_in) / math.sqrt(d_in))
        # D_r, D_c must start SMALL relative to Z Z^T, otherwise Psi ~ I and the
        # inducing weights explain almost none of the variance of W: with
        # gamma = 0 one gets tr(Z^T Psi^{-1} Z) / d_out ~ 16%, so q(U) barely
        # moves the predictive distribution and every choice of q looks alike.
        self.gamma_r = nn.Parameter(torch.full((self.m_out,), -3.0))
        self.gamma_c = nn.Parameter(torch.full((self.m_in,), -3.0))
        self.log_sigma_r = nn.Parameter(torch.tensor(0.5 * math.log(1.0 / d_in)))
        self.log_sigma_c = nn.Parameter(torch.tensor(0.0))

        self.bias = nn.Parameter(torch.zeros(d_out))
        self.det_W = False

        # ---- variational family for q(U) ----
        if variant == 'ddvi-u':
            self.score = ScoreField(self.m_out, self.m_in, hidden)
        # IMPORTANT: q(U) must be initialised at the scale of the prior
        # p(U) = MN(0, Psi_r, Psi_c), whose entries have unit-ish variance.
        # Initialising q_mean at 0 with a small std gives ||U|| ~ 1.6 instead
        # of ~M, so mu(U) -- and hence W -- comes out an order of magnitude
        # below a standard weight init, the forward signal decays by 10^3
        # across three layers, and the network can only fit its output bias.
        # That failure is invisible in the loss (it just plateaus) and makes
        # every choice of q(U) look identical.
        elif variant == 'ffg-u':
            self.q_mean = nn.Parameter(torch.randn(self.m_out, self.m_in))
            self.q_logstd = nn.Parameter(torch.zeros(self.m_out, self.m_in))
        elif variant == 'fcg-u':
            dim = self.m_out * self.m_in
            self.q_mean = nn.Parameter(torch.randn(dim))
            self.q_ltri = nn.Parameter(torch.eye(dim))
        elif variant == 'ensemble-u':
            self.q_particles = nn.Parameter(
                torch.randn(n_ensemble, self.m_out, self.m_in))
            self._ens_ptr = 0
        elif variant == 'map':
            self.W_map = nn.Parameter(torch.randn(d_out, d_in) / math.sqrt(d_in))
        else:
            raise ValueError(variant)

    # ---- prior quantities -------------------------------------------------
    def psi(self):
        D_r2 = torch.exp(2.0 * self.gamma_r)
        D_c2 = torch.exp(2.0 * self.gamma_c)
        Psi_r = self.Z_r @ self.Z_r.t() + torch.diag(D_r2)
        Psi_c = self.Z_c @ self.Z_c.t() + torch.diag(D_c2)
        eye_r = torch.eye(self.m_out, device=Psi_r.device, dtype=Psi_r.dtype)
        eye_c = torch.eye(self.m_in, device=Psi_c.device, dtype=Psi_c.dtype)
        return Psi_r + 1e-5 * eye_r, Psi_c + 1e-5 * eye_c

    def log_prior_U(self, V):
        """log p(V) = log N(V | 0, I): the prior of the WHITENED variable.

        This is identical to log p_fix, so the two terms of Eq. (17) cancel and
        the bound reduces to E[log p(y|W)] - l_1(phi)."""
        return -0.5 * (V ** 2).sum() \
            - 0.5 * self.m_out * self.m_in * math.log(2 * math.pi)

    def _unused_log_prior_U(self, U):
        """log MN(U | 0, Psi_r, Psi_c)."""
        Psi_r, Psi_c = self.psi()
        Lr = torch.linalg.cholesky(Psi_r)
        Lc = torch.linalg.cholesky(Psi_c)
        # tr(Psi_r^{-1} U Psi_c^{-1} U^T)
        A = torch.cholesky_solve(U, Lr)                     # Psi_r^{-1} U
        B = torch.cholesky_solve(A.t(), Lc).t()             # Psi_r^{-1} U Psi_c^{-1}
        quad = (B * U).sum()
        logdet_r = 2.0 * torch.log(torch.diagonal(Lr)).sum()
        logdet_c = 2.0 * torch.log(torch.diagonal(Lc)).sum()
        const = self.m_out * self.m_in * math.log(2 * math.pi)
        return -0.5 * (quad + self.m_in * logdet_r + self.m_out * logdet_c + const)

    def whitening_factors(self, Psi_r, Psi_c):
        """A_r = L_r^{-1} Z_r,  A_c = L_c^{-1} Z_c  with Psi = L L^T.

        Working in the whitened variable V, defined by U = L_r V L_c^T, is what
        makes this parameterization numerically usable. Two things follow:

        (1) Psi^{-1} L = L^{-T}, so
                mu = sigma_r sigma_c A_r^T V A_c,
            and since Psi = Z Z^T + D^2 >= Z Z^T we have ||A||_2 <= 1. The
            entries of mu are therefore bounded by sigma_r sigma_c, i.e. a
            standard weight-init scale, with no ill-conditioned inverse. In the
            unwhitened form, a small D makes Psi nearly singular (for a
            [32, 50] factor the Marchenko-Pastur lower edge puts the smallest
            eigenvalue near 0.04) and Psi^{-1} amplifies mu by ~25x, which blows
            up the loss and gets the gradients annihilated by clipping.

        (2) The prior becomes p(V) = N(0, I), which is exactly p_fix. Hence
            E[log p(V)] - E[log p_fix] = 0 identically and the bound of
            Eq. (17) collapses to  E[log p(y|W)] - l_1(phi).
        """
        Lr = torch.linalg.cholesky(Psi_r)
        Lc = torch.linalg.cholesky(Psi_c)
        A_r = torch.linalg.solve_triangular(Lr, self.Z_r, upper=False)
        A_c = torch.linalg.solve_triangular(Lc, self.Z_c, upper=False)
        return A_r, A_c

    def _cond_mean(self, V, Psi_r, Psi_c):
        """mu = sigma_r sigma_c A_r^T V A_c  ->  [d_out, d_in], V whitened."""
        sr = torch.exp(self.log_sigma_r)
        sc = torch.exp(self.log_sigma_c)
        A_r, A_c = self.whitening_factors(Psi_r, Psi_c)
        return sr * sc * (A_r.t() @ V @ A_c)

    def sample_W(self, V, deterministic=False):
        """Matheron's rule sample of W ~ p(W | U).

        deterministic=True returns the conditional mean mu(U) with no noise --
        a diagnostic to separate "q(U) carries no signal" from "the conditional
        noise swamps the signal".

        Draw (W~, U~) jointly from the prior via [W~; U~] = L_r E L_c^T, then
        W = mu(U) + W~ - mu(U~).  This avoids any d x d factorisation.
        """
        Psi_r, Psi_c = self.psi()
        sr = torch.exp(self.log_sigma_r)
        sc = torch.exp(self.log_sigma_c)
        dev, dt = self.Z_r.device, self.Z_r.dtype
        if deterministic:
            return self._cond_mean(V, Psi_r, Psi_c)

        E11 = torch.randn(self.d_out, self.d_in, device=dev, dtype=dt)
        E12 = torch.randn(self.d_out, self.m_in, device=dev, dtype=dt)
        E21 = torch.randn(self.m_out, self.d_in, device=dev, dtype=dt)
        E22 = torch.randn(self.m_out, self.m_in, device=dev, dtype=dt)
        D_r = torch.diag(torch.exp(self.gamma_r))
        D_c = torch.diag(torch.exp(self.gamma_c))

        W_t = sr * sc * E11
        U_t = (self.Z_r @ E11 + D_r @ E21) @ self.Z_c.t() \
            + (self.Z_r @ E12 + D_r @ E22) @ D_c
        # whiten the prior draw so that it lives in the same space as V
        Lr = torch.linalg.cholesky(Psi_r)
        Lc = torch.linalg.cholesky(Psi_c)
        V_t = torch.linalg.solve_triangular(Lr, U_t, upper=False)
        V_t = torch.linalg.solve_triangular(Lc, V_t.t(), upper=False).t()

        return self._cond_mean(V, Psi_r, Psi_c) + W_t - self._cond_mean(V_t, Psi_r, Psi_c)

    # ---- q(U) sampling ----------------------------------------------------
    def sample_U(self, n_steps=10, beta_min=1e-4, beta_max=0.02):
        """Returns (U, l1) where l1 is the score-matching penalty of Eq. (17).

        Discrete-time formulation (DDPM ancestral sampling). Writing the
        forward chain as q(U_k | U_{k-1}) = N(sqrt(1-b_k) U_{k-1}, b_k I), the
        reverse kernel driven by the learned score is
            p_phi(U_{k-1} | U_k) = N( (U_k + b_k s_phi(U_k,k)) / sqrt(1-b_k),
                                       b_k I ),
        while the bridge process has the analytic score -U_k / kappa_k with
        kappa_k = 1 under the variance-preserving schedule. The per-step KL
        between the two reverse kernels is therefore
            b_k / (2 (1 - b_k)) * || U_k + s_phi(U_k, k) ||^2,
        and l_1 is its sum over k -- the exact discrete analogue of
        1/2 \\int beta(t) ||U_t + s_phi||^2 dt. This form is numerically stable
        for the step counts we can afford, unlike an Euler discretisation of
        the reverse SDE with a large beta_max.
        """
        dev, dt_ = self.Z_r.device, self.Z_r.dtype
        if self.variant == 'ddvi-u':
            betas = torch.linspace(beta_min, beta_max, n_steps,
                                   device=dev, dtype=dt_)
            U = torch.randn(self.m_out, self.m_in, device=dev, dtype=dt_)
            l1 = U.new_zeros(())
            for k in range(n_steps - 1, -1, -1):
                b = betas[k]
                t = (k + 1) / n_steps
                s = self.score(U, t)
                l1 = l1 + b / (2.0 * (1.0 - b)) * ((U + s) ** 2).sum()
                mean = (U + b * s) / torch.sqrt(1.0 - b)
                U = mean + torch.sqrt(b) * torch.randn_like(U) if k > 0 else mean
            return U, l1
        if self.variant == 'ffg-u':
            eps = torch.randn(self.m_out, self.m_in, device=dev, dtype=dt_)
            U = self.q_mean + torch.exp(self.q_logstd) * eps
            ent = (self.q_logstd + 0.5 * math.log(2 * math.pi * math.e)).sum()
            return U, -ent                      # -H[q] plays the role of l1
        if self.variant == 'fcg-u':
            dim = self.m_out * self.m_in
            L = torch.tril(self.q_ltri)
            eps = torch.randn(dim, device=dev, dtype=dt_)
            u = self.q_mean + L @ eps
            ent = torch.log(torch.abs(torch.diagonal(L)) + 1e-8).sum() \
                + 0.5 * dim * math.log(2 * math.pi * math.e)
            return u.reshape(self.m_out, self.m_in), -ent
        if self.variant == 'ensemble-u':
            # Deep ensemble in inducing space: every particle must receive
            # gradient. We cycle deterministically through the particles so
            # that over an epoch each one is trained, rather than sampling one
            # at random (which trains them unevenly and collapses diversity).
            idx = self._ens_ptr % self.q_particles.shape[0]
            self._ens_ptr += 1
            return self.q_particles[idx], self.q_particles.new_zeros(())
        raise ValueError(self.variant)

    def forward(self, x, n_steps=10):
        if self.variant == 'map':
            z = self.Z_r.new_zeros(())
            return F.linear(x, self.W_map, self.bias), z, z, z
        U, l1 = self.sample_U(n_steps=n_steps)
        W = self.sample_W(U, deterministic=self.det_W)
        logp_U = self.log_prior_U(U)
        # The -E_q[log p_fix(U_T)] term belongs to the DDVI bound of Eq. (17)
        # ONLY. For an explicitly parameterized q (FFG/FCG/ensemble) the bound
        # is the ordinary ELBO  E[ll] + E[log p(U)] + H[q], and adding
        # -log p_fix = +||U||^2/2 would reward driving ||U|| to infinity --
        # which is exactly what made the FFG-U run diverge.
        if self.variant == 'ddvi-u':
            logp_fix = -0.5 * (U ** 2).sum() \
                - 0.5 * self.m_out * self.m_in * math.log(2 * math.pi)
        else:
            logp_fix = U.new_zeros(())
        return F.linear(x, W, self.bias), logp_U, l1, logp_fix


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------
class InducingWeightMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=2, m=16,
                 variant='ddvi-u', hidden=128, n_ensemble=5):
        super().__init__()
        dims = [d_in] + [d_hidden] * n_layers + [d_out]
        self.layers = nn.ModuleList([
            InducingWeightLayer(dims[i], dims[i + 1], m, m,
                                variant=variant, hidden=hidden,
                                n_ensemble=n_ensemble)
            for i in range(len(dims) - 1)
        ])
        self.log_noise = nn.Parameter(torch.tensor(-2.0))
        self.variant = variant
        # diagnostic knobs: w_prior scales (log p(U) - log p_fix), w_l1 scales
        # the score-matching penalty. Both are 1.0 for the bound of Eq. (17).
        self.w_prior = 1.0
        self._det_W = False
        self.w_l1 = 1.0
        self._last_terms = (0.0, 0.0, 0.0, 0.0)

    def forward(self, x, n_steps=10):
        logp_U = x.new_zeros(())
        l1 = x.new_zeros(())
        logp_fix = x.new_zeros(())
        h = x
        for i, layer in enumerate(self.layers):
            h, a, b, c = layer(h, n_steps=n_steps)
            logp_U, l1, logp_fix = logp_U + a, l1 + b, logp_fix + c
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h, logp_U, l1, logp_fix

    def loss(self, x, y, n_data, n_steps=10, mc=1):
        """Negative of the bound in Eq. (17), for regression."""
        total = 0.0
        for _ in range(mc):
            pred, logp_U, l1, logp_fix = self.forward(x, n_steps=n_steps)
            noise = torch.exp(self.log_noise)
            ll = (-0.5 * ((y - pred) ** 2) / noise ** 2
                  - torch.log(noise) - 0.5 * math.log(2 * math.pi)).sum()
            scale = n_data / x.shape[0]
            bound = scale * ll + self.w_prior * logp_U - self.w_l1 * l1 \
                - self.w_prior * logp_fix
            total = total + (-bound / n_data)
            self._last_terms = (float(scale * ll), float(logp_U),
                                float(l1), float(logp_fix))
        return total / mc

    @torch.no_grad()
    def predict(self, x, n_samples=32, n_steps=10):
        preds = torch.stack([self.forward(x, n_steps=n_steps)[0]
                             for _ in range(n_samples)])
        return preds.mean(0), preds.std(0), preds


# --------------------------------------------------------------------------
# wheel bandit (Riquelme et al., 2018), as used in the paper
# --------------------------------------------------------------------------
def wheel_contexts(n, delta, rng):
    """Sample n contexts uniformly from the unit ball, with their optimal
    actions and the full reward vector.

    Region rule (Riquelme et al., 2018):
      * ||x|| <= delta : action 1 is optimal, mean reward mu_high is NOT
        available; all of actions 2..5 pay mu_low.
      * ||x|| >  delta : the action indexed by the quadrant of x pays mu_high;
        action 1 still pays mu_med, the remaining three pay mu_low.
    """
    MU_HIGH, MU_MED, MU_LOW, SIGMA = 50.0, 1.2, 1.0, 0.01
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = np.sqrt(rng.uniform(0, 1, size=n))
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    means = np.full((n, 5), MU_LOW)
    means[:, 0] = MU_MED
    norms = np.linalg.norm(x, axis=1)
    outer = norms > delta
    quad = np.where((x[:, 0] > 0) & (x[:, 1] > 0), 1,
           np.where((x[:, 0] > 0) & (x[:, 1] <= 0), 2,
           np.where((x[:, 0] <= 0) & (x[:, 1] > 0), 3, 4)))
    means[outer, quad[outer]] = MU_HIGH

    rewards = means + rng.randn(n, 5) * SIGMA
    return (x.astype(np.float32), rewards.astype(np.float32),
            means.astype(np.float32))


def run_wheel(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_all, r_all, mu_all = wheel_contexts(args.n_steps_bandit, args.delta, rng)
    x_all_t = torch.tensor(x_all, device=dev)

    # one head per action
    model = InducingWeightMLP(2, args.width, 5, n_layers=args.layers,
                              m=args.m, variant=args.variant,
                              hidden=args.hidden,
                              n_ensemble=args.n_ensemble).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    seen_x, seen_a, seen_r = [], [], []
    cum_reward, cum_opt = 0.0, 0.0
    simple_window = []
    t0 = time.time()

    for t in range(args.n_steps_bandit):
        xt = x_all_t[t:t + 1]
        # Thompson sampling: ONE posterior draw, act greedily under it
        with torch.no_grad():
            q, _, _, _ = model.forward(xt, n_steps=args.flow_steps)
        a = int(q.argmax().item())

        cum_reward += float(r_all[t, a])
        cum_opt += float(mu_all[t].max())
        if t >= args.n_steps_bandit - 500:
            simple_window.append(float(mu_all[t].max() - mu_all[t, a]))

        seen_x.append(x_all[t]); seen_a.append(a); seen_r.append(r_all[t, a])

        # periodic retraining on the replay buffer
        if (t + 1) % args.train_every == 0 and len(seen_x) >= args.batch_size:
            bx = torch.tensor(np.array(seen_x), device=dev)
            ba = torch.tensor(np.array(seen_a), device=dev, dtype=torch.long)
            br = torch.tensor(np.array(seen_r), device=dev)
            for _ in range(args.train_iters):
                idx = torch.randint(len(seen_x), (min(args.batch_size,
                                                      len(seen_x)),),
                                    device=dev)
                opt.zero_grad()
                pred, logp_U, l1, logp_fix = model.forward(
                    bx[idx], n_steps=args.flow_steps)
                pred_a = pred.gather(1, ba[idx].unsqueeze(1)).squeeze(1)
                noise = torch.exp(model.log_noise)
                ll = (-0.5 * ((br[idx] - pred_a) ** 2) / noise ** 2
                      - torch.log(noise) - 0.5 * math.log(2 * math.pi)).sum()
                scale = len(seen_x) / len(idx)
                bound = scale * ll + logp_U - l1 - logp_fix
                (-bound / max(len(seen_x), 1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()

    wall = time.time() - t0
    # normalised against a uniformly random agent, as in the paper
    rand_reward = float(mu_all.mean(axis=1).sum())
    opt_reward = float(mu_all.max(axis=1).sum())
    cum_regret = 100.0 * (opt_reward - cum_reward) / max(opt_reward - rand_reward, 1e-9)
    simple_regret = float(np.mean(simple_window)) if simple_window else float('nan')

    print(f'[{args.variant}] delta={args.delta} seed={args.seed} '
          f'cum_regret={cum_regret:.2f} simple_regret={simple_regret:.2f} '
          f'wall={wall:.1f}s', flush=True)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        np.savez(os.path.join(args.save_dir,
                              f'wheel_{args.variant}_d{args.delta}_s{args.seed}.npz'),
                 cum_regret=cum_regret, simple_regret=simple_regret, wall=wall)
    return cum_regret, simple_regret


# --------------------------------------------------------------------------
# toy 1-D task of Foong et al. (2019), as used in the paper
# --------------------------------------------------------------------------
def make_toy(n=100, seed=0):
    g = np.random.RandomState(seed)
    x1 = g.uniform(-1.0, -0.7, size=n // 2)
    x2 = g.uniform(0.5, 1.0, size=n // 2)
    x = np.concatenate([x1, x2])
    y = np.cos(4.0 * x + 0.8) + g.randn(len(x)) * 0.1     # N(cos(4x+0.8), 0.01)
    return x[:, None].astype(np.float32), y[:, None].astype(np.float32)


def run_toy(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    x, y = make_toy(args.n_data, seed=args.seed)
    xt = torch.tensor(x, device=dev)
    yt = torch.tensor(y, device=dev)

    model = InducingWeightMLP(1, args.width, 1, n_layers=args.layers,
                              m=args.m, variant=args.variant,
                              hidden=args.hidden,
                              n_ensemble=args.n_ensemble).to(dev)
    model.w_prior, model.w_l1 = args.w_prior, args.w_l1
    for _l in model.layers:
        _l.det_W = bool(args.det_W)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for it in range(args.iters):
        opt.zero_grad()
        loss = model.loss(xt, yt, n_data=len(x), n_steps=args.flow_steps,
                          mc=args.mc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if (it + 1) % max(1, args.iters // 5) == 0:
            ll_, pu_, l1_, pf_ = model._last_terms
            print(f'  iter {it+1:5d}  loss {loss.item():+.4f}  '
                  f'[ll {ll_:+.1f} logpU {pu_:+.1f} l1 {l1_:+.1f} '
                  f'logpfix {pf_:+.1f}]', flush=True)
    wall = time.time() - t0

    # report the learned q(U) spread: if the entropy term dominates, q never
    # contracts away from the prior (logstd stays ~0) and every sample of U is
    # essentially prior noise.
    for i, _l in enumerate(model.layers):
        if hasattr(_l, 'q_logstd'):
            print('   layer%d q_logstd mean=%+.3f  |q_mean|=%.3f'
                  % (i, _l.q_logstd.mean().item(), _l.q_mean.norm().item()),
                  flush=True)

    # (1) fit quality: RMSE on the training inputs themselves
    fit_mean, _, _ = model.predict(xt, n_samples=args.eval_samples,
                                   n_steps=args.flow_steps)
    fit_rmse = ((fit_mean - yt) ** 2).mean().sqrt().item()

    # (2) in-between uncertainty (Foong et al., 2019): the whole point of this
    #     task is whether q can express HIGH uncertainty in the gap between the
    #     two data clusters. We compare the gap band against the data bands
    #     only -- extrapolation outside [-1, 1] is not what is being tested.
    xs = torch.linspace(-1.0, 1.0, 400, device=dev)[:, None]
    mean, std, _ = model.predict(xs, n_samples=args.eval_samples,
                                 n_steps=args.flow_steps)
    gap = ((xs > -0.7) & (xs < 0.5)).squeeze()
    gap_std = std[gap].mean().item()
    data_std = std[~gap].mean().item()

    print(f'[{args.variant}] seed={args.seed} fit_rmse={fit_rmse:.4f} '
          f'gap_std={gap_std:.4f} data_std={data_std:.4f} '
          f'ratio={gap_std/max(data_std,1e-8):.2f} wall={wall:.1f}s', flush=True)
    in_rmse = fit_rmse

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        np.savez(os.path.join(args.save_dir,
                              f'toy_{args.variant}_s{args.seed}.npz'),
                 xs=xs.cpu().numpy(), mean=mean.cpu().numpy(),
                 std=std.cpu().numpy(), x=x, y=y,
                 rmse=in_rmse, gap_std=gap_std, data_std=data_std, wall=wall)
    return in_rmse, gap_std, data_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', default='ddvi-u',
                   choices=['ddvi-u', 'ffg-u', 'fcg-u', 'ensemble-u', 'map'])
    p.add_argument('--task', default='toy')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--n_data', type=int, default=100)
    p.add_argument('--width', type=int, default=50)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--m', type=int, default=16)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument("--flow_steps", type=int, default=50)
    p.add_argument('--mc', type=int, default=1)
    p.add_argument('--eval_samples', type=int, default=64)
    p.add_argument('--n_ensemble', type=int, default=5)
    p.add_argument('--save_dir', default='results')
    # wheel bandit
    p.add_argument('--det_W', type=int, default=0,
                   help='use mu(U) with no conditional noise (diagnostic)')
    p.add_argument('--w_prior', type=float, default=1.0)
    p.add_argument('--w_l1', type=float, default=1.0)
    p.add_argument('--delta', type=float, default=0.5)
    p.add_argument('--n_steps_bandit', type=int, default=2000)
    p.add_argument('--train_every', type=int, default=50)
    p.add_argument('--train_iters', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()

    if args.task == 'toy':
        run_toy(args)
    elif args.task == 'wheel':
        run_wheel(args)
    else:
        raise ValueError(args.task)


if __name__ == '__main__':
    main()
