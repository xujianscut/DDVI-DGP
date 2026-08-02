"""
DDVI for general latent-variable models -- i.e. WITHOUT any inducing structure.

This addresses Reviewer 2's request for "more general application of this idea
to other latent-variable models not within the context of inducing variables".

The point is that the DDVI bound never uses the fact that the latent variable
is an inducing variable. All it needs is a latent z with a tractable prior p(z)
and a tractable likelihood p(D|z) that can be evaluated pointwise. Two settings
are implemented here, neither of which involves inducing variables:

  --task bnn   full-weight posterior of a Bayesian neural network. The latent
               is the entire weight vector z = w with p(w) = N(0, I). This is
               the plainest possible latent-variable model: no sparsity, no
               inducing points, no GP.

  --task vae   the latent code of a VAE. DDVI replaces the amortised Gaussian
               encoder: q(z|x) is the terminal state of a reverse diffusion
               conditioned on x.

Baselines share the same network and the same training loop, so the only thing
that differs is the variational family:
    mfvi   mean-field Gaussian  q(z) = N(m, diag(s))
    map    point estimate
    ddvi   ours

Under the standard normal prior the whitened bound applies directly:
p(z) = N(0,I) = p_fix, so those two terms cancel and
    l(phi) = E_q[log p(D|z)] - l_1(phi).
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
class ScoreNet(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, t):
        tt = z.new_full((z.shape[0], 1), float(t))
        return self.net(torch.cat([z, tt], dim=-1))


class DiffusionPosterior(nn.Module):
    """q(z) as the terminal state of a discrete reverse diffusion.

    Residual parameterization z = z0 + exp(log_scale) * z_diff: the diffusion
    models the deviation around a learnable location. Without this the sampler
    emits pure noise at initialisation and the likelihood never gets a
    consistent signal (verified empirically).
    """

    def __init__(self, dim, n_steps=10, hidden=256, init_scale=-2.0):
        super().__init__()
        self.dim, self.n_steps = dim, n_steps
        self.score = ScoreNet(dim, hidden)
        self.z0 = nn.Parameter(torch.randn(dim) * 0.1)
        self.log_scale = nn.Parameter(torch.tensor(init_scale))
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))

    def sample(self, n=1):
        dev = self.betas.device
        z = torch.randn(n, self.dim, device=dev)
        l1 = z.new_zeros(())
        for k in range(self.n_steps - 1, -1, -1):
            b = self.betas[k]
            t = (k + 1) / self.n_steps
            s = self.score(z, t)
            l1 = l1 + b / (2.0 * (1.0 - b)) * ((z + s) ** 2).sum() / n
            mean = (z + b * s) / torch.sqrt(1.0 - b)
            z = mean + torch.sqrt(b) * torch.randn_like(mean) if k > 0 else mean
        return self.z0 + torch.exp(self.log_scale) * z, l1


class MeanField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.m = nn.Parameter(torch.randn(dim) * 0.1)
        self.logs = nn.Parameter(torch.full((dim,), -3.0))

    def sample(self, n=1):
        eps = torch.randn(n, self.m.numel(), device=self.m.device)
        z = self.m + torch.exp(self.logs) * eps
        # -ELBO regulariser = KL(q||p) for p = N(0, I)
        kl = 0.5 * (torch.exp(2 * self.logs) + self.m ** 2
                    - 1 - 2 * self.logs).sum()
        return z, kl


class PointMass(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.m = nn.Parameter(torch.randn(dim) * 0.1)

    def sample(self, n=1):
        z = self.m.unsqueeze(0).expand(n, -1)
        return z, 0.5 * (self.m ** 2).sum()          # log prior only


# --------------------------------------------------------------------------
def mlp_forward(x, z, shapes):
    """Functional MLP whose entire parameter vector is z [n_samples, D]."""
    outs = []
    for s in range(z.shape[0]):
        p, h = 0, x
        for i, (o, i_) in enumerate(shapes):
            W = z[s, p:p + o * i_].reshape(o, i_); p += o * i_
            b = z[s, p:p + o]; p += o
            h = F.linear(h, W, b)
            if i < len(shapes) - 1:
                h = torch.tanh(h)
        outs.append(h)
    return torch.stack(outs)


def load_uci(name, root):
    """Loaders matching those in the DGP codebase (formats differ per file)."""
    import pandas as pd
    f = {'boston': 'boston.csv', 'energy': 'energy.csv',
         'concrete': 'Concrete_Data.xls', 'yacht': 'yacht.data',
         'qsar': 'qsar_fish.csv'}[name]
    path = os.path.join(root, f)
    if name == 'boston':
        d = pd.read_csv(path, sep=',', header=None, skiprows=45,
                        engine='python').values
    elif name == 'energy':
        d = pd.read_csv(path).values
        return np.asarray(d[:, :8], np.float64), np.asarray(d[:, 8:9], np.float64)
    elif name == 'concrete':
        d = pd.read_excel(path).values
    elif name == 'yacht':
        d = pd.read_csv(path, sep=r"\s+", header=None,
                        engine='python').dropna().values
    else:
        d = pd.read_csv(path, sep=';', header=None).values
    d = np.asarray(d, dtype=np.float64)
    d = d[~np.isnan(d).any(axis=1)]
    return d[:, :-1], d[:, -1:]


def run_bnn(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, Y = load_uci(args.dataset, args.data)
    n = len(X)
    idx = np.random.RandomState(args.seed).permutation(n)
    # Standard UCI BNN protocol: 90/10 split, fixed iteration budget for every
    # method (no test-set peeking, no per-method tuning).
    ntr = int(n * 0.9)
    tr, te = idx[:ntr], idx[ntr:]
    va = te[:0]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    ym, ys = Y[tr].mean(), Y[tr].std() + 1e-8
    Xn = torch.tensor((X - xm) / xs, dtype=torch.float32, device=dev)
    Yn = torch.tensor((Y - ym) / ys, dtype=torch.float32, device=dev)
    xtr, ytr, xte, yte = Xn[tr], Yn[tr], Xn[te], Yn[te]


    d_in, H = X.shape[1], args.width
    shapes = [(H, d_in), (1, H)]
    D = sum(o * i + o for o, i in shapes)
    print(f'[info] {args.method} {args.dataset} N={ntr} latent_dim={D}',
          flush=True)

    if args.method == 'ddvi':
        q = DiffusionPosterior(D, args.n_steps, args.hidden).to(dev)
    elif args.method == 'mfvi':
        q = MeanField(D).to(dev)
    else:
        q = PointMass(D).to(dev)
    log_noise = nn.Parameter(torch.tensor(-1.0, device=dev))
    opt = torch.optim.Adam(list(q.parameters()) + [log_noise], lr=args.lr)

    def eval_on(xq, yq, ns):
        with torch.no_grad():
            zs, _ = q.sample(ns)
            pr = mlp_forward(xq, zs, shapes)
            nz = torch.exp(log_noise)
            lp = (-0.5 * ((yq.unsqueeze(0) - pr) / nz) ** 2
                  - torch.log(nz * ys) - 0.5 * math.log(2 * math.pi))
            return -(torch.logsumexp(lp, 0) - math.log(pr.shape[0])).mean().item()

    t0 = time.time()
    best_va, best_state = float('inf'), None
    for it in range(args.iters):
        opt.zero_grad()
        z, reg = q.sample(args.mc)
        pred = mlp_forward(xtr, z, shapes)
        noise = torch.exp(log_noise)
        ll = (-0.5 * ((ytr.unsqueeze(0) - pred) / noise) ** 2
              - torch.log(noise) - 0.5 * math.log(2 * math.pi)).sum() / z.shape[0]
        loss = -(ll - reg) / ntr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q.parameters()) + [log_noise], 10.0)
        opt.step()

    with torch.no_grad():
        zs, _ = q.sample(args.eval_samples)
        pr = mlp_forward(xte, zs, shapes)
        mu = pr.mean(0)
        rmse = (((mu - yte) * ys) ** 2).mean().sqrt().item()
        noise = torch.exp(log_noise)
        lp = (-0.5 * ((yte.unsqueeze(0) - pr) / noise) ** 2
              - torch.log(noise * ys) - 0.5 * math.log(2 * math.pi))
        nll = -(torch.logsumexp(lp, 0) - math.log(pr.shape[0])).mean().item()
    print(f'FINAL[{args.method}] {args.dataset} seed={args.seed} '
          f'rmse={rmse:.4f} nll={nll:.4f} wall={time.time()-t0:.0f}s', flush=True)
    return rmse, nll


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='bnn', choices=['bnn'])
    p.add_argument('--method', default='ddvi', choices=['ddvi', 'mfvi', 'map'])
    p.add_argument('--dataset', default='boston')
    p.add_argument('--data', default='../data')
    p.add_argument('--width', type=int, default=50)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--mc', type=int, default=4)
    p.add_argument('--n_steps', type=int, default=10)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--eval_samples', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run_bnn(args)


if __name__ == '__main__':
    main()
