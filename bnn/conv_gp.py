"""
Convolutional GP classification (van der Wilk et al., NeurIPS 2017) with DSVI
and DDVI posteriors over the inducing variables.

Reviewer 2: "The current classification experiments rely on pre-extracted
neural network features -- it will be interesting to see pure-GP based
approaches such as convolutional GP experiments."

This is a PURE GP model: no neural feature extractor anywhere. The convolutional
kernel sums a patch kernel over all patch pairs,

    k(x, x') = (1 / P^2) sum_{p, p'} k_g(x^{[p]}, x'^{[p']}),

and the inducing inputs Z live in PATCH space (inter-domain inducing points),
so that

    K_xz = (1 / P) sum_p k_g(x^{[p]}, z),      K_zz = k_g(z, z).

Multi-class classification uses C independent latent GPs sharing the kernel and
the inducing inputs, with a softmax likelihood estimated by Monte Carlo.

Variants:
    dsvi   q(U) = N(m, LL^T)  per class      (the standard baseline)
    ddvi   q(U) from the diffusion posterior (ours)
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


class PatchRBF(nn.Module):
    def __init__(self, patch_dim):
        super().__init__()
        self.log_ls = nn.Parameter(torch.zeros(()))
        self.log_var = nn.Parameter(torch.zeros(()))

    def forward(self, A, B):
        ls = torch.exp(self.log_ls)
        a, b = A / ls, B / ls
        d = (a * a).sum(-1, keepdim=True) + (b * b).sum(-1) - 2 * a @ b.t()
        return torch.exp(self.log_var) * torch.exp(-0.5 * d.clamp_min(0))


class ConvGP(nn.Module):
    def __init__(self, img_ch, img_sz, patch, n_class, M, Z_init,
                 method='dsvi', n_steps=10, hidden=128):
        super().__init__()
        self.patch, self.img_ch, self.img_sz = patch, img_ch, img_sz
        self.n_class, self.M, self.method = n_class, M, method
        self.kern = PatchRBF(img_ch * patch * patch)
        self.Z = nn.Parameter(Z_init.clone())
        self.n_patch = (img_sz - patch + 1) ** 2

        if method == 'ddvi':
            self.score = nn.Sequential(
                nn.Linear(M * n_class + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, M * n_class))
            nn.init.zeros_(self.score[-1].weight)
            nn.init.zeros_(self.score[-1].bias)
            self.u0 = nn.Parameter(torch.zeros(n_class, M))
            self.log_scale = nn.Parameter(torch.tensor(-1.0))
            self.n_steps = n_steps
            self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        else:
            self.qu_mu = nn.Parameter(torch.zeros(n_class, M))
            self.qu_L = nn.Parameter(torch.eye(M).unsqueeze(0)
                                     .repeat(n_class, 1, 1))

    def patches(self, x):
        p = F.unfold(x, self.patch)                     # [B, C*p*p, n_patch]
        return p.transpose(1, 2)                        # [B, n_patch, D]

    def sample_u(self):
        if self.method == 'ddvi':
            dev = self.betas.device
            u = torch.randn(1, self.M * self.n_class, device=dev)
            l1 = u.new_zeros(())
            for k in range(self.n_steps - 1, -1, -1):
                b = self.betas[k]
                t = (k + 1) / self.n_steps
                s = self.score(torch.cat([u, u.new_full((1, 1), float(t))], -1))
                l1 = l1 + b / (2 * (1 - b)) * ((u + s) ** 2).sum()
                mean = (u + b * s) / torch.sqrt(1 - b)
                u = mean + torch.sqrt(b) * torch.randn_like(mean) if k > 0 else mean
            u = self.u0 + torch.exp(self.log_scale) * u.reshape(self.n_class, self.M)
            return u, l1
        L = torch.tril(self.qu_L)
        eps = torch.randn(self.n_class, self.M, 1, device=self.qu_mu.device)
        u = self.qu_mu + (L @ eps).squeeze(-1)
        kl = 0.5 * ((L ** 2).sum() + (self.qu_mu ** 2).sum()
                    - self.n_class * self.M
                    - 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).abs()
                                    + 1e-8).sum())
        return u, kl

    def forward(self, x):
        P = self.patches(x)                             # [B, n_patch, D]
        B, npat, D = P.shape
        Kzz = self.kern(self.Z, self.Z)
        Kzz = Kzz + 1e-4 * torch.eye(self.M, device=x.device)
        Lz = torch.linalg.cholesky(Kzz)
        Kpz = self.kern(P.reshape(-1, D), self.Z).reshape(B, npat, self.M)
        Kxz = Kpz.mean(1)                               # inter-domain
        A = torch.linalg.solve_triangular(Lz, Kxz.t(), upper=False).t()
        u, reg = self.sample_u()                        # [C, M]
        mean = A @ u.t()                                # [B, C]
        kxx = torch.exp(self.kern.log_var)
        var = (kxx - (A ** 2).sum(-1)).clamp_min(1e-6).unsqueeze(-1)
        return mean, var, reg


def get_data(name, root, n_train, bs):
    tf = T.Compose([T.ToTensor()])
    if name == 'mnist':
        tr = torchvision.datasets.MNIST(root, True, tf, download=True)
        te = torchvision.datasets.MNIST(root, False, tf, download=True)
        ch, sz = 1, 28
    else:
        tr = torchvision.datasets.CIFAR10(root, True, tf, download=True)
        te = torchvision.datasets.CIFAR10(root, False, tf, download=True)
        ch, sz = 3, 32
    if n_train and n_train < len(tr):
        tr = torch.utils.data.Subset(tr, list(range(n_train)))
    mk = lambda d, b, s: torch.utils.data.DataLoader(d, b, shuffle=s,
                                                     num_workers=4)
    return mk(tr, bs, True), mk(te, 500, False), ch, sz, len(tr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='dsvi', choices=['dsvi', 'ddvi'])
    p.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'])
    p.add_argument('--data', default='./data')
    p.add_argument('--patch', type=int, default=5)
    p.add_argument('--M', type=int, default=100)
    p.add_argument('--n_train', type=int, default=10000)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--n_steps', type=int, default=10)
    p.add_argument('--mc', type=int, default=4)
    p.add_argument('--eval_samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda'
    tr, te, ch, sz, ntr = get_data(a.dataset, a.data, a.n_train, a.batch)
    # initialise inducing patches from random training patches
    xb = next(iter(tr))[0].to(dev)
    pt = F.unfold(xb, a.patch).transpose(1, 2).reshape(-1, ch * a.patch ** 2)
    Z0 = pt[torch.randperm(pt.shape[0])[:a.M]].clone()
    model = ConvGP(ch, sz, a.patch, 10, a.M, Z0, a.method, a.n_steps).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f'[info] convGP {a.method} {a.dataset} patch={a.patch} M={a.M} '
          f'N={ntr} patches/img={(sz-a.patch+1)**2}', flush=True)

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            mean, var, reg = model(x)
            # MC softmax likelihood
            ll = 0.0
            for _ in range(a.mc):
                f = mean + var.squeeze(-1).sqrt().unsqueeze(-1) * torch.randn_like(mean)
                ll = ll - F.cross_entropy(f, y, reduction='sum')
            ll = ll / a.mc
            loss = -(ll * (ntr / y.numel()) - reg) / ntr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            model.eval()
            corr = tot = 0
            with torch.no_grad():
                for x, y in te:
                    x, y = x.to(dev), y.to(dev)
                    ps = []
                    for _ in range(a.eval_samples):
                        m, v, _ = model(x)
                        ps.append(F.softmax(m, -1))
                    pred = torch.stack(ps).mean(0).argmax(-1)
                    corr += (pred == y).sum().item(); tot += y.numel()
            print(f'  ep {ep+1:3d} test_acc {100*corr/tot:.2f}', flush=True)
    print(f'FINAL[convgp-{a.method}] {a.dataset} acc={100*corr/tot:.2f} '
          f'wall={(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
