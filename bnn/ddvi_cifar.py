"""
DDVI with inducing weights on CIFAR-10/100 (Table 4/5 of the manuscript).

Design notes
------------
* Every Conv2d/Linear weight is reshaped to a matrix W in R^{d_out x d_in}
  (d_in = c_in*k*k for convolutions) and given the inducing-weight prior of
  Ritter et al. (2021):  p(U) = MN(0, Psi_r, Psi_c),
  Psi_r = Z_r Z_r^T + D_r^2,  Psi_c = Z_c Z_c^T + D_c^2.

* Inference runs in the WHITENED variable V, with U = L_r V L_c^T and
  Psi = L L^T. This is essential: Psi^{-1} is ill-conditioned when D is small,
  and the whitened conditional mean
        mu = sigma_r sigma_c A_r^T V A_c,   A = L^{-1} Z,  ||A||_2 <= 1
  is automatically at standard-init scale. It also makes p(V) = N(0, I) equal
  to p_fix, so the prior and p_fix terms of the bound cancel and the objective
  reduces to   E[log p(y|W)] - l_1(phi).

* q(V) is the terminal state of a reverse-time diffusion driven by a score
  network, in discrete-time (DDPM ancestral) form:
        p_phi(V_{k-1}|V_k) = N((V_k + b_k s_phi(V_k,k))/sqrt(1-b_k), b_k I)
        l_1 = sum_k b_k/(2(1-b_k)) ||V_k + s_phi(V_k,k)||^2

* SPEED: one score network is SHARED across layers, conditioned on a learned
  layer embedding, and all layers' V are processed as a single batch. A
  per-layer score net with a 50-step chain would need depth*50 tiny forward
  passes per step, which is untenable for a 28-layer net.

* BatchNorm affine parameters and biases stay deterministic (weight noise in
  BN scale/shift destroys the normalisation statistics).
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


# --------------------------------------------------------------------------
# shared score network over the whitened inducing variables
# --------------------------------------------------------------------------
class SharedScore(nn.Module):
    """s_phi(V, k, layer) with V zero-padded to a common width."""

    def __init__(self, max_dim, n_layers, hidden=256, emb=32):
        super().__init__()
        self.max_dim = max_dim
        self.layer_emb = nn.Embedding(n_layers, emb)
        self.net = nn.Sequential(
            nn.Linear(max_dim + emb + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, max_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, Vpad, t, idx):
        """Vpad [L, max_dim], t scalar, idx [L] -> [L, max_dim]."""
        e = self.layer_emb(idx)
        tt = Vpad.new_full((Vpad.shape[0], 1), float(t))
        return self.net(torch.cat([Vpad, e, tt], dim=-1))


# --------------------------------------------------------------------------
# inducing-weight wrapper for one Conv2d / Linear
# --------------------------------------------------------------------------
class InducingWeight(nn.Module):
    def __init__(self, d_out, d_in, m_out, m_in, gamma_init=-1.0):
        super().__init__()
        self.d_out, self.d_in = d_out, d_in
        self.m_out, self.m_in = min(m_out, d_out), min(m_in, d_in)
        self.Z_r = nn.Parameter(torch.randn(self.m_out, d_out) / math.sqrt(d_out))
        self.Z_c = nn.Parameter(torch.randn(self.m_in, d_in) / math.sqrt(d_in))
        self.gamma_r = nn.Parameter(torch.full((self.m_out,), gamma_init))
        self.gamma_c = nn.Parameter(torch.full((self.m_in,), gamma_init))
        # Scale so that W = sigma_r sigma_c A_r^T V A_c lands at He init scale.
        # Var[W_ij] ~ (sigma_r sigma_c)^2 * (m_out/d_out) * (m_in/d_in); setting
        # this equal to 2/d_in gives (sigma_r sigma_c)^2 = 2 d_out/(m_out m_in).
        # Without this the middle layers come out ~6x below He and the forward
        # signal decays before it reaches the classifier.
        s2 = 2.0 * d_out / max(self.m_out * self.m_in, 1)
        self.log_sigma_r = nn.Parameter(torch.tensor(0.5 * math.log(s2)))
        self.log_sigma_c = nn.Parameter(torch.tensor(0.0))

    def whitening(self):
        eye_r = torch.eye(self.m_out, device=self.Z_r.device)
        eye_c = torch.eye(self.m_in, device=self.Z_c.device)
        Psi_r = self.Z_r @ self.Z_r.t() + torch.diag(torch.exp(2 * self.gamma_r)) \
            + 1e-5 * eye_r
        Psi_c = self.Z_c @ self.Z_c.t() + torch.diag(torch.exp(2 * self.gamma_c)) \
            + 1e-5 * eye_c
        Lr = torch.linalg.cholesky(Psi_r)
        Lc = torch.linalg.cholesky(Psi_c)
        A_r = torch.linalg.solve_triangular(Lr, self.Z_r, upper=False)
        A_c = torch.linalg.solve_triangular(Lc, self.Z_c, upper=False)
        return A_r, A_c

    def weight(self, V):
        """mu(V) = sigma_r sigma_c A_r^T V A_c  ->  [d_out, d_in]."""
        A_r, A_c = self.whitening()
        sr, sc = torch.exp(self.log_sigma_r), torch.exp(self.log_sigma_c)
        return sr * sc * (A_r.t() @ V @ A_c)


# --------------------------------------------------------------------------
# WRN with inducing weights
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.bn2 = nn.BatchNorm2d(cout)
        self.stride, self.cin, self.cout = stride, cin, cout
        self.need_short = (stride != 1 or cin != cout)

    def forward(self, x, w1, w2, ws):
        o = F.relu(self.bn1(x))
        s = x if not self.need_short else F.conv2d(o, ws, stride=self.stride)
        o = F.conv2d(o, w1, stride=self.stride, padding=1)
        o = F.conv2d(F.relu(self.bn2(o)), w2, stride=1, padding=1)
        return o + s


class DDVIWideResNet(nn.Module):
    def __init__(self, depth=16, widen=4, n_classes=10, M=64, n_steps=5,
                 hidden=256, gamma_init=-1.0):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        self.n, self.widths, self.n_classes = n, widths, n_classes
        self.n_steps = n_steps

        self.blocks = nn.ModuleList()
        self.iw = nn.ModuleList()
        self.shapes = []                       # (d_out, d_in, kernel) per weight

        def add(d_out, d_in, k):
            self.iw.append(InducingWeight(d_out, d_in * k * k, M, M, gamma_init))
            self.shapes.append((d_out, d_in, k))

        add(widths[0], 3, 3)                                   # stem
        cin = widths[0]
        for i, w in enumerate(widths[1:]):
            for j in range(n):
                stride = 2 if (j == 0 and i > 0) else 1
                blk = BasicBlock(cin, w, stride)
                self.blocks.append(blk)
                add(w, cin, 3)                                 # conv1
                add(w, w, 3)                                   # conv2
                if blk.need_short:
                    add(w, cin, 1)                             # shortcut
                cin = w
        self.bn = nn.BatchNorm2d(cin)
        self.iw.append(InducingWeight(n_classes, cin, M, M, gamma_init))
        self.shapes.append((n_classes, cin, 0))                # fc
        self.fc_bias = nn.Parameter(torch.zeros(n_classes))

        self.max_dim = max(l.m_out * l.m_in for l in self.iw)
        self.score = SharedScore(self.max_dim, len(self.iw), hidden)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        # diagnostic / ablation: 'point' makes V a free parameter (no diffusion)
        self.mode = 'ddvi'
        self.V_point = nn.ParameterList([
            nn.Parameter(torch.randn(l.m_out, l.m_in)) for l in self.iw])
        # Residual parameterization: V = V_point + exp(log_diff_scale) * V_diff.
        # With s_phi initialised at zero the diffusion output is pure N(0,I)
        # noise, so a network reading V directly sees a different random weight
        # every forward pass and never receives a consistent learning signal.
        # Letting the diffusion model the DEVIATION around a learnable location
        # keeps the diffusion posterior (and l_1) intact while giving the
        # likelihood a signal to latch onto from step one.
        self.log_diff_scale = nn.Parameter(torch.tensor(-2.0))
        self.fixed_scale = None   # if set, diffusion magnitude is held there

    # ---- sample all V jointly through the reverse chain ------------------
    def sample_V(self, n_steps=None):
        if self.mode == 'point':
            z = self.V_point[0].new_zeros(())
            return [v for v in self.V_point], z
        L = len(self.iw)
        dev = self.betas.device
        steps = n_steps or self.n_steps
        betas = torch.linspace(1e-4, 0.02, steps, device=dev)
        Vpad = torch.randn(L, self.max_dim, device=dev)
        mask = torch.zeros(L, self.max_dim, device=dev)
        for i, l in enumerate(self.iw):
            mask[i, :l.m_out * l.m_in] = 1.0
        Vpad = Vpad * mask
        idx = torch.arange(L, device=dev)
        l1 = Vpad.new_zeros(())
        for k in range(steps - 1, -1, -1):
            b = betas[k]
            t = (k + 1) / steps
            s = self.score(Vpad, t, idx) * mask
            l1 = l1 + b / (2.0 * (1.0 - b)) * ((Vpad + s) ** 2).sum()
            mean = (Vpad + b * s) / torch.sqrt(1.0 - b)
            if k > 0:
                Vpad = (mean + torch.sqrt(b) * torch.randn_like(mean)) * mask
            else:
                Vpad = mean * mask
        scale = (torch.exp(self.log_diff_scale) if self.fixed_scale is None
                 else self.log_diff_scale.new_tensor(self.fixed_scale))
        Vs = []
        for i, l in enumerate(self.iw):
            Vd = Vpad[i, :l.m_out * l.m_in].reshape(l.m_out, l.m_in)
            Vs.append(self.V_point[i] + scale * Vd)
        return Vs, l1

    def weights_from_V(self, Vs):
        ws = []
        for l, V, (d_out, d_in, k) in zip(self.iw, Vs, self.shapes):
            W = l.weight(V)
            ws.append(W if k == 0 else W.reshape(d_out, d_in, k, k))
        return ws

    def forward(self, x, Vs=None, n_steps=None):
        l1 = x.new_zeros(())
        if Vs is None:
            Vs, l1 = self.sample_V(n_steps)
        ws = self.weights_from_V(Vs)
        p = 0
        o = F.conv2d(x, ws[p], padding=1); p += 1
        for blk in self.blocks:
            w1, w2 = ws[p], ws[p + 1]; p += 2
            wsh = None
            if blk.need_short:
                wsh = ws[p]; p += 1
            o = blk(o, w1, w2, wsh)
        o = F.relu(self.bn(o))
        o = F.adaptive_avg_pool2d(o, 1).flatten(1)
        return F.linear(o, ws[p], self.fc_bias), l1


# --------------------------------------------------------------------------
def get_loaders(root, bs, val_frac=0.02, workers=8, dataset='cifar10'):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    D = torchvision.datasets.CIFAR100 if dataset == 'cifar100' \
        else torchvision.datasets.CIFAR10
    train = D(root, True, tr, download=True)
    test = D(root, False, te, download=True)
    n_val = int(len(train) * val_frac)
    g = torch.Generator().manual_seed(1234)
    tr_set, va_set = torch.utils.data.random_split(
        train, [len(train) - n_val, n_val], generator=g)
    mk = lambda d, b, sh: torch.utils.data.DataLoader(
        d, b, shuffle=sh, num_workers=workers, drop_last=sh, pin_memory=True)
    return mk(tr_set, bs, True), mk(va_set, 500, False), mk(test, 500, False), \
        len(tr_set)


@torch.no_grad()
def evaluate(model, loader, dev, n_samples=8, n_steps=None):
    model.eval()
    probs, labels = [], []
    Vs_list = [model.sample_V(n_steps)[0] for _ in range(n_samples)]
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        ps = [F.softmax(model(x, Vs=V)[0], dim=1) for V in Vs_list]
        probs.append(torch.stack(ps).mean(0)); labels.append(y)
    probs = torch.cat(probs); labels = torch.cat(labels)
    conf, pred = probs.max(1)
    acc = (pred == labels).float().mean().item() * 100
    nll = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels).item()
    ece, n = 0.0, labels.numel()
    bins = torch.linspace(0, 1, 16, device=probs.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.float().sum() / n) * ((pred[m] == labels[m]).float().mean()
                                            - conf[m].mean()).abs()
    return acc, float(ece), nll


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--depth', type=int, default=16)
    p.add_argument('--widen', type=int, default=4)
    p.add_argument('--M', type=int, default=64)
    p.add_argument('--n_steps', type=int, default=5)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--gamma_init', type=float, default=-1.0)
    p.add_argument('--l1_weight', type=float, default=1.0)
    p.add_argument('--eval_samples', type=int, default=8)
    p.add_argument('--data', default='./data')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mode', default='ddvi', choices=['ddvi', 'point'])
    p.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--fixed_scale', type=float, default=-1.0,
                   help='>0 pins the diffusion magnitude (keeps the posterior '
                        'from collapsing to a point estimate)')
    p.add_argument('--tag', default='')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda'
    n_cls = 100 if args.dataset == 'cifar100' else 10
    tr_loader, va_loader, te_loader, n_train = get_loaders(
        args.data, args.batch_size, dataset=args.dataset)
    model = DDVIWideResNet(args.depth, args.widen, n_cls, args.M,
                           args.n_steps, args.hidden, args.gamma_init).to(dev)
    model.mode = args.mode
    if args.fixed_scale > 0:
        model.fixed_scale = args.fixed_scale
    npar = sum(q.numel() for q in model.parameters())
    print(f'[info] DDVI WRN-{args.depth}-{args.widen} {args.dataset} M={args.M} '
          f'steps={args.n_steps} params={npar/1e6:.2f}M l1w={args.l1_weight} '
          f'lr={args.lr} tag={args.tag}', flush=True)

    if args.optimizer == 'sgd':
        # standard WRN recipe: SGD + momentum + weight decay + cosine
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.wd, nesterov=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best_val, best = -1.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = seen = 0
        for x, y in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, l1 = model(x)
            ll = -F.cross_entropy(logits, y, reduction='sum')
            # bound: E log p(y|W) - l1   (prior and p_fix cancel under whitening)
            loss = -(ll * (n_train / y.numel()) - args.l1_weight * l1) / n_train
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item() * y.numel(); seen += y.numel()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            va = evaluate(model, va_loader, dev, 4)
            if va[0] > best_val:
                best_val = va[0]
                best = evaluate(model, te_loader, dev, args.eval_samples)
            print(f'  ep {ep+1:3d} loss {tot/seen:+.4f} val_acc {va[0]:.2f} '
                  f'[best_val {best_val:.2f} -> test {best[0]:.2f}]', flush=True)
    acc, ece, nll = evaluate(model, te_loader, dev, args.eval_samples)
    print(f'FINAL[{args.tag}] DDVI {args.dataset} last: acc={acc:.2f} '
          f'ece={ece:.4f} nll={nll:.4f} | val-selected: acc={best[0]:.2f} '
          f'ece={best[1]:.4f} nll={best[2]:.4f} | wall={(time.time()-t0)/60:.1f}min',
          flush=True)


if __name__ == '__main__':
    main()
