"""
Noisy Adam (Zhang et al., ICML 2018, "Noisy Natural Gradient as Variational
Inference", Algorithm 1), as the baseline requested by Reviewer 2.

Update rule, verbatim from Alg. 1 (differences from standard Adam are exactly
the weight sampling and the absence of a square root on f):

    gamma_in = lambda / (N * eta),   gamma = gamma_in + gamma_ex
    w    ~  N(mu, (lambda/N) * diag(f + gamma_in)^{-1})
    v    <- grad_w log p(y|x,w) - gamma_in * w
    m    <- beta1 * m + (1 - beta1) * v
    f    <- beta2 * f + (1 - beta2) * (grad_w log p(y|x,w))^2
    mhat <- m / (1 - beta1^k)
    mhat <- mhat / (f + gamma)
    mu   <- mu + alpha * mhat

Note the sampling variance depends on f, so the weight noise is adaptive --
this is what makes it "noisy natural gradient" rather than plain weight noise.

IMPORTANT (Zhang et al., Table 2 footnote): noisy Adam is reported as N/A for
networks with batch normalisation, being "extremely unstable and work only with
a very small lambda". Our Table 4/5 architecture (WRN-28-10) uses BN, so this
script supports --bn 0/1 to measure that effect directly rather than assume it.
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# --------------------------------------------------------------------------
# Wide ResNet 28-10 (as in the manuscript), with optional BN
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, use_bn):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin) if use_bn else nn.Identity()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(cout) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=not use_bn)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Conv2d(cin, cout, 1, stride, 0, bias=not use_bn)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        s = x if self.short is None else self.short(o)
        o = self.conv1(o)
        o = self.conv2(F.relu(self.bn2(o)))
        return o + s


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, n_classes=10, use_bn=True):
        super().__init__()
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=not use_bn)
        layers = []
        cin = widths[0]
        for i, w in enumerate(widths[1:]):
            for j in range(n):
                layers.append(BasicBlock(cin, w, 2 if (j == 0 and i > 0) else 1,
                                         use_bn))
                cin = w
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin) if use_bn else nn.Identity()
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        o = self.conv1(x)
        o = self.blocks(o)
        o = F.relu(self.bn(o))
        o = F.adaptive_avg_pool2d(o, 1).flatten(1)
        return self.fc(o)


# --------------------------------------------------------------------------
# Noisy Adam optimizer (Alg. 1)
# --------------------------------------------------------------------------
class NoisyAdam:
    """Noisy Adam over the *weight* tensors only.

    BatchNorm affine parameters and biases are kept deterministic and updated
    with plain Adam. Injecting weight noise into BN scale/shift destroys the
    normalisation statistics and drives the network to chance accuracy; this is
    the practical content of the instability Zhang et al. report for BN nets.
    """

    def __init__(self, params, n_data, alpha=1e-3, beta1=0.9, beta2=0.999,
                 lam=1.0, eta=1.0, gamma_ex=0.0, use_sqrt=False,
                 det_params=None):
        self.params = [p for p in params if p.requires_grad]
        self.det = [p for p in (det_params or []) if p.requires_grad]
        self.det_opt = torch.optim.Adam(self.det, lr=alpha) if self.det else None
        self.N = n_data
        self.alpha, self.beta1, self.beta2 = alpha, beta1, beta2
        self.lam, self.eta = lam, eta
        self.gamma_in = lam / (n_data * eta)
        self.gamma = self.gamma_in + gamma_ex
        self.m = [torch.zeros_like(p) for p in self.params]
        self.f = [torch.zeros_like(p) for p in self.params]
        self.mu = [p.detach().clone() for p in self.params]
        self.use_sqrt = use_sqrt
        self.k = 0

    @torch.no_grad()
    def sample_weights(self):
        """w ~ N(mu, (lambda/N) diag(f + gamma_in)^{-1}); returns nothing, writes
        the sample into the live parameters."""
        for p, mu, f in zip(self.params, self.mu, self.f):
            var = (self.lam / self.N) / (f + self.gamma_in)
            p.copy_(mu + var.sqrt() * torch.randn_like(mu))

    @torch.no_grad()
    def step(self):
        """One Alg. 1 update. Assumes p.grad holds grad of the LOSS
        (= -log p(y|x,w) averaged over the batch), so grad of the
        log-likelihood is -p.grad."""
        self.k += 1
        for i, p in enumerate(self.params):
            g = -p.grad                                   # grad log p(y|x,w)
            v = g - self.gamma_in * p
            self.m[i].mul_(self.beta1).add_(v, alpha=1 - self.beta1)
            self.f[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            mhat = self.m[i] / (1 - self.beta1 ** self.k)
            # Zhang et al. Sec. 3.3 note the square root is "inessential ...
            # doesn't change the fixed points"; it does help conditioning, and
            # the reference implementation uses it, so we allow both.
            denom = (self.f[i].sqrt() + self.gamma) if self.use_sqrt \
                else (self.f[i] + self.gamma)
            mhat = mhat / denom
            self.mu[i].add_(mhat, alpha=self.alpha)
        if self.det_opt is not None:
            self.det_opt.step()

    @torch.no_grad()
    def load_mean(self):
        for p, mu in zip(self.params, self.mu):
            p.copy_(mu)


def get_loaders(root, batch_size, workers=8):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10(root, True, tr, download=True)
    test = torchvision.datasets.CIFAR10(root, False, te, download=True)
    return (torch.utils.data.DataLoader(train, batch_size, shuffle=True,
                                        num_workers=workers, drop_last=True),
            torch.utils.data.DataLoader(test, 500, shuffle=False,
                                        num_workers=workers),
            len(train))


@torch.no_grad()
def evaluate(model, loader, dev, n_samples=1, opt=None):
    """Accuracy, ECE and NLL. With n_samples>1 and opt given, averages the
    predictive distribution over posterior samples."""
    model.eval()
    probs_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        ps = []
        for _ in range(n_samples):
            if opt is not None and n_samples > 1:
                opt.sample_weights()
            ps.append(F.softmax(model(x), dim=1))
        probs_all.append(torch.stack(ps).mean(0))
        labels_all.append(y)
    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all)
    conf, pred = probs.max(1)
    acc = (pred == labels).float().mean().item()
    nll = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels).item()
    # 15-bin ECE
    ece, n = 0.0, labels.numel()
    bins = torch.linspace(0, 1, 16, device=probs.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.float().sum() / n) * ((pred[m] == labels[m]).float().mean()
                                            - conf[m].mean()).abs()
    return acc * 100, float(ece), nll


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', default='noisy-adam',
                   choices=['noisy-adam', 'adam'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--eta', type=float, default=1.0)
    p.add_argument('--gamma_ex', type=float, default=0.0)
    p.add_argument('--sqrt', type=int, default=1,
                   help='use sqrt(f) in the denominator (Zhang et al. Sec 3.3)')
    p.add_argument('--depth', type=int, default=28)
    p.add_argument('--widen', type=int, default=10)
    p.add_argument('--bn', type=int, default=1)
    p.add_argument('--eval_samples', type=int, default=1)
    p.add_argument('--data', default='./data')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = 'cuda'
    train_loader, test_loader, n_train = get_loaders(args.data, args.batch_size)
    model = WideResNet(args.depth, args.widen, 10, bool(args.bn)).to(dev)
    n_par = sum(q.numel() for q in model.parameters())
    print(f'[info] {args.optimizer} WRN-{args.depth}-{args.widen} bn={args.bn} '
          f'params={n_par/1e6:.1f}M lam={args.lam} lr={args.lr}', flush=True)

    if args.optimizer == 'noisy-adam':
        bayes, det = [], []
        for mod in model.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                bayes.append(mod.weight)
                if mod.bias is not None:
                    det.append(mod.bias)
            elif isinstance(mod, nn.BatchNorm2d):
                det += [q for q in mod.parameters(recurse=False)]
        print(f'[info] bayesian tensors={len(bayes)} deterministic={len(det)}',
              flush=True)
        opt = NoisyAdam(bayes, n_train, alpha=args.lr,
                        lam=args.lam, eta=args.eta, gamma_ex=args.gamma_ex,
                        use_sqrt=bool(args.sqrt), det_params=det)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if args.optimizer == 'noisy-adam':
                opt.sample_weights()
                if opt.det_opt is not None:
                    opt.det_opt.zero_grad()
            else:
                opt.zero_grad()
            for q in model.parameters():
                if q.grad is not None:
                    q.grad = None
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if not torch.isfinite(loss):
                print(f'[diverged] epoch {ep} loss={loss.item()}', flush=True)
                return
            opt.step()
            tot += loss.item() * y.numel(); seen += y.numel()
        if args.optimizer == 'noisy-adam':
            opt.load_mean()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            acc, ece, nll = evaluate(model, test_loader, dev,
                                     args.eval_samples,
                                     opt if args.optimizer == 'noisy-adam' else None)
            print(f'  epoch {ep+1:3d}  train_loss {tot/seen:.4f}  '
                  f'test_acc {acc:.2f}  ece {ece:.4f}  nll {nll:.4f}',
                  flush=True)

    acc, ece, nll = evaluate(model, test_loader, dev, args.eval_samples,
                             opt if args.optimizer == 'noisy-adam' else None)
    print(f'FINAL [{args.optimizer}] bn={args.bn} lam={args.lam} '
          f'acc={acc:.2f} ece={ece:.4f} nll={nll:.4f} '
          f'wall={(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
