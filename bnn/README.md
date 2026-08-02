# DDVI for Bayesian Neural Networks with Inducing Weights

Code for the BNN half of

> Jian Xu, Delu Zeng, John Paisley.
> *Denoising Diffusion Variational Inference for Bayesian Deep Models with Inducing Variables.*

and for the baselines added during review. The DGP half lives in the parent
directory.

## Model

Each `Conv2d`/`Linear` weight is reshaped to a matrix `W` and given the
inducing-weight prior of Ritter et al. (2021),

```
[ W   U_c ]
[ U_r  U  ] ~ MN(0, Sigma_r, Sigma_c),    p(U) = MN(0, Psi_r, Psi_c)
```

with `Psi_r = Z_r Z_r^T + D_r^2`, `Psi_c = Z_c Z_c^T + D_c^2`, and

```
mu    = sigma_r sigma_c  Z_r^T Psi_r^{-1} U Psi_c^{-1} Z_c
Sigma = [sigma_c^2 (I - Z_c^T Psi_c^{-1} Z_c)] kron [sigma_r^2 (I - Z_r^T Psi_r^{-1} Z_r)]
```

`q(U)` is the terminal state of a reverse-time diffusion driven by a learned
score network, in discrete (DDPM ancestral) form.

## Two implementation details that matter

Both were found while writing this code and are documented in the paper; without
them the model trains to a constant predictor without any visible error.

**1. Whiten the inducing variable.** Work in `V` with `U = L_r V L_c^T`,
`Psi = L L^T`. Then `Psi^{-1} L = L^{-T}` gives

```
mu = sigma_r sigma_c A_r^T V A_c,     A = L^{-1} Z,    ||A||_2 <= 1
```

so `mu` sits at standard-init scale with no ill-conditioned inverse. In the
unwhitened form a small `D` makes `Psi` nearly singular (for a 32x50 factor the
Marchenko-Pastur lower edge puts the smallest eigenvalue near 0.04, so `Psi^{-1}`
amplifies `mu` about 25x): initialising `U` small makes the forward signal decay
by ~10^3 across three layers, while initialising it at the prior scale blows the
loss up to ~10^4 and the gradients get annihilated by clipping. Both ends fail
and there is no safe middle.

Whitening also makes `p(V) = N(0, I)` coincide with `p_fix`, so those two terms
of the bound cancel identically and the objective reduces to

```
l(phi) = E[log p(y|W)] - l_1(phi)
```

**2. Residual parameterization of the diffusion.** With the score network
initialised at zero the sampler emits pure noise, so the network sees a
different random weight every forward pass and never gets a consistent signal.
Letting the diffusion model the deviation around a learnable location,
`V = V_point + s * V_diff`, keeps the diffusion posterior and `l_1` intact while
giving the likelihood something to latch onto from the first step. Holding `s`
fixed (`--fixed_scale`) prevents the posterior from collapsing to a point
estimate late in training, which is what preserves the calibration advantage.

## Files

| File | Purpose |
|---|---|
| `ddvi_cifar.py` | DDVI + inducing weights on CIFAR-10/100 with Wide ResNet |
| `bnn_ddvi.py` | toy 1-D regression and wheel bandit; FFG/FCG/ensemble baselines |
| `ddvi_general.py` | DDVI for a general latent-variable model: full-weight BNN posterior, **no inducing structure** |
| `solve_gp.py` | DSVI / SOLVE-GP / DDVI on UCI, single-layer and deep GP, compute-matched |
| `conv_gp.py` | convolutional-kernel GP classification (no neural feature extractor) |
| `noisy_adam.py` | noisy Adam (Zhang et al., 2018, Alg. 1) baseline |

## Quick start

```bash
pip install torch torchvision pandas numpy xlrd openpyxl

# CIFAR-10, Wide ResNet 28-10
python ddvi_cifar.py --mode ddvi --dataset cifar10 --depth 28 --widen 10 \
    --M 64 --epochs 200 --fixed_scale 0.15

# same architecture without the diffusion posterior (point estimate control)
python ddvi_cifar.py --mode point --dataset cifar10 --depth 28 --widen 10

# general latent-variable model: full weight posterior, no inducing variables
python ddvi_general.py --method ddvi --dataset boston --iters 1000
python ddvi_general.py --method mfvi --dataset boston --iters 1000

# compute-matched comparison against SOLVE-GP (deep GP)
python solve_gp.py --method solvegp --dataset yacht --layers 3 --M 64 --M2 64
python solve_gp.py --method ddvi    --dataset yacht --layers 2 --M 64

# purely GP-based CIFAR-10 classification, no feature extractor
python conv_gp.py --method ddvi --dataset cifar10 --n_train 50000 --M 300

# noisy Adam baseline
python noisy_adam.py --optimizer noisy-adam --depth 16 --widen 4 --eta 1e-6
```

## Note on noisy Adam

`noisy_adam.py` follows Algorithm 1 of Zhang et al. (2018) and applies weight
noise only to `Conv2d`/`Linear` weights; BatchNorm affine parameters and biases
stay deterministic, since noise in the normalisation statistics drives the
network to chance accuracy. Even so, on a Wide ResNet it reaches roughly 28%
against 88% for plain Adam under an identical budget. This matches the original
paper, which reports noisy Adam as N/A for batch-normalised networks
("extremely unstable and work only with a very small lambda") and recommends
noisy K-FAC instead.
