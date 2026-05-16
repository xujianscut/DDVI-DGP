# DDVI-DGP

A clean, gpytorch-free implementation of **Denoising Diffusion Variational
Inference (DDVI)** for Deep Gaussian Processes (DGPs), as introduced in our paper:

> Jian Xu, Delu Zeng, John Paisley.
> *Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference.*
> ICML 2024 (Oral). [arXiv:2407.17033](https://arxiv.org/abs/2407.17033)

---

## Method overview

For DGP inference with inducing variables $\mathbf U=\{U^{(\ell)}\}_{\ell=1}^L$,
classical mean-field VI (DSVI) approximates the posterior by a factorised Gaussian
$q(\mathbf U)=\prod_\ell \mathcal N(m_\ell, S_\ell)$. This assumption is restrictive — the
true posterior over inducing variables is generally **non-Gaussian** in deep models.

**DDVI** instead defines the variational posterior implicitly as the terminal of a
**reverse-time stochastic differential equation (SDE)**:

1. A fixed forward noising process (variance-preserving SDE) maps $q(\mathbf U)$ to a
   simple Gaussian $p_{\text{fix}}$ at $t=T$:
   $$dU_t = -\tfrac{1}{2}\beta(t)U_t\,dt + \sqrt{\beta(t)}\,dW_t,\qquad U_0\sim q,\ U_T\sim p_{\text{fix}}.$$
2. The reverse-time SDE parameterised by a learned score network $s_\phi(U_t,t)\approx\nabla\log p_t(U)$ recovers a sample of $q$ from a noise draw at $t=T$.
3. Training combines the **ELBO data term** (via the reverse-SDE sample) with a
   **denoising score-matching (DSM) regularizer**:
   $$\mathcal L_{\text{DSM}}=\mathbb E_{t,U_0,\varepsilon}\Big\|s_\phi(U_t,t)+\tfrac{\varepsilon}{\sigma_t}\Big\|^2,\quad U_t=\alpha_t U_0+\sigma_t\varepsilon.$$

This sidesteps the mean-field restriction by representing $q(\mathbf U)$ as the
pushforward of noise through a score-based generative process.

---

## Implementation notes

The original prototype distributed alongside the paper used `gpytorch` and grafted
the diffusion machinery onto the standard `VariationalStrategy`. During careful
debugging we found that the gpytorch's
`initialize_variational_distribution` overwrites the diffusion-derived initial
mean at first forward, which left the score network as an **isolated side-network**
with no gradient path to the ELBO. The DDVI claim is sound, but that early code
did not faithfully implement it.

**This repository contains a clean from-scratch implementation in which the score
network genuinely participates in the ELBO**, with the reverse-time SDE used as
the sampling mechanism throughout training and evaluation.

The main script (`ddvi.py`) also includes several other variational
families (mean-field DSVI, flow-based VI, IPVI, Doob-bridge variants) that share
the same DGP backbone for clean methodological comparison; the score-based
variant is selected with `--variant score`.

---

## Quick start

```bash
pip install torch pandas numpy tqdm
```

`ddvi.py` is a single-file script. The DDVI variant is `--variant score`:

```bash
# DDVI (score net + VP DSM) on UCI energy
python ddvi.py --variant score \
    --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256 \
    --dsm_weight 1.0
```

Available datasets (in `data/`): `yacht`, `boston`, `energy`, `qsar`, `concrete`,
`power`, `protein` (standard UCI regression benchmarks).

A mean-field DSVI baseline for comparison:

```bash
python ddvi.py --variant dsvi \
    --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256
```

---

## Key DDVI-specific arguments

| Flag | Default | Purpose |
|---|---|---|
| `--variant score` | — | Select DDVI |
| `--dsm_weight` | 0.0 | Coefficient on DSM auxiliary loss; **set to ≥1.0 for proper DDVI training** |
| `--dsm_samples` | 1 | Number of $(t, \varepsilon)$ MC samples per layer per minibatch |
| `--beta_min`, `--beta_max` | 0.1, 20.0 | VP noise schedule $\beta(t)=\beta_{\min}+t(\beta_{\max}-\beta_{\min})$ |
| `--flow_steps` | 10 | Steps for reverse-SDE integration |
| `--flow_hidden` | 128 | Hidden width of the score network |
| `--num_inducing` | 128 | $M$ per layer |
| `--layers` | 2 | DGP depth |
| `--mc_samples` | 2 | MC samples for ELBO data term |
| `--eval_samples` | 32 | MC samples at evaluation |

Run `python ddvi.py --help` for the full list (also includes flags for the
other variants packaged in the same script).

---

## File map

```
ddvi.py        # main entry point — model + DDVI training + evaluation
aggregate_table.py    # builds RMSE / NLL summary tables across runs
data/                 # bundled UCI regression datasets
```

The DDVI-specific code lives in:

* `SparseGPLayer` — sparse GP layer with RBF–ARD kernel
* `ScoreField` — the score network $s_\phi(U_t, t)$
* `_alpha_sigma` — VP-noising marginal coefficients
* `FlowDGP.sample_U` (the `score` branch) — reverse-SDE sampler
* `FlowDGP.dsm_loss` — DSM regulariser
* `main()` — full DDVI training loop

---

## Citation

```bibtex
@inproceedings{xu2024sparse,
  title={Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference},
  author={Xu, Jian and Zeng, Delu and Paisley, John},
  booktitle={International Conference on Machine Learning},
  year={2024},
}
```
