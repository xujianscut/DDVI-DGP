# Denoising Diffusion Variational Inference for Deep Gaussian Processes (DDVI-DGP)

[![ICML 2024](https://img.shields.io/badge/ICML-2024-blue)](https://proceedings.mlr.press/v235/xu24af.html)
[![PMLR](https://img.shields.io/badge/PMLR-v235-orange)](https://proceedings.mlr.press/v235/xu24af.html)
[![arXiv](https://img.shields.io/badge/arXiv-2407.17033-b31b1b)](https://arxiv.org/abs/2407.17033)

Official implementation of the **ICML 2024 (Oral)** paper
*"Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with
Denoising Diffusion Variational Inference"*.

DDVI replaces the standard mean-field Gaussian posterior over inducing
variables in a Deep GP with the terminal state of a **reverse-time SDE**
driven by a learned score network, optimized jointly with a **denoising
score-matching (DSM)** regularizer. This sidesteps the mean-field restriction
and expresses $q(\mathbf U)$ as the pushforward of noise through a
score-based generative process.

---

## Method overview

For DGP inference with inducing variables $\mathbf U=\{U^{(\ell)}\}_{\ell=1}^L$,
classical mean-field VI (DSVI) approximates the posterior by a factorised Gaussian
$q(\mathbf U)=\prod_\ell \mathcal N(m_\ell, S_\ell)$ — too restrictive since the true
inducing posterior in deep models is generally **non-Gaussian**.

DDVI defines $q(\mathbf U)$ implicitly via:

1. **Forward VP-noising:** a fixed variance-preserving SDE maps $q$ to a
   simple Gaussian $p_{\text{fix}}$ at $t=T$:
   $$dU_t = -\tfrac{1}{2}\beta(t)U_t\,dt + \sqrt{\beta(t)}\,dW_t,\qquad U_0\sim q,\ U_T\sim p_{\text{fix}}.$$
2. **Reverse-time SDE** parameterised by a learned score
   $s_\phi(U_t,t)\approx\nabla\log p_t(U)$ — this is how we *sample* from $q$.
3. **Joint objective** combines the ELBO data term (using the reverse-SDE
   sample) with a DSM regulariser:
   $$\mathcal L_{\text{DSM}}=\mathbb E_{t,U_0,\varepsilon}\Big\|s_\phi(U_t,t)+\tfrac{\varepsilon}{\sigma_t}\Big\|^2,\quad U_t=\alpha_t U_0+\sigma_t\varepsilon.$$

---

## Repository layout

```
.
├── ddvi.py              Main entry — model + DDVI training + evaluation
├── aggregate_table.py   RMSE / NLL summary tables across runs
└── data/                Bundled UCI regression datasets
```

DDVI-specific components inside `ddvi.py`:

- `SparseGPLayer` — sparse GP layer with RBF–ARD kernel
- `ScoreField` — score network $s_\phi(U_t, t)$
- `_alpha_sigma` — VP-noising marginal coefficients
- `FlowDGP.sample_U` (`score` branch) — reverse-SDE sampler
- `FlowDGP.dsm_loss` — DSM regulariser
- `main()` — full training loop

---

## Requirements

```bash
pip install torch pandas numpy tqdm
```

---

## Quick start

`ddvi.py` is a single-file script. Select the DDVI variant with `--variant score`:

```bash
# DDVI (score net + VP-DSM) on UCI energy
python ddvi.py --variant score \
    --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256 \
    --dsm_weight 1.0
```

Available datasets (in `data/`): `yacht`, `boston`, `energy`, `qsar`,
`concrete`, `power`, `protein` (standard UCI regression benchmarks).

Mean-field DSVI baseline for comparison:

```bash
python ddvi.py --variant dsvi \
    --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256
```

---

## Implementation notes

**This repository contains a clean from-scratch implementation in which the
score network genuinely participates in the ELBO**, with the reverse-time SDE
used as the sampling mechanism throughout training and evaluation.

`ddvi.py` also includes several other variational families (mean-field DSVI,
flow-based VI, IPVI, Doob-bridge variants) that share the same DGP backbone
for clean methodological comparison; the score-based variant is `--variant score`.

---

## Key DDVI-specific arguments

| Flag | Default | Purpose |
|---|---|---|
| `--variant score` | — | Select DDVI |
| `--dsm_weight` | 0.0 | Coefficient on DSM auxiliary loss; **set ≥1.0 for proper DDVI training** |
| `--dsm_samples` | 1 | Number of $(t, \varepsilon)$ MC samples per layer per minibatch |
| `--beta_min`, `--beta_max` | 0.1, 20.0 | VP noise schedule $\beta(t)=\beta_{\min}+t(\beta_{\max}-\beta_{\min})$ |
| `--flow_steps` | 10 | Steps for reverse-SDE integration |
| `--flow_hidden` | 128 | Hidden width of the score network |
| `--num_inducing` | 128 | $M$ per layer |
| `--layers` | 2 | DGP depth |
| `--mc_samples` | 2 | MC samples for ELBO data term |
| `--eval_samples` | 32 | MC samples at evaluation |

Run `python ddvi.py --help` for the full list.

---

## Citation

```bibtex
@InProceedings{pmlr-v235-xu24af,
  title     = {Sparse Inducing Points in Deep {G}aussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference},
  author    = {Xu, Jian and Zeng, Delu and Paisley, John},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages     = {55490--55500},
  year      = {2024},
  editor    = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume    = {235},
  series    = {Proceedings of Machine Learning Research},
  month     = {21--27 Jul},
  publisher = {PMLR},
  pdf       = {https://raw.githubusercontent.com/mlresearch/v235/main/assets/xu24af/xu24af.pdf},
  url       = {https://proceedings.mlr.press/v235/xu24af.html}
}
```

---

## Contact

Questions / issues: open a GitHub issue or contact the corresponding author
(see paper).
