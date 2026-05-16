# DDVI-DGP / FBVI-DGP

A unified, gpytorch-free implementation of diffusion- and flow-based variational
inference for **Deep Gaussian Processes (DGPs)**, covering the following methodological
spectrum in a single codebase:

| Variant | Family | Posterior $q(\mathbf U)$ | Training |
|---|---|---|---|
| `dsvi` | mean-field | $\mathcal N(m, LL^\top)$ free | Closed-form KL + ELBO |
| `fbvi` | **flow (velocity)** | ODE $dU_t = v_\phi(U_t,t)\,dt$ from prior | ELBO backprop through ODE |
| `dbvi` | flow + SDE noise | $dU_t = v_\phi\,dt + \sigma\,dW_t$ | ELBO backprop through SDE |
| `fbvi-bridge` | **flow + Doob bridge** | Bridge-anchored start + conditional $v_\phi(U_t,t,\text{ctx})$ | ELBO backprop through bridge ODE |
| `score` | **score-based (DDVI)** | Reverse VP SDE with $s_\phi(U,t)$ | ELBO + denoising score matching |
| `dbvi-s` | **score + Doob bridge** | Reverse Doob-bridged SDE with $s_\phi(U,t,\text{ctx})$ | ELBO + conditional DSM |
| `ipvi` | **GAN-style implicit** | $U = g_\phi(\epsilon)$, $\epsilon\sim\mathcal N(0,I)$ | Best-response dynamics (Gen vs Disc) |

The implementation is **from scratch (no gpytorch dependency)**, allowing each
variational family to plug into the same DGP backbone (sparse GP layers, RBF–ARD
kernel, doubly-stochastic forward pass) for apples-to-apples comparison.

---

## Relation to prior work

This codebase grew out of three of our papers studying inducing-variable VI for DGPs:

* **DDVI** — Xu, Zeng, Paisley.
  *Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference.* ICML 2024. [arXiv:2407.17033](https://arxiv.org/abs/2407.17033)

* **DBVI** — Xu, Zeng, Zhao, Paisley.
  *Diffusion Bridge Variational Inference for Deep Gaussian Processes.* ICLR 2026. [arXiv:2509.19078](https://arxiv.org/abs/2509.19078)

* **FBVI** (new, this repo's focus) — flow-matching counterpart of the above. We implement
  velocity-field VI with optional Doob-bridge structure, providing an apples-to-apples
  comparison between score-based and flow-based DGP-VI in the same framework.

The original prototype implementations of DDVI/DBVI used `gpytorch` and grafted
the SDE/score machinery onto the standard `VariationalStrategy`. We found that
in those reference implementations the SDE/score branch ended up **decoupled
from the ELBO** (because gpytorch's `initialize_variational_distribution`
overwrites the SDE-derived mean and the auxiliary `sde_loss` only trains a
side-network with no gradient path to $q(\mathbf U)$). This repository contains
a clean from-scratch implementation in which all variational families
**genuinely participate in the ELBO**.

---

## Quick start

```bash
pip install torch pandas numpy tqdm
```

`fbvi_native.py` is a single-file script. Run it with any `--variant` and any
of the bundled UCI regression datasets:

```bash
# Velocity-field flow-matching VI (FBVI)
python fbvi_native.py --variant fbvi --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256

# Doob-bridge score VI (proper DBVI)
python fbvi_native.py --variant dbvi-s --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256 \
    --dsm_weight 1.0 --doob_lambda 1.0 --doob_g 1.0 --doob_sigma0 1.0

# Plain DDVI (unconditional VP DSM, score-based)
python fbvi_native.py --variant score --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256 --dsm_weight 1.0

# GAN-style IPVI (Yu et al. 2019)
python fbvi_native.py --variant ipvi --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256

# Mean-field DSVI baseline
python fbvi_native.py --variant dsvi --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256

# Few-step inference at end of training
python fbvi_native.py --variant fbvi --dataset energy --data_path data/energy.csv \
    --epochs 100 --eval_steps_list 1,2,4,10,20
```

Available datasets (in `data/`): `yacht`, `boston`, `energy`, `qsar`, `concrete`,
`power`, `protein`. All are standard UCI regression benchmarks.

---

## Key CLI arguments

| Flag | Default | Purpose |
|---|---|---|
| `--variant {fbvi,dsvi,dbvi,fbvi-bridge,score,dbvi-s,ipvi}` | `fbvi` | Which variational family |
| `--num_inducing` | 128 | $M$ per layer |
| `--layers` | 2 | DGP depth |
| `--flow_steps` | 10 | ODE/SDE integration steps |
| `--flow_hidden` | 128 | Hidden width of velocity / score network |
| `--mc_samples` | 2 | Monte-Carlo samples for ELBO data term |
| `--eval_samples` | 32 | MC samples at evaluation |
| `--dsm_weight` | 0.0 | Coefficient on DSM auxiliary loss (`score`/`dbvi-s`) |
| `--sde_sigma` | 0.1 | Noise scale for the `dbvi` (velocity + SDE) variant |
| `--doob_lambda`, `--doob_g`, `--doob_sigma0` | 1, 1, 1 | Affine forward-SDE schedule for Doob-bridge variants |
| `--shortcut_weight`, `--shortcut_warmup_epochs` | 0.0, 20 | Frans-2024 shortcut self-consistency loss for accelerated few-step inference |
| `--eval_steps_list` | "" | Comma-separated step counts for the few-step inference sweep after training |

Run `python fbvi_native.py --help` for the full list.

---

## File map

```
fbvi_native.py        # main entry point — model + training + evaluation
aggregate_table.py    # builds RMSE / NLL summary tables across runs
data/                 # bundled UCI regression datasets
```

The main file is organized as:

* `SparseGPLayer` — single sparse GP layer (RBF–ARD kernel, learnable $Z$, residual mean)
* `VelocityField` / `ScoreField` / `Generator` / `Discriminator` — per-layer NN modules
* `Amortizer` + `_precompute_doob` + `ConditionalScoreField` / `ConditionalVelocityField` — Doob-bridge plumbing
* `FlowDGP` — the unified model; dispatches on `--variant`
* `eval_metrics` — MC-RMSE / MC-NLL
* `main()` — data loading, training loop (with a separate BRD branch for `ipvi`)

---

## Citation

If you build on this codebase, please cite the underlying papers:

```bibtex
@inproceedings{xu2024sparse,
  title={Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference},
  author={Xu, Jian and Zeng, Delu and Paisley, John},
  booktitle={International Conference on Machine Learning},
  year={2024},
}

@inproceedings{xu2026diffusion,
  title={Diffusion Bridge Variational Inference for Deep Gaussian Processes},
  author={Xu, Jian and Zeng, Delu and Zhao, Qibin and Paisley, John},
  booktitle={International Conference on Learning Representations},
  year={2026},
}
```

For the IPVI baseline:

```bibtex
@inproceedings{yu2019implicit,
  title={Implicit Posterior Variational Inference for Deep Gaussian Processes},
  author={Yu, Haibin and Chen, Yizhou and Dai, Zhongxiang and Low, Bryan Kian Hsiang and Jaillet, Patrick},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
