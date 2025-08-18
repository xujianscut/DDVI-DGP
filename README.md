
# DDVI-DGP: Denoising Diffusion Variational Inference for Deep Gaussian Processes

This repository provides a prototype implementation of  
Denoising Diffusion Variational Inference (DDVI) for Deep Gaussian Processes (DGPs),  
as described in the paper:

> Jian Xu, Delu Zeng, John Paisley.  
> Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference.  
> ICML Oral 2024. [arXiv:2407.17033](https://arxiv.org/abs/2407.17033)

---

## Overview

Deep Gaussian Processes (DGPs) use inducing points to approximate posterior distributions,  
but standard variational inference (DSVI, IPVI) suffers from bias.  
DDVI introduces a denoising diffusion SDE to sample inducing variables,  
combined with a new variational ELBO bound.

This code implements:

- LSDE module: Linear SDE drift/diffusion for inducing points  
- SDEVariationalDistribution: Variational distribution parameterized by diffusion process  
- ToyDeepGPHiddenLayer / DDeepGP: Deep Gaussian Process with multiple layers  
- Training loop: ELBO loss + SDE regularization  
- Datasets: Synthetic sine data or Concrete compressive strength dataset  

Note: This is a research prototype, simplified compared to the full experimental setup in the paper.

---

## Requirements

- Python 3.8+  
- [PyTorch](https://pytorch.org/) >= 1.9  
- [GPyTorch](https://gpytorch.ai/) <= 1.6 (old API, tested)  
- [torchsde](https://github.com/google-research/torchsde)  
- numpy, pandas, tqdm  

Install dependencies:

```bash
pip install torch gpytorch==1.6 torchsde numpy pandas tqdm




## 📂 Code Structure

```
ddvi_dgp.py         # Main training script (entry point)
```

Main components inside `ddvi_dgp.py`:

* `LSDE`: SDE drift/diffusion dynamics for inducing variables
* `CholeskyVariationalDistribution`: Variational distribution with diffusion-based sampling
* `ToyDeepGPHiddenLayer`: Single DGP layer with RBF kernel
* `DDeepGP`: Multi-layer DGP with Gaussian likelihood
* `make_sine_data`, `load_concrete_excel`: Dataset utilities
* `split_and_loaders`: Data split & DataLoader
* `main()`: Training loop

---

## 🚀 Usage

Run with synthetic sine dataset:

```bash
python ddvi_dgp.py --use_sine --layers 2 --epochs 200
```

Run with Concrete dataset (Excel file required):

```bash
python ddvi_dgp.py --data_path Concrete_Data.xls --layers 3 --epochs 500
```

Key arguments:

* `--layers`: number of GP layers (default: 2)
* `--num_inducing`: number of inducing points (default: 128)
* `--t1`: terminal time for diffusion SDE (default: 1e-3)
* `--lr`: learning rate (default: 1e-2)
* `--epochs`: training epochs (default: 500)
* `--batch_size`: training batch size (default: 1024)

For full list:

```bash
python ddvi_dgp.py --help
```

---

## 📖 Relation to Paper

* **Equation (7-9): Diffusion SDE** → `LSDE` and `torchsde.sdeint`
* **Bridge trick & κt (Eq. 18-20)** → implemented in `LSDE.kappa`
* **Variational distribution (Eq. 23)** → approximated by
  `DeepApproximateMLL(VariationalELBO)` + additional **SDE losses**
* **Algorithm 1 (DDVI training loop)** → implemented in `main()`


、

## 🔗 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{xu2024ddvi,
  title={Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference},
  author={Xu, Jian and Zeng, Delu and Paisley, John},
  booktitle={International Conference on Machine Learning},
  year={2024}
}









