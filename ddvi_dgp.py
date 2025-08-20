
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDVI / DeepGP training script with command-line arguments (v2, safer init order).

- Fixes AttributeError: 'SDEVariationalDistribution' object has no attribute 't1'
- Avoids assigning to reserved 'device' attribute on gpytorch variational modules
"""
import argparse
import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gpytorch
import tqdm

from torch.utils.data import TensorDataset, DataLoader
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
import torchsde

from linear_operator.operators import TriangularLinearOperator, CholLinearOperator


def make_sine_data(N=2000, noise=0.1, seed=0, device="cpu"):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    X = torch.linspace(-2.0, 2.0, N, device=device).unsqueeze(1)
    y_true = torch.sin(2*math.pi*X) + 0.3*torch.sin(4*math.pi*X)
    y = y_true + noise*torch.randn_like(y_true)
    return X, y

def load_concrete_excel(path, device="cpu"):
    data = pd.read_excel(path)
    data = torch.tensor(data.values, dtype=torch.float32, device=device)
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0].clamp_min(1e-12)) - 1
    y = data[:, -1]
    y = (y - y.mean()) / y.std().clamp_min(1e-12)
    return X, y

def split_and_loaders(X, y, batch_size=1024, split=0.8, test_batch_size=1024):
    N = X.shape[0]
    perm = torch.randperm(N, device=X.device)
    ntr = int(split * N)
    tr, te = perm[:ntr], perm[ntr:]
    train_x, train_y = X[tr].contiguous(), y[tr].contiguous()
    test_x,  test_y  = X[te].contiguous(), y[te].contiguous()

    train_ds = TensorDataset(train_x, train_y)
    test_ds  = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False)
    return train_x, train_y, test_x, test_y, train_loader, test_loader


class LSDE(nn.Module):
    def __init__(self, num_inducing, batch_shape,device, t1=0.1):
        super().__init__()
        
        self.dev = device  # keep device for local tensors if needed
        self.t1 = float(t1)

        self.nn1 = nn.Sequential(
            nn.Linear(num_inducing, 128),
            nn.SiLU(),
     

            nn.Linear(128, num_inducing)
        ).to(device)

        self.sigma = nn.Parameter(torch.ones(*batch_shape, num_inducing, device=device))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def lam(self, t):
        return self.t1 - (t + 1e-3)

    def f(self, t, x):
        lamda = self.lam(t)
        f = 2 * self.sigma**2 * self.nn1(x) 
        return f + lamda*x

    def g(self, t, x):
        if self.sigma.dim() == 1:
            return self.sigma.unsqueeze(0)
        return self.sigma
    def kappa(self, ts, dt=1e-3, sigma0_sq=1.0):
        lam_vals = self.lam(ts)                  # [steps]
        I = torch.cumsum(lam_vals, dim=0) * dt   # 积分近似
        exp_I = torch.exp(I)

        sigma_scalar = self.sigma.mean()        # 用均值近似一个 g
        g_vals = sigma_scalar * torch.ones_like(ts)  # [steps]
        gr2 = g_vals**2

        J = torch.cumsum(gr2 * exp_I * dt, dim=0)
        kappa = (J + sigma0_sq) * torch.exp(-I)
        return kappa.mean()   # shape [steps]




class SDEVariationalDistribution(gpytorch.variational._variational_distribution._VariationalDistribution):
    def __init__(self, num_inducing_points, device, batch_shape, mean_init_std=1e-3, t1=0.1):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)

        # IMPORTANT: do NOT assign to self.device (gpytorch VariationalDistribution may define a read-only property)
        self._t1 = float(t1)
        self.dt=1e-3

        # SDE components
        self.u0 = nn.Parameter(torch.randn(*batch_shape, num_inducing_points, device=device))
       
        self.usde = LSDE(num_inducing=num_inducing_points, batch_shape=batch_shape, device=device, t1=t1)

        # Use SDE to produce initial mean
        self.uT = self.u()

        # Initialize mean/covar
        self.mean_init = self.uT
        self.covar_init = torch.eye(num_inducing_points, device=device) 
        
        
        self.covar_init = self.covar_init.repeat(*batch_shape, 1, 1)
        

        # Trainable parameters of q(U)
        self.register_parameter(name="variational_mean", parameter=nn.Parameter(self.mean_init))
        self.register_parameter(name="chol_variational_covar", parameter=nn.Parameter(self.covar_init))

    @property
    def t1(self):
        return self._t1

    def u(self):
        # Guard: ensure t1 exists
        t1 = getattr(self, "_t1", 1e-3)
        t0 = 0.0
      
        steps = max(2, int((t1 - t0) / self.dt) + 1)
        ts = torch.linspace(t0, t1, steps=steps, device='cuda')
        y0=self.u0
        if y0.dim() == 1:
            y0 = y0.unsqueeze(0)  # (1, num_inducing_points)
          
        u_s = torchsde.sdeint(self.usde, y0, ts=ts, names={'drift': 'f', 'diffusion': 'g'},
                              method='euler', dt=self.dt)
        u = u_s[-1, :, :]
        return u.squeeze()

    def forward(self):
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        lower_mask = torch.ones(chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        tril = chol_variational_covar.mul(lower_mask)

        tri_op = TriangularLinearOperator(tril)
        var_covar = CholLinearOperator(tri_op)
        return MultivariateNormal(self.variational_mean, var_covar)
       
    def sde_loss(self):


        # 重算一次路径

        u0 = self.u0
        if u0.dim() == 1:
            u0 = u0.unsqueeze(0)

        #self.usde.u0=u0 

        t0, t1 = 0.0, self._t1
        dt = 1e-3
        
        steps = max(2, int((t1 - t0) / self.dt) + 1)
        ts = torch.linspace(t0, t1, steps=steps, device='cuda')

        u_s = torchsde.sdeint(
            self.usde, y0=u0, ts=ts,
            names={'drift': 'f', 'diffusion': 'g'},
            method='euler', dt=self.dt
        )


        g = self.usde.sigma.unsqueeze(0)**2
        kappa_vals = self.usde.kappa(ts, dt=self.dt)
        r = self.usde.nn1(u_s)

        sde_loss = 0.5 * torch.norm(g * (u_s/kappa_vals + r))**2
        return sde_loss


    def initialize_variational_distribution(self, prior_dist):
        self.variational_mean.data.copy_(prior_dist.mean)
        self.variational_mean.data.add_(torch.randn_like(prior_dist.mean), alpha=self.mean_init_std)
        self.chol_variational_covar.data.copy_(prior_dist.lazy_covariance_matrix.cholesky().evaluate())


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, device, mean_type='constant', t1=0.1):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims, device=device)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims, device=device)
            batch_shape = torch.Size([output_dims])

        variational_distribution = SDEVariationalDistribution(
            num_inducing_points=num_inducing,
            device=device,
            batch_shape=batch_shape,
            mean_init_std=1e-3,
            t1=t1
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)

        self.variational_distribution = variational_distribution
        self.variational_strategy = variational_strategy

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape).to(device)
        else:
            self.mean_module = LinearMean(input_dims).to(device)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        ).to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample().to(next(self.parameters()).device)
            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]
            x = torch.cat([x] + processed_inputs, dim=-1)
        return super().__call__(x, are_samples=bool(len(other_inputs)))


import torch.nn as nn
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP

class DDeepGP(DeepGP):
    def __init__(self, input_dims, num_inducing, num_hidden_layers, device="cuda", t1=0.1):
        super().__init__()

        # 用 ModuleList 管理多层
        self.hidden_layers = nn.ModuleList()

        for i in range(num_hidden_layers):
            self.hidden_layers.append(
                ToyDeepGPHiddenLayer(
                    input_dims=input_dims,
                    output_dims=input_dims,
                    num_inducing=num_inducing,
                    device=device,
                    mean_type="linear",
                    t1=t1,
                )
            )

        # 最后一层
        self.last_layer = ToyDeepGPHiddenLayer(
            input_dims=input_dims,
            output_dims=None,
            num_inducing=num_inducing,
            device=device,
            mean_type="constant",
            t1=t1,
        )

        self.likelihood = GaussianLikelihood().to(device)

    def forward(self, inputs):
        h = inputs
        for layer in self.hidden_layers:
            h = layer(h)
        return self.last_layer(h)
    def collect_sde_losses(self, weight=1.0):
        """统一收集所有层的 SDE loss"""
        total_loss = 0.0
        # hidden layers
        for hl in self.hidden_layers:
            total_loss = total_loss + weight * hl.variational_distribution.sde_loss()
        # last layer
        total_loss = total_loss + weight * self.last_layer.variational_distribution.sde_loss()
        return total_loss




def main():
    parser = argparse.ArgumentParser(description="DDVI / DeepGP with CLI (v2)")
    parser.add_argument("--layers", type=int, default=2, help="Number of hidden layers in DDeepGP")
    parser.add_argument('--data_path', type=str, default='Concrete_Data.xls', help='Path to Concrete Excel file')
    parser.add_argument('--use_sine', action='store_true', help='Use synthetic sine dataset instead of Excel')
    parser.add_argument('--sine_N', type=int, default=2000, help='Number of points for sine dataset')
    parser.add_argument('--sine_noise', type=float, default=0.1, help='Noise for sine dataset')
    parser.add_argument('--sine_seed', type=int, default=0, help='Seed for sine dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Train split ratio in [0,1)')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Train batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='Test batch size')
    parser.add_argument('--num_samples', type=int, default=1, help='num_likelihood_samples for DeepGP')

    parser.add_argument('--num_inducing', type=int, default=128, help='Number of inducing points')
    parser.add_argument('--t1', type=float, default=1e-1, help='Terminal time for SDE')
    parser.add_argument('--dt', type=float, default=1e-3, help='Euler step for SDE integration')
    parser.add_argument('--loss_sde_w', type=float, default=1e-3, help='Weight for hidden layer SDE loss')

    
   

    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='Preferred device (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[Info] Using device: {device}")

    if args.use_sine:
        X, y = make_sine_data(N=args.sine_N, noise=args.sine_noise, seed=args.sine_seed, device=device)
    else:
        if not os.path.exists(args.data_path):
            print(f"[Warn] Excel not found at '{args.data_path}'. Falling back to sine dataset.")
            X, y = make_sine_data(N=args.sine_N, noise=args.sine_noise, seed=args.sine_seed, device=device)
        else:
            X, y = load_concrete_excel(args.data_path, device=device)

    train_x, train_y, test_x, test_y, train_loader, test_loader = split_and_loaders(
        X, y, batch_size=args.batch_size, split=args.split, test_batch_size=args.test_batch_size
    )

    input_dims = train_x.shape[-1]
    num_inducing = int(args.num_inducing)

    model = DDeepGP(input_dims=input_dims, num_inducing=num_inducing, num_hidden_layers=args.layers-1, device=device, t1=args.t1).to(device)
    print(model)
    model.train()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

    epochs_iter = tqdm.tqdm(range(args.epochs), desc="Epoch")
    for _ in epochs_iter:
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            with gpytorch.settings.num_likelihood_samples(args.num_samples):
                optimizer.zero_grad()
                output = model(x_batch)
                loss1=-mll(output, y_batch)
                loss = loss1+model.collect_sde_losses(weight=args.loss_sde_w) 
              
                loss.backward()
                optimizer.step()


                minibatch_iter.set_postfix(loss=float(loss1.detach().cpu().item()))

    model.eval()
    preds_all = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            preds = model(x_batch.to(device)).mean
            preds_all.append(preds)
    preds_all = torch.cat(preds_all, dim=0)
    rmse = torch.mean(torch.pow(preds_all - test_y.to(device), 2)).sqrt()

    print(f"RMSE: {rmse.mean().item():.6f}")

if __name__ == "__main__":
    main()
