import copy
import math
from typing import Optional
import os 
os.system('pip install random-fourier-features-pytorch')
import torch
import torch.nn.functional as F
import os 
from rff.layers import GaussianEncoding, PositionalEncoding
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as savgol
import json
import time


def params_to_tensor(params):
    return torch.cat([p.flatten() for p in params]), [p.shape for p in params]


def tensor_to_params(tensor, shapes):
    params = []
    start = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape)).item()
        param = tensor[start : start + size].reshape(shape)
        params.append(param)
        start += size
    return tuple(params)

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)




def wrap_func(func, shapes):
    def wrapped_func(params, *args, **kwargs):
        params = tensor_to_params(params, shapes)
        return func(params, *args, **kwargs)

    return wrapped_func


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.w0 = w0
        self.c = c
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight: torch.Tensor, bias: torch.Tensor, c: float, w0: float):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            # bias.uniform_(-w_std, w_std)
            bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class INR(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        n_layers: int = 3,
        hidden_features: int = 32,
        out_features: int = 1,
        pe_features: Optional[int] = None,
        fix_pe=True,
    ):
        super().__init__()

        if pe_features is not None:
            if fix_pe:
                self.layers = [PositionalEncoding(sigma=10, m=pe_features)]
                encoded_dim = in_features * pe_features * 2
            else:
                self.layers = [
                    GaussianEncoding(
                        sigma=10, input_size=in_features, encoded_size=pe_features
                    )
                ]
                encoded_dim = pe_features * 2
            self.layers.append(Siren(dim_in=encoded_dim, dim_out=hidden_features))
        else:
            self.layers = [Siren(dim_in=in_features, dim_out=hidden_features)]
        for i in range(n_layers - 2):
            self.layers.append(Siren(hidden_features, hidden_features))
        self.layers.append(nn.Linear(hidden_features, out_features))
        self.seq = nn.Sequential(*self.layers)
        self.num_layers = len(self.layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + 0.5


class INRPerLayer(INR):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = [x]
        for layer in self.seq:
            nodes.append(layer(nodes[-1]))
        nodes[-1] = nodes[-1] + 0.5
        return nodes


def make_functional(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values


def train_inr(train_ds, local_rank, num_samples=np.inf, num_iters=200, plot_every=np.inf):
    job_start_time = time.time()
    max_job_time = 23.8 * 3600  # 23.8 hours in seconds
    slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    inr_criterion = torch.nn.MSELoss()
    all_files = os.listdir("/data/TalkingStars/inr_data")
    processed_files = {f for f in all_files if f.endswith('.pth')}
    loss_dict = {}
    loss_file_path = f"/data/TalkingStars/inr_data/losses_{slurm_array_id}.json"
    if f'losses_{slurm_array_id}.json' in all_files:
        try:
            with open(loss_file_path, "r") as f:
                loss_dict = json.load(f)
        except json.JSONDecodeError:
            print(f"Error loading JSON from {loss_file_path}, starting with empty dict")
    # inr_criterion = torch.nn.L1Loss()
    for i, batch in enumerate(train_ds):
        coords, flux, info = batch
        coords, flux = coords.to(local_rank), flux.to(local_rank)
        coords = coords.squeeze().unsqueeze(-1)
        final_loss = np.inf
        n_iters = 0
        kid = info['KID']
        if f"{kid}.pth" in processed_files:
            print(f"Already trained {kid}")
            continue
        while final_loss > 1e-4:
            min_idx = max(coords.shape[0] - 20000, 1)
            start_idx = np.random.randint(0, min_idx)
            end_idx = start_idx + 20000
            coords_slice = coords[start_idx:end_idx].to(local_rank)
            flux_slice = flux[start_idx:end_idx]
            model = INR(hidden_features=2048, n_layers=3,
                        in_features=1,
                        out_features=1).to(local_rank)
            optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-4)
            pbar = tqdm(range(num_iters))
            losses = []
            for t in pbar:
                optimizer.zero_grad()
                pred_values = model(coords_slice.float())
                loss = inr_criterion(pred_values.squeeze(), flux_slice)
                loss.backward()
                optimizer.step()
                pbar.set_description(f'iter: {n_iters}, KID: {info["KID"]}, loss: {loss.item()}')
                losses.append(loss.item())
            final_loss = losses[-1]
            n_iters += 1
            if n_iters > 4:
                break
        loss_dict[str(info['KID'])] = final_loss
        coords = coords_slice
        flux = flux_slice
        if i % plot_every == 0:
            fig, axes = plt.subplots(2, 1)
            smooth_preds = savgol(pred_values.squeeze().detach().cpu().numpy(), 48, 1, mode='mirror', axis=0)
            axes[0].plot(coords.squeeze().cpu().numpy(), flux.squeeze().cpu().numpy())
            axes[0].plot(coords.squeeze().cpu().numpy(), smooth_preds, c='r')
            axes[1].plot(losses)
            fig.suptitle(f"{info['KID']}, final loss: final loss: {losses[-1]:.3e}")
            plt.tight_layout()
            plt.savefig(f"/data/TalkingStars/inr_figs/{info['KID']}.png")
        try:
            state_dict = model.state_dict()
            torch.save(state_dict, f"/data/TalkingStars/inr_data/{info['KID']}.pth")
        except Exception as e:
            print(f"Error saving model for {info['KID']}: {e}")
        if i > num_samples:
            break
        if time.time() - job_start_time > max_job_time:
            print("Approaching job time limit, saving checkpoint and exiting")
            with open(loss_file_path, "w") as f:
                json.dump(loss_dict, f)
            sys.exit(0)
    with open(loss_file_path, "w") as f:  # Use "w" not "a"
        json.dump(loss_dict, f)
        


