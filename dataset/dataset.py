import os
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import pandas as pd
from typing import List
import copy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import traceback
from typing import NamedTuple, Tuple, Union


# from nn import inr
# from nn.inr import INR, Siren
# import torch
# from einops.layers.torch import Rearrange

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.sampler import DistributedUniqueLightCurveSampler, DistributedBalancedSampler, UniqueIDDistributedSampler

mpl.rcParams['axes.linewidth'] = 4
plt.rcParams.update({'font.size': 30, 'figure.figsize': (14,10), 'lines.linewidth': 4})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["gray", "r", "c", 'm', 'brown'])
plt.rcParams.update({'xtick.labelsize': 22, 'ytick.labelsize': 22})
plt.rcParams.update({'legend.fontsize': 22})

T_sun = 5778
T_MIN =3483.11
T_MAX = 7500.0
LOGG_MIN = -0.22
LOGG_MAX = 4.9
FEH_MIN = -2.5
FEH_MAX = 0.964
VSINI_MAX = 100
P_MAX = 70
MAX_AGE = 10

def pad_with_last_element(tensor, seq_len):
    """
    Pad the last dimension of a tensor from size T to size seq_len, 
    where the padding values for each dimension are the last elements of that dimension.
    
    Args:
        tensor: Input tensor of shape (dims, T)
        seq_len: Target sequence length for the last dimension
        
    Returns:
        Padded tensor of shape (dims, seq_len)
    """
    dims, T = tensor.shape
    
    if T >= seq_len:
        # If the tensor is already longer than or equal to seq_len, truncate it
        return tensor[:, :seq_len]
    
    # Create the padded tensor
    padded = torch.zeros((dims, seq_len), dtype=tensor.dtype, device=tensor.device)
    
    # Copy the original data
    padded[:, :T] = tensor
    
    # For each dimension, fill the padding with the last element of that dimension
    for i in range(dims):
        padded[i, T:] = tensor[i, -1]
    
    return padded

def create_boundary_values_dict(df):
  boundary_values_dict = {}
  for c in df.columns:
    if isinstance(df[c].values[0], str):
      continue
    if c not in boundary_values_dict.keys():
      if c == 'Butterfly':
        boundary_values_dict[c] = bool(df[c].values[0])
      else:
        min_val, max_val = df[c].min(), df[c].max()
        boundary_values_dict[f'min {c}'] = float(min_val)
        boundary_values_dict[f'max {c}'] = float(max_val)
  return boundary_values_dict


class BatchSiren(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_layers=3,
        hidden_features=32,
        img_shape=None,
        input_init=None,
    ):
        super().__init__()
        inr_module = INR(
            in_features=in_features,
            n_layers=n_layers,
            hidden_features=hidden_features,
            out_features=out_features,
        )
        fmodel, params = inr.make_functional(inr_module)

        vparams, vshapes = inr.params_to_tensor(params)
        self.sirens = torch.vmap(inr.wrap_func(fmodel, vshapes))

        inputs = (
            input_init if input_init is not None else make_coordinates(img_shape, 1)
        )
        self.inputs = torch.nn.Parameter(inputs, requires_grad=False)

        self.reshape_weights = Rearrange("b i o 1 -> b (o i)")
        self.reshape_biases = Rearrange("b o 1 -> b o")

    def forward(self, weights, biases):
        params_flat = torch.cat(
            [
                torch.cat(
                    [self.reshape_weights(w), self.reshape_biases(b)],
                    dim=-1,
                )
                for w, b in zip(weights, biases)
            ],
            dim=-1,
        )

        out = self.sirens(params_flat, self.inputs.expand(params_flat.shape[0], -1, -1))
        return out


class SimulationDataset(Dataset):
    def __init__(self,
                 df,
                labels=['Period', 'Inclination'],
                light_transforms=None,
                spec_transforms=None,
                npy_path = None,
                spec_path = None,
                use_acf=False,
                use_fft=False,
                scale_flux=False,
                meta_columns=None,
                spec_seq_len=4096,
                light_seq_len=34560,
                example_wv_path='/data/lamost/example_wv.npy'
                ):
        self.df = df
        self.spec_seq_len = spec_seq_len
        self.lc_seq_len = light_seq_len
        self.use_acf = use_acf
        self.use_fft = use_fft
        self.range_dict = dict()
        self.labels = labels
        # self.labels_df = self.df[labels]
        self.update_range_dict()
        # self.lc_dir = os.path.join(data_dir, 'lc')
        # self.spectra_dir = os.path.join(data_dir, 'lamost')
        self.lc_transforms = light_transforms
        self.spectra_transforms = spec_transforms
        self.example_wv = np.load(example_wv_path)
        self.Nlc = len(self.df)
        self.scale_flux = scale_flux
        self.meta_columns = meta_columns
        self.boundary_values_dict = create_boundary_values_dict(self.df)

    def fill_nan_inf_np(self, x: np.ndarray, interpolate: bool = True):
        """
        Fill NaN and Inf values in a numpy array

        Args:
            x (np.ndarray): array to fill
            interpolate (bool): whether to interpolate or not

        Returns:
            np.ndarray: filled array
        """
        # Create a copy to avoid modifying the original array
        x_filled = x.copy()
        
        # Identify indices of finite and non-finite values
        finite_mask = np.isfinite(x_filled)
        non_finite_indices = np.where(~finite_mask)[0]

        finite_indices = np.where(finite_mask)[0]
        
        # If there are non-finite values and some finite values
        if len(non_finite_indices) > 0 and len(finite_indices) > 0:
            if interpolate:
                # Interpolate non-finite values using linear interpolation
                interpolated_values = np.interp(
                    non_finite_indices, 
                    finite_indices, 
                    x_filled[finite_mask]
                )
                # Replace non-finite values with interpolated values
                x_filled[non_finite_indices] = interpolated_values
            else:
                # Replace non-finite values with zero
                x_filled[non_finite_indices] = 0
        
        return x_filled

    def update_range_dict(self):
        for name in self.labels:
            min_val = self.df[name].min()
            max_val = self.df[name].max()
            self.range_dict[name] = (min_val, max_val)
        
    def __len__(self):
        return len(self.df)

    def _normalize(self, x, label):
        # min_val = float(self.range_dict[key][0])
        # max_val = float(self.range_dict[key][1])
        # return (x - min_val) / (max_val - min_val)
        if 'period' in label.lower():
            x = x / P_MAX
        elif 'age' in label.lower():
            x = x / MAX_AGE
        return x

        # min_val, max_val = self.boundary_values_dict[f'min {label}'], self.boundary_values_dict[f'max {label}']
        # return (x - min_val)/(max_val - min_val)

    def transform_lc_flux(self, flux, info_lc):
        if self.lc_transforms:
            flux,_,info_lc = self.lc_transforms(flux, info=info_lc)
            if self.use_acf:
                acf = torch.tensor(info_lc['acf']).nan_to_num(0)
                flux = torch.cat((flux, acf), dim=0)
            if self.use_fft:
                fft = torch.tensor(info_lc['fft']).nan_to_num(0)
                flux = torch.cat((flux, fft), dim=0)
        if flux.shape[-1] == 1:
            flux = flux.squeeze(-1)
        if len(flux.shape) == 1:
            flux = flux.unsqueeze(0)
        flux = pad_with_last_element(flux, self.lc_seq_len)
        return flux, info_lc

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        padded_idx = f'{idx:d}'.zfill(int(np.log10(self.Nlc))+1)
        s = time.time()
        try:
            spec = pd.read_parquet(row['spec_data_path']).values
            # spec = self.fill_nan_np(spec)
        except (FileNotFoundError, OSError, IndexError) as e:   
            # print("Error reading file ", idx, e)
            spec = np.zeros((3909, 1))
        try: 
            # print(idx, row['lc_data_path'])
            lc = pd.read_parquet(row['lc_data_path']).values
            # lc[:, 1] = self.fill_nan_inf_np(lc[:, 1])
            # max_val = np.max(np.abs(lc[:, 1]))
            # if max_val > 1e2:
            #     lc[np.abs(lc[:, 1]) > 1e2, 1] = np.random.uniform(0, 2, size=lc[np.abs(lc[:, 1]) > 1e2, 1].shape)
            
            # lc[:, 1] = lc[:, 1] / lc[:, 1].max()
        except (FileNotFoundError, OSError, IndexError) as e:
            print("Error reading file ", idx, e)
            lc = np.zeros((48000, 2))
        spectra = spec[:,-1]
        target_spectra = spectra.copy()
        flux = lc[:,-1]
        target_flux = flux.copy()
        info_s = dict()
        info_lc = dict()
        try:
            # label = row[self.labels].to_dict()
            # print(label)
            # label = {k: self._normalize(v, k) for k, v in label.items()}
            y = torch.tensor([self._normalize(row[k], k) for k in self.labels], dtype=torch.float32)

        except IndexError:
            y = torch.zeros(len(self.labels))
        s1 = time.time()
        info_s['wavelength'] = self.example_wv
        if self.spectra_transforms:
            spectra, _,info_s = self.spectra_transforms(spectra, info=info_s)
        spectra = pad_with_last_element(spectra, self.spec_seq_len)
        spectra = torch.nan_to_num(spectra, nan=0)
        s2 = time.time()
        # print("nans in spectra: ", np.sum(np.isnan(spectra)))
        # spectra = torch.tensor(spectra).float()
        flux, info_lc = self.transform_lc_flux(flux, info_lc)
        target_flux, _ = self.transform_lc_flux(target_flux, info_lc)
        info = {'spectra': info_s, 'lc': info_lc}
        if 'L' in row.keys():
            info['KMAG'] = row['L']
        else:
            info['KMAG'] = 1
        s3 = time.time()
        flux = flux.nan_to_num(0).float()
        spectra = spectra.nan_to_num(0).float()
        target_flux = target_flux.nan_to_num(0).float()
        # print(flux.shape, target_flux.shape, spectra.shape, y.shape)
        # print(s1-s, s2-s1, s3-s2)
        return flux, spectra, y , target_flux,  spectra, info

class SpectraDataset(Dataset):
    """
    dataset for spectra data
    Args:
        data_dir: path to the data directory
        transforms: transformations to apply to the data
        df: dataframe containing the data paths
        max_len: maximum length of the spectra
        use_cache: whether to use a cache file
        id: column name for the observation id
    """
    def __init__(self, data_dir,
                     transforms=None,
                     df=None,
                    max_len=3909,
                    use_cache=True,
                    id='combined_obsid',
                    target_norm='solar'
                    ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.target_norm = target_norm
        self.df = df
        self.id = id
        if df is None:
            cache_file = os.path.join(self.data_dir, '.path_cache.txt')
            
            if use_cache and os.path.exists(cache_file):
                print("Loading cached file paths...")
                with open(cache_file, 'r') as f:
                    self.path_list = np.array([line.strip() for line in f])
            else:
                print("Creating files list...")
                self.path_list = self._file_listing()
                if use_cache:
                    with open(cache_file, 'w') as f:
                        f.write('\n'.join(self.path_list))
        else:
            self.path_list = None
        self.max_len = max_len
        self.mask_transform = RandomMasking()
    
    def _file_listing(self):
        
        def process_chunk(file_names):
            return [self.data_dir / name for name in file_names]
        
        file_names = os.listdir(self.data_dir)
        chunk_size = 100000  
        chunks = [file_names[i:i + chunk_size] for i in range(0, len(file_names), chunk_size)]
        
        with ThreadPoolExecutor() as executor:
            paths = []
            for chunk_paths in executor.map(process_chunk, chunks):
                paths.extend(chunk_paths)
        
        return np.array(paths)
        
    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.path_list) if self.path_list is not None else 0

    def read_lamost_spectra(self, filename):
        with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        rv = header['HELIO_RV']
        meta = {'RV': rv, 'wavelength': wv}
        return x, meta
    
    def read_apogee_spectra(self, filename):
        with fits.open(filename) as hdul:
            data = hdul[1].data.astype(np.float32).squeeze()[None]
        meta = {}
        return data, meta
    
    def create_lamost_target(self, row, info):
        info['Teff'] = row['combined_teff']
        info['rv2'] = row['combined_rv']
        info['logg'] = row['combined_logg']
        info['FeH'] = row['combined_feh']
        info['snrg'] = row['combined_snrg'] / 1000
        info['snri'] = row['combined_snri'] / 1000
        info['snrr'] = row['combined_snrr'] / 1000
        info['snrz'] = row['combined_snrz'] / 1000
        
        # Add joint probability information if available
        if 'joint_prob' in row:
            info['joint_prob'] = row['joint_prob']
            info['joint_weight'] = row['joint_weight']
            
        if self.target_norm =='solar':
            target = torch.tensor([-999, info['Teff'] / T_sun, info['logg'], info['FeH']], dtype=torch.float32)
        elif self.target_norm == 'minmax':
            teff = (info['Teff'] - T_MIN) / (T_MAX - T_MIN)
            logg = (info['logg'] - LOGG_MIN) / (LOGG_MAX - LOGG_MIN)
            feh = (info['FeH'] - FEH_MIN) / (FEH_MAX - FEH_MIN)
            target = torch.tensor([-999, teff, logg, feh], dtype=torch.float32)
        else:
            raise ValueError("Unknown target normalization method")
        return target, info
    
    def create_apogee_target(self, row, info):
        info['Teff'] = row['TEFF']
        info['logg'] = row['LOGG']
        info['FeH'] = row['FE_H']
        info['snrg'] = row['SNR'] / 1000
        info['snri'] = row['SNR'] / 1000
        info['snrr'] = row['SNR'] / 1000
        info['snrz'] = row['SNR'] / 1000
        info['vsini'] = row['VSINI']
        target = torch.tensor([info['vsini'] / VSINI_MAX, info['Teff'] / T_sun, info['logg'], info['FeH']], dtype=torch.float32)
        return target, info

    def create_empty_info(self, info):
        info['Teff'] = np.nan
        info['logg'] = np.nan
        info['FeH'] = np.nan
        info['snrg'] = 1e-4
        info['snri'] = 1e-4
        info['snrr'] = 1e-4
        info['snrz'] = 1e-4
        info['vsini'] = np.nan
        return info

    def __getitem__(self, idx):
        if self.df is not None:
            if 'APOGEE' in self.id:     
                filepath = f"{self.data_dir}/aspcapStar-dr17-{self.df.iloc[idx][self.id]}.fits"
            else:
                filepath = self.df.iloc[idx]['data_path']
            target_size = 4
        else: 
            filepath = self.path_list[idx]
            target_size = self.max_len    
        obsid = os.path.basename(filepath)
        try:
            if 'APOGEE' in self.id:
                spectra, meta = self.read_apogee_spectra(filepath)
            else:
                spectra, meta = self.read_lamost_spectra(filepath)
        except (OSError, FileNotFoundError) as e:
            info = self.create_empty_info({self.id: obsid})
            # print("Error reading file ", filepath, "\n", e)
            return (torch.zeros(self.max_len),
                    torch.zeros(self.max_len),
                    torch.zeros(target_size),\
                    torch.zeros(self.max_len, dtype=torch.bool),
                    torch.zeros(self.max_len, dtype=torch.bool),
                    info)
        meta[self.id] = obsid
        if self.transforms:
            spectra, _, info = self.transforms(spectra, None, meta)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, info)

        if self.df is not None:
            row = self.df.iloc[idx]
            if 'APOGEE' in self.id:
                target, info = self.create_apogee_target(row, meta)
            else:
                target, info = self.create_lamost_target(row, meta)
        else:
            target = torch.zeros_like(mask)
            
        if spectra_masked.shape[-1] < self.max_len:
            pad = torch.zeros(1, self.max_len - spectra_masked.shape[-1])
            spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
            pad_mask = torch.zeros(1, self.max_len  - mask.shape[-1], dtype=torch.bool)
            mask = torch.cat([mask, pad_mask], dim=-1)
            pad_spectra = torch.zeros(1, self.max_len - spectra.shape[-1])
            spectra = torch.cat([spectra, pad_spectra], dim=-1)
        spectra = torch.nan_to_num(spectra, nan=0)
        spectra_masked = torch.nan_to_num(spectra_masked, nan=0)
        
        return (spectra_masked.float().squeeze(0), spectra.float().squeeze(0),\
         target.float(), mask.squeeze(0), mask.squeeze(0), info)


class KeplerDataset():
    """
    A dataset for Kepler data.
    """
    def __init__(self,
                df:pd.DataFrame=None,
                prot_df:pd.DataFrame=None,
                npy_path:str=None,
                transforms:object=None,
                seq_len:int=4800,
                target_len:int=30*48,
                label_len:int=30*48,
                masked_transforms:bool=False,
                use_acf:bool=False,
                use_fft:bool=False,
                scale_flux:bool=True,
                labels:object=None,
                 dims:int=1
                ):
        """
        dataset for Kepler data
        Args:
            df (pd.DataFrame): DataFrame containing Kepler paths
            prot_df (pd.DataFrame): DataFrame containing rotation periods
            npy_path (str): Path to numpy files
            transforms (object): Transformations to apply to the data
            seq_len (int): Sequence length
            target_transforms (object): Transformations to apply to the target
            masked_transforms (bool): Whether to apply masking transformations

        """
        self.df = df
        self.transforms = transforms
        self.npy_path = npy_path
        self.prot_df = prot_df
        self.seq_len = seq_len
        self.target_len = target_len
        self.label_len = label_len
        if df is not None and 'predicted period' not in df.columns:
            if prot_df is not None:
                self.df = pd.merge(df, prot_df[['KID', 'predicted period']], on='KID')
        self.mask_transform = RandomMasking(mask_prob=0.2) if masked_transforms else None
        self.use_acf = use_acf
        self.use_fft = use_fft
        self.scale_flux = scale_flux
        self.labels = labels
        self.dims = 1 + self.use_acf + self.use_fft


    def __len__(self):
        return len(self.df)

    def read_lc(self, filename: str):
        """
        Reads a FITS file and returns the PDCSAP_FLUX and TIME columns as numpy arrays.

        Args:
            filename (str): The path to the FITS file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The PDCSAP_FLUX and TIME columns as numpy arrays.
        """
        with fits.open(filename) as hdulist:
                binaryext = hdulist[1].data
                meta = hdulist[0].header
        df = pd.DataFrame(data=binaryext)
        x = df['PDCSAP_FLUX']
        time = df['TIME'].values
        return x,time, meta

    def read_row(self, idx):
        """
        Reads a row from the DataFrame.
        """
        row = self.df.iloc[idx]

        # read row from npy file
        if self.npy_path is not None:
            try:
                file_path = os.path.join(self.npy_path, f"{int(row['KID'])}.npy")
                x = np.load(file_path)            
                if not isinstance(x, np.ndarray) or x.size == 0:
                    print(f"Warning: Empty or invalid numpy array for {row['KID']}")
                    x = np.zeros((self.seq_len, 1))
            except FileNotFoundError:
                print(f"Error: File not found for {row['KID']}")
                x = np.zeros((self.seq_len, 1))
            except Exception as e:
                print(f"Error loading file for {row['KID']}: {str(e)}")
                x = np.zeros((self.seq_len, 1))
            meta = dict()
            for key in row.keys():
                if key not in ['data_file_path']:
                    meta[key] = row[key]
            return x, meta

        # read row from FITS file
        else:
            paths = row['data_file_path']
            meta = {}
            for i in range(len(paths)):
                x, time, meta = self.read_lc(paths[i])
                meta = dict(meta.items())
                if i == 0:
                    x_tot = x.copy()
                else:
                    border_val = np.nanmean(x) - np.nanmean(x_tot)
                    x -= border_val
                    x_tot = np.concatenate((x_tot, np.array(x)))
                self.cur_len = len(x)
            return x_tot, meta

    def fill_nan_np(self, x:np.ndarray, interpolate:bool=True):
        """
        fill nan values in a numpy array

        Args:
                x (np.ndarray): array to fill
                interpolate (bool): whether to interpolate or not

        Returns:
            np.ndarray: filled array
        """
        non_nan_indices = np.where(~np.isnan(x))[0]
        nan_indices = np.where(np.isnan(x))[0]
        if len(nan_indices) and len(non_nan_indices):
            if interpolate:
                # Interpolate NaN values using linear interpolation
                interpolated_values = np.interp(nan_indices, non_nan_indices, x[non_nan_indices])
                # Replace NaNs with interpolated values
                x[nan_indices] = interpolated_values
            else:
                x[nan_indices] = 0	
        return x
    
    def random_chop(self, x, seq_len, pred_len, label_len=None):
        """
        Chop the input sequence into encoder input and decoder input for Autoformer.
        
        Args:
            x: Input tensor of shape [dims, length]
            seq_len: Length of encoder input sequence (e.g., 4800 for 100 days)
            pred_len: Length of prediction horizon (e.g., 1440 for 30 days)
            label_len: Length of known context for decoder (overlap with encoder sequence)
                       If None, defaults to 20% of seq_len
                       
        Returns:
            x_enc: Encoder input sequence [dims, seq_len]
            x_dec: Decoder input sequence [dims, label_len + pred_len]
            x_mark_enc: Time features for encoder with Kepler's resolution (1/48 days)
            x_mark_dec: Time features for decoder with Kepler's resolution (1/48 days)
            start: Starting index of the chopped sequence
        """
        # Default label_len to 20% of seq_len if not provided
        if label_len is None:
            label_len = int(seq_len * 0.2)
            
        # Ensure we have enough data
        total_required = 2 * seq_len
        if x.shape[0] < total_required:
            raise ValueError(f"Input sequence length {x.shape[0]} is shorter than required length {total_required}")
        
        # Random starting point
        max_start = x.shape[0] - total_required
        start = np.random.randint(0, max(1, max_start))
        
        # Encoder sequence
        end_enc = start + 2 * seq_len
        x_enc = x.clone()[start:end_enc, :]
        target_start = start + seq_len - pred_len // 2
        target_end = start + seq_len + pred_len // 2
        
        # Decoder sequence (overlapping label_len + prediction horizon)
        start_dec = target_start - label_len // 2
        end_dec = target_end + label_len // 2
        x_dec = x.clone()[start_dec:end_dec, :]
        x_enc[seq_len - pred_len // 2:seq_len + pred_len // 2, :] = 0
        
        # Create time features with Kepler's resolution (1/48 days)
        # Generate time stamps in days with resolution of 1/48
        t_enc = torch.arange(start, end_enc, dtype=torch.float32) / 48.0
        t_dec = torch.arange(start_dec, end_dec, dtype=torch.float32) / 48.0
        
        # Convert to time features format similar to Autoformer
        # For simplicity, we'll create a single feature (time in days)
        # In a real implementation, you might want to expand this to include
        # month, day, weekday, hour features from these timestamps
        x_mark_enc = t_enc.unsqueeze(-1)  # [seq_len, 1]
        x_mark_dec = t_dec.unsqueeze(-1)  # [label_len + pred_len, 1]
        
        return x_enc, x_dec, x_mark_enc, x_mark_dec, start
    
    def transform_data(self, x, transforms, info, idx):
        try:
            if transforms is not None:
                x, mask, info = self.transforms(x, mask=None, info=info)
            else:
                x = torch.tensor(x)
                mask = torch.zeros(1, self.seq_len)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            elif (len(x.shape) == 2) and (x.shape[0] == 1):
                x = x.squeeze(0).unsqueeze(-1)
            if self.use_acf:
                acf = torch.tensor(info['acf']).squeeze().unsqueeze(-1).nan_to_num(0)
                x = torch.cat((x, acf), dim=-1)
            if self.use_fft:
                fft = torch.tensor(info['fft']).squeeze().unsqueeze(-1).nan_to_num(0)
                x = torch.cat((x, fft), dim=-1)
        except Exception as e:
            print(f"Error in transforms for index {idx}: {str(e)}")
            traceback.print_exc()
            x = torch.zeros(self.seq_len, self.dims)
            mask = torch.zeros(self.seq_len, 1)
        return x, mask, info



    def __getitem__(self, idx):
        tic = time.time()
        x, info = self.read_row(idx)
        x = self.fill_nan_np(x, interpolate=True)
        info['idx'] = idx
        
        if self.scale_flux:
            x = (x - x.min()) / (x.max() - x.min())
            
        # Apply transformations if any
        x, _, info = self.transform_data(x, self.transforms, info, idx)
                
        # Get Autoformer inputs using the updated random_chop method
        try:
            x_enc, x_dec, x_mark_enc, x_mark_dec, start = self.random_chop(
                x, self.seq_len, self.target_len, label_len=self.label_len)
        except ValueError:
            start = 0
            x_enc, x_mark_enc = torch.zeros(2*self.seq_len, self.dims), torch.zeros(2*self.seq_len, 1)
            x_dec, x_mark_dec =  torch.zeros(self.target_len + self.label_len, self.dims), torch.zeros(self.target_len + self.label_len, 1)
        
        info['chop_start'] = start
        info['label_len'] = self.label_len
        
        # Replace NaN values with zeros
        # x_enc = x_enc.nan_to_num(0)
        # x_dec = x_dec.nan_to_num(0)
        # x_mark_enc = x_mark_enc.nan_to_num(0)
        # x_mark_dec = x_mark_dec.nan_to_num(0)
        
        # Ensure info is always a dictionary
        
        info = info if isinstance(info, dict) else {}

        # Return all Autoformer inputs
        result = (
            x_enc.float(),           # encoder input sequence
            x_mark_enc.float(),      # encoder time features
            x_dec.float(),           # decoder input sequence
            x_mark_dec.float(),      # decoder time features
            info
        )
        return result
class INRDataset(KeplerDataset):
    def __init__(self,
                df:pd.DataFrame=None,
                npy_path:str=None,
                transforms:object=None,
                scale_flux:bool=False
    ):
        self.df = df
        self.npy_path = npy_path
        self.transforms = transforms
        self.scale_flux = scale_flux
    
    def __getitem__(self, idx):
        x, info = self.read_row(idx)
        x = self.fill_nan_np(x, interpolate=True)
        info['idx'] = idx      
        if self.scale_flux:
            x = (x - x.min()) / (x.max() - x.min())
        if self.transforms is not None:
                x, mask, info = self.transforms(x, mask=None, info=info)
        
        if len(x.shape) > 1:
            x = x.squeeze()
        return torch.arange(len(x)).float(), x.float(), info

class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])

class KeplerINRDataset(KeplerDataset):
    def __init__(self,
                inr_folder:str='/data/TalkingStars/inr_data',
                **kwargs
    ):
        super().__init__(**kwargs)
        self.inr_folder = inr_folder
    
    def __getitem__(self, idx):
        x_enc, x_mark_enc, x_dec, x_mark_dec, info = super().__getitem__(idx)
        row = self.df.iloc[idx]
        try:
            state_dict = torch.load(os.path.join(self.inr_folder, f"{row['KID']}.pth"), map_location=lambda storage, loc: storage)
            weights = tuple(
                [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
            )
            biases = tuple([v for w, v in state_dict.items() if "bias" in w])
        except Exception as e:
            print("Error reading state dict: ", row['KID'], e)
            weights = (torch.zeros(1,2048), torch.zeros(2048,2048), torch.zeros(2048,1))
            biases = (torch.zeros(2048), torch.zeros(2048), torch.zeros(1))

        # Add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])
        return (x_enc.float(), x_mark_enc.float(), x_dec.float(), x_mark_dec.float(),
         Batch(weights, biases, row['KID']))
                


class LightSpecDataset(KeplerDataset):
    """
    A Multimodal dataset for spectra and lightcurve.
    Args:
        df (pd.DataFrame): DataFrame containing paths
        prot_df (pd.DataFrame): DataFrame containing rotation periods
        npy_path (str): Path to numpy files
        spec_path (str): Path to spectra files
        light_transforms (object): Transformations to apply to the lightcurve
        spec_transforms (object): Transformations to apply to the spectra
        light_seq_len (int): Sequence length for lightcurve
        spec_seq_len (int): Sequence length for spectra
        meta_columns (List[str]): Columns to use as metadata weights

    """
    def __init__(self, df:pd.DataFrame=None,
                prot_df:pd.DataFrame=None,
                npy_path:str=None,
                spec_path:str=None,
                light_transforms:object=None,
                spec_transforms:object=None,
                light_seq_len:int=34560,
                use_acf:bool=False,
                use_fft:bool=False,
                scale_flux:bool=True,
                spec_seq_len:int=3909,
                meta_columns = ['Teff', 'Mstar', 'logg'], labels=None
                ):
        super().__init__(df, prot_df, npy_path, light_transforms,
                light_seq_len, target_transforms=light_transforms, use_acf=use_acf, use_fft=use_fft, scale_flux=scale_flux)
        self.spec_path = spec_path
        self.spec_transforms = spec_transforms
        self.spec_seq_len = spec_seq_len
        self.meta_columns = meta_columns
        self.masked_transform = RandomMasking()
        self.labels = labels


    def read_spectra(self, filename):
        with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        rv = header['HELIO_RV']
        meta = {'RV': rv, 'wavelength': wv}
        return x, meta

    def __getitem__(self, idx):
        start = time.time()
        
        # Get Autoformer inputs from parent class
        light_enc, light_mark_enc, light_dec, light_mark_dec, info = super().__getitem__(idx)
        light_time = time.time() - start
        
        # Extract additional information
        kid = int(info['KID'])
        obsid = int(self.df.iloc[idx]['ObsID'])
        info['obsid'] = obsid
        obsdir = str(obsid)[:4]
        
        # Load spectra data
        spectra_filename = os.path.join(self.spec_path, f'{obsdir}/{obsid}.fits')
        try:
            spectra, meta = self.read_spectra(spectra_filename)
            spec_time = time.time() - start
        except OSError as e:
            print("Error reading file ", obsid, e)
            spectra = np.zeros((self.spec_seq_len))
            meta = {'RV': 0, 'wavelength': np.zeros(self.spec_seq_len)}
        
        # Transform spectra data
        if self.spec_transforms:
            spectra, _, spec_info = self.spec_transforms(spectra, None, meta)
            spec_transform_time = time.time() - start
            
        if spectra.shape[-1] < self.spec_seq_len:
            spectra = F.pad(spectra, ((0, self.spec_seq_len - spectra.shape[-1],0,0)), "constant", value=0)
            
        spectra = torch.nan_to_num(spectra, nan=0)
        masked_spectra, _, spec_info = self.masked_transform(spectra, None, spec_info)
        
        # Update info with spectra metadata
        info.update(spec_info)
        
        # Add metadata columns if specified
        if self.meta_columns is not None:
            w = torch.tensor([info[c] for c in self.meta_columns], dtype=torch.float32)
            info['w'] = w
            
        # Create target labels
        if self.labels is not None:
            y = torch.tensor([info[label] for label in self.labels], dtype=torch.float32)
        else:
            # Use part of the decoder sequence as the target (prediction horizon)
            label_len = info['label_len']
            y = light_dec[:, label_len:].clone()
        
        # Return all inputs needed for both lightcurve prediction and spectra analysis
        return (
            light_enc.float().squeeze(0),         # encoder input lightcurve
            light_mark_enc.float(),               # encoder time features
            light_dec.float().squeeze(0),         # decoder input lightcurve
            light_mark_dec.float(),               # decoder time features
            spectra.float().squeeze(0),           # spectra input
            masked_spectra.float().squeeze(0),    # masked spectra for training
            y,                                    # target labels or prediction
            info                                  # metadata
        )
    

class LightSpecDatasetV2(KeplerDataset):
    def __init__(self, lc_df:pd.DataFrame=None,
                 lc_data_dir:str=None,
                 spec_df:pd.DataFrame=None,
                 spec_data_dir:str=None,
                 shared_df:pd.DataFrame=None,
                 main_type:str='spectra',
                 spec_col:str='combined_obsid',
                 lc_col:str='KID',
                 light_transforms:object=None,
                 spec_transforms:object=None,
                 light_seq_len:int=13506,
                 spec_seq_len:int=4096,
                 **kwargs):
        self.lc_df = lc_df
        self.lc_data_dir = lc_data_dir
        self.spec_df = spec_df
        self.spec_data_dir = spec_data_dir
        self.shared_df = shared_df
        self.spec_col = spec_col
        self.lc_col = lc_col
        self.main_type = main_type
        self.spec_transforms = spec_transforms
        self.light_transforms = light_transforms
        self.light_seq_len = light_seq_len
        self.spec_seq_len = spec_seq_len
        super().__init__(df=lc_df,
                        transforms=light_transforms,
                         target_transforms=light_transforms,
                        seq_len=light_seq_len,
                         **kwargs)
        self.mask_transform = RandomMasking()
    def read_lamost_spectra(self, filename):
        with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        rv = header['HELIO_RV']
        meta = {'RV': rv, 'wavelength': wv}
        return x, meta
    
    def create_spectra_sample(self, spectra, meta):
        spectra_masked = spectra.copy()
        if self.spec_transforms:
            spectra, _, meta = self.spec_transforms(spectra, None, meta)
            spectra_masked, mask, _ = self.mask_transform(spectra, None, meta)
 
        if spectra_masked.shape[-1] < self.spec_seq_len:
            pad = torch.zeros(1, self.spec_seq_len - spectra_masked.shape[-1])
            spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
            pad_mask = torch.zeros(1, self.spec_seq_len  - mask.shape[-1], dtype=torch.bool)
            mask = torch.cat([mask, pad_mask], dim=-1)
            pad_spectra = torch.zeros(1, self.spec_seq_len - spectra.shape[-1])
            spectra = torch.cat([spectra, pad_spectra], dim=-1)
        spectra = torch.nan_to_num(spectra, nan=0)
        spectra_masked = torch.nan_to_num(spectra_masked, nan=0)
        return spectra, spectra_masked, mask, meta        

    def __getitem__(self, idx):
        if self.main_type == 'spectra':
            spec_id = self.spec_df.iloc[idx][self.spec_col]
            spec_dir = str(spec_id)[:4]
            filepath = f"{self.spec_data_dir}/{spec_dir}/{spec_id}.fits"
            spectra, meta = self.read_lamost_spectra(filepath)
            spectra, spectra_masked, mask, info_spec = self.create_spectra_sample(spectra, meta)
            if spec_id in self.shared_df[self.spec_col].values:
                shared_idx = self.shared_df[self.shared_df[self.spec_col] == spec_id].index[0]
                lc , lc_target, _, _, info_lc, _ = super().__getitem__(shared_idx)
            else: 
                lc = torch.zeros(1, self.seq_len) if not self.use_acf else torch.zeros(2, self.seq_len)
                lc_target = torch.zeroslike(lc)
                info_lc = {'data_dir': self.data_dir}
        else:
            lc, lc_target, _, _, info_lc, _ = super().__getitem__(idx)
            lc_id = self.lc_df.iloc[idx][self.lc_col]
            if lc_id in self.shared_df[self.lc_col].values:
                shared_idx = self.shared_df[self.shared_df[self.lc_col] == lc_id].index[0]
                filepath = f"{self.lc_data_dir}/{lc_id}.fits"
                spectra, meta = self.read_lamost_spectra(filepath)
                spectra, spectra_masked, mask, info_spec = self.create_spectra_sample(spectra,meta)
            else:
                spectra = torch.zeros(1, self.spec_seq_len)
                spectra_masked = torch.zeros(1, self.spec_seq_len)
                mask = torch.zeros(1, self.spec_seq_len)
                info_spec = {}
        return (lc.float().squeeze(0), spectra.float().squeeze(0), lc_target.float().squeeze(0),
            spectra_masked.float().squeeze(0), info_lc, info_spec)

class FineTuneDataset(LightSpecDataset):
    """
    A dataset for fine-tuning lightcurve and spectra models
    """
    def __init__(self,
                labels = ['inc'],
                **kwargs
                ):
        super().__init__(**kwargs)
        self.labels = labels
    
    def __getitem__(self, idx):
        light, spectra, _, light_target, spectra_target, info = super().__getitem__(idx)
        row = self.df.iloc[idx]
        y = torch.tensor([row[label] for label in self.labels], dtype=torch.float32)
        # print(f"Light time: {light_time}, Spec time: {spec_time}, Spec transform time: {spec_transform_time}")
        return (light, spectra, y, light_target, spectra_target, info)

def create_unique_loader(dataset, batch_size, num_workers=4, **kwargs):
    """
    Create a distributed data loader with the custom sampler
    """
    # sampler = DistributedUniqueLightCurveSampler(
    #     dataset, 
    #     batch_size=batch_size
    # )
    sampler = UniqueIDDistributedSampler(
        dataset, 
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        **kwargs
    )
