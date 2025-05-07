import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import json
from collections import OrderedDict
import warnings
import datetime
from tqdm import tqdm
import os 
# os.system('pip install omegaconf hydra-core')
# from omegaconf import OmegaConf


warnings.filterwarnings("ignore")

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.dataset import *
from dataset.sampler import BalancedDistributedSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from Autoformer.models.Autoformer import Model as Autoformer
from nn.models import *
from nn.rl_transf import RelationalTransformer
from nn.simsiam import SimSiam, projection_MLP
from nn.utils import init_model, load_checkpoints_ddp, deepnorm_init
from util.utils import *
# from PaddleSpatial.research.D3VAE.model import denoise_net as d3vae 

MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder,
            'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

DATASETS = {'LightSpec': LightSpecDataset, 'FineTune': FineTuneDataset, 'Simulation': SimulationDataset, 'Kepler':
                KeplerDataset, 'Spectra': SpectraDataset}

SIMULATION_DATA_DIR = '/data/simulations/dataset_big/lc'
SIMULATION_LABELS_PATH = '/data/simulations/dataset_big/simulation_properties.csv'


def get_kepler_data(data_args, df, transforms):

    return KeplerDataset(df=df,
                        transforms=transforms,
                        npy_path = '/data/lightPred/data/raw_npy',
                        seq_len=int(data_args.max_len_lc),
                        target_len=int(data_args.max_len_target_lc),
                        masked_transforms = data_args.masked_transform,
                        use_acf=data_args.use_acf,
                        use_fft=data_args.use_fft,
                        scale_flux=data_args.scale_flux,
                        labels=data_args.prediction_labels_lc,
                        dims=data_args.dim_lc,
                                )
def get_kepler_inr_data(data_args, df, transforms):

    return KeplerINRDataset(df=df,
                        transforms=transforms,
                npy_path= '/data/lightPred/data/raw_npy',
                scale_flux=data_args.scale_flux,
                seq_len=data_args.max_len_lc, 
                target_len=int(data_args.max_len_target_lc),
                label_len=data_args.max_len_label_lc,
                use_acf=data_args.use_acf,
                inr_folder='/data/TalkingStars/inr_data',
                                )
def get_lamost_data(data_args, df, transforms):

    return SpectraDataset(data_args.data_dir, transforms=transforms, df=df, 
                                 max_len=int(data_args.max_len_spectra),
                                 target_norm=data_args.target_norm,)

def get_lightspec_data(data_args, df, light_transforms, spec_transforms):

    return LightSpecDataset(df=df,
                            light_transforms=light_transforms,
                            spec_transforms=spec_transforms,
                            npy_path = '/data/lightPred/data/raw_npy',
                            spec_path = data_args.spectra_dir,
                            light_seq_len=int(data_args.max_len_lc),
                            spec_seq_len=int(data_args.max_len_spectra),
                            use_acf=data_args.use_acf,
                            use_fft=data_args.use_fft,
                            meta_columns=data_args.meta_columns_lightspec,
                            scale_flux=data_args.scale_flux,
                            labels=data_args.prediction_labels_lightspec
                            )

def get_data(data_args, data_generation_fn, dataset_name='FineTune', config=None):


    light_transforms = Compose([
                            MovingAvg(13),
                            ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                            Normalize(['none']),
                            ToTensor(), ])
    spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                                ToTensor()
                            ])
    if dataset_name == 'Kepler':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_lc)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        spec_transforms = None
        train_dataset = get_kepler_data(data_args, train_df, light_transforms)
        val_dataset = get_kepler_data(data_args, val_df, light_transforms)
        test_dataset = get_kepler_data(data_args, test_df, light_transforms)
    
    elif dataset_name == 'KeplerINR':
        light_transforms = Compose([
                            MovingAvg(48),
                            ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                            Normalize(['std']),
                            ToTensor(), ])
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_lc)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        spec_transforms = None
        train_dataset = get_kepler_inr_data(data_args, train_df, light_transforms)
        val_dataset = get_kepler_inr_data(data_args, val_df, light_transforms)
        test_dataset = get_kepler_inr_data(data_args, test_df, light_transforms)

    elif dataset_name == 'Spectra':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_spec)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                                ToTensor()
                            ])
        light_transforms = None
        train_dataset = get_lamost_data(data_args, train_df, spec_transforms)
        val_dataset = get_lamost_data(data_args, val_df, spec_transforms)
        test_dataset = get_lamost_data(data_args, test_df, spec_transforms)
    

    elif dataset_name == 'LightSpec':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_lightspec)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        train_dataset = get_lightspec_data(data_args, train_df, light_transforms, spec_transforms)
        val_dataset = get_lightspec_data(data_args, val_df, light_transforms, spec_transforms)
        test_dataset = get_lightspec_data(data_args, test_df, light_transforms, spec_transforms)

        

    else: 
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    if config is None:
        config = {}
    config.update({
        "light_transforms": str(light_transforms),
        "spec_transforms": str(spec_transforms),
        "train_dataset": str(train_dataset),
        "val_dataset": str(val_dataset),
        "test_dataset": str(test_dataset)
    }
    )
    return train_dataset, val_dataset, test_dataset, config

def get_model(data_args,
            args_dir,
            config,
             local_rank,
             load_individuals=False,
             freeze=False):
    
    light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'AutoFormer_lc'])
    optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'Optimization'])
    rl_transformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'rl_transformer'])
    graph_constructor_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'GraphConstructor'])
    # graph_constructor_args = OmegaConf.create(graph_constructor_args.get_dict())

    # inr_model = RelationalTransformer(
    #                               graph_constructor=graph_constructor_args.get_dict(),
    #                              args=rl_transformer_args).to(local_rank)
    model = Autoformer(light_model_args).to(local_rank)
    # model = INREncoderDecoder(encdec_model, inr_model).to(local_rank)
    if data_args.load_checkpoint:
        datetime_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
        exp_num = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
        print(datetime_dir)
        print("loading checkpoint from: ", data_args.checkpoint_path)
        moco = load_checkpoints_ddp(model, data_args.checkpoint_path)
        print("loaded checkpoint from: ", data_args.checkpoint_path)
    else:
        deepnorm_init(model, light_model_args)
        
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainble parameters: {num_params}")
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {all_params}")

    if config is None:
        config = {}
    config.update(
        {
            "model_args": str(light_model_args),
            "model": str(model),
    }
    )
    # model.apply(init_weights)
    return model, optim_args, light_model_args, config


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        # gain = torch.nn.init.calculate_gain('gelu')
        # print(gain)
        torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
        # m.bias.data.fill_(0.01)





