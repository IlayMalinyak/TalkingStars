
import os
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
from astropy.table import Table
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import json
from collections import OrderedDict
import warnings
import datetime
warnings.filterwarnings("ignore")

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.dataset import KeplerDataset, INRDataset
from nn.astroconf import Astroconformer
from nn.models import *
from Autoformer.models.Autoformer import Model as Autoformer
from nn.train import *
from nn.utils import deepnorm_init, load_checkpoints_ddp
from nn.optim import ForecastLoss
from util.utils import *
import generator

R_SUN_KM = 6.957e5

def set_seed(fix_seed):
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

def priority_merge_prot(dataframes, target_df):
    """
    Merge 'Prot' values from multiple dataframes into target dataframe in priority order,
    using 'KID' as the merge key. Much more efficient implementation.
    
    Args:
        dataframes: List of dataframes, each containing 'Prot' and 'KID' columns (in decreasing priority order)
        target_df: Target dataframe to merge 'Prot' values into (must contain 'KID' column)
    
    Returns:
        DataFrame with aggregated 'Prot' values merged into target_df
    """
    # Create a copy of the target dataframe
    result = target_df.copy()
    
    # Create an empty dataframe with just KID and Prot columns
    prot_values = pd.DataFrame({'KID': [], 'Prot': [], 'Prot_ref': []})
    
    # Process dataframes in priority order
    for df in dataframes:
        print(f"Processing dataframe with {len(df)} rows. currently have {len(prot_values)} prot values")
        # Extract just the KID and Prot columns
        current = df[['KID', 'Prot', 'Prot_ref']].copy()
        
        # Only add keys that aren't already in our prot_values dataframe
        missing_keys = current[~current['KID'].isin(prot_values['KID'])]
        
        # Concatenate with existing values
        prot_values = pd.concat([prot_values, missing_keys])
    
    # Merge the aggregated Prot values into the result dataframe
    result = result.merge(prot_values, on='KID', how='left')
    
    return result

def create_train_test_dfs(meta_columns):
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    berger_cat = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
    kepler_meta = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    activity_cat = pd.read_csv('/data/logs/activity_proxies/proxies_full_dist.csv')
    kepler_df = kepler_df.merge(berger_cat, on='KID').merge(kepler_meta[['KID', 'KMAG']], on='KID', how='left').merge(activity_cat, on='KID', how='left')
    kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5

    lightpred_df = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
    lightpred_df['Prot_ref'] = 'lightpred'
    lightpred_df.rename(columns={'predicted period': 'Prot'}, inplace=True)

    santos_df = pd.read_csv('/data/lightPred/tables/santos_periods_19_21.csv')
    santos_df['Prot_ref'] = 'santos'
    mcq14_df = pd.read_csv('/data/lightPred/tables/Table_1_Periodic.txt')
    mcq14_df['Prot_ref'] = 'mcq14'
    reinhold_df = pd.read_csv('/data/lightPred/tables/reinhold2023.csv')
    reinhold_df['Prot_ref'] = 'reinhold'

    p_dfs = [lightpred_df, santos_df, mcq14_df, reinhold_df]
    kepler_df = priority_merge_prot(p_dfs, kepler_df)

    drop_cols = [c for c in kepler_df.columns if 'Unnamed' in c]
    kepler_df = kepler_df.drop(columns=drop_cols)
    kepler_df = kepler_df[kepler_df['num_qs'] > 2]

    kois = pd.read_csv('/data/lightPred/tables/kois.csv')

    kepler_df = kepler_df[~kepler_df['KID'].isin(kois['KID'])]

    file_kids = set()
    for filename in os.listdir('/data/TalkingStars/inr_data'):
        if filename.endswith('.pth'):
            kid = filename.replace('.pth', '')
            file_kids.add(kid)

    # Step 2: Filter the dataframe based on file KIDs
    kepler_df = kepler_df[kepler_df['KID'].astype(str).isin(file_kids)]

    print("total number of samples: ", len(kepler_df))

    train_df, test_df = train_test_split(kepler_df, test_size=0.2, random_state=1234)
    return train_df, test_df


set_seed(1234)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder,
            "CNNEncoderDecoder": CNNEncoderDecoder, 'CNNRegressor': CNNRegressor}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"light_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/TalkingStars/nn/full_config.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

train_df, val_df = create_train_test_dfs(data_args.meta_columns_lc)

light_transforms = Compose([
                            MovingAvg(13),
                            ToTensor(), ])

# inr_dataset = INRDataset(df=train_df, npy_path='/data/lightPred/data/raw_npy',
#                          transforms=light_transforms, scale_flux=data_args.scale_flux
#                            )


train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
                                                                            data_generation_fn=create_train_test_dfs,
                                                                            dataset_name='Kepler')

for i in range(10):
    lc_enc, t_enc, lc_dec, t_dec, info = train_dataset[i]
    print(lc_enc.shape, t_enc.shape, lc_dec.shape, t_dec.shape)
    plt.plot(t_enc[:, 0], lc_enc[:, 0], label='enc')
    plt.plot(t_dec[:, 0], lc_dec[:, 0], label='dec')
    plt.legend()
    plt.savefig(f'/data/TalkingStars/figs/dataset_{i}.png')
    plt.close()

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_dataset,
                              batch_size=int(data_args.batch_size), \
                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                              collate_fn=kepler_collate_fn,
                              sampler=train_sampler)


val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)
val_dataloader = DataLoader(val_dataset,
                            batch_size=int(data_args.batch_size),
                            sampler=val_sampler, 
                            collate_fn=kepler_collate_fn,
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

test_dataloader = DataLoader(test_dataset,
                            batch_size=int(data_args.batch_size),
                            collate_fn=kepler_collate_fn,
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

model, optim_args, model_args, complete_config = generator.get_model(data_args, args_dir, complete_config, local_rank)

if data_args.create_umap:
    umap_df = create_umap(model.module.simsiam.encoder, test_dataloader, local_rank, use_w=False, dual=False)
    print("umap created: ", umap_df.shape)
    umap_df.to_csv(f"{data_args.log_dir}/{checkpoint_dir}/umap_{checkpoint_exp}_ssl.csv", index=False)
    print(f"umap saved at {data_args.log_dir}/{checkpoint_dir}/umap_{checkpoint_exp}_ssl.csv")
    exit()
    
loss_fn = ForecastLoss(reduction='none')
optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()
total_steps = int(data_args.num_epochs) * len(train_dataloader)
scheduler = OneCycleLR(
        optimizer,
        max_lr=float(optim_args.max_lr),
        epochs= int(data_args.num_epochs),
        steps_per_epoch = len(train_dataloader),
        pct_start=float(optim_args.warmup_pct),
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=10.0,
        final_div_factor=100.0
    )


trainer = AutoformerTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, output_dim=1, scaler=scaler,
                    scheduler=scheduler, train_dataloader=train_dataloader, use_inr=False,
                    val_dataloader=val_dataloader, device=local_rank, pred_len=model_args.pred_len,
                            label_len=model_args.label_len, alpha=data_args.alpha, beta= data_args.beta,
                        exp_num=datetime_dir, log_path=data_args.log_dir, 
                        max_iter=np.inf,
                        exp_name=f"lc_{exp_num}") 


# Save the complete configuration to a YAML file
complete_config.update(
    {"trainer": trainer.__dict__,
    "loss_fn": str(loss_fn),
    "optimizer": str(optimizer)}
)
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

print(f"Configuration (with model structure) saved at {config_save_path}.")


fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=40, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/lc_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_lc_{exp_num}.png")
plt.clf()

preds, targets, sigmas, xs, ts, ts_dec, losses = trainer.predict(val_dataloader, device=local_rank)
# v_loss, v_acc = trainer.eval_epoch(local_rank, epoch=1000)
# print('val loss: ', np.mean(v_loss))
print(preds.shape, targets.shape, len(sigmas), xs.shape, ts.shape, ts_dec.shape)

seq_len = xs.shape[1]
target_start = seq_len // 2 - model_args.pred_len // 2
target_end = seq_len // 2 + model_args.pred_len // 2
print("shapes: ", preds.shape, targets.shape, xs.shape, ts.shape, ts_dec.shape)
print("target start: ", target_start, "target end: ", target_end)
print(ts_dec[i, target_start:target_end].shape, targets[i].shape)
for i in range(10):
    # plt.plot(ts[i], xs[i], label='context')
    plt.plot(ts_dec[i, model_args.label_len//2:-model_args.label_len//2].squeeze(), targets[i], label='target')
    plt.plot(ts_dec[i, model_args.label_len//2:-model_args.label_len//2].squeeze(), preds[i], label='pred')
    plt.legend()
    plt.savefig(f"/data/TalkingStars/figs/{data_args.exp_num}_test_{i}.png")
    plt.close()

# predict_results(trainer, val_dataloader, test_dataloader, loss_fn,
#                  data_args.prediction_labels_lc,
#                  data_args, optim_args, 'light', exp_num,
#                   datetime_dir, local_rank, world_size)

# umap_df = create_umap(model.module.simsiam.encoder, test_dataloader, local_rank, use_w=False, dual=False)
# print("umap created: ", umap_df.shape)
# umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
# print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
