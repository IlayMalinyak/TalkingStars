import argparse
import torch
import datetime
import json
import yaml
import os
os.system('pip install -r requirements.txt')
from torch.nn.parallel import DistributedDataParallel as DDP


from main_model import CSDI_Forecasting
from dataset_kepler import get_dataloader
from utils import train, evaluate, setup_ddp

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="kepler")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true", default=False)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.datatype == 'kepler':
    target_dim = 1

config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.datatype + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

if config['train']['ddp']:
    device, world_size, gpus_per_node = setup_ddp()
else:
    device = args.device
    world_size = 1
    gpus_per_node = 1



train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= device,
    batch_size=config["train"]["batch_size"],
)

model = CSDI_Forecasting(config, device, target_dim).to(device)
if config['train']['ddp']:
    model = DDP(
        model, device_ids=[device], output_device=device, find_unused_parameters=True
    )

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
model.target_dim = target_dim
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    max_iter=2,
    foldername=foldername,
    config=config['train'],
)
