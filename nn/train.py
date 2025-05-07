import torch
from torch.cuda.amp import autocast
import numpy as np
import time
import os
import yaml
import json
from matplotlib import pyplot as plt
import glob
from collections import OrderedDict
from tqdm import tqdm
import torch.distributed as dist
import umap
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import io
import zipfile
# import wandb

def count_occurence(x,y):
  coord_counts = {}
  for i in range(len(x)):
      coord = (x[i], y[i])
      if coord in coord_counts:
          coord_counts[coord] += 1
      else:
          coord_counts[coord] = 1

def save_compressed_checkpoint(model, save_path, results, use_zip=True):
        """
        Save model checkpoint with compression
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        if use_zip:
            # Save model state dict to buffer first
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer, 
                    _use_new_zipfile_serialization=True,
                    pickle_protocol=4)
            
            # Save buffer to compressed zip
            model_path = str(Path(save_path).with_suffix('.zip'))
            with zipfile.ZipFile(model_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('model.pt', buffer.getvalue())
                
            # Save results separately
            results_path = str(Path(save_path).with_suffix('.json'))
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
                
        else:
            # Save with built-in compression
            model_path = str(Path(save_path).with_suffix('.pt'))
            torch.save(model.state_dict(), model_path,
                    _use_new_zipfile_serialization=True,
                    pickle_protocol=4)
                
            results_path = str(Path(save_path).with_suffix('.json'))
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
                
        return model_path, results_path

def load_compressed_checkpoint(model, save_path):
    """
    Load model checkpoint from compressed format
    """
    if save_path.endswith('.zip'):
        with zipfile.ZipFile(save_path) as zf:
            with zf.open('model.pt') as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer)
    else:
        state_dict = torch.load(save_path)
    
    model.load_state_dict(state_dict)
    return model



class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    def __init__(self, model, optimizer, criterion, train_dataloader, device, world_size=1, output_dim=2,
                 scheduler=None, val_dataloader=None,   max_iter=np.inf, scaler=None,
                  grad_clip=False, exp_num=None, log_path=None, exp_name=None, plot_every=None,
                   cos_inc=False, range_update=None, accumulation_step=1, wandb_log=False, num_quantiles=1,
                   update_func=lambda x: x):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.grad_clip = grad_clip
        self.cos_inc = cos_inc
        self.output_dim = output_dim
        self.scheduler = scheduler
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.train_sampler = self.get_sampler_from_dataloader(train_dataloader)
        self.val_sampler = self.get_sampler_from_dataloader(val_dataloader)
        self.max_iter = max_iter
        self.device = device
        self.world_size = world_size
        self.exp_num = exp_num
        self.exp_name = exp_name
        self.log_path = log_path
        self.best_state_dict = None
        self.plot_every = plot_every
        self.logger = None
        self.range_update = range_update
        self.accumulation_step = accumulation_step
        self.wandb = wandb_log
        self.num_quantiles = num_quantiles
        self.update_func = update_func
        # if log_path is not None:
        #     self.logger =SummaryWriter(f'{self.log_path}/exp{self.exp_num}')
        #     # print(f"logger path: {self.log_path}/exp{self.exp_num}")

        # print("logger is: ", self.logger)
    
    def get_sampler_from_dataloader(self, dataloader):
        if hasattr(dataloader, 'sampler'):
            if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
                return dataloader.sampler
            elif hasattr(dataloader.sampler, 'sampler'):
                return dataloader.sampler.sampler
        
        if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'sampler'):
            return dataloader.batch_sampler.sampler
        
        return None
    
    def fit(self, num_epochs, device,  early_stopping=None, start_epoch=0, best='loss', conf=False):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        lrs = []
        epochs = []
        self.train_aux_loss_1 = []
        self.train_aux_loss_2 = []
        self.val_aux_loss_1 = []
        self.val_aux_loss_2 = []
        self.train_logits_mean = []
        self.train_logits_std = []
        self.val_logits_mean = []
        self.val_logits_std = []
        # self.optim_params['lr_history'] = []
        epochs_without_improvement = 0
        main_proccess = (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or self.device == 'cpu'

        print(f"Starting training for {num_epochs} epochs")
        print("is main process: ", main_proccess, flush=True)
        global_time = time.time()
        self.epoch = 0
        for epoch in range(start_epoch, start_epoch + num_epochs):
            epochs.append(epoch)
            self.epoch = epoch
            start_time = time.time()
            plot = (self.plot_every is not None) and (epoch % self.plot_every == 0)
            t_loss, t_acc = self.train_epoch(device, epoch=epoch)
            t_loss_mean = np.nanmean(t_loss)
            train_loss.extend(t_loss)
            global_train_accuracy, global_train_loss = self.process_loss(t_acc, t_loss_mean)
            if main_proccess:  # Only perform this on the master GPU
                train_acc.append(global_train_accuracy.mean().item())
                
            v_loss, v_acc = self.eval_epoch(device, epoch=epoch)
            v_loss_mean = np.nanmean(v_loss)
            val_loss.extend(v_loss)
            global_val_accuracy, global_val_loss = self.process_loss(v_acc, v_loss_mean)
            if main_proccess:  # Only perform this on the master GPU                
                val_acc.append(global_val_accuracy.mean().item())
                
                current_objective = global_val_loss if best == 'loss' else global_val_accuracy.mean()
                improved = False
                
                if best == 'loss':
                    if current_objective < min_loss:
                        min_loss = current_objective
                        improved = True
                else:
                    if current_objective > best_acc:
                        best_acc = current_objective
                        improved = True
                
                if improved:
                    model_name = f'{self.log_path}/{self.exp_num}/{self.exp_name}.pth'
                    print(f"saving model at {model_name}...")
                    torch.save(self.model.state_dict(), model_name)
                    self.best_state_dict = self.model.state_dict()
                    # model_path, output_filename = save_compressed_checkpoint(
                    #                            self.model, model_name, res, use_zip=True )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                res = {"epochs": epochs, "train_loss": train_loss, "val_loss": val_loss,
                        "train_acc": train_acc, "val_acc": val_acc, "train_aux_loss_1": self.train_aux_loss_1,
                        "train_aux_loss_2":self.train_aux_loss_2, "val_aux_loss_1":self.val_aux_loss_1,
                        "val_aux_loss_2": self.val_aux_loss_2, "train_logits_mean": self.train_logits_mean,
                         "train_logits_std": self.train_logits_std, "val_logits_mean": self.val_logits_mean,
                          "val_logits_std": self.val_logits_std, "lrs": lrs}

                current_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler is None \
                            else self.scheduler.get_last_lr()[0]
                
                lrs.append(current_lr)
                
                output_filename = f'{self.log_path}/{self.exp_num}/{self.exp_name}.json'
                with open(output_filename, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"saved results at {output_filename}")
                
                print(f'Epoch {epoch}, lr {current_lr}, Train Loss: {global_train_loss:.6f}, Val Loss:'\
                
                        f'{global_val_loss:.6f}, Train Acc: {global_train_accuracy.round(decimals=4).tolist()}, '\
                f'Val Acc: {global_val_accuracy.round(decimals=4).tolist()},'\
                  f'Time: {time.time() - start_time:.2f}s, Total Time: {(time.time() - global_time)/3600} hr', flush=True)
                if epoch % 10 == 0:
                    print(os.system('nvidia-smi'))

                if epochs_without_improvement == early_stopping:
                    print('early stopping!', flush=True)
                    break
                if time.time() - global_time > (23.83 * 3600):
                    print("time limit reached")
                    break 

        return {"epochs":epochs, "train_loss": train_loss,
                 "val_loss": val_loss, "train_acc": train_acc,
                "val_acc": val_acc, "train_aux_loss_1": self.train_aux_loss_1,
                "train_aux_loss_2": self.train_aux_loss_2, "val_aux_loss_1":self.val_aux_loss_1,
                "val_aux_loss_2": self.val_aux_loss_2, "train_logits_mean": self.train_logits_mean,
                 "train_logits_std": self.train_logits_std, "val_logits_mean": self.val_logits_mean,
                  "val_logits_std": self.val_logits_std, "lrs": lrs}

    def process_loss(self, acc, loss_mean):
        if  torch.cuda.is_available() and torch.distributed.is_initialized():
            global_accuracy = torch.tensor(acc).cuda()  # Convert accuracy to a tensor on the GPU
            torch.distributed.reduce(global_accuracy, dst=0, op=torch.distributed.ReduceOp.SUM)
            global_loss = torch.tensor(loss_mean).cuda()  # Convert loss to a tensor on the GPU
            torch.distributed.reduce(global_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            
            # Divide both loss and accuracy by world size
            world_size = torch.distributed.get_world_size()
            global_loss /= world_size
            global_accuracy /= world_size
        else:
            global_loss = torch.tensor(loss_mean)
            global_accuracy = torch.tensor(acc)
        return global_accuracy, global_loss

    def load_best_model(self, to_ddp=True, from_ddp=True):
        data_dir = f'{self.log_path}/exp{self.exp_num}'
        # data_dir = f'{self.log_path}/exp29' # for debugging

        state_dict_files = glob.glob(data_dir + '/*.pth')
        print("loading model from ", state_dict_files[-1])
        
        state_dict = torch.load(state_dict_files[-1]) if to_ddp else torch.load(state_dict_files[0],map_location=self.device)
    
        if from_ddp:
            print("loading distributed model")
            # Remove "module." from keys
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    while key.startswith('module.'):
                        key = key[7:]
                new_state_dict[key] = value
            state_dict = new_state_dict
        # print("state_dict: ", state_dict.keys())
        # print("model: ", self.model.state_dict().keys())

        self.model.load_state_dict(state_dict, strict=False)

    def check_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 100:
                   print(f"Large gradient in {name}: {grad_norm}")

    def train_epoch(self, device, epoch):
        """
        Trains the model for one epoch.
        """
        if self.train_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        self.model.train()
        train_loss = []
        train_acc = 0
        total = 0
        all_accs = torch.zeros(self.output_dim, device=device)
        pbar = tqdm(self.train_dl)
        for i, batch in enumerate(pbar):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            loss, acc , y = self.train_batch(batch, i + epoch * len(self.train_dl), device)
            train_loss.append(loss.item())
            all_accs = all_accs + acc
            total += len(y)
            pbar.set_description(f"train_acc: {acc}, train_loss:  {loss.item():.4f}")      
            if i > self.max_iter:
                break
        print("number of train_accs: ", all_accs, "total: ", total)
        return train_loss, all_accs/total
    
    def train_batch(self, batch, batch_idx, device):
        lc,spec,_,_, y,info = batch
        b, _, _ = lc.shape
        spec = spec.to(device)
        lc = lc.to(device)
        if isinstance(y, tuple):
            y = torch.stack(y)
        y = y.to(device)
        y_pred = self.model(lc.float(), spec.float())
        y_pred = y_pred.reshape(b, -1, self.output_dim)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        loss = 0
        for i in range(self.output_dim):
            y_pred_i = y_pred[:, :, i]
            y_i = y[:, i]
            loss += self.criterion(y_pred_i, y_i)
        loss /= self.output_dim
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        y_pred_mean = y_pred[:, y_pred.shape[1]//2, :]
        diff = torch.abs(y_pred_mean - y)
        acc = (diff < (y/10)).sum(0)
        # if self.wandb:
            # wandb.log({"train_loss": loss.item(), "train_acc": acc})
        return loss, acc, y

    def eval_epoch(self, device, epoch):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        total = 0
        all_accs = torch.zeros(self.output_dim, device=device)
        pbar = tqdm(self.val_dl)
        for i,batch in enumerate(pbar):
            loss, acc, y = self.eval_batch(batch, i + epoch * len(self.val_dl), device)
            val_loss.append(loss.item())
            all_accs = all_accs + acc
            total += len(y)
            pbar.set_description(f"val_acc: {acc}, val_loss:  {loss.item():.4f}")
            if i > self.max_iter:
                break
        if epoch % 10 == 0:
            print("running predict fn")
            preds, targets, sigmas, xs, ts, ts_dec, test_loss = self.predict(self.val_dl, device=device)
            print("loss from predict: ", np.mean(test_loss))
            seq_len = xs.shape[1]
            target_start = seq_len // 2 - self.pred_len // 2
            target_end = seq_len // 2 + self.pred_len // 2
            print("shapes: ", preds.shape, targets.shape, xs.shape, ts.shape)
            print("target start: ", target_start, "target end: ", target_end)
            for i in range(10):
                # plt.plot(ts[i], xs[i], label='context')
                plt.plot(ts_dec[i, self.label_len//2:-self.label_len//2].squeeze(), targets[i], label='target')
                plt.plot(ts_dec[i, self.label_len//2:-self.label_len//2].squeeze(), preds[i], label='pred')
                plt.legend()
                plt.savefig(f"/data/TalkingStars/figs/{self.exp_num}_{self.exp_name}_epoch_{self.epoch}_{i}.png")
                plt.close()
                print(f"fig {i} saved in /data/TalkingStars/figs/{self.exp_num}_{self.exp_name}_epoch_{self.epoch}_{i}.png")

        return val_loss, all_accs/total

    def eval_batch(self, batch, batch_idx, device):
        lc,spec,_,_,y,info = batch
        spec = spec.to(device)
        lc = lc.to(device)
        b, _, _ = lc.shape
        if isinstance(y, tuple):
            y = torch.stack(y)
        y = y.to(device)
        with torch.no_grad():
            y_pred= self.model(lc.float(), spec.float())
            y_pred = y_pred.reshape(b, -1, self.output_dim)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        loss = 0
        for i in range(self.output_dim):
            y_pred_i = y_pred[:, :, i]
            y_i = y[:, i]
            loss += self.criterion(y_pred_i, y_i)
        loss /= self.output_dim
        y_pred_mean = y_pred[:, y_pred.shape[1]//2, :]
        diff = torch.abs(y_pred_mean - y)
        acc = (diff < (y/10)).sum(0)
        # if self.wandb:
        #     wandb.log({"val_loss": loss.item(), "val_acc": acc})
        return loss, acc, y

    def predict(self, test_dataloader, device, load_best=True):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model(from_ddp=False)
        self.model.eval()
        preds = np.zeros((0, self.output_dim))
        targets = np.zeros((0, self.output_dim))
        confs = np.zeros((0, self.output_dim))
        tot_kic = []
        tot_teff = []
        for i,(lc_spec,_,_, y,info) in enumerate(test_dataloader):
            spec = spec.to(device)
            lc = lc.to(device)
            if isinstance(y, tuple):
                y = torch.stack(y)
            y = y.to(device)
            with torch.no_grad():
                y_pred = self.model(spec.float(), lc.float())
            y_pred = y_pred.reshape(b, -1, self.output_dim)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.output_dim:
                targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets

class AutoformerTrainer(Trainer):
    def __init__(self, pred_len, label_len, use_inr, alpha=0.5, beta=0.01, **kwargs):
        super().__init__(**kwargs)
        self.pred_len = pred_len
        self.label_len = label_len
        self.use_inr=use_inr
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha
        self.beta = beta

    def normalize_to_distribution(self, x):
        # Ensure all values are positive (you may need to shift data)
        shifted = x - x.min() + 1e-10
        # Normalize to sum to 1
        return shifted / shifted.sum()
    def train_batch(self, batch, batch_idx, device):
        x_enc, x_enc_mark, x_dec, x_dec_mark, info = batch
        x_enc = x_enc.to(device)
        x_enc_mark = x_enc_mark.to(device)
        x_dec = x_dec.to(device)
        x_dec_mark = x_dec_mark.to(device)
        if self.use_inr:
            inr = info.to(device)
        seq_len = x_enc.shape[1]
        dec_inp = torch.zeros_like(x_dec, device=device)
        dec_inp[:, :self.label_len // 2 :] = x_dec[:, :self.label_len // 2 :]
        dec_inp[:, -self.pred_len // 2 :] = x_dec[:, -self.pred_len // 2 :]
        dec_inp = dec_inp.float().to(self.device)
        if self.use_inr:
            preds = self.model(x_enc, x_enc_mark,dec_inp, x_dec_mark, inr)
        else:
            preds = self.model(x_enc, x_enc_mark,dec_inp, x_dec_mark)
        if isinstance(preds, tuple):
            dec_pred, log_sigma = preds[0], preds[1]
        else:
            dec_preds, log_sigma = preds, torch.zeros(x_enc.shape[0])

        boundary_loss = torch.abs(dec_pred[:, 0, 0] - x_dec[:, self.label_len // 2 - 1, 0]).mean()
        boundary_loss += torch.abs(dec_pred[:, -1, 0] - x_dec[:, -self.label_len // 2 + 1, 0]).mean()
        
        dec_pred = dec_pred[:, :, 0]
        x_dec = x_dec[:, self.label_len // 2 : -self.label_len // 2, 0]
        log_sigma = log_sigma.squeeze()
        mse_loss = self.criterion(dec_pred.squeeze(), x_dec.squeeze())
        loss = 0.5 * torch.exp(-log_sigma) * mse_loss + log_sigma / 2
        
        dist_pred = torch.log(self.normalize_to_distribution(dec_pred.squeeze()))
        dist_true = self.normalize_to_distribution(x_dec.squeeze())
        kl_loss = self.kl_loss(dist_pred, dist_true)
        loss = (1-self.beta) * loss.mean() + self.beta * kl_loss + boundary_loss
        # print("losses: " , log_sigma.mean().item(), mse_loss.mean().item(), kl_loss.mean().item(), boundary_loss.mean().item())
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()            
        return loss, 0, x_dec

    def eval_batch(self, batch, batch_idx, device):
        
        self.model.eval()
        x_enc, x_enc_mark, x_dec, x_dec_mark, info = batch
        x_enc = x_enc.to(device)
        x_enc_mark = x_enc_mark.to(device)
        x_dec = x_dec.to(device)
        x_dec_mark = x_dec_mark.to(device)
        if self.use_inr:
            inr = info.to(device)
        seq_len = x_enc.shape[1]
        dec_inp = torch.zeros_like(x_dec, device=device)
        dec_inp[:, :self.label_len // 2 :] = x_dec[:, :self.label_len // 2 :]
        dec_inp[:, -self.pred_len // 2 :] = x_dec[:, -self.pred_len // 2 :]
        dec_inp = dec_inp.float().to(self.device)
        with torch.no_grad():
            if self.use_inr:
                preds = self.model(x_enc, x_enc_mark, dec_inp, x_dec_mark, inr)
            else:
                preds = self.model(x_enc, x_enc_mark,dec_inp, x_dec_mark)
            if isinstance(preds, tuple):
                dec_pred, log_sigma = preds[0], preds[1].squeeze()
            else:
                dec_preds, log_sigma = preds, torch.zeros(x_enc.shape[0])
            
            boundary_loss = torch.abs(dec_pred[:, 0, 0] - x_dec[:, self.label_len // 2 - 1, 0]).mean()
            boundary_loss += torch.abs(dec_pred[:, -1, 0] - x_dec[:, -self.label_len // 2 + 1, 0]).mean()
            
            dec_pred = dec_pred[:, :, 0]
            x_dec = x_dec[:, self.label_len // 2 : -self.label_len // 2, 0]
            log_sigma = log_sigma.squeeze()
            mse_loss = self.criterion(dec_pred.squeeze(), x_dec.squeeze())
            loss = 0.5 * torch.exp(-log_sigma) * mse_loss + log_sigma / 2
            
            dist_pred = torch.log(self.normalize_to_distribution(dec_pred.squeeze()))
            dist_true = self.normalize_to_distribution(x_dec.squeeze())
            kl_loss = self.kl_loss(dist_pred, dist_true)
            loss = (1-self.beta) * loss.mean() + self.beta * kl_loss + boundary_loss
        
        return loss, 0, x_dec

    def predict_sample(self, batch, device):
        self.model.eval()
        x_enc, x_enc_mark, x_dec, x_dec_mark, info = batch
        x_enc = x_enc.to(device)
        x_enc_mark = x_enc_mark.to(device)
        x_dec = x_dec.to(device)
        x_dec_mark = x_dec_mark.to(device)
        with torch.no_grad():
            dec_inp = torch.zeros_like(x_dec[:, -self.pred_len:, :]).float()
            dec_inp = torch.cat([x_dec[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
            preds = self.model(x_enc, x_enc_mark,dec_inp, x_dec_mark)
            if isinstance(preds, tuple):
                dec_pred, log_sigma = preds[0], preds[1].squeeze()
            else:
                dec_preds, log_sigma = preds, torch.zeros(x.shape[0])
            dec_pred = dec_pred[:, -self.pred_len:, 0]
        return dec_pred, x_dec, log_sigma
    
    def predict(self, test_dataloader, device):
        self.model.eval()
        preds = np.zeros((0, self.pred_len))
        targets = np.zeros((0, self.pred_len))
        xs = []
        marks = []
        marks_dec = []
        sigmas = []
        losses = []
        pbar = tqdm(test_dataloader)
        for i,(x_enc, x_enc_mark, x_dec, x_dec_mark, info) in enumerate(pbar):
            x_enc = x_enc.to(device)
            x_enc_mark = x_enc_mark.to(device)
            x_dec = x_dec.to(device)
            x_dec_mark = x_dec_mark.to(device)
            with torch.no_grad():
                dec_inp = torch.zeros_like(x_dec[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([x_dec[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                preds_ = self.model(x_enc, x_enc_mark, dec_inp, x_dec_mark)
                if isinstance(preds_, tuple):
                    dec_pred, log_sigma = preds_[0], preds_[1].squeeze()
                else:
                    dec_preds, log_sigma = preds_, torch.zeros(x_enc.shape[0])
                
                seq_len = dec_pred.shape[1]
            
                target_start = seq_len // 2 - self.pred_len // 2
                target_end = seq_len // 2 + self.pred_len // 2
                boundary_loss = torch.abs(dec_pred[:, 0, 0] - x_dec[:, self.label_len // 2 - 1, 0]).mean()
                boundary_loss += torch.abs(dec_pred[:, -1, 0] - x_dec[:, -self.label_len // 2 + 1, 0]).mean()
                
                dec_pred = dec_pred[:, :, 0]
                x_dec = x_dec[:, self.label_len // 2 : -self.label_len // 2, 0]
                log_sigma = log_sigma.squeeze()
                mse_loss = self.criterion(dec_pred.squeeze(), x_dec.squeeze())
                loss = 0.5 * torch.exp(-log_sigma) * mse_loss + log_sigma / 2

                dist_pred = torch.log(self.normalize_to_distribution(dec_pred.squeeze()))
                dist_true = self.normalize_to_distribution(x_dec.squeeze())
                kl_loss = self.kl_loss(dist_pred, dist_true)
                loss = (1-self.beta) * loss.mean() + self.beta * kl_loss
                losses.append(loss.item())

                sigmas.extend(log_sigma.cpu().tolist())
                xs.append(x_enc[:, :, 0].cpu().numpy())
                marks.append(x_enc_mark.cpu().numpy())
                marks_dec.append(x_dec_mark.cpu().numpy())
                preds = np.append(preds, dec_pred.cpu().numpy(), axis=0)
                targets = np.append(targets, x_dec.cpu().numpy(), axis=0)

                pbar.set_description(f"test_loss:  {loss.item():.4f}")
            if i > self.max_iter:
                break
        xs = np.concatenate(xs, axis=0)
        marks = np.concatenate(marks, axis=0)
        marks_dec = np.concatenate(marks_dec, axis=0)
        sigmas = np.array(sigmas)
        return preds, targets, sigmas, xs, marks, marks_dec, losses