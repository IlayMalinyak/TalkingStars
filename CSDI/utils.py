import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import matplotlib.pyplot as plt


def filter_non_zeros(x, mask):
    first_mask = mask[0, 0]
    if torch.all(mask == first_mask.view(1, 1, -1)):
        mask = first_mask
    else:
        raise ValueError("Mask is not the same across batch and sequence dimensions")
    non_zero_indices = torch.nonzero(mask.view(-1), as_tuple=True)[0]

    # Extract the same indices from every (b,k) pair in the tensor
    b, k, _ = x.shape

    # Reshape tensor to (b*k, l)
    tensor_flat = x.view(b * k, -1)

    # Index the flattened tensor using the non-zero indices
    filtered_flat = tensor_flat[:, non_zero_indices]

    # Reshape back to (b, k, num_nonzeros)
    filtered_tensor = filtered_flat.view(b, k, -1)

    return filtered_tensor

def setup_ddp():
    """
    Setup the distributed training environment.
    """
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node. ", flush=True)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")
    return local_rank, world_size, gpus_per_node

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "" and (not config.get('ddp', False) or dist.get_rank() == 0):
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        
        # Set train sampler's epoch for proper shuffling in distributed training
        if config.get('ddp', False) and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch_no)
            
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0, 
                  disable=config.get('ddp', False) and dist.get_rank() != 0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                
                # Aggregate loss from all processes if using DDP
                if config.get('ddp', False):
                    loss_tensor = torch.tensor([loss.item()], device=loss.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss_item = loss_tensor.item() / dist.get_world_size()
                else:
                    avg_loss_item = loss.item()
                    
                avg_loss += avg_loss_item
                optimizer.step()
                
                if not config.get('ddp', False) or dist.get_rank() == 0:
                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": avg_loss / batch_no,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                # Set valid sampler's epoch for proper shuffling in distributed training
                if config.get('ddp', False) and hasattr(valid_loader.sampler, 'set_epoch'):
                    valid_loader.sampler.set_epoch(epoch_no)
                    
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0,
                          disable=config.get('ddp', False) and dist.get_rank() != 0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        
                        # Aggregate validation loss from all processes if using DDP
                        if config.get('ddp', False):
                            loss_tensor = torch.tensor([loss.item()], device=torch.device('cuda'))
                            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                            avg_loss_valid_item = loss_tensor.item() / dist.get_world_size()
                        else:
                            avg_loss_valid_item = loss.item()
                            
                        avg_loss_valid += avg_loss_valid_item
                        
                        if not config.get('ddp', False) or dist.get_rank() == 0:
                            it.set_postfix(
                                ordered_dict={
                                    "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                    "epoch": epoch_no,
                                },
                                refresh=False,
                            )
            
            # Ensure all processes have the same validation loss for comparison
            if config.get('ddp', False):
                avg_loss_valid_tensor = torch.tensor([avg_loss_valid], device=torch.device('cuda'))
                dist.all_reduce(avg_loss_valid_tensor, op=dist.ReduceOp.SUM)
                avg_loss_valid = avg_loss_valid_tensor.item() / dist.get_world_size()
                
                # Also sync batch_no across processes
                batch_no_tensor = torch.tensor([batch_no], device=torch.device('cuda'))
                dist.all_reduce(batch_no_tensor, op=dist.ReduceOp.MAX)
                batch_no = batch_no_tensor.item()
                
            # Only rank 0 saves the model and prints updates
            if (not config.get('ddp', False) or dist.get_rank() == 0) and best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                # Save the best model if foldername is provided
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
                    print("model saved at ", output_path)

    # Save final model (only rank 0 if using DDP)
    if foldername != "" and (not config.get('ddp', False) or dist.get_rank() == 0):
        torch.save(model.state_dict(), output_path)
        print("Final model saved at ", output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, max_iter=np.inf, foldername="", config=None):
    is_ddp = config is not None and config.get('ddp', False)
    rank = dist.get_rank() if is_ddp else 0
    world_size = dist.get_world_size() if is_ddp else 1

    print("is ddp:", is_ddp)

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        # Each process will collect its own results
        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        
        # Only rank 0 shows progress bar in DDP mode
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0, 
                  disable=is_ddp and rank != 0) as it:

            for batch_no, test_batch in enumerate(it, start=1):

                # fig, axes = plt.subplots(1, 2, figsize=(26, 16))
                # axes[0].plot(test_batch['timepoints'][0].squeeze().cpu(), test_batch['observed_data'][0],label="observed")
                # axes[1].plot(test_batch['timepoints'][0].squeeze().cpu(), test_batch['gt_mask'][0], label="target")
                # axes[0].legend()
                # axes[1].legend()
                # plt.savefig(f'/data/TalkingStars/figs/CSDI_input_{batch_no}.png')
                # plt.close()

                if is_ddp:
                    output = model.module.evaluate(test_batch, nsample)
                else:
                    output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                filtered_time = filter_non_zeros(observed_time.unsqueeze(1), eval_points)
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                samples_std = samples.std(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                filtered_samples_median = filter_non_zeros(samples_median.values.permute(0, 2, 1),
                                                           eval_points.permute(0, 2, 1))
                filtered_samples_std = filter_non_zeros(samples_std.permute(0, 2, 1),
                                                        eval_points.permute(0, 2, 1))

                plt.plot(observed_time[0].squeeze().cpu(), c_target[0].squeeze().cpu(), label='True')
                plt.plot(filtered_time[0].squeeze().cpu(), filtered_samples_median[0].squeeze().cpu(), label='Median')
                min_std = filtered_samples_median[0] - filtered_samples_std[0]
                max_std = filtered_samples_median[0] + filtered_samples_std[0]
                plt.fill_between(filtered_time[0].squeeze().cpu(),
                                 min_std.squeeze().cpu(),
                                 max_std.squeeze().cpu(),
                                 alpha=0.5,
                                 color='salmon',
                                 label='±1 std')
                plt.legend()
                plt.savefig(os.path.join(foldername, f"sample_prediction_{batch_no}.png"))
                plt.close()

                
                # fig, axes = plt.subplots(1, 3, figsize=(26, 16))
                # print(samples_median.values.shape, c_target.shape, eval_points.shape, observed_points.shape)
                # axes[0].plot(observed_time[0].squeeze().cpu(), observed_points[0].squeeze().cpu(), label="observed")
                # axes[1].plot(observed_time[0].squeeze().cpu(), c_target[0].squeeze().cpu(), label="target")
                # axes[2].plot(observed_time[0].squeeze().cpu(), samples_median.values[0].squeeze().cpu(), label="median prediction")
                # axes[0].legend()
                # axes[1].legend()
                # axes[2].legend()
                # plt.savefig(f'/data/TalkingStars/figs/CSDI_eval_{batch_no}.png')
                # plt.close()
                # Calculate the mean squared error and mean absolute error

                mse_current = (
                    ((samples_median.values - c_target) * (eval_points)) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * (eval_points))
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                if not is_ddp or rank == 0:
                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )
                if batch_no >= max_iter:
                    break
        
        # If using DDP, gather all evaluation metrics from all processes
        if is_ddp:
            # Create tensors for gathering metrics
            mse_tensor = torch.tensor([mse_total], device='cuda')
            mae_tensor = torch.tensor([mae_total], device='cuda')
            evalpoints_tensor = torch.tensor([evalpoints_total], device='cuda')
            
            # All-reduce to sum metrics across processes
            dist.all_reduce(mse_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mae_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(evalpoints_tensor, op=dist.ReduceOp.SUM)
            
            # Update local variables with global sums
            mse_total = mse_tensor.item()
            mae_total = mae_tensor.item()
            evalpoints_total = evalpoints_tensor.item()
            
            # We need to gather all the samples, targets, etc. from all processes
            # But this is complex and would require serialization/deserialization
            # So for simplicity, we'll only save results from rank 0 process
            # Or we can use all_gather to collect everything if needed

        # Only rank 0 saves results when using DDP
        if not is_ddp or rank == 0:
            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)
            
            return np.sqrt(mse_total / evalpoints_total), mae_total / evalpoints_total, CRPS
        
        # For other ranks in DDP, just return the computed metrics
        elif is_ddp:
            return np.sqrt(mse_total / evalpoints_total), mae_total / evalpoints_total, 0  # CRPS not computed on non-zero ranks