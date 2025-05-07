import multiprocessing as mp
from multiprocessing import Process, Manager
import os
import numpy as np
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as savgol
from inr import INR
import time

def train_process(process_id, batch_indices, all_batches, result_dict, 
                 num_iters=200, output_dir="inr_data", figs_dir="inr_figs"):
    """Worker process function that trains multiple INR models."""
    # Set up process-specific parameters
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    
    process_results = {}
    
    # Process each batch assigned to this process
    for batch_idx in batch_indices:
        batch = all_batches[batch_idx]
        coords, flux, info = batch
        
        # Skip if batch is invalid
        if coords is None or flux is None or info is None:
            continue
            
        coords = coords.squeeze().unsqueeze(-1)
        inr_criterion = torch.nn.MSELoss()
        
        # Initialize variables
        final_loss = np.inf
        n_iters = 0
        kid_id = info['KID']
        
        print(f"Process {process_id}: Starting training for KID {kid_id}")
        start_time = time.time()
        
        # Training loop
        while final_loss > 1e-4:
            # Sample a subset of data points to speed up training
            min_idx = max(coords.shape[0] - 20000, 0)
            start_idx = np.random.randint(0, min_idx)
            end_idx = start_idx + 20000
            coords_slice = coords[start_idx:end_idx]
            flux_slice = flux[start_idx:end_idx]
            
            # Initialize model
            model = INR(hidden_features=2048, n_layers=3, in_features=1, out_features=1)
            optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-4)
            
            # We'll use a list to store losses instead of tqdm for faster execution
            losses = []
            
            # Training loop for this attempt - no tqdm for speed
            for t in range(num_iters):
                optimizer.zero_grad()
                pred_values = model(coords_slice).float()
                loss = inr_criterion(pred_values.squeeze(), flux_slice)
                loss.backward()
                optimizer.step()
                
                # Only append loss occasionally to reduce overhead
                if t % 10 == 0 or t == num_iters - 1:
                    losses.append(loss.item())
                    if t % 50 == 0:
                        print(f"Process {process_id} - KID {kid_id}, iter {t}/{num_iters}, loss: {loss.item():.3e}")
            
            final_loss = losses[-1]
            n_iters += 1
            
            if n_iters > 4 or final_loss <= 1e-4:
                break
        
        # Save plot without blocking
        try:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            smooth_preds = savgol(pred_values.squeeze().detach().cpu().numpy(), 48, 1, mode='mirror', axis=0)
            axes[0].plot(coords_slice.squeeze().cpu().numpy(), flux_slice.squeeze().cpu().numpy())
            axes[0].plot(coords_slice.squeeze().cpu().numpy(), smooth_preds, c='r')
            axes[1].plot(losses)
            fig.suptitle(f"{kid_id}, final loss: {losses[-1]:.3e}")
            plt.tight_layout()
            plt.savefig(f"{figs_dir}/{kid_id}.png")
            plt.close(fig)
        except Exception as e:
            print(f"Error saving figure for KID {kid_id}: {str(e)}")
        
        # Save model state
        state_dict = model.state_dict()
        torch.save(state_dict, f"{output_dir}/{kid_id}.pth")
        
        # Store result
        process_results[str(kid_id)] = final_loss
        elapsed = time.time() - start_time
        print(f"Process {process_id}: Completed KID {kid_id} in {elapsed:.2f}s with loss {final_loss:.3e}")
    
    # Update the shared result dictionary
    result_dict.update(process_results)
    return process_results

def parallel_training(
    train_ds, 
    num_processes=None, 
    num_samples=float('inf'), 
    num_iters=200, 
    output_dir="inr_data", 
    figs_dir="inr_figs"
):
    """
    Train INR models in parallel using regular Python multiprocessing.
    
    Args:
        train_ds: Dataset containing (coords, flux, info) tuples
        num_processes: Number of CPU cores to use. If None, uses all available cores
        num_samples: Maximum number of samples to process
        num_iters: Number of iterations for each INR training
        output_dir: Directory to save trained models
        figs_dir: Directory to save figures
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    num_processes = min(num_processes, mp.cpu_count())
    
    print(f"Starting parallel training with {num_processes} CPU processes")
    
    # Pre-load all batches into memory to prevent loading in each process
    all_batches = []
    for i, batch in enumerate(train_ds):
        if i >= num_samples:
            break
        all_batches.append(batch)
    
    total_samples = len(all_batches)
    print(f"Loaded {total_samples} samples for processing")
    
    # Split batches among processes
    batch_indices_per_process = [[] for _ in range(num_processes)]
    for i in range(total_samples):
        batch_indices_per_process[i % num_processes].append(i)
    
    # Use a Manager to share results between processes
    with Manager() as manager:
        # Create a shared dictionary to collect results
        result_dict = manager.dict()
        
        # Create and start processes
        processes = []
        for i in range(num_processes):
            p = Process(
                target=train_process,
                args=(
                    i, 
                    batch_indices_per_process[i],
                    all_batches,
                    result_dict,
                    num_iters,
                    output_dir,
                    figs_dir
                )
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Convert manager dict to regular dict
        all_results = dict(result_dict)
    
    # Save combined results
    with open(os.path.join(output_dir, "losses.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Finished training {total_samples} INR models")
    return all_results

# Example usage
# if __name__ == "__main__":
    # Sample code to run the training
    # import your_dataset_module
    # train_ds = your_dataset_module.get_dataset()
    # results = parallel_training(train_ds, num_processes=4)