#! /usr/env/bin python3
import torch as th
import matplotlib.pyplot as plt
import wandb
import numpy as np

def plot_rmse_per_gridpoint(outputs_np, targets_np, epoch, lead_time):
    """Plotting the unweighted RMSE per gridpoint, taking the mean over the batch and time - first channel only"""
    if isinstance(outputs_np, np.ndarray):
        outputs = th.from_numpy(outputs_np)
    if isinstance(targets_np, np.ndarray):
        targets = th.from_numpy(targets_np)

    rmse_gridpoint = th.sqrt(th.mean((outputs - targets) ** 2, dim=[0])).cpu().numpy()
    
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot RMSE gridpoint
    im = ax.imshow(rmse_gridpoint, cmap='viridis', vmin=-1, vmax=2.5)
    ax.set_title(f"unweighted RMSE (lat-weighted), MSLP for {lead_time} days")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.close(fig)
    
    # Log the figure to wandb
    wandb.log({f"RMSE_validation/epoch_{epoch}_day_{lead_time+1}": wandb.Image(fig)})

    output = outputs[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot RMSE gridpoint
    im = ax.imshow(output, cmap='viridis')
    ax.set_title(f"MSLP at {lead_time} days")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.close(fig)

    wandb.log({f"Predictions/epoch_{epoch}_day_{lead_time+1}": wandb.Image(fig)})

    target = targets[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot RMSE gridpoint
    im = ax.imshow(target, cmap='viridis')
    ax.set_title(f"Target MSLP at {lead_time} days")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.close(fig)

    wandb.log({f"Predictions/TARGET_epoch_{epoch}_day_{lead_time+1}": wandb.Image(fig)})

    # Close the plot to free up memory
   

def plot_output_vs_target(outputs_cat, targets_cat, variable_list, epoch):

    # plot only the target variable
    channel = 0
    # Create a matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot output
    im1 = ax1.imshow(outputs_cat[0, -1, channel, :, :].cpu().numpy(), cmap='viridis')
    ax1.set_title(f"{variable_list[channel]} Output")
    fig.colorbar(im1, ax=ax1)
    
    # Plot target
    im2 = ax2.imshow(targets_cat[0, -1, channel, :, :].cpu().numpy(), cmap='viridis')
    ax2.set_title(f"{variable_list[channel]} Target")
    fig.colorbar(im2, ax=ax2)
    
    # Set a common title for the entire figure
    fig.suptitle(f"Variable: {variable_list[channel]}, Epoch {epoch}", fontsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Log the figure to wandb
    wandb.log({f"channel_{channel}_epoch{epoch}": wandb.Image(fig, caption=f"Predicted {channel}, single sample from batch")})
    
    # Close the plot to free up memory
    plt.close(fig)