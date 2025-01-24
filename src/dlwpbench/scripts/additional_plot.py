#! /usr/env/bin python3
import torch as th
import matplotlib.pyplot as plt
import wandb

def plot_rmse_per_gridpoint(outputs_cat, targets_cat, epoch):
                
    # Calculate RMSE per gridpoint
    rmse_gridpoint = th.sqrt(th.mean((outputs_cat[:, 0, :, :, :] - targets_cat[:, 0, :, :, :]) ** 2, dim=[0,1])).cpu().numpy()
    
    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot RMSE gridpoint
    im = ax.imshow(rmse_gridpoint, cmap='viridis')
    ax.set_title(f"RMSE per Gridpoint")
    fig.colorbar(im, ax=ax)

    
    # Adjust layout
    plt.tight_layout()
    
    # Log the figure to wandb
    wandb.log({f"RMSE_validation/rmse_gridpoint_epoch{epoch}": wandb.Image(fig)})
    
    # Close the plot to free up memory
    plt.close(fig)

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