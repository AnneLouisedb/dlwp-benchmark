#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import threading
import xarray as xr
import hydra
import numpy as np
import torch as th

sys.path.append("")
from data.datasets import *
from models import *
import utils.utils as utils
import wandb

import matplotlib.pyplot as plt
import io

# internal import 
from losses import CustomMSELoss

@hydra.main(config_path='../configs/', config_name='config', version_base=None)
def run_training(cfg):
    """
    Trains a model with the given configuration, printing progress to console and tensorboard and writing checkpoints
    to file.

    :param cfg: The hydra-configuration for the training
    """

    if cfg.seed:
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
    device = th.device(cfg.device)

    wandb.init(project="dlwp-benchmark_scalingup") 

    if cfg.verbose: print("\nInitializing model")

    # Set up model
    model = eval(cfg.model.type)(**cfg.model).to(device=device)
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters")

    #print(model)
    #exit()

    # Initialize training modules
    criterion = CustomMSELoss() #th.nn.MSELoss()

    optimizer = th.optim.Adam(params=model.parameters(), lr=cfg.training.learning_rate)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.training.epochs)

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_last.ckpt")
    if cfg.training.continue_training:
        if cfg.verbose: print(f"\tRestoring model from {checkpoint_path}")
        checkpoint = th.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        iteration = checkpoint["iteration"]
        best_val_error = checkpoint["best_val_error"]
    else:
        epoch = 0
        iteration = 0
        best_val_error = np.infty

    # Write the model configurations to the model save path
    os.makedirs(os.path.join("outputs", cfg.model.name), exist_ok=True)

    
    if cfg.verbose: print("\nInitializing datasets")

    print("DATA", cfg.data)

    # Initializing dataloaders for training and validation
    train_dataset = hydra.utils.instantiate(
        cfg.data,
        start_date=cfg.data.train_start_date,
        stop_date=cfg.data.train_stop_date,
        sequence_length=cfg.training.sequence_length
    )
    print("Train dataset")
    print(train_dataset)
    val_dataset = hydra.utils.instantiate(
        cfg.data,
        start_date=cfg.data.val_start_date,
        stop_date=cfg.data.val_stop_date,
        sequence_length=cfg.validation.sequence_length
    )
    train_dataloader = th.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    val_dataloader = th.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # Perform training by iterating over all epochs
    if cfg.verbose: print("\nStart training.")
    for epoch in range(epoch, cfg.training.epochs):

        wandb.log({"Epoch": epoch, "Learning Rate": optimizer.state_dict()["param_groups"][0]["lr"]}, step=iteration)


        start_time = time.time()

        # Train: iterate over all training samples
        outputs = list()
        targets = list()
        
        for train_idx, (constants, prescribed, prognostic, target) in enumerate(train_dataloader):
            # Prepare inputs and targets
            split_size = max(1, prognostic.shape[0]//cfg.training.gradient_accumulation_steps)
            constants = constants.to(device=device).split(split_size) if not constants.isnan().any() else None
            prescribed = prescribed.to(device=device).split(split_size) if not prescribed.isnan().any() else None
            prognostic = prognostic.to(device=device).split(split_size)
            target = target.to(device=device).split(split_size)

            # Perform optimization step and record outputs
            optimizer.zero_grad()
            for accum_idx in range(len(prognostic)):
                output = model(
                    constants=constants[accum_idx] if not constants == None else None,
                    prescribed=prescribed[accum_idx] if not prescribed == None else None,
                    prognostic=prognostic[accum_idx]
                )
                
                train_loss = criterion(output, target[accum_idx])
                train_loss.backward()
                if cfg.training.clip_gradients:
                    curr_lr = optimizer.param_groups[-1]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
                    th.nn.utils.clip_grad_norm_(model.parameters(), curr_lr)
                outputs.append(output.detach().cpu())
                targets.append(target[accum_idx].detach().cpu())
            optimizer.step()
            wandb.log({"MSE/training": train_loss}, step=iteration)
            iteration += 1
        with th.no_grad(): epoch_train_loss = criterion(th.cat(outputs), th.cat(targets)).numpy()

        # Validate (without gradients)
        with th.no_grad():
            outputs = list()
            targets = list()

            for constants, prescribed, prognostic, target in val_dataloader:
                split_size = max(1, prognostic.shape[0]//cfg.validation.gradient_accumulation_steps)
                constants = constants.to(device=device).split(split_size) if not constants.isnan().any() else None
                prescribed = prescribed.to(device=device).split(split_size) if not prescribed.isnan().any() else None
                prognostic = prognostic.to(device=device).split(split_size)
                target = target.to(device=device).split(split_size)
                for accum_idx in range(len(prognostic)):
                    output = model(
                        constants=constants[accum_idx] if not constants == None else None,
                        prescribed=prescribed[accum_idx] if not prescribed == None else None,
                        prognostic=prognostic[accum_idx]
                    )
                    outputs.append(output.cpu())
                    targets.append(target[accum_idx].cpu())

            
            losses = []
            outputs_cat = th.cat(outputs)
            targets_cat = th.cat(targets)
            variable_list = list(cfg.data.prognostic_variable_names_and_levels.keys())

            criterion_wo_reduction = th.nn.MSELoss(reduction='none')

            # MSE LOSS PER TIMESTEP - VALIDATION
            mean_loss_per_time_step = criterion_wo_reduction(outputs_cat, targets_cat).mean(dim=(0, 2, 3, 4)).cpu().numpy()    
            
            log_dict = {
                f"MSE_validation/time_{i}": value 
                for i, value in enumerate(mean_loss_per_time_step)
            }

            # Log to wandb with the current iteration as the step
            wandb.log(log_dict, step=iteration)

            # Compute the mean loss over the first time step
            if epoch % 10 == 0:
                
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
                

                # Find the ACC scores
                #path_to_climatology = os.path.join("outputs", "climatology", "evaluation", "outputs.nc")
                # path_to_climatology = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/climatology_1981-2010'
                # print("\tComputing ACC...")

                # ds_climatology = xr.open_dataset(path_to_climatology)

                #outputs_cat, targets_cat #).mean(dim=(0, 2, 3, 4)).cpu().numpy()


                # diff_out_clim = ds_outputs - ds_climatology
                # diff_tar_clim = ds_targets - ds_climatology

                    
                # nom = (lat_weights*diff_out_clim*diff_tar_clim).mean(dim=mean_over)
                # denom = np.sqrt(
                #         (lat_weights*diff_out_clim**2).mean(dim=mean_over) * (lat_weights*diff_tar_clim**2).mean(dim=mean_over)
                #     )
                # accs = nom/denom
                    

            
            for channel in range(outputs_cat.shape[2]):  # Iterate over channels

                channel_output = outputs_cat[:, :, channel, :, :] 
                channel_target = targets_cat[:, :, channel, :, :]
                channel_loss = criterion(channel_output, channel_target).item()
                losses.append(channel_loss)
                wandb.log({f"MSE/validation_all_times_rollout/channel_{variable_list[channel]}":channel_loss}, step=iteration)
                
                
            epoch_val_loss = criterion(outputs_cat, targets_cat).numpy()

        wandb.log({"MSE/validation": epoch_val_loss}, step=iteration)
       
        # Write model checkpoint to file, using a separate thread
        if cfg.training.save_model:
            if epoch_val_loss > best_val_error or epoch == cfg.training.epochs - 1:
                dst_path = checkpoint_path
            else:
                best_val_error = epoch_val_loss
                dst_path = f"{checkpoint_path.replace('last', 'best')}"
            thread = threading.Thread(
                target=utils.write_checkpoint,
                args=(model, optimizer, scheduler, epoch, iteration, best_val_error, dst_path, ))
            thread.start()
            

            #Log checkpoint information with wandb
            wandb.log({
                "checkpoint_saved": True,
                "checkpoint_path": dst_path,
                "best_val_error": best_val_error,
                "epoch": epoch,
                "iteration": iteration
            }, step=iteration)
            
        
        # Print training progress to console
        if cfg.verbose:
            epoch_time = round(time.time() - start_time, 2)
            print(f"Epoch {str(epoch).zfill(3)}/{str(cfg.training.epochs)}\t"
                  f"{epoch_time}s\t"
                  f"MSE train: {'%.2E' % epoch_train_loss}\t"
                  f"MSE val: {'%.2E' % epoch_val_loss}")

        # Update learning rate
        scheduler.step()

  

if __name__ == "__main__":
    run_training()
    print("Done.")