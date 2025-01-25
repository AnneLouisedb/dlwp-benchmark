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
from diffusers.schedulers import DDPMScheduler

sys.path.append("")
from data.datasets import *
from models import *
import utils.utils as utils
import wandb

import matplotlib.pyplot as plt
import io

# internal import 
from losses import CustomMSELoss
from additional_plot import *

@hydra.main(config_path='../configs/', config_name='config', version_base=None)
def run_training(cfg):
    """
    Trains a model with the given configuration, printing progress to console and tensorboard and writing checkpoints
    to file.

    :param cfg: The hydra-configuration for the training
    """
    assert cfg.training.sequence_length > cfg.model.context_size, 'No time steps to predict, increase the prediction window.'

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
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters\n")

    optimizer = th.optim.Adam(params=model.parameters(), lr=cfg.training.learning_rate)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.training.epochs)
    
    
    if cfg.training.type == 'diffusion':
        #self.ema = ExponentialMovingAverage(self.model, decay=self.hparams.ema_decay)
        # We use the Diffusion implementation here. Alternatively, one could
        # implement the denoising manually.
        betas = [cfg.training.min_noise_std ** (k / cfg.training.num_refinement_steps) for k in reversed(range(cfg.training.num_refinement_steps + 1))]
        # scheduling the addition of noise
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.training.num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction", # shouldnt this be epsilon?
            clip_sample=False,
        )
        # Multiplies k before passing to frequency embedding.
        time_multiplier = 1000 / cfg.training.num_refinement_steps

    

    #print(model)
    #exit()

    # Initialize training modules
    criterion = CustomMSELoss() 

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

            print("target size", target[0].shape, len(target))
            print("size prognostic??", prognostic[0].shape, len(prognostic))
            
            # target size torch.Size([16, 1, 3, 32, 64]) 
            #size prescribed?? torch.Size([16, 3, 1, 32, 64]) - takes 3 time steps 
            #torch.Size([16, 5, 3, 32, 64])
          
            # Perform optimization step and record outputs
            optimizer.zero_grad()
            for accum_idx in range(len(prognostic)):
            
                if cfg.training.type == 'diffusion':
                    
                    k = th.randint(0, cfg.training.num_refinement_steps, (1,), device=device)
                    k_scalar = k.item()
                    batch_size = prognostic[accum_idx].shape[0]
                    time_tensor = th.full((batch_size,), k_scalar, device=device)
                    # constructing the noise factor
                    noise_factor = noise_scheduler.alphas_cumprod.to(device)[k]
                    noise_factor = noise_factor.view(-1, *[1 for _ in range(prognostic[accum_idx].ndim - 1)])
                    signal_factor = 1 - noise_factor

                    target_new = target[accum_idx] 
                    noise = th.randn_like(target_new)
                    y_noised = noise_scheduler.add_noise(target_new, noise, k)
                    
                    #x_in = th.cat([prognostic[accum_idx][:, 0:cfg.model.context_size], y_noised], axis=1)
                    # [B, T, C, (F), H, W]
                    output = model.single_forward(constants[accum_idx], prescribed[accum_idx], prognostic[accum_idx], y_noised, time = time_tensor)

                    # output = model(
                    #     constants=constants[accum_idx] if not constants == None else None,
                    #     prescribed=prescribed[accum_idx] if not prescribed == None else None,
                    #     prognostic=x_in,
                    #     time = time_tensor * time_multiplier)
                    
                    if isinstance(target, tuple):
                        target = list(target)
                    
                    target[accum_idx] = (noise_factor**0.5) * noise - (signal_factor**0.5) * target[accum_idx]
                    target = tuple(target)
                

                else:

                    output = model(
                        constants=constants[accum_idx] if not constants == None else None,
                        prescribed=prescribed[accum_idx] if not prescribed == None else None,
                        prognostic=prognostic[accum_idx]
                    )
                
                train_loss = criterion(output, target[accum_idx])
                train_loss.backward()
                if cfg.training.clip_gradients:
                    try:
                        curr_lr = optimizer.param_groups[-1]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
                    except Exception:
                        curr_lr = optimizer.param_groups[-1]["lr"] # 0.001 - fix this for diffusion?
                        
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

                    if cfg.training.type == 'diffusion':
                        # implement all diffusion steps
                        output = model(
                        constants=constants[accum_idx] if not constants == None else None,
                        prescribed=prescribed[accum_idx] if not prescribed == None else None,
                        prognostic=prognostic[accum_idx],
                        noise_scheduler = noise_scheduler)


                    else:
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

                plot_rmse_per_gridpoint(outputs_cat, targets_cat, epoch)
                # plot final target predictions
                #plot_output_vs_target(outputs_cat, targets_cat, variable_list, epoch)

                
            for channel in range(outputs_cat.shape[2]):  
                # Log the loss per channel over the entire rollout

                channel_output = outputs_cat[:, :, channel, :, :] 
                channel_target = targets_cat[:, :, channel, :, :]
                channel_loss = criterion(channel_output, channel_target).item()
                losses.append(channel_loss)
                try:
                    wandb.log({f"MSE/validation_all_times_rollout/channel_{variable_list[channel]}":channel_loss}, step=iteration)
                except:
                    wandb.log({f"MSE/validation_all_times_rollout/channel_{channel}":channel_loss}, step=iteration)
                
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