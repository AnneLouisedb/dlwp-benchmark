#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import os
import sys
import gc
import threading
import xarray as xr
import hydra
import numpy as np
import torch as th
from diffusers.schedulers import DDPMScheduler, FlowMatchEulerDiscreteScheduler
import itertools
sys.path.append("")
from data.datasets import *
from models import *
import utils.utils as utils
import wandb
import einops
from omegaconf import OmegaConf
import time
import matplotlib.pyplot as plt
import io
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy

from scripts.helper_scripts.evaluation_helper import make_biweekly_inits
# internal import 
from losses import CustomMSELoss, MELRCalculator
from additional_plot import *
from helper_scripts.ema import ExponentialMovingAverage
from evaluate import remap
from helper_scripts.healpix_regridding import regrid_healpix_face


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

    wandb_config = OmegaConf.to_container(cfg)
    wandb.init(project="Final 2.0 Experiments", config = wandb_config) 

    if cfg.verbose: print("\nInitializing model")

    # Set up model
    model = eval(cfg.model.type)(**cfg.model).to(device=device)
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters\n")


    optimizer = th.optim.AdamW(params=model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.optimizer_weight_decay) 
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.training.epochs)
    train_timesteps = None

    if cfg.training.type == 'diffusion':

        log_dict = {}

        if cfg.training.flow_matching:

            noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=50)
            train_timesteps = 50

        elif cfg.training.ACDM:

            # def linear_beta_schedule(timesteps):
            #     if timesteps < 10:
            #         raise ValueError("Warning: Less than 10 timesteps require adjustments to this schedule!")

            #     beta_start = 0.0001 * (500/timesteps) # 0.0001 adjust reference values determined for 500 steps
            #     beta_end = 0.02 * (500/timesteps) # 
            #     betas = th.linspace(beta_start, beta_end, timesteps)

            #     return th.clip(betas, 0.0001, 0.9999)

            # betas = linear_beta_schedule(cfg.training.num_refinement_steps)

            # print("BETAS", betas)
        
            # noise_scheduler = DDPMScheduler(
            #     num_train_timesteps=cfg.training.num_refinement_steps ,
            #     trained_betas=betas,
            #     prediction_type="v_prediction", 
            #     clip_sample=False )
            #     # how to do ? squaredcos_cap_v2
                
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.training.num_refinement_steps,
                beta_schedule="squaredcos_cap_v2",  # Key parameter for cosine schedule
                prediction_type="v_prediction",
                clip_sample=False,
                # Remove trained_betas parameter to use predefined schedule
            )


            train_timesteps = cfg.training.num_refinement_steps

              
        else:
            
            betas = [cfg.training.min_noise_std ** (k / cfg.training.num_refinement_steps) for k in reversed(range(cfg.training.num_refinement_steps + 1))]
            # scheduling the addition of noise
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.training.num_refinement_steps + 1,
                trained_betas=betas,
                prediction_type="v_prediction", 
                clip_sample=False, 
            )
            train_timesteps = cfg.training.num_refinement_steps

            # ema = ExponentialMovingAverage(model, 0.995)
            # ema.register()

        # For Diffusion models and models in general working on small errors,
        # it is better to evaluate the exponential average of the model weights
        # instead of the current weights. If an appropriate scheduler with
        # cooldown is used, the test results will be not influenced.
        
    print(model)
    
    melr = MELRCalculator(device = device)
    # used for training
    criterion = CustomMSELoss(cfg,weighted = False, channel_weights=cfg.data.channel_weights, num_train_timesteps= train_timesteps)
    
    # used for creating latitude-weighted RMSE and ACC
    val_criterion = CustomMSELoss(cfg,weighted = True, reduction=None)
    # used for creating latitude-weighted per-variable loss
    val_criterion_red = CustomMSELoss(cfg, weighted = True)

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_best.ckpt")
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
    
    val_dataset = hydra.utils.instantiate( 
        cfg.data,
        statistics = cfg.data.train_start_date,
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

    # Initializing dataloader for testing
    init_dates = make_biweekly_inits(
            start=cfg.data.test_start_date,
            end=cfg.data.test_stop_date,
            sequence_length=cfg.testing.sequence_length,
            timedelta=cfg.data.timedelta
    )
    
    # Perform training by iterating over all epochs
    if cfg.verbose: print("\nStart training.")
    for epoch in range(epoch, cfg.training.epochs):

        wandb.log({"Epoch": epoch, "Learning Rate": optimizer.state_dict()["param_groups"][0]["lr"]}, step=iteration)

        # Train: iterate over all training samples
        outputs = list()
        targets = list()

        print("length of dataloader", len(train_dataloader))


        for train_idx, (constants, prescribed, prognostic, target) in enumerate(train_dataloader):
            # Prepare inputs and targets
            
            split_size = max(1, prognostic.shape[0]//cfg.training.gradient_accumulation_steps)
            constants = constants.to(device=device).split(split_size) if not constants.isnan().any() else None
            prescribed = prescribed.to(device=device).split(split_size) if not prescribed.isnan().any() else None

            assert prescribed != None

            prognostic = prognostic.to(device=device).split(split_size)
            target = target.to(device=device).split(split_size)


            # Perform optimization step and record outputs
            optimizer.zero_grad()

            if cfg.training.type == 'pushforward':
                
                print("Training with pushforward trick..")
              
                for accum_idx in range(len(prognostic)):

                
                    first_step = model(
                                constants=constants[accum_idx] if not constants == None else None,
                                prescribed=prescribed[accum_idx][:, :2] if not prescribed == None else None,
                                prognostic=prognostic[accum_idx][:, :2]
                            )

                    progs = th.cat([first_step, prognostic[accum_idx][:, 1:2]], dim=1)

                    second_step = model(
                                constants=constants[accum_idx] if not constants == None else None,
                                prescribed=prescribed[accum_idx][:, 1:3] if not prescribed == None else None,
                                prognostic=progs)


                    output = model(
                                constants=constants[accum_idx] if not constants == None else None,
                                prescribed=prescribed[accum_idx][:, 1:3] if not prescribed == None else None,
                                prognostic=prognostic[accum_idx][:, 1:3]
                            )
                    
                    train_loss = criterion(second_step, target[accum_idx][:, 1:3]) + criterion(output, target[accum_idx][:, 1:3])
                    
                    if isinstance(model, BARNNMUNetHPX):
                        print("Barnn + Pushforward trick")
                        train_loss = train_loss + model.kl

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


            else:
                
                for accum_idx in range(len(prognostic)):
                    # List of tensors?
                   
                    if cfg.training.input_perturbation:
                        # Code from BARNN paper
                        print("Train on perturbed input?")      
                        input_prog = prognostic[accum_idx]
                        perturbation_val = cfg.training.input_pertubation_val 
                        
                        perturbation = (perturbation_val * input_prog.abs().max() * 
                                        th.randn_like(input_prog))
                        input_prog =  input_prog + perturbation

                    else:
                        input_prog = prognostic[accum_idx]


                    if cfg.training.type == 'diffusion':
                        if cfg.training.arches:
                            with th.no_grad():

                                ensemble_mean_next_state = model.backbone(
                                constants=constants[accum_idx] if not constants == None else None,
                                prescribed=prescribed[accum_idx] if not prescribed == None else None,
                                prognostic=input_prog)
                                # [B, C, F, W, H] -> [B, T, C, F, W, H]
                            print("made residual target like ArchesWeather... ")
                            second_item =  ensemble_mean_next_state
                              
                        else:
                            second_item =  prognostic[accum_idx]
                            
                            
                        input_prog = einops.rearrange(second_item, "b t c f h w -> (b f) t c h w")
                        input_target = einops.rearrange(target[accum_idx], "b t c f h w -> (b f) t c h w")

                        
                        if input_target.shape[1] ==1:
                            target_res = (input_target - input_prog[:, cfg.model.context_size-1:cfg.model.context_size])
                            res_std = target_res.std()  # Measure variability in residuals
                            print('DIFFERENCE WEIGHT', res_std)
                            # i want to know what std i should rescale by, can you print the recommended difference weight?
                            target_res = target_res/ cfg.training.difference_weight

                           
                        else:
                            raise ValueError("Target time dimension (input_target.shape[1]) must be 1 ")

                         
                        k = th.randint(0, cfg.training.num_refinement_steps, (1,), device=device)
                        k_scalar = k.item()
                        batch_size = input_prog.shape[0]
                        time_tensor = th.full((batch_size,), k_scalar, device=device)
                        
                        # constructing the noise factor
                        if not cfg.training.flow_matching:
                            noise_factor = noise_scheduler.alphas_cumprod.to(device)[k]
                            noise_factor = noise_factor.view(-1, *[1 for _ in range(input_prog.ndim - 1)])
                            signal_factor = 1 - noise_factor

                        noise = th.randn_like(target_res)

                        if isinstance(noise_scheduler, DDPMScheduler):
                            y_noised = noise_scheduler.add_noise(target_res, noise, k)  
                            diffusion_loss = cfg.training.num_refinement_steps

                        else:
                            noise_scheduler.set_timesteps(50)
                            diffusion_loss = 50 #cfg.training.num_refinement_steps
                            
                            u = th.normal(mean=0, std=1, size= (batch_size,), device="cpu").sigmoid()
                            indices = (u * 50).long()
                            #indices = timesteps.cpu()

                            timesteps = noise_scheduler.timesteps[indices].to(device)

                            schedule_timesteps = noise_scheduler.timesteps.to(device)

                            sigmas = noise_scheduler.sigmas.to(device=device)

                            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                            sigma = sigmas[step_indices].flatten()[:, None, None, None, None]  

                            # used in the forward
                            y_noised = noise * sigma + target_res *(1.0 - sigma)

                            time_tensor = timesteps

                
                        # TO DO: add flow matching option here
                        # https://github.com/INRIA/geoarches/blob/main/geoarches/lightning_modules/diffusion.py

                        # Forward to model
                        if cfg.training.arches:
                            print("Training with arches setup... ")
                            output = model.single_forward(constants[accum_idx], prescribed[accum_idx][:, 0:cfg.model.context_size], ensemble_mean_next_state , y_noised, time = time_tensor)
                        
                        else:
                            print('Training with direct diffusion setup... ')
                            output = model.single_forward(constants[accum_idx], prescribed[accum_idx] [:, 0:cfg.model.context_size], prognostic[accum_idx][:, 0:cfg.model.context_size], y_noised, time = time_tensor)

                        
                        output = output.unsqueeze(1) 
              
                        if isinstance(target, tuple):
                            target = list(target)

                        if isinstance(noise_scheduler, DDPMScheduler):
                            target[accum_idx] = (noise_factor**0.5) * noise - (signal_factor**0.5) * target_res
                        else:
                            target[accum_idx] = target_res


                        target = tuple(target)
                        
                        assert output.shape == target[accum_idx].shape, f"{output.shape} != {target[accum_idx].shape}"


                        if np.isnan(output.detach().cpu().numpy()).any():
                            print(f"Error: Output contains only NaN values during training, {k_scalar} Quitting the Snellius job.")
                            sys.exit(1)  # Exit with an error code

                        train_loss = criterion(output, target[accum_idx], timesteps= time_tensor, diffusion= diffusion_loss)

                        del target_res, input_prog, noise, y_noised, time_tensor
                    
                        th.cuda.empty_cache()

                            
                    else:

                        if cfg.training.arches:
                            with th.no_grad():

                                ensemble_mean_next_state = model.backbone(
                                constants=constants[accum_idx] if not constants == None else None,
                                prescribed=prescribed[accum_idx] if not prescribed == None else None,
                                prognostic=input_prog)
                                # [B, C, F, W, H] -> [B, T, C, F, W, H]

                            print("made residual target like ArchesWeather... ")
                            input_prog =  ensemble_mean_next_state

                        output = model(
                                constants=constants[accum_idx] if not constants == None else None,
                                prescribed=prescribed[accum_idx] if not prescribed == None else None,
                                prognostic=input_prog
                            )

                        if isinstance(model, BARNNMUNetHPX):
                            print("is barn unet?")
                            train_loss = criterion(output, target[accum_idx]) + model.kl
                            
                        else:

                            train_loss = criterion(output, target[accum_idx])

                        
                        
                    # unweighted-mse loss for training
                    # output: [16, 7, 12, 8, 8] -> [B, C, F, W, H]
                    
                    train_loss.backward() 
                    
                    
                    if cfg.training.type == 'diffusion':
                        
                        log_dict[f"MSE_training/time_{k_scalar}"] = train_loss

                        # Log to wandb with the current iteration as the step
                        wandb.log(log_dict, step=iteration)

                        
                    if cfg.training.clip_gradients:
                        try:
                            curr_lr = optimizer.param_groups[-1]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
                        except Exception:
                            curr_lr = optimizer.param_groups[-1]["lr"] # 0.001 - fix this for diffusion?
                            
                        th.nn.utils.clip_grad_norm_(model.parameters(), curr_lr)
                        
                    outputs.append(output.detach().cpu())
                    targets.append(target[accum_idx].detach().cpu())
                        
                    optimizer.step()

                    # if cfg.training.type == 'diffusion':
                    #     ema.update()

                 

                    wandb.log({"MSE/training": train_loss}, step=iteration)
                    iteration += 1

            
        total_loss = 0
        num_samples = 0
        with th.no_grad():
            for output, target in zip(outputs, targets):
                output = output.cpu()
                target = target.cpu()
                # unweighted mse loss for trainnig
                batch_loss = criterion(output, target).item()
                total_loss += batch_loss * output.size(0)
                num_samples += output.size(0)
            epoch_train_loss = total_loss / num_samples

        # if cfg.training.type == 'diffusion':
        #     ema.apply_shadow()
            

        # Validate (without gradients)
        with th.no_grad():
            outputs = list()
            targets = list()
            persistence = list()
            print("Start Validation")
            
            for constants, prescribed, prognostic, target in val_dataloader:
                split_size = max(1, prognostic.shape[0]//cfg.validation.gradient_accumulation_steps)
                constants = constants.to(device=device).split(split_size) if not constants.isnan().any() else None
                prescribed = prescribed.to(device=device).split(split_size) if not prescribed.isnan().any() else None
                prognostic = prognostic.to(device=device).split(split_size)
                target = target.to(device=device).split(split_size)
                
                
                for accum_idx in range(len(prognostic)):
                    if accum_idx == 0:
                        original_prog = prognostic[accum_idx][:, -1, :, :, :]

                    if cfg.training.type == 'diffusion':

                        inference_scheduler = deepcopy(noise_scheduler)
                        inference_scheduler.set_timesteps(cfg.model.num_refinement_step)
                    
                        print("Validating Diffusion Model")
                         
                        output = model(
                        constants=constants[accum_idx] if not constants == None else None,
                        prescribed=prescribed[accum_idx] if not prescribed == None else None,
                        prognostic=prognostic[accum_idx],
                        noise_scheduler = inference_scheduler, target = target[accum_idx])

                        # np.random.seed(cfg.seed+1)
                        # th.manual_seed(cfg.seed+1)

                        # output2 = model(
                        # constants=constants[accum_idx] if not constants == None else None,
                        # prescribed=prescribed[accum_idx] if not prescribed == None else None,
                        # prognostic=prognostic[accum_idx],
                        # noise_scheduler = inference_scheduler, target = target[accum_idx])

                        # np.random.seed(cfg.seed+2)
                        # th.manual_seed(cfg.seed+2)

                        # output3 = model(
                        # constants=constants[accum_idx] if not constants == None else None,
                        # prescribed=prescribed[accum_idx] if not prescribed == None else None,
                        # prognostic=prognostic[accum_idx],
                        # noise_scheduler = inference_scheduler, target = target[accum_idx])

                        # np.random.seed(cfg.seed)
                        # th.manual_seed(cfg.seed)

                        # # take the mean over the outputs?
                        # out_ = th.stack([output1, output2, output3], dim=0)
                        # output = out_.mean(dim=0)
        
                        
                    else:
                        # The output is tensor of length 6
                        if isinstance(model, BARNNMUNetHPX):
                            # BARNN
                            output_list = []
                            print("Taking ensemble of 25 members")
                            for i in range(25):

                                output = model(
                                constants=constants[accum_idx] if not constants == None else None,
                                prescribed=prescribed[accum_idx] if not prescribed == None else None,
                                prognostic=prognostic[accum_idx]
                                )
                                output_list.append(output)

                            # take the mean over the outputs?
                            out_ = th.stack(output_list , dim=0)
                            output = out_.mean(dim=0)

                        # Deterministic UNET
                        else:
                            output = model(
                                constants=constants[accum_idx] if not constants == None else None,
                                prescribed=prescribed[accum_idx] if not prescribed == None else None,
                                prognostic=prognostic[accum_idx]
                            )

                        
                    # persistence
                    persistence.append(original_prog.cpu())
                    outputs.append(output.cpu())
                    targets.append(target[accum_idx].cpu())

            
            persistence_cat = th.cat(persistence)
            outputs_cat = th.cat(outputs)
            targets_cat = th.cat(targets)
            variable_list = list(cfg.data.prognostic_variable_names_and_levels.keys())
            ### MSE

            r2_scores = {}
            mse_scores = {}
            persistence_r2 = {}
            persistence_mse = {}

            for var_idx, var_name in enumerate(variable_list):

                r2_scores[var_name] = []
                mse_scores[var_name] = []
                persistence_r2[var_name] = []
                persistence_mse[var_name] = []

                ### R^2
                # Calculate R² for each variable and timestep
            
                for t in range(outputs_cat.shape[1]):

                    # True and predicted values
                    y_true = targets_cat[:, t, var_idx, :, :, :].flatten().numpy() 
                    y_pred = outputs_cat[:, t, var_idx, :, :, :].flatten().numpy()
                
                    # B, C, F, H, W
                    y_persist = persistence_cat[:, var_idx, :,:, :].flatten().numpy()

                    # Model scores
                    r2 = r2_score(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    
                    # Persistence scores
                    persist_r2 = r2_score(y_true, y_persist)
                    persist_mse = mean_squared_error(y_true, y_persist)

                    r2_scores[var_name].append(r2)
                    mse_scores[var_name].append(mse)
                    persistence_r2[var_name].append(persist_r2)
                    persistence_mse[var_name].append(persist_mse)

            if cfg.validation.sequence_length > 7:
                # Log comparison metrics
                for var_name in variable_list:
                
                    xs = [1, 2, 3, 5, 7]  # Forecast days
                    ys = [
                        [mse_scores[var_name][0], mse_scores[var_name][1], mse_scores[var_name][2], mse_scores[var_name][4], mse_scores[var_name][6]],  # Validation MSE
                        [persistence_mse[var_name][0], persistence_mse[var_name][1], persistence_mse[var_name][2],persistence_mse[var_name][4],persistence_mse[var_name][6]]  # Persistence MSE
                    ]

                    wandb.log({
                    f"MSE/{var_name}": wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=["Validation", "Persistence"],
                        title=f"MSE Comparison: {var_name}",
                        xname="Forecast Day"
                    )
                    }, step=iteration)

                    xs = [1,2,3]
                    ys = [
                        [np.mean(mse_scores[var_name][0:7]),np.mean(mse_scores[var_name][7:14]), np.mean(mse_scores[var_name][14:21])],  # Validation MSE
                        [np.mean(persistence_mse[var_name][0:7]),np.mean(persistence_mse[var_name][7:14]), np.mean(persistence_mse[var_name][14:21])]  # Persistence MSE
                    ]

                    wandb.log({
                    f"MSE/{var_name}": wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=["Validation", "Persistence"],
                        title=f"Weekly MSE: {var_name}",
                        xname="Forecast Week"
                    )
                    }, step=iteration)

                 
                    
            # MSE LOSS PER TIMESTEP - VALIDATION 
            if cfg.model.mesh == 'healpix': 
                # [B, T, C, (F), H, W]
                # leaves only the mean per time
                mean_loss_per_time_step = val_criterion(outputs_cat, targets_cat).mean(dim=(0, 2, 3, 4, 5)).cpu().numpy()

                mean_loss_per_time_step_msl = val_criterion(outputs_cat[:,:,0,:,:,:], targets_cat[:,:,0,:,:,:]).mean(dim=(0, 2, 3, 4)).cpu().numpy()
    
            else: 
                # [B, T, C, W, H]
                mean_loss_per_time_step = val_criterion(outputs_cat, targets_cat).mean(dim=(0, 2, 3, 4)).cpu().numpy() 
            
            if cfg.validation.sequence_length > 7:
                log_dict = {
                "MSE_validation/time_1": mean_loss_per_time_step[0],
                "MSE_validation/time_2": mean_loss_per_time_step[1],
                "MSE_validation/time_3": mean_loss_per_time_step[2],
                "MSE_validation/time_4": mean_loss_per_time_step[3],
                "MSE_validation/time_5": mean_loss_per_time_step[4],
                "MSE_validation/time_7": mean_loss_per_time_step[6],
                "MSE_validation/week2_mean": np.mean(mean_loss_per_time_step[7:14]),
                "MSE_validation/week3_mean": np.mean(mean_loss_per_time_step[15:22])
                }
                wandb.log(log_dict, step=iteration)
            else:
                log_dict = {
                "MSE_validation/time_1": mean_loss_per_time_step[0]}
                wandb.log(log_dict, step=iteration)

            # Create lead days (x-axis)
            lead_days = np.arange(1, len(mean_loss_per_time_step) + 1)  # Assuming time_step is 1 day

            # Create the plot data
            table_val = wandb.Table(columns=["lead_day", "MSE_loss"])
            for i, loss in enumerate(mean_loss_per_time_step):
                table_val.add_data(lead_days[i], loss)

            # Create the plot data
            table_val_msl = wandb.Table(columns=["lead_day", "MSE_loss MSLP"])
            for i, loss in enumerate(mean_loss_per_time_step_msl):
                table_val_msl.add_data(lead_days[i], loss)
                
            # Log the table to W&B
            wandb.log({
                "MSE_loss_vs_lead_day": wandb.plot.line(
                    table_val,
                    x="lead_day",
                    y="MSE_loss",
                    title="(all variables) MSE Loss vs. Lead Day"
                )
            })

            # Log the table to W&B
            wandb.log({
                "MSE_loss_vs_lead_day_MSLP": wandb.plot.line(
                    table_val_msl,
                    x="lead_day",
                    y="MSE_loss MSLP",
                    title="MSLP MSE Loss vs. Lead Day"
                )
            })

            # Usage in your code
            if epoch % 5 == 0 and cfg.model.mesh == 'healpix' and cfg.validation.sequence_length > 7:
                print("REMAPPING")
       
                # remapping the MSLP from HPX to lat-lon representation (TIME 0)
                #if L < 10:
                vals = [0, 2, 4, 6, 8]
                if cfg.validation.sequence_length > 40:
                    vals = [0, 6, 11, 18, 25, 32, 39]
                

                for time in vals: # day 1, day 3, day 5, day 7, day 14
                    output_ = outputs_cat[:, time, :, :, :, :]
                    target_ = targets_cat[:, time, :, :, :, :]

                    print("storing the tensorts!!")
                    # Example: store output_ and target_ tensors for each time point
                    th.save(output_, f"output_time{time}.pt")
                    th.save(target_, f"target_time{time}.pt")

                    # store this dataset somewhere and break the loop

                    if cfg.data.degree < 5.6:
                        output_ = regrid_healpix_face(outputs_cat[:, time, :, :, :, :], target_size=8)
                        target_ = regrid_healpix_face(targets_cat[:, time, :, :, :, :], target_size=8)
                        print("regridding data!")
                    
                    outputs_right_0 = remap(cfg=cfg, data=output_, name="Outputs") #   # [B, T, C, (F), H, W]
                    targets_right_0 = remap(cfg=cfg, data=target_, name="Targets") 
                    
                    # Plotting the MELR metric to lead-day [time]
                    print("shape reshaped output")
                    print(outputs_right_0.shape) # (32, 13, 90, 180)

                    melr.apply(outputs_right_0[:, 0, :, :], targets_right_0[:, 0, :, :], variable_name=f'msl_day_{time}', epoch = epoch)
                    melr.apply(outputs_right_0[:, 1, :, :], targets_right_0[:, 1, :, :], variable_name=f'geopotential_1000_day_{time}', epoch = epoch)
                    try:
                        melr.apply(outputs_right_0[:, 6, :, :], targets_right_0[:, 6, :, :], variable_name=f'geopotential_500_day_{time}', epoch = epoch)
                    except:
                        pass
 
                    plot_rmse_per_gridpoint(outputs_right_0[:, 0, :, :], targets_right_0[:, 0, :, :], epoch, time) 

                    print(f"DONE RMSE {time}") 

        epoch_val_loss = val_criterion_red(outputs_cat, targets_cat).numpy()
        wandb.log({"MSE/validation": epoch_val_loss}, step=iteration)
        # if cfg.training.type == 'diffusion':
        #     ema.restore()
       
    
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

            print(f"Epoch {str(epoch).zfill(3)}/{str(cfg.training.epochs)}\t"
                  f"MSE train: {'%.2E' % epoch_train_loss}\t"
                  f"MSE val: {'%.2E' % epoch_val_loss}")

        # Update learning rate
        scheduler.step()
        th.cuda.empty_cache()                
        gc.collect()                            

  

if __name__ == "__main__":
    run_training()
    print("Done.")