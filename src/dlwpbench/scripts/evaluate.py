#! bin/env/python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import gc  # Garbage collector
import sys
import time
import shutil
import argparse
import threading
import subprocess
import multiprocessing
from tqdm import tqdm
import wandb
import hydra
import numpy as np
import torch as th
import pandas as pd
import xarray as xr


from omegaconf import DictConfig
from dask.diagnostics import ProgressBar
import dask.array as da

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from diffusers.schedulers import DDPMScheduler
sys.path.append("")
from data.datasets import *
from data.processing.healpix_mapping import HEALPixRemap 
from data.processing.istarmap import istarmap
from scripts.build_baselines import persistence_forecast

from scripts.helper_scripts.evaluation_helper import get_adjusting_weights, make_biweekly_inits, remap, generate_mp4
from models import *

from scripts.chaosbench.criterion import RMSE
from scripts.losses import MELRCalculator



MODEL_NAME_PLOT_ARGS = {
    "persistence": {"c": "dimgray", "ls": "solid", "label": "Persistence"},
    "climatology": {"c": "darkgrey", "ls": "solid", "label": "Climatology"},
    "clstm16m_cyl_4x228_v2": {"c": "yellowgreen", "ls": "solid", "label": "ConvLSTM (16M)"},
    "clstm16m_hpx8_4x228_v1": {"c": "yellowgreen", "ls": "dashed", "label": "ConvLSTM HPX (16M)"},
    "unet128m_cyl_128-256-512-1024-2014_v2": {"c": "darkgreen", "ls": "solid", "label": "U-Net (128M)"},
    "unet16m_hpx8_92-184-368-736_v0": {"c": "darkgreen", "ls": "dashed", "label": "U-Net HPX (16M)"},
    "swint2m_cyl_d88_l2x4_h2x4_v0": {"c": "darkorange", "ls": "solid", "label": "SwinTransformer (2M)"},
    "swint16m_hpx8_d120_l3x4_h3x4_v2": {"c": "darkorange", "ls": "dashed", "label": "SwinTransformer HPX (16M)"},
    "fno2d64m_cyl_d307_v0": {"c": "lightcoral", "ls": "solid", "label": "FNO2D (64M)"},
    "tfno2d128m_cyl_d477_v0": {"c": "darkturquoise", "ls": "solid", "label": "TFNO2D (128M)"},
    "fcnet4m_emb272_nopos_l6_v1": {"c": "firebrick", "ls": "solid", "label": r"FourCastNet $p=1x1$ (4M)"},
    "fcnet8m_emb384_nopos_p2x4_l6_v2": {"c": "goldenrod", "ls": "solid", "label": r"FourCastNet $p=2x4$ (8M)"},
    "fcnet64m_emb940_nopos_p4x4_l8_v0": {"c": "orangered", "ls": "solid", "label": r"FourCastNet $p=4x4$ (64M)"},
    "sfno2d128m_cyl_d686_equi_nonorm_nopos_v0": {"c": "steelblue", "ls": "solid", "label": "SFNO (128M)"},
    "pangu32m_d216_h6-12-12-6_v1": {"c": "deepskyblue", "ls": "solid", "label": "Pangu-Weather (32M)"},
    "mgn32m_l8_d470_v0": {"c": "blueviolet", "ls": "solid", "label": "MeshGraphNet (32M)"},
    "gcast16m_p4_b1_d565_v2": {"c": "darkblue", "ls": "solid", "label": "GraphCast (16M)"},
}

def build_dataset_ensemble(
    cfg: DictConfig,
    outputs: np.array,
    statistics: dict,
    init_dates: np.array,
    file_path: str,
    complevel: int = 7
):
    """
    Creates a netCDF dataset for initializations, outputs, and targets and writes them to file.
    
    :param cfg: The hydra configuration of the model
    :param inits: The first frame of the prognostic inputs to the model
    :param outputs: The outputs of the model (predictions)
    :param targets: The ground truth and target for prediction
    :param statistics: Dictionary containing mean and standard deviations per variable and level
    :param init_dates: The dates where the forecasts are initialized
    :param file_path: The path to the directory where the datasets are written to
    """

    # Determine data dimensions and set resolution in degree
    M, B, T, D, H, W = outputs.shape

    deg = cfg.data.degree 

    dt = f"{cfg.data.timedelta*24}h"
    timedeltas = pd.timedelta_range(start=dt, periods=T, freq=dt)

    # Set up netCDF dataset
    coords = {}
    coords['member'] = np.arange(M)
    coords["sample"] = init_dates
    coords["time"] = timedeltas
    if deg == 2.0:
        coords["lat"] = np.flip(np.array(np.arange(start=-90, stop=90, step=deg), dtype=np.float32))
    else:
        coords["lat"] = np.flip(np.array(np.arange(start=-87.1875, stop=90, step=deg), dtype=np.float32))
    coords["lon"] = np.array(np.arange(start=0, stop=360, step=deg), dtype=np.float32)
    chunkdict = {coord: len(coords[coord]) for coord in coords}
    chunkdict["sample"] = 1
    chunkdict["member"] = 1

    # Prepare dictionaries for initializations, outputs, and targets to create according Datasets below
    v_idx = 0
    inits_dict = dict()
    outputs_dict = dict()
    for p in cfg.data.prognostic_variable_names_and_levels:
        if len(cfg.data.prognostic_variable_names_and_levels[p]) > 0:
            for l in cfg.data.prognostic_variable_names_and_levels[p]:
                vname = f"{p}{l}"
                attrs = statistics[p]["level"][l]
                outputs_dict[vname] = xr.DataArray(data=outputs[:, :, v_idx], dims=["member","sample", "time", "lat", "lon"], attrs=attrs)
                v_idx += 1
        else:
            vname = p
            attrs = statistics[p]
            outputs_dict[vname] = xr.DataArray(data=outputs[:, :, v_idx], dims=["member", "sample", "time", "lat", "lon"], attrs=attrs)
            v_idx += 1

    # Create datasets and write them to file
    def write_to_file(ds: xr.Dataset, dst_path_name: str, compress_dict: dict):
        if os.path.exists(dst_path_name): os.remove(dst_path_name)  # Delete file if it exists
        print(f"\tWriting to {dst_path_name}")
        if "outputs" in dst_path_name:  # Display progress bar when writing the targets.nc to file
            write_job = ds.to_netcdf(dst_path_name, compute=False, encoding=compress_dict)
            with ProgressBar(): write_job.compute()
        else:
            ds.to_netcdf(dst_path_name, encoding=compress_dict)  # Silently write inits.nc and targets.nc

    print("\nWriting datasets to file. This may take a while.")# Optionally, reduce compression level via the -z flag")
    # Remove threading, write sequentially
    # Extract the model name from the path

    write_to_file(
        xr.Dataset(coords=coords, data_vars=outputs_dict).chunk(chunkdict),
        os.path.join(file_path, "ensemble.nc"), 
        compress_dict
    )
    print("stored ensemble.nc")

def rmse_cal(ds_outputs_oct2017, ds_targets_oct2017, ds_ec46):
    # Select October 2017 from your model outputs and targets
    ds_outputs_oct2017 = ds_outputs_oct2017.isel(time=slice(0, 46))
    ds_targets_oct2017 = ds_targets_oct2017.isel(time=slice(0, 46))
    ds_ec46 = ds_ec46.isel(time=slice(0, 46))

    rmse_model = np.sqrt(np.mean((ds_outputs_oct2017.msl.values - ds_targets_oct2017.msl.values)**2, axis=(1, 2,3))) 
    # Compute RMSE between EC46 and targets
    rmse_ec46 = np.sqrt( np.mean(((ds_ec46.msl.values - ds_targets_oct2017.msl.values)**2), axis = (1, 2,3)))
    diff = np.sqrt(np.mean(((ds_ec46.msl.values - ds_outputs_oct2017.msl.values)**2) , axis = (1, 2,3)))
    # Compute relative improvement
    relative_improvement = (rmse_ec46 - rmse_model) / rmse_ec46 * 100

    rmse_model_raw = np.sqrt( np.mean(((ds_outputs_oct2017.msl.values - ds_targets_oct2017.msl.values)**2), axis = (2,3)))

    rmse_ec46_raw = np.sqrt( np.mean(((ds_ec46.msl.values - ds_targets_oct2017.msl.values)**2) ,axis = (2,3) ))

    return rmse_model, rmse_ec46, diff, relative_improvement, rmse_model_raw, rmse_ec46_raw


def make_accs(ds_targets, ds_outputs, ds_climatology, lat_weights_np):
           
            accs_list = []
            mean_over_idx = (0, 2, 3)

            L = min(ds_climatology.sizes["time"], ds_outputs.sizes["time"])
            F = min(ds_outputs.sizes['sample'], ds_climatology.sizes['sample'])
            ds_outputs = ds_outputs.sortby('lat')

            ds_outputs = ds_outputs.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, F))
            ds_targets = ds_targets.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, F))
            ds_climatology = ds_climatology.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, F))

            for var in ds_targets.var():
                
                diff_out_clim = ds_outputs[f'{var}'].values - ds_climatology[f'{var}'].values
                diff_tar_clim = ds_targets[f'{var}'].values - ds_climatology[f'{var}'].values

                print(diff_out_clim.shape)
            
                nom = np.nanmean((lat_weights_np*diff_out_clim*diff_tar_clim), axis = mean_over_idx )

                denom = np.sqrt(
                    np.nanmean((lat_weights_np*diff_out_clim**2), axis = mean_over_idx) * np.nanmean((lat_weights_np*diff_tar_clim**2), axis =mean_over_idx)
                )

                acc = nom/denom
                print("ACC CHECK?")
                print(acc)

                new_xarray = xr.DataArray(
                data=acc,
                dims=['time'],
                coords={'time': ds_outputs[f'{var}'].time},
                attrs=ds_outputs[f'{var}'].attrs
                )

                accs_list.append(new_xarray)

            # Create an empty dataset
            ds_combined_accs = xr.Dataset()
            var_list = [var for var in ds_outputs.var()]
            # Add each DataArray as a variable to the dataset
            for i, data_array in enumerate(accs_list):
                var_name =  var_list[i]
                # or use a list of predefined variable names if available
                ds_combined_accs[var_name] = data_array

            return ds_combined_accs


def make_rmses(ds_targets, ds_outputs, lat_weights, persistence = False):
    """ Function makes lat-weighted RMSE and returns a frame with RMSE for each variable """
    rmses_list = []
    L = min(ds_targets.sizes["time"], ds_outputs.sizes["time"])
    F = min(ds_outputs.sizes['sample'], ds_targets.sizes['sample'])
    ds_outputs = ds_outputs.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, F))
    ds_targets = ds_targets.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, F))

    # i want to make sure the latitude and longitudes are aligned and ordered
    for var in ds_targets.var():

        print(f"\tComputing RMSE... {var}")
        
        mean_over = ["sample", "lat", "lon"]
 
        if persistence:

            ds_outputs_reshaped = np.transpose(ds_outputs[f'{var}'].values, (1, 0, 2, 3))

            diff_out_tar = ds_outputs_reshaped - ds_targets[f'{var}'].values

        else:
            diff_out_tar = ds_outputs[f'{var}'].values - ds_targets[f'{var}'].values

       
        rmse = np.sqrt(lat_weights*diff_out_tar**2) 

        new_xarray = xr.DataArray(
            data=rmse,
            dims=ds_targets[f'{var}'].dims,
            coords=ds_targets[f'{var}'].coords,
            attrs=ds_targets[f'{var}'].attrs
        ).mean(dim=mean_over)
    
        rmses_list.append(new_xarray)

    # Create an empty dataset
    ds_combined = xr.Dataset()
    var_list = [var for var in ds_outputs.var()]
    # Add each DataArray as a variable to the dataset
    for i, data_array in enumerate(rmses_list):
        var_name =  var_list[i]
        # or use a list of predefined variable names if available
        ds_combined[var_name] = data_array

    return ds_combined


def evaluate_model(cfg: DictConfig, file_path: str, dataset: WeatherBenchDataset = None, complevel: int = 7) -> None:
    """
    Evaluates a single model for a given configuration.

    :param cfg: The hydra configuration for the model
    :param file_path: The destination path for the datasets
    :param dataloader: The PyTorch dataloader if it exists already
    :param complevel: The level of compression when writing datasets to disk (higher is stronger compression).
    :return: A list of model inputs, outputs, and targets, each of shape [B, T, D, H, W]
    """

    if cfg.verbose: print("\nInitializing model")

    if cfg.seed:
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
    device = th.device(cfg.device)

    print('CFG data', cfg.data)

    


    # Set up model
    model = eval(cfg.model.type)(**cfg.model).to(device=device)
    # set model to train?

    #model.train() # REMOVE

    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters")

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_best.ckpt")
    if cfg.verbose: print(f"\tRestoring model from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Initializing dataloader for testing
    init_dates = make_biweekly_inits(
            start=cfg.data.test_start_date,
            end=cfg.data.test_stop_date,
            sequence_length=cfg.testing.sequence_length,
            timedelta=cfg.data.timedelta
    )
    
    if dataset == None:
        print("\nInitializing dataset...")
        dataset = hydra.utils.instantiate(
            cfg.data,
            start_date=cfg.data.test_start_date,
            stop_date=cfg.data.test_stop_date,
            sequence_length= cfg.testing.sequence_length,
            init_dates=init_dates
        )
    print(dataset)
       
    dataloader1 = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.testing.batch_size,
        shuffle=False,
        num_workers= 16 # 0?
    )
    print("loaded dataset")
    print()

    print(cfg.training.type)


    if cfg.training.type == 'diffusion':

        betas = [cfg.training.min_noise_std ** (k / cfg.training.num_refinement_steps) for k in reversed(range(cfg.training.num_refinement_steps + 1))]
        
        # scheduling the addition of noise
        # noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=cfg.training.num_refinement_steps + 1,
        #     trained_betas=betas,
        #     prediction_type="v_prediction", # shouldnt this be "epsilon"
        #     clip_sample=False)

        # PDE REFINER DDPM
        noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4, # 0.001
        beta_end=1e-1, # 0.012
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=False
        )

        noise_scheduler.set_timesteps(cfg.model.num_refinement_step)

        if cfg.training.ACDM:

           

            # def linear_beta_schedule(timesteps):
            #     if timesteps < 10:
            #         raise ValueError("Warning: Less than 10 timesteps require adjustments to this schedule!")

            #     beta_start = 0.0001 * (500/timesteps) 
            #     beta_end = 0.02 * (500/timesteps) 
            #     betas = th.linspace(beta_start, beta_end, timesteps)

            #     return th.clip(betas, 0.0001, 0.9999)

            # betas = linear_beta_schedule(cfg.training.num_refinement_steps)

            # print("BETAS", betas)
        
            # noise_scheduler = DDPMScheduler(
            #     num_train_timesteps=cfg.training.num_refinement_steps ,
            #     trained_betas=betas,
            #     prediction_type="v_prediction", 
            #     clip_sample=False )

            print('Validating with Cosine noise scheduler.. ')

            noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.training.num_refinement_steps,
                beta_schedule="squaredcos_cap_v2",  # Key parameter for cosine schedule
                prediction_type="v_prediction",
                clip_sample=False,
                # Remove trained_betas parameter to use predefined schedule
            )


            train_timesteps = cfg.training.num_refinement_steps

        
    # Evaluate (without gradients): iterate over all test samples
    # Evaluate (without gradients): iterate over all test samples
    with th.no_grad():
        inits = list()
        outputs = list()
        targets = list()
        for constants, prescribed, prognostic, target in tqdm(dataloader1, desc="Generating forecasts"):
            # Load data and generate predictions
            constants = constants.to(device=device) if not constants.isnan().any() else None
            prescribed = prescribed.to(device=device) if not prescribed.isnan().any() else None
            prognostic = prognostic.to(device=device)
            target = target.to(device=device)

              
            # input_prog = prognostic

            # perturbation_val = cfg.testing.noise 
                
            # perturbation = (perturbation_val * input_prog.abs().max() * 
            #                     th.randn_like(input_prog))
                                
            # prognostic =  input_prog + perturbation


            if cfg.training.type == 'diffusion':
                output = model(
                    constants=constants if not constants == None else None,
                    prescribed=prescribed if not prescribed == None else None,
                    prognostic=prognostic,
                    noise_scheduler = noise_scheduler, target = target)

            
            # elif isinstance(model, BARNNMUNetHPX):
            #     # BARNN
            #     output_list = []
            #     print("Taking ensemble of 3 members")
            #     for i in range(3):

            #         output = model(
            #         constants=constants if not constants == None else None,
            #         prescribed=prescribed if not prescribed == None else None,
            #         prognostic=prognostic
            #         )
            #         output_list.append(output)

            #     # take the mean over the outputs?
            #     out_ = th.stack(output_list , dim=0)
            #     output = out_.mean(dim=0)

            else:

                output = model(
                    constants=constants if not constants == None else None,
                    prescribed=prescribed if not prescribed == None else None,
                    prognostic=prognostic
                )
              
            inits.append(prognostic[:, 0].cpu()) # over the entire batch, take the first input day?
            outputs.append(output.cpu())
            targets.append(target.cpu())

        inits = th.cat(inits).numpy()
        outputs = th.cat(outputs).numpy()
        targets = th.cat(targets).numpy()


    if cfg.data.normalize:
        v_idx = 0
        for p in cfg.data.prognostic_variable_names_and_levels:
            if len(cfg.data.prognostic_variable_names_and_levels[p]) > 0:
                for l in cfg.data.prognostic_variable_names_and_levels[p]:
                    mean, std = dataset.stats[p]["level"][l]["mean"], dataset.stats[p]["level"][l]["std"]
                    inits[:, v_idx] = inits[:, v_idx]*std + mean # Cut out the time.. 
                    targets[:, :, v_idx] = targets[:, :, v_idx]*std + mean
                    outputs[:, :, v_idx] = outputs[:, :, v_idx]*std + mean
                    v_idx += 1
            else:
                mean, std = dataset.stats[p]["mean"], dataset.stats[p]["std"]
                inits[:, v_idx] = inits[:, v_idx]*std + mean
                targets[:, :, v_idx] = targets[:, :, v_idx]*std + mean
                outputs[:, :, v_idx] = outputs[:, :, v_idx]*std + mean
                v_idx += 1
    
   
    if cfg.model.mesh == 'healpix': 
        if cfg.verbose: print("\nMapping initial conditions, outputs, and targets from HEALPix to LatLon")
        inits = remap(cfg=cfg, data=inits, name="Initial conditions")
        outputs = remap(cfg=cfg, data=outputs, name="Outputs")
        targets = remap(cfg=cfg, data=targets, name="Targets")
        print()
        
    
    
    build_dataset(
        cfg=cfg,
        inits=inits,
        outputs=outputs,
        targets=targets,
        statistics=dataset.stats,
        file_path=file_path,
        init_dates=init_dates,
        complevel=complevel
    )

    # if isinstance(model, BARNNMUNetHPX):

    #     ensemble_outputs = out_.cpu().numpy()

    #     ensemble_outputs_latlon = []
    #     for member_idx in range(ensemble_outputs.shape[0]):
    #         member_data = ensemble_outputs[member_idx]  # shape: (batch_size, ...)
    #         remapped_member = remap(cfg=cfg, data=member_data, name=f"Ensemble member {member_idx}")
    #         ensemble_outputs_latlon.append(remapped_member)

    #     ensemble_outputs_latlon = np.stack(ensemble_outputs_latlon, axis=0)  # shape: (25, batch_size, lat, lon, ...)
        
    #     build_dataset_ensemble(
    #         cfg=cfg,
    #         outputs=ensemble_outputs_latlon,
    #         statistics=dataset.stats,
    #         file_path=file_path,
    #         init_dates=init_dates,
    #         complevel=complevel)

    
    melr = MELRCalculator(device = cfg.device)

    vals = [0, 2, 4, 6, 8]         
    for time in vals: # day 1, day 3, day 5, day 7
        
        melr.apply(outputs[:,time, 0, :, :], targets[:, time, 0, :, :], variable_name=f'msl_day_{time}', epoch = 1)
        melr.apply(outputs[:, time, 1, :, :], targets[:, time, 1, :, :], variable_name=f'geopotential_1000_day_{time}', epoch = 1)
        try:
            melr.apply(outputs[:, time, 6, :, :], targets[:, time, 6, :, :], variable_name=f'geopotential_500_day_{time}', epoch = 1)
        except:
            pass

    return dataset


def build_dataset(
    cfg: DictConfig,
    inits: np.array,
    outputs: np.array,
    targets: np.array,
    statistics: dict,
    init_dates: np.array,
    file_path: str,
    complevel: int = 7
):
    """
    Creates a netCDF dataset for initializations, outputs, and targets and writes them to file.
    
    :param cfg: The hydra configuration of the model
    :param inits: The first frame of the prognostic inputs to the model
    :param outputs: The outputs of the model (predictions)
    :param targets: The ground truth and target for prediction
    :param statistics: Dictionary containing mean and standard deviations per variable and level
    :param init_dates: The dates where the forecasts are initialized
    :param file_path: The path to the directory where the datasets are written to
    """
    print(outputs.shape)

    # Determine data dimensions and set resolution in degree
    B, T, D, H, W = outputs.shape

    deg = cfg.data.degree #5.625 # 1.0 

    dt = f"{cfg.data.timedelta*24}h"
    timedeltas = pd.timedelta_range(start=dt, periods=T, freq=dt)

    # Set up netCDF dataset
    coords = {}
    coords["sample"] = init_dates
    coords["time"] = timedeltas
    if deg == 2.0:
        coords["lat"] = np.flip(np.array(np.arange(start=-90, stop=90, step=deg), dtype=np.float32))
    else:
        coords["lat"] = np.flip(np.array(np.arange(start=-87.1875, stop=90, step=deg), dtype=np.float32))
    coords["lon"] = np.array(np.arange(start=0, stop=360, step=deg), dtype=np.float32)
    chunkdict = {coord: len(coords[coord]) for coord in coords}
    chunkdict["sample"] = 1

    # Prepare dictionaries for initializations, outputs, and targets to create according Datasets below
    v_idx = 0
    inits_dict = dict()
    outputs_dict = dict()
    targets_dict = dict()
    compress_dict = dict()
    for p in cfg.data.prognostic_variable_names_and_levels:
        if len(cfg.data.prognostic_variable_names_and_levels[p]) > 0:
            for l in cfg.data.prognostic_variable_names_and_levels[p]:
                vname = f"{p}{l}"
                attrs = statistics[p]["level"][l]
                #if vname != "z500": v_idx+=1; continue
                inits_dict[vname] = xr.DataArray(data=inits[:, v_idx], dims=["sample", "lat", "lon"], attrs=attrs)
                outputs_dict[vname] = xr.DataArray(data=outputs[:, :, v_idx], dims=["sample", "time", "lat", "lon"], attrs=attrs)
                targets_dict[vname] = xr.DataArray(data=targets[:, :, v_idx], dims=["sample", "time", "lat", "lon"], attrs=attrs)
                #compress_dict[vname] = {"scale_factor": 0.1, "zlib": True, "complevel": complevel}
                v_idx += 1
        else:
            vname = p
            
            attrs = statistics[p]
            inits_dict[vname] = xr.DataArray(data=inits[:, v_idx], dims=["sample", "lat", "lon"], attrs=attrs)
            outputs_dict[vname] = xr.DataArray(data=outputs[:, :, v_idx], dims=["sample", "time", "lat", "lon"], attrs=attrs)
            targets_dict[vname] = xr.DataArray(data=targets[:, :, v_idx], dims=["sample", "time", "lat", "lon"], attrs=attrs)
            #compress_dict[vname] = {"scale_factor": 0.1, "zlib": True, "complevel": complevel}
            v_idx += 1

    # Create datasets and write them to file
    def write_to_file(ds: xr.Dataset, dst_path_name: str, compress_dict: dict):
        if os.path.exists(dst_path_name): os.remove(dst_path_name)  # Delete file if it exists
        print(f"\tWriting to {dst_path_name}")
        if "outputs" in dst_path_name:  # Display progress bar when writing the targets.nc to file
            write_job = ds.to_netcdf(dst_path_name, compute=False, encoding=compress_dict)
            with ProgressBar(): write_job.compute()
        else:
            ds.to_netcdf(dst_path_name, encoding=compress_dict)  # Silently write inits.nc and targets.nc

    print("\nWriting datasets to file. This may take a while.")# Optionally, reduce compression level via the -z flag")
    # Remove threading, write sequentially
    # Extract the model name from the path

    write_to_file(
        xr.Dataset(coords=coords, data_vars=inits_dict).chunk(chunkdict),
        os.path.join(file_path, "inits.nc"), 
        compress_dict
    )
    print("stored inits.nc")
    write_to_file(
        xr.Dataset(coords=coords, data_vars=outputs_dict).chunk(chunkdict),
        os.path.join(file_path, "outputs.nc"),
        compress_dict
    )
    print('stored outputs.nc')
    write_to_file(
        xr.Dataset(coords=coords, data_vars=targets_dict).chunk(chunkdict),
        os.path.join(file_path, "targets.nc"),
        compress_dict
    )
    
    print("\tDatasets successfully written to file\n")



def plot_acc_over_time(
    cfg: DictConfig,
    performance_dict: dict,
    plot_title: str = "Model comparison"
):
    """
    Plot anomaly correlation coefficient of all models (averaged over samples, dimensions, height, width) over time.
    """

    file_path = "./plots"
    os.makedirs(file_path, exist_ok=True)
    dt = cfg.data.timedelta * 24 # Days!

    vnames = list(performance_dict[list(performance_dict.keys())[0]]["outputs"].keys())
    for vname in vnames:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for m_idx, model_name in enumerate(performance_dict):
            if model_name == "climatology": continue
            acc_path = os.path.join("outputs", model_name, "evaluation", "accs.nc")
            if not os.path.exists(acc_path): continue
            acc = xr.open_dataset(acc_path)[vname]
            x_range = np.arange(start=dt, stop=len(acc)*dt + 1, step=dt) / 24
            if model_name in list(MODEL_NAME_PLOT_ARGS.keys()): kwargs = MODEL_NAME_PLOT_ARGS[model_name]
            else: kwargs = {"label": model_name}
            ax.plot(x_range, acc, **kwargs)

        if not "x_range" in locals(): continue
        ax.grid()
        ax.set_ylabel("ACC")
        ax.set_xlabel("Lead time [days]")
        ax.set_xlim([x_range[0], x_range[-1]])
        ax.set_ylim([0.1, 1.0])
        ax.legend(ncol=2, fontsize=9)
        #fig.suptitle(plot_title)
        fig.tight_layout()
        fig.savefig(os.path.join(file_path, f"acc_plot_{vname}.pdf"))

        # Log to Weights & Biases
        wandb.log({
            f"acc_plot_{vname}": wandb.Image(fig)})
        
        plt.close()

    
def plot_relative_improvement(cfg, performance_dict, file_path_comparison, plot_title='RMSE of Models vs. EC46', with_climatology = False):
    """This validation is done in 5.625 degrees!
    1. needs climatology
    2. Needs EC46 
    """

    color_dict = {
            'model1': 'blue',
            'model2': 'orange',
            'model3': 'green',
            'model4': 'grey'}
    
    colors =  list(color_dict.values())

    months = [1,2,3,4,5,6,7,8,9,10] #,11,12] 
    years = [2022]
    rmse_ec46 = None
    rmse_clim = None

    caption = f"Tested on biweekly values in months: {str(months)}; year: {str(years)}"

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=10, style='italic')
    model_names = list(performance_dict.keys())

    num_models = len(model_names)  
    x = np.arange(num_models + 2 if with_climatology else num_models + 1)
    width =  1 

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    rmse_max = -np.infty

    for m_idx, model_name in enumerate(model_names):
        file_path = os.path.join("outputs", model_name, "evaluation")
        
        ec46L = []
        modelL = []
        clim = []
        

        for year in years:
            for month in months:
                if with_climatology:
                    file_path_comparison_climatology = os.path.join(file_path, f"climatology_comparison_with_ec46_{str(month)}-{str(year)}.nc")
                    if os.path.exists(file_path_comparison_climatology):
                        df_clim = xr.open_dataset(file_path_comparison_climatology)
                        # mean for this month of the year?
                        rmse_clim = df_clim.rmse_climatology.mean()
                        clim.append(rmse_clim)

                file_path_comparison = os.path.join(file_path, f"comparison_with_ec46_{str(month)}-{str(year)}.nc")
                if os.path.exists(file_path_comparison):
                    df = xr.open_dataset(file_path_comparison)
                    rmse_ec46 = np.mean(df.rmse_ec46.values)
                    rmse_model = df.rmse_model.mean()

                    ec46L.append(rmse_ec46)
                    modelL.append(rmse_model)


        modelL = np.mean(modelL)
        rmse_ec46 = np.nanmean(ec46L) 
        rmse_clim = np.mean(clim)
        
        rmse_max = max(rmse_max, modelL, rmse_ec46)
        print(rmse_max, 'rmsemax!!')

        color = list(color_dict.values())[m_idx]
        ax.bar(x[m_idx], rmse_model, width, align='edge', color=colors[m_idx], label=model_name, capsize=5)

    # Plot EC46 bar with error bar
    ax.bar(x[-1], rmse_ec46, width, align='edge',label='RMSE EC46', capsize=5)

    if with_climatology: 
        ax.bar(x[-2] , rmse_clim, width, align='edge',label='RMSE Climatology', capsize=5)

    ax.set_xticks(x)
    if with_climatology:
        ax.set_xticklabels(model_names + ['Climatology']+ ['EC46'], rotation=45, ha='right')
    else:
        ax.set_xticklabels(model_names + ['EC46'], rotation=45, ha='right')

    ax.grid(axis='y')
    ax.set_title("RMSE Comparison: Models vs EC46 (MSL only)")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, rmse_max * 1.1)  # Add 10% padding to the top
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 0), loc='lower left')
    fig.suptitle(plot_title)
    fig.tight_layout()
    fig.savefig(os.path.join(file_path, f"rmse_plot.pdf"))
    wandb.log({f"RMSE_comparison": wandb.Image(fig)})

    plt.close()


def plot_skill_per_day(cfg,performance_dict,file_path_comparison,plot_title, climatology = False): 
    """This validation is done in 5.625 degrees!"""
    color_dict = {
            'model1': 'blue',
            'model2': 'orange',
            'model3': 'green',
            'model4': 'grey'}
    colors =  list(color_dict.values())

    months = [1,2,3,4,5,6,7,8,9,10] #11,12] 
    years = [2022]

    caption = f"Tested on biweekly values in months: {str(months)}; year: {str(years)}"

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=10, style='italic')
    model_names = list(performance_dict.keys())

    for m_idx, model_name in enumerate(model_names):
        file_path = os.path.join("outputs", model_name, "evaluation")
        
        all_skill_scores = []

        for year in years:
            for month in months:
                if climatology:
                    file_path_comparison = os.path.join(file_path, f"climatology_comparison_with_ec46_{str(month)}-{str(year)}.nc")
                else:
                    file_path_comparison = os.path.join(file_path, f"comparison_with_ec46_{str(month)}-{str(year)}.nc")

                if os.path.exists(file_path_comparison):
                    df = xr.open_dataset(file_path_comparison)
                    skill_score = df.relative_per_day
                    all_skill_scores.append(skill_score)

        if all_skill_scores:
            combined_skill_scores = xr.concat(all_skill_scores, dim='sample')
            mean_skill_score = combined_skill_scores.mean(dim='sample')
            
            time_values = np.arange(1, mean_skill_score.shape[1] + 1)
            color = colors[m_idx]
            
            ax.plot(time_values, mean_skill_score[0], color=color, label=model_name, linewidth=2)

        ax.set_title("Relative Improvement vs. Time for Each Sample")
        ax.set_xlabel("Day")
        ax.set_ylabel("Relative Improvement (%)")

        # Add a horizontal line at y=0
        ax.axhline(0, color='red', linestyle='--', label='No Improvement')

        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add grid
        ax.grid(True, linestyle=':', alpha=0.7)

    # Adjust layout
    fig.tight_layout()
    if climatology:
        wandb.log({
                f"climatology_relativeimprovement_per_day": wandb.Image(fig, caption='Relative improvement of climatology over EC46') })
        plt.close()
    else:

        wandb.log({
                f"relativeimprovement_per_day": wandb.Image(fig, caption=caption) })
        plt.close()

      

def plot_rmse_over_time(
    cfg: DictConfig,
    performance_dict: dict,
    plot_title: str = "Model comparison",
    with_climatology = False
):
    """
    Plot the root mean squared error of all models (averaged over samples, dimensions, height, width) over time.
    """

    file_path = "./plots"
    os.makedirs(file_path, exist_ok=True)
    dt = cfg.data.timedelta * 24 # DAYS!

    vnames = list(performance_dict[list(performance_dict.keys())[0]]["outputs"].keys())
    for vname in vnames:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        rmse_max = -np.infty

        for m_idx, model_name in enumerate(performance_dict):
            
            rmse = xr.open_dataset(os.path.join("outputs", model_name, "evaluation", "rmses.nc"))[vname]

            if with_climatology:
                rmse_climatology = xr.open_dataset(os.path.join("outputs", model_name, "evaluation", "rmses_climatology.nc"))[vname]
                #rmse_weekly_climatology = xr.open_dataset(os.path.join("outputs", model_name, "evaluation", "rmses_weekly_climatology.nc"))[vname]
                 
            x_range = np.arange(start=dt, stop=len(rmse)*dt + 1, step=dt) / 24
            if model_name in list(MODEL_NAME_PLOT_ARGS.keys()): kwargs = MODEL_NAME_PLOT_ARGS[model_name]
            else: kwargs = {"label": model_name}
            ax.plot(x_range, rmse, **kwargs)

            if with_climatology:
                kwargs = {"label": f'monthly climatology, {vname}'}
                ax.plot(x_range, rmse_climatology, **kwargs)

                #kwargs = {"label": f'weekly climatology, {vname}'}
                #ax.plot(rmse_weekly_climatology, **kwargs)

            # adding persistence
            rmse_persistence = xr.open_dataset(os.path.join("outputs", model_name, "evaluation", "rmses_persistence.nc"))[vname]
            kwargs = {"label": f'persistence, {vname}'}
            ax.plot(x_range, rmse_persistence, **kwargs)

            rmse_max = max(rmse_max, rmse.max())

        ax.grid()
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Lead time [days]")
        ax.set_xlim([x_range[0], x_range[-1]])
        #ax.set_xlim([x_range[0], x_range[55]])
        #ax.set_ylim([200, 2000])
        #ax.set_ylim([50, 1200])
        ax.legend(ncol=2, fontsize=9)
        #fig.suptitle(plot_title)
        fig.tight_layout()
        fig.savefig(os.path.join(file_path, f"rmse_plot_{vname}.pdf"))

        # Log to Weights & Biases
        wandb.log({
            f"rmse_plot_{vname}": wandb.Image(fig) })
        plt.close()
        


def compute_metrics(
    cfg: DictConfig,
    ds_outputs: xr.Dataset,
    ds_targets: xr.Dataset,
    file_path: str,
    overide: bool = False,
) -> None:
    """
    Compute RMSE and Frobenius Norm (accumulated error) and print them to console.

    :param cfg: The configuration of the model
    :param ds_outputs: The dataset containing the model outputs (predictions)
    :param ds_targets: The dataset containing the targets (ground truth)
    :param file_path: The destination path where to write results
    """

    if cfg.data.timedelta == 1:
        if cfg.data.degree == 2.0:
            path_to_climatology = "/projects/prjs1254/climatology1D_32/outputs.nc"
        else:
            path_to_climatology = '/projects/prjs1254/climatology/outputs.nc'

    elif cfg.data.timedelta == 2:
        path_to_climatology  = '/projects/prjs1254/climatology2D/outputs.nc'

    elif cfg.data.timedelta == 3:
        path_to_climatology  = '/projects/prjs1254/climatology3D/outputs.nc'

    else:
        print("not yet implemented climatology")
        ValueError

        
    
    ds_climatology = xr.open_dataset(path_to_climatology)

    T = min(ds_outputs.sizes["time"], ds_climatology.sizes['time']) 
    L = T

    ds_climatology = ds_climatology.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, 194)) #CHANGE

    ds_outputs = ds_outputs.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, 194))
    ds_targets = ds_targets.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, 194))

    print("\nChecking whether to compute metrics for", cfg.model.name, "model", 'width:', cfg.data.width)

    # Number of time steps
    S = ds_outputs.sizes["sample"]

    # Compute latitude-area weighting factors cos(lat_j) / (1/N_lat * sum(lat_j))
    # Equation (2) in https://arxiv.org/abs/2002.00469

    lat_weights = get_adjusting_weights(cfg.data.degree).unsqueeze(0).unsqueeze(0).expand(S,T,ds_outputs.sizes["lat"],ds_outputs.sizes["lon"])  # sample: 194 time: 47
    # Compute anomaly correlation coefficient per variable and write to file
    # Equation (A1) in https://arxiv.org/abs/2002.00469
    print('LAT WEIGHTS', lat_weights.shape)

    path_to_weekly_climatology = '/projects/prjs1254/climatology_weekly/outputs.nc'
    file_path_ = os.path.join(file_path, "accs.nc")
   
    #ds_weekly_climatology = xr.open_dataset(path_to_weekly_climatology)
    #ds_weekly_climatology = ds_weekly_climatology.sortby('lat').isel(time=slice(0, 46))

    # ds_ensemble = xr.open_dataset('/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/bigensemble_5/ensemble_big_model_5_new.nc') #'/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/Ensemble_Small/outputs.nc')
    # ds_ensemble = ds_ensemble.sortby('lat').isel(time=slice(0, 46))

    # Compute the RMSE for climatology
    path_to_persistence = os.path.join(file_path, "persistence.nc")
    ds_persistence = xr.open_dataset(path_to_persistence)
    ds_persistence  = ds_persistence.sortby('lat').isel(time=slice(0, L)).isel(sample=slice(0, 194))

    lat_weights_np = lat_weights.cpu().detach().numpy()

    if os.path.exists(path_to_climatology): #and (not os.path.exists(file_path_) or overide):
        print("\tComputing ACC...")

        ds_combined_accs = make_accs(ds_targets, ds_outputs, ds_climatology, lat_weights_np)
    
        ds_combined_accs.to_netcdf(os.path.join(file_path, "accs.nc"))

        ds_combined = make_rmses(ds_targets, ds_outputs, lat_weights)
        ds_combined.to_netcdf(os.path.join(file_path, "rmses.nc"))

        # Compute the RMSE for climatology
        ds_combined = make_rmses(ds_targets, ds_climatology, lat_weights)
        ds_combined.to_netcdf(os.path.join(file_path, "rmses_climatology.nc"))

        # ds_combined = make_rmses(ds_targets, ds_weekly_climatology, lat_weights)
        # ds_combined.to_netcdf(os.path.join(file_path, "rmses_weekly_climatology.nc"))

        ds_combined = make_rmses(ds_targets, ds_persistence, lat_weights, persistence =True)
        ds_combined.to_netcdf(os.path.join(file_path, "rmses_persistence.nc"))

        # ensemble rmse
        # ds_combined = make_rmses(ds_targets, ds_ensemble, lat_weights)
        # ds_combined.to_netcdf(os.path.join(file_path, "big_ensemble_5items.nc"))


    # Compute annually averaged RMSE for U10

    # file_path_ = os.path.join(file_path, "rmse_months_01-12_global.nc")
    # if not os.path.exists(file_path_) or overide:
    #     # Global RMSE
    #     print("\tComputing RMSE for physical soundness of global winds...")
    #     avg_tar = ds_targets.mean(dim=("time", "lon"))
    #     avg_out = ds_outputs.mean(dim=("time", "lon"))
    #     rmse_global = np.sqrt(((avg_out-avg_tar)**2).mean())
    #     rmse_global.to_netcdf(file_path_)

    #     # Trade Winds RMSE (near north and south of equator)
    #     file_path_ = os.path.join(file_path, "rmse_months_01-12_trade-winds.nc")
    #     print("\tComputing RMSE for physical soundness of Trade Winds...")
    #     avg_tar_ = (xr.merge([avg_tar.sel(lat=slice(-20, -10)), avg_tar.sel(lat=slice(10, 20))]))
    #     avg_out_ = (xr.merge([avg_out.sel(lat=slice(-20, -10)), avg_out.sel(lat=slice(10, 20))]))
    #     rmse_trade_winds = np.sqrt(((avg_out_-avg_tar_)**2).mean())
    #     rmse_trade_winds.to_netcdf(file_path_)

    #     # South Westerlies RMSE (in southern extratropics)
    #     file_path_ = os.path.join(file_path, "rmse_months_01-12_south-westerlies.nc")
    #     print("\tComputing RMSE for physical soundness of South Westerlies...")
    #     avg_tar_ = avg_tar.sel(lat=slice(-55, -45))
    #     avg_out_ = avg_out.sel(lat=slice(-55, -45))
    #     rmse_south_westerlies = np.sqrt(((avg_out_-avg_tar_)**2).mean())
    #     rmse_south_westerlies.to_netcdf(file_path_)
        
    #     # Clear memory
    #     del avg_tar, avg_tar_, avg_out, avg_out_

    # #
    # # Compute average RMSE over lead times of 11 and 12 months for Z500
    # file_path_ = os.path.join(file_path, "rmse_months_11-12.nc")
    # if not os.path.exists(file_path_) or overide:
    #     print("\tComputing RMSE in months 11 and 12 of one-year rollout...")
    #     avg_tar = ds_targets.sel(time=slice(pd.Timedelta(334, "d"), pd.Timedelta(365, "d"))).mean(dim=("time"))
    #     avg_out = ds_outputs.sel(time=slice(pd.Timedelta(334, "d"), pd.Timedelta(365, "d"))).mean(dim=("time"))
    #     rmse = np.sqrt(((avg_out-avg_tar)**2).mean())
    #     rmse.to_netcdf(file_path_)
    #     del avg_tar, avg_out
    
    ## Model compared to EC46
    EC46 = False
    if EC46:
    
        ec46_folder = '/projects/prjs1254/EC46/EC46/msl' #'/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/EC46/msl'
        
        months = [1,2,3,4,5,6,7,8,9,10] #12, 11
        years = [2022]
        print("where is month in the original predicted output 12??")
        
        for year in years:
            for month in months:
                file_path_comparison = os.path.join(file_path, f"comparison_with_ec46_{str(month)}-{str(year)}.nc")
                ec46_file = os.path.join(ec46_folder, f"{month}-{year}.nc")

                if not os.path.exists(file_path_comparison) or overide:
                    print("\tComputing comparison (MSL only!) with EC46 for October 2017...")
                    
                    # Load EC46 data
                    ds_ec46 = xr.open_dataset(ec46_file)
                    ds_climatology = xr.open_dataset(path_to_climatology)
                
                    # Select October 2017 from your model outputs and targets
                    ds_outputs_oct2017 = ds_outputs.sel(sample=((ds_outputs.sample.dt.year == year) & (ds_outputs.sample.dt.month == month)))
                    ds_targets_oct2017 = ds_targets.sel(sample=((ds_targets.sample.dt.year == year) & (ds_targets.sample.dt.month == month)))
                    ds_climatology_oct2017 = ds_climatology.sel(sample=((ds_climatology.sample.dt.year == year) & (ds_climatology.sample.dt.month == month)))
                    
                    # Select October 2017 from your model outputs and targets
                    ds_climatology = ds_climatology.isel(time=slice(0, 46))
                    ds_outputs_oct2017 = ds_outputs_oct2017.isel(time=slice(0, 46))
                    ds_targets_oct2017 = ds_targets_oct2017.isel(time=slice(0, 46))
                    ds_ec46 = ds_ec46.isel(time=slice(0, 46))

                    ########
                    # Compute RMSE between your model and targets
                    
                    rmse_model, rmse_ec46, diff, relative_improvement, rmse_model_raw, rmse_ec46_raw = rmse_cal(ds_outputs_oct2017, ds_targets_oct2017, ds_ec46)

                    # Compute relative improvement
                    relative_improvement_raw = (rmse_ec46_raw - rmse_model_raw) / rmse_ec46_raw * 100

                    num_samples = len(ds_outputs_oct2017.sample)# pass all other variables with jus sample dimsension

                    # Create DataArrays with 'sample' dimension
                    da_rmse_model = xr.DataArray(rmse_model, dims=['sample'], coords={'sample': range(num_samples)})
                    da_rmse_ec46 = xr.DataArray(rmse_ec46, dims=['sample'], coords={'sample': range(num_samples)})
                    da_relative_improvement = xr.DataArray(relative_improvement, dims=['sample'], coords={'sample': range(num_samples)})
                    da_diff = xr.DataArray(diff, dims=['sample'], coords={'sample': range(num_samples)})
                    da_relative_improvement_raw = xr.DataArray(relative_improvement_raw, dims=['sample', 'time'], coords={'sample': range(num_samples), 'time': ds_outputs_oct2017.time})
                    # Create a dataset with the comparison results
                    ds_comparison = xr.Dataset({
                        "rmse_model": da_rmse_model, # each of these should be given the dimension sample
                        "rmse_ec46": da_rmse_ec46, 
                        "relative_improvement": da_relative_improvement, 
                        'difference_model_ec': da_diff,
                        'relative_per_day': da_relative_improvement_raw,
                    })

                    # Save the comparison results
                    ds_comparison.to_netcdf(file_path_comparison)
                    
                    print(f"\tComparison results saved to {file_path_comparison}")
                    # Assuming your DataArray is named `skill_score`
                    
                    file_path_ = os.path.join(file_path, "accs_climatology.nc")


                file_path_comparison = os.path.join(file_path, f"climatology_comparison_with_ec46_{str(month)}-{str(year)}.nc")
                # Inside the loop where you're processing EC46 data
                if os.path.exists(path_to_climatology) and (not os.path.exists(file_path_comparison) or overide):
                    print("\tComputing RMSE for EC46 vs climatology...")
                    # climatology vs. ec46
                    
                    rmse_climatology, rmse_ec46, diff, relative_improvement, rmse_climatology_raw, rmse_ec46_raw = rmse_cal(ds_climatology_oct2017, ds_targets_oct2017, ds_ec46 )
        
                    # Compute relative improvement
                    relative_improvement_raw = (rmse_ec46_raw - rmse_climatology_raw) / rmse_ec46_raw * 100


                    num_samples = len(ds_outputs_oct2017.sample)# pass all other variables with jus sample dimsension
                    
                    # Create DataArrays with 'sample' dimension
                    da_rmse_climatology = xr.DataArray(rmse_climatology, dims=['sample'], coords={'sample': range(num_samples)})
                    da_rmse_ec46 = xr.DataArray(rmse_ec46, dims=['sample'], coords={'sample': range(num_samples)})
                    da_relative_improvement = xr.DataArray(relative_improvement, dims=['sample'], coords={'sample': range(num_samples)})
                    da_diff = xr.DataArray(diff, dims=['sample'], coords={'sample': range(num_samples)})
                    da_relative_improvement_raw = xr.DataArray(relative_improvement_raw, dims=['sample', 'time'], coords={'sample': range(num_samples), 'time': ds_outputs_oct2017.time})
                    # Create a dataset with the comparison results
                    ds_comparison = xr.Dataset({
                        "rmse_climatology": da_rmse_climatology, # each of these should be given the dimension sample
                        "rmse_ec46": da_rmse_ec46, 
                        "relative_improvement": da_relative_improvement, 
                        'difference_model_ec': da_diff,
                        'relative_per_day': da_relative_improvement_raw,
                    })

                    # Save the comparison results
                    ds_comparison.to_netcdf(file_path_comparison)
                    
                    print(f"\tComparison results saved to {file_path_comparison}")
                    # Assuming your DataArray is named `skill_score`

                    file_path_ = os.path.join(file_path, "accs_climatology.nc")

                        
def run_evaluations(
    configuration_dir_list: str,
    device: str,
    overide: bool = False,
    batch_size: int = None,
    sequence_length: int = None,
    plot_title: str = "Model comparison",
    normalize_video: bool = False,
    complevel: int = 7
):
    """
    Evaluates a model with the given configuration.

    :param configuration_dir_list: A list of hydra configuration directories to the models for evaluation
    :param device: The device where the evaluations are performed
    """

    
    performance_dict = {}
    dataset_hpx = None
    dataset_cyl = None
    #overide = True
    wandb.init(project="Evaluation_dlwpbenchmark", name=f"evaluation_all_models") # replace with model name

    
    # Iterate over all configuration directories and perform evaluations
    for configuration_dir in configuration_dir_list:
        
        # If default configuration path has been overridden, append .hydra since then a custom path to a specific model
        # has been provided by the user
        if configuration_dir != "configs": configuration_dir = os.path.join(configuration_dir, ".hydra")

        # Initialize the hydra configurations for this forecast
        with hydra.initialize(version_base=None, config_path=os.path.join("..", configuration_dir)):
            cfg = hydra.compose(config_name="config")
            cfg.device = device
            if batch_size: cfg.testing.batch_size = batch_size
            if sequence_length: cfg.testing.sequence_length = sequence_length

        
        # Generate forecasts if they do not exist and load them
        output_fname = "outputs.nc"
        file_path = os.path.join("outputs", str(cfg.model.name), "evaluation")
        if os.path.exists(os.path.join(file_path, output_fname)):
            ds = xr.open_dataset(os.path.join(file_path, output_fname))


        print("(1) LOADDED THE DATASET?")
        if not os.path.exists(os.path.join(file_path, output_fname)) or overide:
            os.makedirs(file_path, exist_ok=True)
            dataset = dataset_hpx if "hpx" in file_path else dataset_cyl
            # this function returns the remapped outputs - lat-lon representation
            dataset = evaluate_model(cfg=cfg, file_path=file_path, dataset=dataset, complevel=complevel)
            if "hpx" in file_path: dataset_hpx = dataset
            else: dataset_cyl = dataset

        print("(2) Loading targets..")
        ds_inits = xr.open_dataset(os.path.join(file_path, "inits.nc"))
        ds_outputs = xr.open_dataset(os.path.join(file_path, "outputs.nc")).isel(time=slice(0, 1460))
        ds_targets = xr.open_dataset(os.path.join(file_path, "targets.nc"))

        print("(3) Computing Metrics")
        # Compute forecast error metrics if they don't yet exist and write results to file
        if not os.path.exists(os.path.join(file_path, "rmses.nc")) or overide:
            print("Computing persistence.. ")
            persistence_forecast(ds_inits, ds_outputs, ds_targets, os.path.join(file_path, "persistence.nc"))
            compute_metrics(cfg=cfg, ds_outputs=ds_outputs, ds_targets=ds_targets, file_path=file_path, overide=overide)

        # Add the current model's datasets to the performance dict for cross model evaluation later
        performance_dict[cfg.model.name] = dict(inits=ds_inits, outputs=ds_outputs, targets=ds_targets)

        #print("(4) Generating Videos")
        # Generate video showcasing model forecast
        # if not os.path.exists(os.path.join(file_path, "videos")) or overide:
        #     generate_mp4(
        #             cfg=cfg,
        #             ds_outputs=ds_outputs,
        #             ds_targets=ds_targets,
        #             file_path=file_path,
        #             normalize=normalize_video)

        # Clear RAM by deleting the datasets used here and calling the garbage collector subsequently
        del ds_inits, ds_outputs, ds_targets
        gc.collect()

    print("(5) Plotting RMSE and ACC")
    #if overide: plot_rmse_over_time(cfg-cfg, performance_dict=performance_dict)
    plot_rmse_over_time(cfg=cfg, performance_dict=performance_dict, plot_title=plot_title, with_climatology=True)
    plot_acc_over_time(cfg=cfg, performance_dict=performance_dict, plot_title=plot_title)
    #RMSE over EC period
    plot_relative_improvement(cfg,performance_dict,"comparison_with_ec46_october2017.nc",plot_title,with_climatology=True )

    plot_skill_per_day(cfg, performance_dict,"comparison_with_ec46_october2017.nc",plot_title)
    plot_skill_per_day(cfg, performance_dict,"comparison_with_ec46_october2017.nc",plot_title,climatology=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with a given configuration. Particular properties of the configuration can be "
                    "overwritten, as listed by the -h flag.")
    parser.add_argument("-c", "--configuration-dir-list", nargs="*", default=['outputs/acdm50cosine'], #'outputs/PDErefhpx321241_60epoch_long3'] , #outputs/PDErefhpx321238', 'outputs/PDErefhpx321239', 'outputs/PDErefhpx321240'], #PDErefhpx321239_60epoch_long3', 'outputs/PDErefhpx321239_60epoch_long8', 'outputs/PDErefhpx321239_60epoch_long'],# 'outputs/PDErefhpx81240','outputs/PDErefhpx81241'], #C2PDErefhpx321241','outputs/C2PDErefhpx321239'], #''C2PDErefhpx321240'outputs/PDErefhpx81241' , 'outputs/PDErefhpx81240', 'outputs/PDErefhpx81239'], ##'outputs/PDErefhpx321238','outputs/PDErefhpx321239', 'outputs/PDErefhpx321240', 'outputs/PDErefhpx321241'], #C2PDErefhpx321240','outputs/C2PDErefhpx321241','outputs/C2PDErefhpx321239'], #,'outputs/PDErefhpx321240','outputs/PDErefhpx321241', 'outputs/PDErefhpx321239'], # ] # 'outputs/big_seed1235_drop0.2', 'outputs/big_seed1235'],# 'outputs/3daysteps', 'outputs/3daysteps'], #big_seed100', 'outputs/big_seed3000', 'outputs/PDEref'], # , 'outputs/C2PDEref','outputs/C2PDErefsmall_seed20', 'outputs/C2PDErefsmall_seed3000'], #,#'outputs/Big_Bz64'Big_Bz32_C2, 'outputs/Small_Bz64',  plt.plot(x, msl3, label='Climatology', marker='^')'outputs/LongContext',  'outputs/MUunet_inverted_C2_hpx88','outputs/DiffMUNetHPX_smallest_98R', 'outputs/DiffMUNetHPX_test', 'outputs/unet_inverted_C2_hpx'], #'outputs/speccheckCHECK2INV0.7', 'outputs/modernunet_inverted'],#  ], # #, , 'outputs/modunet_inverted_32B_COMP_check'], #, , ,'outputs/unet_inverted'], # 'outputs/MUnet_w_diff', 'outputs/MUnet_w_diff_SpectralLoss', 'outputs/MUnet_w_diff_ADJ', 'outputs/MUnet_w_diff_ADJ_50'], # modernunet_inverted'], #swintransformer'], #unet'], #=["configs"], 'outputs/panguweather', 'outputs/unet_inverted', 'outputs/unet', 'outputs/swintransformer',
                        help="List of directories where the configuration files of all models to be evaluated lie.")
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="The device to run the evaluation. Any of ['cpu' (default), 'cuda:0', 'mpg'].")
    parser.add_argument("-o", "--overide", action="store_true",
                        help="Overide model forecasts and evaluation files if they exist already.")
    parser.add_argument("-b", "--batch-size", type=int, default=None,
                        help="Batch size used for evaluation. Defaults to None to take entire test set in one batch.")
    parser.add_argument("-s", "--sequence-length", type=int, default=None,
                        help="Sequence length for the evaluation. Use 14 to generate 14-days forecasts.")
    parser.add_argument("-pt", "--plot-title", type=str, default="Model comparison",
                        help="The title for the RMSE plot.")
    parser.add_argument("-nv", "--normalize-video", action="store_false",
                        help="Whether to normalize values for the .mp4 visualization. Default true.")
    parser.add_argument("-z", "--complevel", type=int, default=7,
                        help="Compression level when writing netcdf datasets to file.")

    run_args = parser.parse_args()
    run_evaluations(configuration_dir_list=run_args.configuration_dir_list,
                    device=run_args.device,
                    overide=run_args.overide,
                    batch_size=run_args.batch_size,
                    sequence_length=run_args.sequence_length,
                    plot_title=run_args.plot_title,
                    normalize_video=run_args.normalize_video,
                    complevel=run_args.complevel)
    
    print("Done.")