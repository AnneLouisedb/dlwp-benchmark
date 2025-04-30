import numpy as np
import pandas as pd

import multiprocessing
from tqdm import tqdm
import os
import gc  # Garbage collector
import sys
import time
import shutil
import argparse
import threading
import subprocess

from tqdm import tqdm
import wandb
import hydra
import numpy as np
import torch as th
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data.processing.healpix_mapping import HEALPixRemap
from data.processing.istarmap import istarmap


def get_adjusting_weights(deg):
    if deg == 5.625:
        latitudes = th.arange(-87.1875, 90, 5.625)
        expand = 64
        
    elif deg == 2.0:
        latitudes = th.arange(-90, 90, 2.0)
        expand = 180

    latitudes_rad = th.deg2rad(latitudes)
    weights = th.cos(latitudes_rad)

    weights = (weights / th.sum(weights)) * weights.size(0)

    return weights.unsqueeze(1).expand(-1, expand)
   
def generate_mp4(
    cfg: DictConfig,
    ds_outputs: xr.Dataset,
    ds_targets: xr.Dataset,
    file_path: str,
    normalize: bool = False
):
    """
    Generates mp4 video visualizing model output, target, and the difference between those.

    :param cfg: The hydra configuration of the model
    :param ds: An xarray dataset containing model inputs, outputs, and targets
    """

    sample = 0
    file_path = os.path.join(file_path, "videos")
    os.makedirs(os.path.join(file_path, "frames"), exist_ok=True)

    for vname in tqdm(list(ds_outputs.keys()), desc="Generating frames and a video of the model forecasts"):
        outputs, targets = ds_outputs[vname].isel(sample=sample), ds_targets[vname].isel(sample=sample)
        if normalize:
            outputs = (outputs-outputs.attrs["mean"])/outputs.attrs["std"]
            targets = (targets-targets.attrs["mean"])/targets.attrs["std"]
        outputs, targets = outputs.values, targets.values

        # Visualize results
        diff = outputs - targets
        diffmax = max(abs(np.min(diff[cfg.model.context_size:])),
                    abs(np.max(diff[cfg.model.context_size:])))
        vmin, vmax = np.min(targets), np.max(targets)
        for t in range(outputs.shape[0]):
            fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)
            
            ax[0].imshow(outputs[t], origin="lower", vmin=vmin, vmax=vmax, extent=[-180, 180, -90, 90])
            ax[0].set_title(r"Prediction ($\hat{y}$)")
            ax[0].set_xlabel("Longitude")
            ax[0].set_ylabel("Latitude")

            im1 = ax[1].imshow(targets[t], origin="lower", vmin=vmin, vmax=vmax, extent=[-180, 180, -90, 90])
            ax[1].set_title(r"Ground truth ($y$)")
            ax[1].set_xlabel("Longitude")
            divider1 = make_axes_locatable(ax[1])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im1, cax=cax1, orientation='vertical')

            im2 = ax[2].imshow(diff[t], origin="lower", vmin=-diffmax, vmax=diffmax, cmap="bwr",
                               extent=[-180, 180, -90, 90])
            ax[2].set_title(r"Difference ($\hat{y}-y$)")
            ax[2].set_xlabel("Longitude")
            divider2 = make_axes_locatable(ax[2])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im2, cax=cax2, orientation='vertical')

            hour = str(pd.to_timedelta(ds_outputs.isel(time=t).time.values))
            init_date = str(pd.Timestamp(ds_outputs.isel(sample=sample).sample.values).date())
            fig.suptitle(f"{vname}, time step = {t+1}/{outputs.shape[0]}, "
                         f"init date = {init_date}, lead time = {hour} hours")
            fig.tight_layout()
            fig.savefig(os.path.join(file_path, "frames", f"state_{str(t).zfill(4)}.png"))
            plt.close()

        # Generate a video from the just generated frames with ffmpeg
        subprocess.run(["ffmpeg",  #"/usr/bin/ffmpeg",
                        "-f", "image2",
                        "-hide_banner",
                        "-loglevel", "error",
                        "-r", "15",
                        #"-vf", "setpts=1.5*PTS",
                        "-pattern_type", "glob",
                        "-i", f"{os.path.join(file_path, 'frames', '*.png')}",
                        #"-vcodec", "libx264",
                        "-crf", "22",
                        "-pix_fmt", "yuv420p",
                        "-y",
                        f"{os.path.join(file_path, f'{vname}.mp4')}"])
        
        video_path = os.path.join(file_path, f'{vname}.mp4')
        wandb.log({f"video/{cfg.model.name}": wandb.Video(video_path, fps=15, format="mp4")})
        print("VIDEO")
        
    # Cleaning up
    shutil.rmtree(os.path.join(file_path, "frames"))


def make_biweekly_inits(
    start: str = "2022-01-01T00:00:00.000000000",
    end: str = "2024-02-29T00:00:00.000000000",
    sequence_length: int = 15,
    timedelta: int = 1 
):
    # Convert start and end to pandas Timestamp objects with UTC timezone
    start_date = pd.Timestamp(start, tz='UTC')  #+ pd.Timedelta(hours=sequence_length*timedelta*24)
    end_date = pd.Timestamp(end, tz='UTC') - pd.Timedelta(hours=sequence_length*timedelta*24)
    
    # Generate date range for Mondays at 11:00 UTC
    mondays = pd.date_range(start=start_date, end=end_date, freq='W-MON', tz='UTC') 
    
    # Generate date range for Thursdays at 11:00 UTC
    thursdays = pd.date_range(start=start_date, end=end_date, freq='W-THU', tz='UTC')
    
    # Combine Mondays and Thursdays
    all_dates = mondays.union(thursdays).sort_values()

    naive_timestamp = all_dates.tz_localize(None)

    return naive_timestamp.to_numpy()

def remap(cfg, data, name=None): # 32, 64
    latitudes = cfg.data.height
    longitudes = cfg.data.width
    
    hpx_remapper = HEALPixRemap(
        latitudes=latitudes,
        longitudes=longitudes,
        nside=data.shape[-1],
        verbose=cfg.verbose
    )
    if len(data.shape) == 5:
        # Inits
        B, C, _, _, _ = data.shape
        arguments = []
        for b_idx in range(B):
            for c_idx in range(C):
                arguments.append([data[b_idx][c_idx]])
        data = mp_hpx2ll(remapper=hpx_remapper, arguments=arguments, name=name)
        data = np.reshape(data, (B, C, latitudes, longitudes))  # [(b c) lat lon] -> [b c lat lon]
    else:
        # Outputs and targets
        B, T, C, _, _, _ = data.shape
        arguments = []
        for b_idx in range(B):
            for t_idx in range(T):
                for c_idx in range(C):
                    arguments.append([data[b_idx][t_idx][c_idx]])
        data = mp_hpx2ll(remapper=hpx_remapper, arguments=arguments, name=name)
        data = np.reshape(data, (B, T, C, latitudes, longitudes))  # [(b t c) lat lon] -> [b t c lat lon]
    return data


def mp_hpx2ll(remapper, arguments, name=None):
    # Run the remapping in parallel
    poolsize = 5
    with multiprocessing.Pool(poolsize) as pool:
        data = np.array(list(tqdm(pool.istarmap(remapper.hpx2ll, arguments), total=len(arguments), desc=name)))
        pool.terminate()
        pool.join()
    return data