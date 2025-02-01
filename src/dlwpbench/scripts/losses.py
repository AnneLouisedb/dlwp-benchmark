import os
import sys
import time
import threading
import xarray as xr
import numpy as np
import torch as th
import wandb

class CustomSpectralLoss(th.nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        
        path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/weatherbench/latitude_weights/latitude_weights_5.6degree.nc'
        weights_map = xr.open_dataset(path) 
        self.weights = th.tensor(weights_map.weights.values).T

    def forward(self, input, target):
        self.spatial_weights = th.as_tensor(self.weights, device=input.device)

        # Apply FFT
        
        input_fft = th.fft.fft(input.float(), dim=-1)
        target_fft = th.fft.fft(target.float(), dim=-1)

        input_spectrum = th.abs(input_fft)**2
        target_spectrum = th.abs(target_fft)**2

        print('shape target', target_spectrum.shape)

        spectral_diff = input_spectrum - target_spectrum
        weighted_spectral_diff = spectral_diff * self.spatial_weights.unsqueeze(0).unsqueeze(0)

        loss = th.mean(weighted_spectral_diff **2)
        
        return loss
        

class CustomMSELoss(th.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean", healpix=False) -> None:
        super().__init__()
        self.reduction = reduction

        if healpix:
            path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/zarr/weatherbench_hpx8/latitude_weights/latitude_weights_5.625deg.zarr'
            weights_map = xr.open_dataset(path)
            weights_values = np.nan_to_num(weights_map.weights.values, nan=0.0)
            self.weights = th.tensor(weights_values) 

        else:
            path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/weatherbench/latitude_weights/latitude_weights_5.6degree.nc'
            weights_map = xr.open_dataset(path) 
            self.weights = th.tensor(weights_map.weights.values).T
            
    def forward(self, input, target):
        
        self.spatial_weights = th.as_tensor(self.weights, device=input.device)

        d = ((target-input)**2) #*self.spatial_weights 

        return th.mean(d)
    