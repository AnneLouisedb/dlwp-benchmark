import os
import sys
import time
import threading
import xarray as xr
import numpy as np
import torch as th
import wandb


class CustomMSELoss(th.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        
        path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/weatherbench/latitude_weights/latitude_weights_5.6degree.nc'
        weights_map = xr.open_dataset(path) 
        self.weights = th.tensor(weights_map.weights.values).T
        
    def forward(self, input, target):
        
        self.spatial_weights = th.tensor(self.weights, device = input.device)

        d = ((target-input)**2)*self.spatial_weights 

        return th.mean(d)