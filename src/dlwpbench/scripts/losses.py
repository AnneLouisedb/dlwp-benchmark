import os
import sys
import glob
import time
import re
import threading
import xarray as xr
import numpy as np
import torch as th
import wandb
import math
import matplotlib.pyplot as plt
from scripts.chaosbench.criterion import SpectralDiv
  
# Code from WeatherBench2
EARTH_RADIUS_M = 1000 * (6357 + 6378) / 2

class ZonalSpectrum:
    def __init__(self, variable_name: str | list[str]):
        self.variable_name = variable_name

    def _circumference(self, dataset: xr.Dataset) -> xr.DataArray:
        """Earth's circumference as a function of latitude."""
        circum_at_equator = 2 * np.pi * EARTH_RADIUS_M
        return np.cos(dataset.lat * np.pi / 180) * circum_at_equator

    def lon_spacing_m(self, dataset: xr.Dataset) -> xr.DataArray:
        """Spacing (meters) between longitudinal values in `dataset`."""
        diffs = dataset.lon.diff('lon')
        if np.max(np.abs(diffs - diffs[0])) > 1e-3:
            raise ValueError(
                f'Expected uniform longitude spacing. {dataset.lon.values}'
            )
        return self._circumference(dataset) * diffs[0].data / 360

    def compute(self, dataset: xr.Dataset) -> xr.DataArray:
        """Computes zonal power at wavenumber and frequency."""
        spacing = self.lon_spacing_m(dataset)

        def simple_power(f_x):
            f_k = np.fft.rfft(f_x, axis=-1, norm='forward')
            # freq > 0 should be counted twice in power since it accounts for both
            # positive and negative complex values.
            one_and_many_twos = np.concatenate(([1], [2] * (f_k.shape[-1] - 1)))
            return np.real(f_k * np.conj(f_k)) * one_and_many_twos

        spectrum = xr.apply_ufunc(
            simple_power,
            dataset,
            input_core_dims=[['lon']],
            output_core_dims=[['lon']],
            exclude_dims={'lon'},
        ).rename_dims({'lon': 'zonal_wavenumber'})[self.variable_name]

        spectrum = spectrum.assign_coords(
            zonal_wavenumber=('zonal_wavenumber', spectrum.zonal_wavenumber.data)
        )
        
        base_frequency = xr.DataArray(
            np.fft.rfftfreq(len(dataset.lon)),
            dims='zonal_wavenumber',
            coords={'zonal_wavenumber': spectrum.zonal_wavenumber},
        )
        
        spectrum = spectrum.assign_coords(frequency=base_frequency / spacing)
        spectrum['frequency'] = spectrum.frequency.assign_attrs(units='1 / m')

        spectrum = spectrum.assign_coords(wavelength=1 / spectrum.frequency)
        spectrum['wavelength'] = spectrum.wavelength.assign_attrs(units='m')

        # This last step ensures the sum of spectral components is equal to the
        # (discrete) integral of data around a line of latitude.
        return spectrum  * self._circumference(spectrum)


def compute_zonal_spectrum(dataset: xr.Dataset, variable_name: str | list[str]) -> xr.Dataset:
    zonal_spectrum = ZonalSpectrum(variable_name)
    return zonal_spectrum.compute(dataset)


class MELRCalculator:
    def __init__(self, device, wandb=True):
        self.coords = {}
        self.device= device
        self.wandb = wandb
        
    def apply(self, pred_np, true_np, variable_name, epoch):
        # Convert tensors to numpy arrays
        sample_dim, lat_dim, lon_dim = pred_np.shape

        self.coords['sample'] = np.arange(sample_dim)
        self.coords['lat'] = np.linspace(-90, 90, lat_dim)
        self.coords['lon'] = np.linspace(0, 360, lon_dim)

        SpecDiv = SpectralDiv(device = self.device, percentile=0.6, is_train=True)
        sdiv = SpecDiv(true_np, pred_np).cpu().numpy()

        if self.wandb:

            wandb.log({f"SpectralDiv_0.6_{variable_name}": sdiv})
        
        # Create xarray Datasets
        pred_ds = xr.Dataset(
            {variable_name: (list(self.coords.keys()), pred_np)},
            coords=self.coords
        )
        
        true_ds = xr.Dataset(
            {variable_name: (list(self.coords.keys()), true_np)},
            coords=self.coords
        )

        # Compute zonal spectra
        pred_spectrum = compute_zonal_spectrum(pred_ds, variable_name)
        true_spectrum = compute_zonal_spectrum(true_ds, variable_name)
        
        # Compute MELR
        E_pred = pred_spectrum.mean(dim='sample').mean('lat')
        E_pred = E_pred / np.nansum(E_pred.values)

        E_true = true_spectrum.mean(dim='sample').mean('lat')
        E_true = E_true/ np.nansum(E_true.values)

        # Use np.maximum to clamp values
        ratio = np.maximum(E_true / E_pred, 1e-9)

        # Calculate the divergence
        div = E_true * np.log(ratio)

        wavenumbers = true_spectrum.wavelength.mean('lat') # / 1000

        print(E_true)
        print(wavenumbers)

        if not self.wandb:
            # Create a normal plot using Matplotlib
            plt.figure(figsize=(10, 6))
            
            # Plot predicted and true energy spectra
            plt.plot(
                [math.log10(i) for i in wavenumbers.values],
                [math.log10(float(i)) for i in E_pred.values],
                label="Log10_E_pred",
                color="blue",
                linestyle="-"
            )
            plt.plot(
                [math.log10(i) for i in wavenumbers.values],
                [math.log10(float(i)) for i in E_true.values],
                label="Log10_E_true",
                color="orange",
                linestyle="--"
            )
            
            plt.title(f"Zonal Power Spectrum: Predicted vs True ({variable_name})", fontsize=14)
            plt.xlabel("Log10(Zonal Wavenumber [km])", fontsize=12)
            plt.ylabel("Log10(Power)", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            
            # Show the plot
            plt.show()

        if self.wandb:

            # Create a W&B Table
            table = wandb.Table(columns=["Zonal Wavenumber (km)", "Value", "Variable", "Epoch"])


            for i in range(len(wavenumbers.values)):
                assert E_pred[i] is not None
                
                # Add both E_pred and E_true values for each wavenumber
                table.add_data( math.log10(wavenumbers.values[i]), math.log10(float(E_pred[i])), "Log10_E_pred", epoch)
                table.add_data( math.log10(wavenumbers.values[i]), math.log10(float(E_true[i])), "Log10_E_true", epoch)


            # Create a line plot using 'series' for line grouping
            melr_plot = wandb.plot.line(
                table,
                x="Zonal Wavenumber (km)",
                y="Value",
                stroke="Variable",  # Combines variable and epoch in stroke
                title=f"Zonal Power Spectrum: Predicted vs True ({variable_name})")

            # Log the plot to W&B
            wandb.log({f"energy_spectrum {variable_name}": melr_plot})

        
        threshold = E_pred.quantile(0.6, dim=['zonal_wavenumber'])
        E_pred_thres = E_pred.sel(zonal_wavenumber=E_pred['zonal_wavenumber'] > threshold)
        E_true_thres = E_true.sel(zonal_wavenumber=E_true['zonal_wavenumber'] > threshold)

        # Use np.maximum to clamp values
        ratio = np.maximum(E_true_thres / E_pred_thres, 1e-9)

        # Calculate the divergence
        div_thres = np.nansum(E_true_thres * np.log(ratio))

        if self.wandb:
        
            wandb.log({f"spectral_divergence_above_0.6": np.mean(div_thres), "epoch":epoch//5})
       
        return table
        

class CustomMSELoss(th.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, cfg, reduction: str = "mean", weighted = False, channel_weights=False, num_train_timesteps = None) -> None:
        super().__init__()
        self.reduction = reduction
        self.weighted = weighted
        self.channel_weights = channel_weights
        data_path = cfg.data.data_path + 'constants/'
        zarr_files = glob.glob(os.path.join(data_path, 'constants*.zarr'))
        dataset = xr.open_zarr(zarr_files[0])
        weights_values = np.nan_to_num(dataset.latitude_weights.values, nan=0.0)

        self.weights = th.tensor(weights_values)
        self.timesteps = num_train_timesteps
        
   
    def forward(self, input, target, timesteps = None, diffusion=None):

        if diffusion:
            # Loss weighting strategy using SNR
            sigmas = timesteps / self.timesteps 
            snr_weights = (1 - sigmas) / sigmas
            snr_weights = snr_weights.to(input.device)[:, None, None, None]  # Adjust dimensions for broadcasting
            
        if self.channel_weights:
            # Create linear weights for channels
            def normalized_level_weights(input):
                
                pressure_levels = [1013.25, 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100] 
                #pressure_levels = [1013.25, 850, 500, 250] 
                #pressure_levels = [1013.25, 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 2*1013.25, 850, 850, 500]
                
                channel_weights = pressure_levels / np.sum(pressure_levels)
                channel = th.tensor(channel_weights, device = input.device)
                return channel.view(1, -1, 1, 1, 1) # B, C, F, W, H
             
            d = ((target-input)**2)

            if diffusion:
                d *= snr_weights

            d *= normalized_level_weights(input)
            
        
        elif self.weighted:
            self.spatial_weights = th.as_tensor(self.weights, device=input.device)
            
            d = ((target-input)**2)*self.spatial_weights
            
        else:
            d = ((target-input)**2)

        if self.reduction == 'mean':
            return th.mean(d)
        else:
            # No reduction
            return d
    