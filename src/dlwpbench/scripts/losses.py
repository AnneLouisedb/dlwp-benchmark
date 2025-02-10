import os
import sys
import time
import threading
import xarray as xr
import numpy as np
import torch as th
import wandb
import matplotlib.pyplot as plt

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
                f'Expected uniform longitude spacing. {dataset.lon.values=}'
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
        return spectrum * self._circumference(spectrum)


def compute_zonal_spectrum(dataset: xr.Dataset, variable_name: str | list[str]) -> xr.Dataset:
    zonal_spectrum = ZonalSpectrum(variable_name)
    return zonal_spectrum.compute(dataset)


class MELRCalculator:
    def __init__(self):
        # Load ERA5 file to get dimensions
        era5file = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/zarr/weatherbench/msl/msl_1979_5.625deg.zarr'
        self.era5_ds = xr.open_dataset(era5file)
        
        # Define dimensions
        self.dims = ['time', 'level', 'lat', 'lon']
        self.coords = {dim: self.era5_ds[dim] for dim in self.dims if dim in self.era5_ds.dims}
        # Reset time coordinate to array of 16
        self.coords['time'] = np.arange(16)

    def apply(self, pred_np, true_np, variable_name):
        # Convert tensors to numpy arrays
        
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
        E_pred = pred_spectrum.mean(dim='time').mean('lat')
        E_true = true_spectrum.mean(dim='time').mean('lat')

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(E_pred, label='Predicted')
        plt.plot(E_true, label='True')
        plt.xlabel('Zonal Wavenumber')
        plt.ylabel('Energy')
        plt.title(f'Energy vs Zonal Wavenumber for {variable_name}')
        plt.xscale('log', base=2)
        plt.yscale('log', base=2) 
        plt.legend()
        plt.grid(True)
        
        # Save the plot and log it to wandb
        plt.savefig(f'Energy_plot_{variable_name}.png')
        wandb.log({"Energy_plot": wandb.Image(f'Energy_plot_{variable_name}.png')})
        
        # Close the plot to free up memory
        plt.close()

        # Add a small epsilon to avoid division by zero or log of zero
        epsilon = 1e-10
        ratio = np.log((E_pred + epsilon) / (E_true + epsilon))
        
        # Average over all dimensions
        melr = ratio.mean().values

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(ratio)
        plt.xlabel('Zonal Wavenumber')
        plt.ylabel('MELR')
        plt.title(f'MELR vs Zonal Wavenumber for {variable_name}')
        plt.xscale('log', base=2)
        plt.grid(True)
        
        # Save the plot and log it to wandb
        plt.savefig(f'melr_plot_{variable_name}.png')
        wandb.log({"MELR_plot": wandb.Image(f'melr_plot_{variable_name}.png')})
        
        # Close the plot to free up memory
        plt.close()
        
        return melr
        

class CustomMSELoss(th.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean", healpix=False, weighted = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.weighted = weighted

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
        

        if self.weighted:
            self.spatial_weights = th.as_tensor(self.weights, device=input.device)
            d = ((target-input)**2)*self.spatial_weights 
        else:
            d = ((target-input)**2)

        if self.reduction == 'mean':
            return th.mean(d)
        else:
            # No reduction
            return d
    