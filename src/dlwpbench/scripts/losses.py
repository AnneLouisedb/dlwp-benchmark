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
        pass
       
    def apply(self, pred_np, true_np, variable_name, epoch):
        # Convert tensors to numpy arrays
        sample_dim, lat_dim, lon_dim = pred_np.shape

        self.coords['sample'] = np.arange(sample_dim)
        self.coords['lat'] = np.linspace(-90, 90, lat_dim)
        self.coords['lon'] = np.linspace(0, 360, lon_dim)
        
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
        E_true = true_spectrum.mean(dim='sample').mean('lat')

    
        # Transform data to log base 2 for both axes (zonal wavenumber and energy/MELR values)
        log_wavenumbers = np.log2(np.arange(len(E_pred)) + 1)  # Add 1 to avoid log2(0)
        log_E_pred = np.log2(E_pred + epsilon)
        log_E_true = np.log2(E_true + epsilon)
        
        # Add a small epsilon to avoid division by zero or log of zero
        epsilon = 1e-10
        ratio = np.log((E_pred + epsilon) / (E_true + epsilon))
        
        # Average over all dimensions
        melr = ratio.mean().values

        table = wandb.Table(columns=["Log2(Zonal Wavenumber)", "Log2(E_pred)", "Log2(E_true)", "MELR", "Epoch"])

        for i in range(len(log_wavenumbers)):
            table.add_data(log_wavenumbers[i], log_E_pred[i].values, log_E_true[i].values, ratio[i].values, epoch)

        wandb.log({f"{variable_name}_MELR_Log2_Table": table}) 
        # filter the table for the latest epoch
        energy_plot = wandb.plot.line(
        table,
        x="Log2(Zonal Wavenumber)",
        y=["Log2(E_pred)", "Log2(E_true)"],  
        keys=["Predicted Energy", "True Energy"], 
        title=f"Energy vs. Zonal Wavenumber for {variable_name}",
        xname="Log2(Zonal Wavenumber)",
        
        )
        # Log the custom plot to W&B
        wandb.log({f"{variable_name}_Energy_vs_Zonal_Wavenumber": energy_plot})

        melr_plot = wandb.plot.line(
            table,
            x="Log2(Zonal Wavenumber)",
            y="MELR",
            title=f"MELR vs. Zonal Wavenumber for {variable_name}",
            xname="Log2(Zonal Wavenumber)",
            keys=["Epoch"]
        )
        wandb.log({f"{variable_name}_MELR_vs_Zonal_Wavenumber": melr_plot})

        return melr
        

class CustomMSELoss(th.nn.Module):
    """Custom MSE loss for PDEs.

    MSE but summed over time and fields, then averaged over space and batch.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, cfg, reduction: str = "mean", weighted = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.weighted = weighted

        data_path = cfg.data.data_path + 'constants/'
        zarr_files = glob.glob(os.path.join(data_path, 'constants*.zarr'))
        dataset = xr.open_zarr(zarr_files[0])
        weights_values = np.nan_to_num(dataset.latitude_weights.values, nan=0.0)
        self.weights = th.tensor(weights_values)
            
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
    