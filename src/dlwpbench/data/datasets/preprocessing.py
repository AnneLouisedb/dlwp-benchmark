# script to load and plot each variable at "1W" in its respective spatial spectrum
"""Preprocessor for s2spy workflow."""

import warnings
from typing import Literal
from typing import Union
import numpy as np
import scipy.stats
import xarray as xr
from geogif import gif
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _linregress(
    x: np.ndarray,
    y: np.ndarray,
    nan_mask: Literal["individual", "complete"] = "individual",
) -> tuple[float, float]:
    """Calculate the slope and intercept between two arrays using scipy's linregress.

    Used to make linregress more ufunc-friendly.


    Args:
        x: First array.
        y: Second array.
        nan_mask: How to handle NaN values. If 'complete', returns nan if x or y contains
            1 or more NaN values. If 'individual', fit a trend by masking only the
            indices with the NaNs.

    Returns:
        slope, intercept
    """
    if nan_mask == "individual":
        mask = np.logical_or(np.isnan(x), np.isnan(y))
        if np.all(mask):
            slope, intercept = np.nan, np.nan
        elif np.any(mask):
            slope, intercept, _, _, _ = scipy.stats.linregress(x[~mask], y[~mask])
        else:
            slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
        return slope, intercept
    elif nan_mask == "complete":
        if np.logical_or(np.isnan(x), np.isnan(y)).any():
            slope, intercept = np.nan, np.nan  # any NaNs in timeseries, return NaN.
        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
        return slope, intercept


def _trend_linear(
    data: Union[xr.DataArray, xr.Dataset], nan_mask: str = "complete"
) -> dict:
    """Calculate the linear trend over time.

    Args:
        data: The input data of which you want to know the trend.
        nan_mask: How to handle NaN values. If 'complete', returns nan if x or y contains
            1 or more NaN values. If 'individual', fit a trend by masking only the
            indices with the NaNs.

    Returns:
        Dictionary containing the linear trend information (slope and intercept)
    """
    assert nan_mask in [
        "complete",
        "individual",
    ], "nan_mask should be 'complete' or 'individual'"
    slope, intercept = xr.apply_ufunc(
        _linregress,
        data["time"].astype(float),
        data,
        nan_mask,
        input_core_dims=[["time"], ["time"], []],
        output_core_dims=[[], []],
        vectorize=True,
    )
    return {"slope": slope, "intercept": intercept}


def _get_lineartrend_timeseries(data: Union[xr.DataArray, xr.Dataset], trend: dict):
    """Calculate the linear trend timeseries from the trend dictionary."""
    trend_timeseries = trend["intercept"] + trend["slope"] * (
        data["time"].astype(float)
    )
    return trend_timeseries.transpose(*list(data.dims) + [...])


def _trend_poly(data: Union[xr.DataArray, xr.Dataset], degree: int = 2) -> dict:
    """Calculate the polynomial trend over time and return coefficients.

    Args:
        data: Input data.
        degree: Degree of the polynomial for detrending.

    Returns:
        Dictionary containing polynomial trend coefficients.
    """
    fixed_timestamp = np.datetime64("1900-01-01")
    data.coords["ordinal_day"] = (
        ("time",),
        (data.time - fixed_timestamp).values.astype("timedelta64[D]").astype(int),
    )
    coeffs = data.swap_dims({"time": "ordinal_day"}).polyfit(
        "ordinal_day", deg=degree, skipna=True
    )
    return {"coefficients": coeffs}


def _get_polytrend_timeseries(data: Union[xr.DataArray, xr.Dataset], trend: dict):
    """Calculate the polynomial trend timeseries from the trend dictionary."""
    fixed_timestamp = np.datetime64("1900-01-01")
    polynomial_trend = (
        xr.polyval(
            data.assign_coords(
                ordinal_day=(
                    "time",
                    (data.time - fixed_timestamp)
                    .values.astype("timedelta64[D]")
                    .astype(int),
                )
            ).swap_dims({"time": "ordinal_day"})["ordinal_day"],
            trend["coefficients"],
        ).swap_dims({"ordinal_day": "time"})
    ).drop_vars("ordinal_day")
    # data = data # remove the ordinal_day coordinate
    # rename f"{data_var}_polyfit_coeffiencts" to orginal data_var name
    if isinstance(data, xr.Dataset):
        da_names = list(data.data_vars)
        rename = dict(
            map(
                lambda i, j: (i, j),
                list(polynomial_trend.data_vars),
                da_names,
            )
        )
        polynomial_trend = polynomial_trend.rename(rename)
    if isinstance(
        data, xr.DataArray
    ):  # keep consistent with input data and _get_lineartrend_timeseries
        polynomial_trend = (
            polynomial_trend.to_array().squeeze("variable").drop_vars("variable")
        )
        polynomial_trend.name = (
            data.name if data.name is not None else "timeseries_polyfit"
        )
    return polynomial_trend.transpose(*list(data.dims) + [...])


def _subtract_linear_trend(data: Union[xr.DataArray, xr.Dataset], trend: dict):
    """Subtract a previously calclulated linear trend from (new) data."""
    return data - _get_lineartrend_timeseries(data, trend)


def _subtract_polynomial_trend(
    data: Union[xr.DataArray, xr.Dataset],
    trend: dict,
):
    """Subtract a previously calculated polynomial trend from (new) data.

    Args:
        data: The data from which to subtract the trend (either an xarray DataArray or Dataset).
        trend: A dictionary containing the polynomial trend coefficients.

    Returns:
        The data with the polynomial trend subtracted.
    """
    # Subtract the polynomial trend from the data
    return data - _get_polytrend_timeseries(data, trend)


def _get_trend(
    data: Union[xr.DataArray, xr.Dataset],
    method: str,
    nan_mask: str = "complete",
    degree=2,
):
    """Calculate the trend, with a certain method. Only linear is implemented."""
    if method == "linear":
        return _trend_linear(data, nan_mask)

    if method == "polynomial":
        if nan_mask != "complete":
            raise ValueError("Polynomial currently only supports 'complete' nan_mask")
        return _trend_poly(data, degree)
    raise ValueError(f"Unkown detrending method '{method}'")


def _subtract_trend(data: Union[xr.DataArray, xr.Dataset], method: str, trend: dict):
    """Subtract the previously calculated trend from (new) data. Only linear is implemented."""
    if method == "linear":
        detrended = _subtract_linear_trend(data, trend)
    if method == "polynomial":
        detrended = _subtract_polynomial_trend(data, trend)
    if method not in ["linear", "polynomial"]:
        raise NotImplementedError(f"Detrending method '{method}' not implemented.")
    #detrended.name = data.name
    return detrended


def _get_climatology(
    data: Union[xr.Dataset, xr.DataArray],
    timescale: Literal["monthly", "weekly", "daily"],
):
    """Calculate the climatology of timeseries data."""
    _check_data_resolution_match(data, timescale)
    if timescale == "monthly":
        climatology = data.groupby("time.month").mean("time")
    elif timescale == "weekly":
        climatology = data.groupby(data["time"].dt.isocalendar().week).mean("time")
    elif timescale == "daily":
        climatology = data.groupby("time.dayofyear").mean("time")
    else:
        raise ValueError("Given timescale is not supported.")

    return climatology



def _subtract_climatology(
    data: Union[xr.Dataset, xr.DataArray],
    timescale: Literal["monthly", "weekly", "daily"],
    climatology: Union[xr.Dataset, xr.DataArray],
):
    if timescale == "monthly":
        deseasonalized = data.groupby("time.month") - climatology
    elif timescale == "weekly":
        deseasonalized = data.groupby(data["time"].dt.isocalendar().week) - climatology
    elif timescale == "daily":
        deseasonalized = data.groupby("time.dayofyear") - climatology
    else:
        raise ValueError("Given timescale is not supported.")

    # restore name
    # #climatology.name = data.name
    # if isinstance(data, xr.DataArray):
    #     climatology.name = data.name
    # elif isinstance(data, xr.Dataset):
    #     # Optionally set a default name or handle specific variables
    #     climatology.name = 'geopotential-500'  # Example: using the first variable's name

    return deseasonalized


def _check_input_data(data: Union[xr.DataArray, xr.Dataset]):
    """Check the input data for compatiblity with the preprocessor.

    Args:
        data: Data to validate.

    Raises:
        ValueError: If the input data is of the wrong type.
        ValueError: If the input data does not have a 'time' dimension.
    """
    if not any(isinstance(data, dtype) for dtype in (xr.DataArray, xr.Dataset)):
        raise ValueError(
            "Input data has to be an xarray-DataArray or xarray-Dataset, "
            f"not {type(data)}"
        )
    if "time" not in data.dims:
        raise ValueError(
            "Analysis is done of the 'time' dimension, but the input data"
            f" only has dims: {data.dims}"
        )


def _check_temporal_resolution(
    timescale: Literal["monthly", "weekly", "daily"],
) -> Literal["monthly", "weekly", "daily"]:
    support_temporal_resolution = ["monthly", "weekly", "daily"]
    if timescale not in support_temporal_resolution:
        raise ValueError(
            "Given temporal resoltuion is not supported."
            "Please choose from 'monthly', 'weekly', 'daily'."
        )
    return timescale


def _check_data_resolution_match(
    data: Union[xr.DataArray, xr.Dataset],
    timescale: Literal["monthly", "weekly", "daily"],
):
    """Check if the temporal resolution of input is the same as given timescale."""
    timescale_dict = {
        "monthly": np.timedelta64(1, "M"),
        "weekly": np.timedelta64(1, "W"),
        "daily": np.timedelta64(1, "D"),
    }
    time_intervals = np.diff(data["time"].to_numpy())
    temporal_resolution = np.median(time_intervals).astype("timedelta64[D]")
    if timescale == "monthly":
        temporal_resolution = temporal_resolution.astype(int)
        min_days, max_days = (28, 31)
        if not max_days >= temporal_resolution >= min_days:
            warnings.warn(
                "The temporal resolution of data does not completely match "
                "the target timescale. Please check your input data.",
                stacklevel=1,
            )

    elif timescale in timescale_dict:
        if timescale_dict[timescale].astype("timedelta64[D]") != temporal_resolution:
            warnings.warn(
                "The temporal resolution of data does not completely match "
                "the target timescale. Please check your input data.",
                stacklevel=1,
            )


class Preprocessor:
    """Preprocessor for s2s data."""

    def __init__(  # noqa: PLR0913
        self,
        rolling_window_size: Union[int, None],
        timescale: Literal["monthly", "weekly", "daily"],
        rolling_min_periods: int = 1,
        subtract_climatology: bool = True,
        detrend: Union[str, None] = "linear",
        nan_mask: str = "complete",
    ):
        """Preprocessor for s2s data. Can detrend as well as deseasonalize.

        On calling `.fit(data)`, the preprocessor will:
         - Calculate the rolling mean of the input data.
         - Calculate and store the climatology of the rolling mean.
         - Calculate and store the trend of the rolling mean.

        When calling `.transform(data)`, the preprocessor will:
         - Remove the climatology from a copy of the data.
         - Remove the (stored) trend from this deseasonalized data.
         - Return the detrended and deseasonalized data.

        Args:
            rolling_window_size: The size of the rolling window that will be applied
                before calculating the trend and climatology. Setting this to None will
                skip this step.
            rolling_min_periods: The minimum number of periods within a rolling window.
                If higher than 1 (the default), NaN values will be present at the start
                and end of the preprocessed data.
            subtract_climatology (optional): If you want to calculate and remove the
                climatology of the data. Defaults to True.
            detrend (optional): Which method to use for detrending. Choose from "linear"
                or "polynomial". Defaults to "linear". If you want to skip detrending,
                set this to None.
            timescale: Temporal resolution of input data.
            nan_mask: How to handle NaN values. If 'complete', returns nan if x or y contains
                1 or more NaN values. If 'individual', fit a trend by masking only the
                indices with the NaNs.
        """
        self._window_size = rolling_window_size
        self._min_periods = rolling_min_periods
        self._detrend = detrend
        self._subtract_climatology = subtract_climatology
        self._nan_mask = nan_mask

        if subtract_climatology:
            self._timescale = _check_temporal_resolution(timescale)

        self._climatology: Union[xr.DataArray, xr.Dataset]
        self._trend: dict
        self._is_fit = False

    def fit(self, data: Union[xr.DataArray, xr.Dataset]) -> None:
        """Fit this Preprocessor to input data.

        Args:
            data: Input data for fitting.
        """
        _check_input_data(data)
        if self._window_size not in [None, 1]:
            data_rolling = data.rolling(
                dim={"time": self._window_size},  # type: ignore
                min_periods=self._min_periods,
                center=True,
            ).mean()
        # TODO: give option to be a gaussian-like window, instead of a block.
        else:
            data_rolling = data

        if self._subtract_climatology:
            self._climatology = _get_climatology(data_rolling, self._timescale)
        if self._detrend is not None:
            if self._subtract_climatology:
                deseasonalized = _subtract_climatology(
                    data_rolling, self._timescale, self._climatology
                )
                self._trend = _get_trend(deseasonalized, self._detrend, self._nan_mask)
            else:
                self._trend = _get_trend(data_rolling, self._detrend, self._nan_mask)

        self._is_fit = True

    def transform(
        self, data: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Apply the preprocessing steps to the input data.

        Args:
            data: Input data to perform preprocessing.

        Returns:
            Preprocessed data.
        """
        if not self._is_fit:
            raise ValueError(
                "The preprocessor has to be fit to data before a transform"
                " can be applied"
            )

        if self._subtract_climatology:
            d = _subtract_climatology(data, self._timescale, self._climatology)
        else:
            d = data

        if self._detrend is not None:
            return _subtract_trend(d, self._detrend, self.trend)

        return d

    def fit_transform(
        self, data: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Fit this Preprocessor to input data, and then apply the steps to the data.

        Args:
            data: Input data for fit and transform.

        Returns:
            Preprocessed data.
        """
        self.fit(data)
        return self.transform(data)

    def get_trend_timeseries(self, data, align_coords: bool = False):
        """Get the trend timeseries from the data.

        Args:
            data (xr.DataArray or xr.Dataset): input data.
            align_coords (bool): align coordinates to data.
        """
        if not self._is_fit:
            raise ValueError(
                "The preprocessor has to be fit to data before the trend"
                " timeseries can be requested."
            )
        if self._detrend is None:
            raise ValueError("Detrending is set to `None`, so no trend is available")
        if align_coords:
            trend = self.align_coords(data)
        else:
            trend = self.trend
        if self._detrend == "linear":
            return _get_lineartrend_timeseries(data, trend)
        elif self._detrend == "polynomial":
            return _get_polytrend_timeseries(data, trend)
        raise ValueError(f"Unkown detrending method '{self._detrend}'")

    @property
    def trend(self) -> dict:
        """Return the stored trend (dictionary)."""
        if not self._detrend:
            raise ValueError("Detrending is set to `None`, so no trend is available")
        if not self._is_fit:
            raise ValueError(
                "The preprocessor has to be fit to data before the trend"
                " can be requested."
            )
        return self._trend

    @property
    def climatology(self) -> Union[xr.DataArray, xr.Dataset]:
        """Return the stored climatology data."""
        if not self._subtract_climatology:
            raise ValueError(
                "`subtract_climatology is set to `False`, so no climatology "
                "data is available"
            )
        if not self._is_fit:
            raise ValueError(
                "The preprocessor has to be fit to data before the"
                " climatology can be requested."
            )
        return self._climatology

    def align_coords(self, data):
        """Align coordinates between data and trend.

        Args:
            data (xr.DataArray or xr.Dataset): time, (lat), (lon) array.
            trend ():
        """
        if self._detrend == "linear":
            align_trend = self.trend.copy()
            align_trend["intercept"], align_trend["slope"] = xr.align(
                *[align_trend["intercept"], align_trend["slope"], data]
            )[:2]
        elif self._detrend == "polynomial":
            align_trend = self.trend.copy()
            align_trend["coefficients"] = xr.align(align_trend["coefficients"], data)[0]
        else:
            raise ValueError(f"Unkown detrending method '{self._detrend}'")
        return align_trend
    

# Function for preprocessing
def daily_preprocessing(dataset, var = 'geopotential-500', winter_mask = False):

    deseasonalizer = Preprocessor(
            timescale="daily",
            rolling_window_size=10,
            detrend="linear",
            subtract_climatology=True,
        )
    da = deseasonalizer.fit_transform(dataset)
    da = da.resample(time='1D').mean()
    
    if winter_mask:

        winter_mask = da.time.dt.month.isin([12,1,2])
        winter_weekly_mean = da.where(winter_mask, drop=True)
        da = winter_weekly_mean

    # Select the latitude range
    variable = da[var] #.sel(latitude=slice(50,40))

    # Calculate the mean along the latitude dimension
    variable_mean = variable.mean(dim='latitude')

    # Create a new coordinate for latitude
    new_lat = xr.DataArray([45], dims='latitude', coords={'latitude': [45]})

    # Assign the mean values to the new latitude coordinate
    variable = variable_mean.expand_dims({'latitude': new_lat})

    # Apply FFT along the longitude dimension
    fft_result = np.fft.fft(variable.values, axis=variable.get_axis_num('longitude')) # over second dimension

    # Compute amplitude spectrum
    amplitude_spectrum =  np.abs(fft_result)

    # Create a new DataArray with the amplitude spectrum
    fft_da = xr.DataArray(amplitude_spectrum, coords=variable.coords, dims=variable.dims)

    n_lon = variable.shape[variable.get_axis_num('longitude')]
    wavenumbers = np.fft.fftfreq(n_lon, d=1.0)[:n_lon//2]
    time_step = np.diff(variable.time.values).mean().astype('timedelta64[s]').astype(float)
    frequencies = np.fft.fftfreq(variable.time.size, d=time_step)
    phase_speed = np.outer(frequencies, 1/wavenumbers)
    phase_speed_da = xr.DataArray(phase_speed,coords={'frequency': frequencies, 'wavenumber': wavenumbers}, dims=['frequency', 'wavenumber'])
        
    return variable, fft_da, phase_speed_da
    

def weekly_preprocessing(dataset, var = 'geopotential-500', winter_mask = False):

    deseasonalizer = Preprocessor(
            timescale="daily",
            rolling_window_size=25,
            detrend="linear",
            subtract_climatology=False, #True,
        )
    da = deseasonalizer.fit_transform(dataset)
    da = da.resample(time='1W').mean()
    
    if winter_mask:

        winter_mask = da.time.dt.month.isin([12,1,2])
        winter_weekly_mean = da.where(winter_mask, drop=True)
        da = winter_weekly_mean

    # Select the latitude range
    variable = da[var].sel(latitude=slice(50,40))

    # Calculate the mean along the latitude dimension
    variable_mean = variable.mean(dim='latitude')

    # Create a new coordinate for latitude
    new_lat = xr.DataArray([45], dims='latitude', coords={'latitude': [45]})

    # Assign the mean values to the new latitude coordinate
    variable = variable_mean.expand_dims({'latitude': new_lat})

    # Apply FFT along the longitude dimension
    fft_result = np.fft.fft(variable.values, axis=variable.get_axis_num('longitude')) # over second dimension

    # Compute amplitude spectrum
    amplitude_spectrum =  np.abs(fft_result)

    # Create a new DataArray with the amplitude spectrum
    fft_da = xr.DataArray(amplitude_spectrum, coords=variable.coords, dims=variable.dims)
   
    return variable, fft_da

def plot_val_at_lat(da, var = 'geopotential-500', start_date = '2015-01-01', end_date = '2018-12-31'):

    da = da.sel(time=slice(start_date, end_date))
    temperature_array = da.values

    # Get the longitude values
    longitudes = da.longitude.values
 
    # Create a time array (assuming equal time steps)
    time_steps = np.arange(da.shape[da.get_axis_num('time')])

    # Transpose the temperature array
    temperature_array_transposed = temperature_array.T

    # Plot
    plt.figure(figsize=(15, 8))
  
    plt.imshow(temperature_array_transposed, aspect='auto', cmap='viridis', 
            extent=[time_steps.min(), time_steps.max(), longitudes.min(), longitudes.max()])
    plt.colorbar(label= var)
    plt.title(f'{var}, Latitude mean 48.75 - 46.25 - 43.75 - 41.25')
    plt.xlabel(f'Time Step 1W, {start_date} - {end_date}')
    plt.ylabel('Longitude (spatial dimension)')
    plt.show()

def make_gif(name, dataset, start_date = '2015-01-01', end_date = '2018-12-31'):
    """ Store a GIF """
    image =   gif(dataset.sel(time=slice(start_date, end_date)), to=f"{name}.gif")
    return image


def plot_latitude(fft_da, var = 'geopotential-500'):
    """The range 0-180 corresponds to the number of longitude grid points"""
    for i in range(300, 303):
        # Select the first time step
        fft_da_t0 = fft_da.isel(time=i)  #take 10 random time steps in the dataset

        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot for each latitude
        for lat in fft_da_t0.latitude.values:
            spectrum = fft_da_t0.sel(latitude=lat)
            ax.plot(spectrum.longitude, spectrum, label=f'Latitude {lat:.2f}°')

        # Customize the plot
        ax.set_xlabel('Longitude') 
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Zonal FFT Amplitude Spectrum {var}')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(0, 14000)

        ax.set_xlim(-180, 180)

        plt.tight_layout()
        plt.show()

def plot_wavenumbers(fft_da, ylim = 80000, var =  'geopotential-500'):
    # Select the time range
    fft_da_subset = fft_da 

    fig, ax = plt.subplots(figsize=(20, 10))

    # Prepare data for boxplot
    data = []
    for lat in fft_da_subset.latitude.values:
        for wn in range(1, 15):  # Adjust range as needed
            amplitudes = fft_da_subset.sel(latitude=lat).isel(longitude=wn).values
            data.extend([(lat, wn, amp) for amp in amplitudes])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Latitude', 'Wavenumber', 'Amplitude'])

    # Create boxplot
    sns.boxplot(x='Wavenumber', y='Amplitude', hue='Latitude', data=df, ax=ax)

    # Customize the plot
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Zonal FFT Amplitude Spectrum Distribution, {var}')
    ax.legend(title='Latitude')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, ylim)

    plt.tight_layout()
    plt.show()


