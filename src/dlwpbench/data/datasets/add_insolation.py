import os
import glob
import numpy as np
import pandas as pd
import torch as th
import xarray as xr


def insolation(dates, lat, lon, S=1., daily=False, enforce_2d=True, clip_zero=True):
    """
    Calculate the approximate solar insolation for given dates.

    For an example reference, see:
    https://brian-rose.github.io/ClimateLaboratoryBook/courseware/insolation.html

    :param dates: 1d array: datetime or Timestamp
    :param lat: 1d or 2d array of latitudes
    :param lon: 1d or 2d array of longitudes (0-360deg). If 2d, must match the shape of lat.
    :param S: float: scaling factor (solar constant)
    :param daily: bool: if True, return the daily max solar radiation (lat and day of year dependent only)
    :param enforce_2d: bool: if True and lat/lon are 1-d arrays, turns them into 2d meshes.
    :param clip_zero: bool: if True, set values below 0 to 0
    :return: 3d array: insolation (date, lat, lon)
    """
    # pylint: disable=invalid-name
    if len(lat.shape) != len(lon.shape):
        raise ValueError("'lat' and 'lon' must have the same number of dimensions")
    if len(lat.shape) >= 2 and lat.shape != lon.shape:
        raise ValueError(f"shape mismatch between lat ({lat.shape} and lon ({lon.shape})")
    if len(lat.shape) == 1 and enforce_2d:
        lon, lat = np.meshgrid(lon, lat)
    n_dim = len(lat.shape)

    # Constants for year 1995 (standard in climate modeling community)
    # Obliquity of Earth
    eps = 23.4441 * np.pi / 180.
    # Eccentricity of Earth's orbit
    ecc = 0.016715
    # Longitude of the orbit's perihelion (when Earth is closest to the sun)
    om = 282.7 * np.pi / 180.
    beta = np.sqrt(1 - ecc ** 2.)

    # Get the day of year as a float.
    start_years = np.array([pd.Timestamp(pd.Timestamp(d).year, 1, 1) for d in dates], dtype='datetime64')
    days_arr = (np.array(dates, dtype='datetime64') - start_years) / np.timedelta64(1, 'D')
    for d in range(n_dim):
        days_arr = np.expand_dims(days_arr, -1)
    # For daily max values, set the day to 0.5 and the longitude everywhere to 0 (this is approx noon)
    if daily:
        days_arr = 0.5 + np.round(days_arr)
        new_lon = lon.copy().astype(np.float32)
        new_lon[:] = 0.
    else:
        new_lon = lon.astype(np.float32)
    # Longitude of the earth relative to the orbit, 1st order approximation
    lambda_m0 = ecc * (1. + beta) * np.sin(om)
    lambda_m = lambda_m0 + 2. * np.pi * (days_arr - 80.5) / 365.
    lambda_ = lambda_m + 2. * ecc * np.sin(lambda_m - om)
    # Solar declination
    dec = np.arcsin(np.sin(eps) * np.sin(lambda_))
    # Hour angle
    h = 2 * np.pi * (days_arr + new_lon / 360.)
    # Distance
    rho = (1. - ecc ** 2.) / (1. + ecc * np.cos(lambda_ - om))

    # Insolation
    sol = S * (np.sin(np.pi / 180. * lat[None, ...]) * np.sin(dec) -
               np.cos(np.pi / 180. * lat[None, ...]) * np.cos(dec) *
               np.cos(h)) * rho ** -2.
    if clip_zero:
        sol[sol < 0.] = 0.

    return sol.astype(np.float32)
