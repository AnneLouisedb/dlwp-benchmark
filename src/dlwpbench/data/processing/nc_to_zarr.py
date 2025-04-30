#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from dlwp-hpx repo Copyright (c) Matthias Karlbauer

import os
import glob
import numpy as np
import xarray as xr
from netCDF4 import Dataset

if __name__ == "__main__":

    # levels = [50,100,150,200,250,300,400,500,600,700,850,925,1000] #

    # for level in levels:
    #     folder_path = f'/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/weatherbench_hpx8/geopotential-{level}'

    #     # List all .nc files in the folder
    #     nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]

    #     # Rename 'q' to 'geopotential' in each file
    #     for file in nc_files:
    #         file_path = os.path.join(folder_path, file)
        
    #         # Open the NetCDF file in read-write mode
    #         with Dataset(file_path, 'r+') as nc:
    #             if 'q' in nc.variables:
    #                 nc.renameVariable('q', f'geopotential-{level}')
    #                 print(f"{file_path} - Variable renamed!")

    ############################
    # import xarray as xr
    # import pandas as pd

    # path = f'/projects/prjs0981/ewalt/Xaurora/data/era5_wb2/1979-2022-1d-1440x721/era5_wb2_q-1979-2022-1D-1440x721.zarr'
    # p = xr.open_dataset(path)
    # # Shift all timestamps by 11 hours
    # p = p.assign_coords(time=p.time + pd.Timedelta(hours=11))

    # new_ds = xr.Dataset()
    # ds = p
    # for level in ds.level.values:
    #     var_name = f'geopotential-{level}'
    #     new_ds[var_name] = ds.q.sel(level=level).drop_vars('level')

    # # Copy coordinates and attributes
    # new_ds.coords['latitude'] = ds.latitude
    # new_ds.coords['longitude'] = ds.longitude
    # new_ds.coords['time'] = ds.time


    # path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/climatology_1.0_1981-2010/msl/climatology_msl.nc'
    # target_grid = xr.open_dataset(path)

    # ds = new_ds
    # #reversed_vars = list(ds.data_vars)[::-1] # 'geopotential-250','geopotential-300', 'geopotential-500', 
    # reversed_vars = [  'geopotential-300'] #[::-1] # 'geopotential-300','geopotential-400',
    # for var in reversed_vars:
    #     var_ds = ds[[var]]
    #     # Filter for the year 2022
    #     #year_ds = var_ds
    
    #     for year, year_ds in var_ds.groupby('time.year'):
    #         # Create the filename
    #         filename = f'/home/adboer/dlwp-benchmark/src/dlwpbench/{var}/{var}_{year}_1deg.nc'
    #         if os.path.exists(filename):
    #             print(f"File {filename} already exists. Skipping.")
    #             continue

    #         else:
    #             year_ds = year_ds.interp(latitude=target_grid.lat, longitude=target_grid.lon, method='linear')
                
    #             # Save the year's data as a NetCDF file
    #             year_ds.to_netcdf(filename)
    #             print(f"Saved {filename}")
    # ################

  
    # #src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/weatherbench_hpx32'
    # src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/weatherbench_hpx8'
    src_path = os.path.join("data", "netcdf","weatherbench_hpx8") # "ERA5_5.625")
    src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_5.625_hpx8'
    src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/weatherbench_hpx32' #/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/weatherbench_hpx8'


    # src_path = "/home/adboer/dlwp-benchmark/src/dlwpbench/geopotential-50"
    dir_paths = glob.glob(os.path.join(src_path, "*"))

    for dir_path in dir_paths:
        print('dir path', dir_path)
        dir_name = os.path.basename(dir_path)

        os.makedirs(dir_name.replace("netcdf", "zarr"), exist_ok=True)

        nc_file_paths = np.sort(glob.glob(os.path.join(dir_path, "*")))
        print("nc files?", nc_file_paths)
        for nc_file_path in nc_file_paths:
            zarr_file_path = nc_file_path.replace("netcdf", "zarr").replace(".nc", ".zarr")
            print(zarr_file_path)
            if os.path.exists(zarr_file_path): continue
            xr.open_dataset(nc_file_path).to_zarr(zarr_file_path).close()
