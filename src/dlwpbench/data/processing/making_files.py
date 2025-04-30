import os
import xarray as xr
import numpy as np
# Directory containing the zarr datasets and potentially other files
# path = '/home/adboer/dlwp-benchmark/src/dlwpbench/geopotential_1940_1deg.nc'
# path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_1.0/msl/msl_1940_1deg.nc'
# target_grid = xr.open_dataset(path)

var = 'z'
deg = 2.0

path = f'/projects/prjs0981/ewalt/Xaurora/data/era5_wb2/1979-2022-1d-1440x721/era5_wb2_{var}-1979-2022-1D-1440x721.zarr'
#path = '/projects/prjs0981/ewalt/Xaurora/data/era5_wb2/1979-2022-1d-1440x721/era5_wb2_2t-1979-2022-1D-1440x721.zarr'
px = xr.open_dataset(path)
# regrid the longitude and latidue to 180 by 360
print("got here?")

for year, year_ds in px.groupby('time.year'):
    output_path = f'/projects/prjs1254/data2.0/{var}/{var}_{year}_1deg.nc'
    if not os.path.exists(output_path):

        if deg == 2.0:
            target_lat = np.flip(np.array(np.arange(start=-90, stop=90, step=deg), dtype=np.float32))
            target_lon = np.array(np.arange(start=0, stop=360, step=deg), dtype=np.float32)

        frame = year_ds.interp(latitude=target_lat, longitude=target_lon, method='linear')
        frame.to_netcdf(
            output_path)
        print('written to', output_path)