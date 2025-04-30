import xarray as xr
import os
import numpy as np


target_path = '/projects/prjs1254/netcdf/ERA5_5.625/msl/msl_1940_5.625deg.nc'
target_grid= xr.open_dataset(target_path)
# Directory containing the NetCDF files

for var in ['z-250']: #'msl', 'geopotential-1000', 'geopotential-925', 'geopotential-850', 'geopotential-700', 'geopotential-600', 'geopotential-500', 'geopotential-400', 'geopotential-300', 'geopotential-200', 'geopotential-250', 'geopotential-150', 'geopotential-100', 'geopotential-50']:
    directory = f'/projects/prjs1254/{var}'

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        
        file_path = os.path.join(directory, filename)
        # Check if the file is a NetCDF file
        if os.path.isfile(file_path) and filename.endswith('.nc'):
            print(f"Processing file: {file_path}")
            # Load the NetCDF file
            # check if the file contains nan values!
            ds = xr.open_dataset(file_path)
            # Check for NaN values
            has_nan = np.isnan(ds.to_array()).any()
            if has_nan:
                print(f"Warning: {file_path} contains NaN values")
                # Apply the regrid function

            # Perform the regriddingz
            # Perform the regridding using xarray's interp method
            ds_regridded = ds.interp(lat=target_grid.lat, lon=target_grid.lon, method='linear')

            output_name = file_path.split('/')[-1]
            
            output_name = output_name.replace('1deg', '5.625deg')


            # Save the regridded dataset if needed
            ds_regridded.to_netcdf(f'/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_5.625/{var}/{output_name}')

            print(f"Regridded file saved as")
            print(f'/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_5.625/{var}/{output_name}')
