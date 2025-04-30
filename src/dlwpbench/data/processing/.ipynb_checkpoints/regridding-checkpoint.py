import xarray as xr
import os
import numpy as np


# Create the target grid
target_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(-89, 90, 90)),
    'lon': (['lon'], np.linspace(0, 359, 180))
})


# Directory containing the NetCDF files
var = 'toa_incident_solar_radiation'
directory = f'/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_1.0/{var}'

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
        
        output_name = output_name.replace('1deg', '2deg')


        # Save the regridded dataset if needed
        ds_regridded.to_netcdf(f'/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_2.0/{var}/{output_name}')

        print(f"Regridded file saved as")
        print(f'/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_2.0/{var}/{output_name}')
