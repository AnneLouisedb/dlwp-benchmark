import xarray as xr
import numpy as np
import os

def regrid(ds, ec46=True):
    """Load an EC46 file and store it"""

    def convert_longitude_180_to_360(lon):
        return (lon + 360) % 360
    
    # if ds['longitude']:

    #     ds['longitude'] = convert_longitude_180_to_360(ds['longitude'])

    #     ds = ds.sortby('longitude')
    #     ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

    # Your target latitude coordinates
    target_lat = np.array([-87.1875, -81.5625, -75.9375, -70.3125, -64.6875, -59.0625, -53.4375,
        -47.8125, -42.1875, -36.5625, -30.9375, -25.3125, -19.6875, -14.0625,
            -8.4375,  -2.8125,   2.8125,   8.4375,  14.0625,  19.6875,  25.3125,
            30.9375,  36.5625,  42.1875,  47.8125,  53.4375,  59.0625,  64.6875,
            70.3125,  75.9375,  81.5625,  87.1875])

    target_lon = [  0.   ,   5.625,  11.25 ,  16.875,  22.5  ,  28.125,  33.75 ,  39.375,
            45.   ,  50.625,  56.25 ,  61.875,  67.5  ,  73.125,  78.75 ,  84.375,
            90.   ,  95.625, 101.25 , 106.875, 112.5  , 118.125, 123.75 , 129.375,
        135.   , 140.625, 146.25 , 151.875, 157.5  , 163.125, 168.75 , 174.375,
        180.   , 185.625, 191.25 , 196.875, 202.5  , 208.125, 213.75 , 219.375,
        225.   , 230.625, 236.25 , 241.875, 247.5  , 253.125, 258.75 , 264.375,
        270.   , 275.625, 281.25 , 286.875, 292.5  , 298.125, 303.75 , 309.375,
        315.   , 320.625, 326.25 , 331.875, 337.5  , 343.125, 348.75 , 354.375]

    # Assuming your dataset is named 'ds'
    ds = ds.interp(lat=target_lat, lon=target_lon, method='linear')

    #month = np.unique(ds.time.dt.month.values).item()
    year= np.unique(ds.time.dt.year.values).item()

    if ec46:
        # find the month of the timestamps inside this datafram
        return ds.to_netcdf(f"{month}-{year}.nc")
    else:
        return  ds.to_netcdf(f"toa_incident_solar_radiation/toa_incident_solar_radiation_{year}_5.625deg.nc")


# Directory containing the NetCDF files
directory = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_1.0/toa_incident_solar_radiation'

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

        regridded_ds = regrid(ds, ec46=False)
       
        print(f"Regridded file saved")

    # if os.path.isfile(file_path) and filename.endswith('.grib'):
    #     print(f"Processing file: {file_path}")
    #     # Load the NetCDF file
    #     ds = xr.open_dataset(file_path, engine='cfgrib')
    #     # Apply the regrid function
    #     regridded_ds = regrid(ds, ec46=True)
       
    #     print(f"Regridded file saved")