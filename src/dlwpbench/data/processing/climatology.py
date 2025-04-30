import os
import os
import re
import glob
import tqdm
import numpy as np
import pandas as pd
import xarray as xr

import os
import os
import re
import glob
import tqdm
import numpy as np
import pandas as pd
import xarray as xr

def climatology_forecast(vname, deg): #ds_inits: xr.Dataset): # ds_outputs: xr.Dataset, ds_targets: xr.Dataset
	"""Creating a monthly climatology."""
	print("Creating climatology forecast...")

	# Specs according to climatological standard normal from 1981 through 2010
	# https://en.wikipedia.org/wiki/Climatological_normal
	start_date = "1981-01-01"
	stop_date = "2010-12-31"
	data_src_path = f'/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_{deg}/{vname}'
	zarr_file_paths = glob.glob(os.path.join(data_src_path, "**", "*.nc"), recursive=True)

	# Lazy load all data to base climatology calculation on
	print("Lazy loading all data")
	ds_climatology = xr.open_mfdataset(
		zarr_file_paths).chunk(dict(time=-1, lat=180, lon=360)).sel(time=slice(start_date, stop_date))  
	
	# # Calculate climatological standard normal per variable over specified period	               
	print(f"Computing climatology for {vname} and loading it to memory")
	
	print('size', ds_climatology['time'].size)
	ds_climatology = ds_climatology[f'{vname}'].rolling(
			dim={"time": 25},  
			min_periods = 1,
			center=True,
		).mean()

	ds_climatology = ds_climatology.groupby(ds_climatology.time.dt.month).mean().load()
	ds_climatology.to_netcdf(f"MonthlyClimatology_{vname}.nc")

	
climatology_forecast('stream500', 5.625)