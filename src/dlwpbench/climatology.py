import os
import os
import re
import glob
import tqdm
import numpy as np
import pandas as pd
import xarray as xr

def climatology_forecast(): #ds_inits: xr.Dataset): # ds_outputs: xr.Dataset, ds_targets: xr.Dataset
	"""Creating a monthly climatology."""
	print("Creating climatology forecast...")

	# Specs according to climatological standard normal from 1981 through 2010
	# https://en.wikipedia.org/wiki/Climatological_normal 
	start_date = "1981-01-01"
	stop_date = "2010-12-31"
	data_src_path = os.path.join("data", "zarr", "weatherbench")
	zarr_file_paths = glob.glob(os.path.join(data_src_path, "**", "*.zarr"), recursive=True)

	# Lazy load all data to base climatology calculation on
	print("Lazy loading all data")
	ds_climatology = xr.open_mfdataset(
		zarr_file_paths,
		engine="zarr"
	).chunk(dict(time=1, lat=32, lon=64)).sel(time=slice(start_date, stop_date))
	

	# # Calculate climatological standard normal per variable over specified period
	for vname in list(['stream']):
					               
		# Select data array variable from climatology dataset
		if vname in list(ds_climatology.keys()):
			print("is in key")
			da_climatology = ds_climatology[vname]
		else:
			v, l = re.match(r"([a-z]+)([0-9]+)", vname, re.I).groups()
			da_climatology = ds_climatology[v].sel(level=int(l))
			
		print(f"Computing climatology for {vname} and loading it to memory")
		da_climatology = da_climatology.groupby(da_climatology.time.dt.month).mean().load()
		da_climatology.to_netcdf(f"MonthlyClimatology.nc")
	
	
climatology_forecast()