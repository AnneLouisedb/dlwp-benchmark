#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import glob
import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import os


def write_to_file(ds_inits: xr.Dataset, ds_outputs: xr.Dataset, ds_targets: xr.Dataset, dst_path: str):
	print("Writing to file...")
	
	os.makedirs(dst_path, exist_ok=True)
	ds_inits.to_netcdf(os.path.join(dst_path, "inits.nc"))
	ds_outputs.to_netcdf(os.path.join(dst_path, "outputs.nc"))
	ds_targets.to_netcdf(os.path.join(dst_path, "targets.nc"))


def persistence_forecast(ds_inits: xr.Dataset, ds_outputs: xr.Dataset, ds_targets: xr.Dataset, dst_path):
	print("Creating persistence forecast...")
	
	ds_inits = ds_inits.sortby('lat', ascending=False)
	ds_targets = ds_targets.sortby('lat', ascending=False)
	ds_inits = ds_inits.drop_dims('time')
	times = ds_targets.time

	# Step 2: Expand the dataset along the new time dimension
	ds_targets_new  = ds_inits.expand_dims(time=times) 
	# sport the data by the latitude index
	print("done writing to file persistence ..")

	ds_targets_new.to_netcdf(dst_path)
	

# def climatology_forecast(ds_inits: xr.Dataset, ds_outputs: xr.Dataset, ds_targets: xr.Dataset):
#     target_lat = np.flip(np.array(np.arange(start=-90, stop=90, step=2.0), dtype=np.float32))
#     target_lon = np.array(np.arange(start=0, stop=360, step=deg), dtype=np.float32)

# 	start_date = "1981-01-01" 
# 	stop_date = "2010-12-31"
# 	data_src_path = '/projects/prjs1254/netcdf/ERA5_5.625'
# 	zarr_file_paths = np.sort(glob.glob(os.path.join("/projects/prjs1254/", 'msl', f"*.zarr")))

#     print("Creating climatology forecast...")

    
# 	#zarr_file_paths = glob.glob(os.path.join(data_src_path, "**", "*.nc"), recursive=True)

# 	def ensure_lat_lon_coords(ds):
# 		return ds.drop_vars(['lat', 'lon'], errors='ignore')

# 	# Lazy load all data to base climatology calculation on
# 	print("Lazy loading all data")
# 	ds_climatology = xr.open_mfdataset(
# 			zarr_file_paths,
# 			preprocess=ensure_lat_lon_coords,
# 		).sel(time=slice(start_date, stop_date))

# 	# Lazy load all data to base climatology calculation on
# 	print("Lazy loading all data")
	
# 	ds_climatology = ds_climatology.sortby('lat', ascending=False)
#     ds_climatology = ds_climatology.interp(latitude=target_lat, longitude=target_lon, method='linear')

# 	# Calculate climatological standard normal per variable over specified period
# 	for vname in list(ds_inits.keys()):
		
# 		# Select data array variable from climatology dataset
# 		if vname in list(ds_climatology.keys()):
# 			da_climatology = ds_climatology[vname]
			
# 		else:
# 			ValueError

		
# 		print(f"Computing climatology for {vname} and loading it to memory")

# 		da_climatology_ = da_climatology.groupby(da_climatology.time.dt.month).mean().load()
# 		for s_idx, s in enumerate(tqdm.tqdm(ds_outputs.sample, desc="Overwriting ds_outputs with climatology")):
# 			for t_idx, t in enumerate(ds_outputs.time):
				
# 				ds_out_sel = ds_outputs[vname][s_idx, t_idx]
# 				month = pd.Timestamp((ds_out_sel.sample + ds_out_sel.time).values).month
# 				ds_outputs[vname][s_idx, t_idx] = da_climatology_ .sel(month=month)

# 		#dst_path = os.path.join("outputs", "climatology", "evaluation")
# 		dst_path = '/projects/prjs1254/climatology1D_32'
# 		write_to_file(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets, dst_path=dst_path)

def climatology_forecast(ds_inits: xr.Dataset, ds_outputs: xr.Dataset, ds_targets: xr.Dataset):
		deg = 2.0
		start_date = "1981-01-01" 
		stop_date = "2010-12-31"
		target_lat = np.flip(np.array(np.arange(start=-90, stop=90, step=2.0), dtype=np.float32))
		target_lon = np.array(np.arange(start=0, stop=360, step=deg), dtype=np.float32)

		# data_src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_5.625'
		# data_src_path ='/projects/prjs1254/z-100'

		# Base path for all variables
		base_data_path = '/projects/prjs1254/'

		# List of variables to load
		variables = ["z-100", "z-150", "z-200", "z-250", "z-300", 
			"z-400", "z-500", "z-600", "z-700", "z-850", 
			"z-925", "z-1000", 'msl'
		]
		
		# Time range for selecting data
		start_date = '2000-01-01'
		stop_date = '2020-12-31'

		# Dictionary to store datasets for each variable
		datasets = {}

		# Loop through each variable and load its data
		for var in variables:
			data_src_path = os.path.join(base_data_path, var)
			#Find all .nc files for the current variable
			zarr_file_paths = glob.glob(os.path.join(data_src_path, "**", "*.nc"), recursive=True)
			print(f"Found {len(zarr_file_paths)} files for {var}.")

			if var == 'msl':
				ds = xr.open_dataset('/projects/prjs1254/data_ERA5_1.0/MSLP_era5_Global_1degr_19400101_20240229.nc').sel(time=slice(start_date, stop_date))
				ds= ds.rename({'longitude': "lon", 'latitude': "lat"})
				ds = ds.sortby('lat', ascending=False)
				ds= ds.interp(lat=target_lat, lon=target_lon, method='linear')
				datasets[var] = ds
			else:

				# Lazy load all files for the current variable
				try:
					ds = xr.open_mfdataset(
						zarr_file_paths,
						engine="netcdf4"      
					).sel(time=slice(start_date, stop_date))

					ds = ds.drop_vars('level', errors='ignore')
					#ds = ds.rename({'z': f"{var}"})
					
					# Store the dataset in the dictionary with the variable name as the key
					ds = ds.sortby('lat', ascending=False)
					ds= ds.interp(lat=target_lat, lon=target_lon, method='linear')
					datasets[var] = ds
				
					# delete the coordinate level

					print(f"Loaded dataset for {var}.")
				
				except:
					print(f"Error loading data for {var}")

				
		combined_dataset = xr.merge(datasets.values())
		# descending latitudes
		ds_climatology = combined_dataset.sortby('lat', ascending=False)
		ds_climatology = ds_climatology.interp(lat=target_lat, lon=target_lon, method='linear')

		# Calculate climatological standard normal per variable over specified period
		for vname in variables: #list(ds_inits.keys()):
			
			da_climatology = ds_climatology[vname]
				
			print(f"Computing climatology for {vname} and loading it to memory")
			# # Create weekly climatology
			da_climatology_  = da_climatology.groupby(da_climatology.time.dt.month).mean().load()
			
			for s_idx, s in enumerate(tqdm.tqdm(ds_outputs.sample, desc="Overwriting ds_outputs with weekly climatology")):
				for t_idx, t in enumerate(ds_outputs.time):
					print('replace', t)
					ds_out_sel = ds_outputs[vname][s_idx, t_idx]
					#month = ds_out_sel.time.dt.month.values.item()
					month = pd.Timestamp((ds_out_sel.sample + ds_out_sel.time).values).month
					ds_outputs[vname][s_idx, t_idx] = da_climatology_.sel(month=month)

					 
		#dst_path = os.path.join("outputs", "climatology", "evaluation")
		dst_path = '/projects/prjs1254/climatology1D_32'
		write_to_file(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets, dst_path=dst_path)

	
	

def climatology_weekly_forecast(ds_inits: xr.Dataset, ds_outputs: xr.Dataset, ds_targets: xr.Dataset):
		
		start_date = "1981-01-01" 
		stop_date = "2010-12-31"

		#data_src_path = os.path.join("data", "zarr", "weatherbench")
		#data_src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/zarr/ERA5_5.625'
		data_src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/data/netcdf/ERA5_5.625'
		
		zarr_file_paths = glob.glob(os.path.join(data_src_path, "**", "*.nc"), recursive=True)

		# Lazy load all data to base climatology calculation on
		print("Lazy loading all data")
		ds_climatology = xr.open_mfdataset(
			zarr_file_paths,
			#engine="netcdf"
		).sel(time=slice(start_date, stop_date))
		
		# descending latitudes
		ds_climatology = ds_climatology.sortby('lat', ascending=False)

		# Calculate climatological standard normal per variable over specified period
		for vname in list(ds_inits.keys()):
			
			# Select data array variable from climatology dataset
			if vname in list(ds_climatology.keys()):
				da_climatology = ds_climatology[vname]
				
			else:
				ValueError

			print(f"Computing climatology for {vname} and loading it to memory")
	
			# # Create weekly climatology
			da_climatology_ = da_climatology.groupby(da_climatology.time.dt.isocalendar().week).mean().load()

			for s_idx, s in enumerate(tqdm.tqdm(ds_outputs.sample, desc="Overwriting ds_outputs with weekly climatology")):
				for t_idx, t in enumerate(ds_outputs.time):
					ds_out_sel = ds_outputs[vname][s_idx, t_idx]
					week = pd.Timestamp((ds_out_sel.sample + ds_out_sel.time).values).isocalendar().week
					ds_outputs[vname][s_idx, t_idx] = da_climatology_.sel(week=week)

		#dst_path = os.path.join("outputs", "climatology", "evaluation")
		dst_path = '/projects/prjs1254/climatology_weekly'
		write_to_file(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets, dst_path=dst_path)

	
	


if __name__ == "__main__":
	# Specs
	src_model_name = "unet"

	
	#src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/2/evaluation'
	src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/2daysteps/evaluation'
	src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/hpx32_big_seed1236_2000_full/evaluation'
	src_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321240/evaluation'
	# Load data	
	ds_inits = xr.open_dataset(os.path.join(src_path, "inits.nc"))
	ds_outputs = xr.open_dataset(os.path.join(src_path, "outputs.nc"))
	ds_targets = xr.open_dataset(os.path.join(src_path, "targets.nc"))

	#persistence_forecast(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_target, dst_path='/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/2daysteps')
	#climatology_weekly_forecast(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets)
	climatology_forecast(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets)
