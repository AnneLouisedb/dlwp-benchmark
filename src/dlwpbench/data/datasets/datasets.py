#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import time
import numpy as np
import torch as th
import pandas as pd
import xarray as xr
from datetime import datetime


class WeatherBenchDataset(th.utils.data.Dataset):

    # Statistics are computed over the training period from 1979-01-01 to 2014-12-31 using the
    # self._compute_statistics() of this class
    STATISTICS = {
       
        
        'msl': {"file_name": "msl",
            "mean": 100958.9342887361, "std": 1302.149826386961},
        'sst': {"file_name": "sst",
            "mean": 290.31263675008245,"std": 11.372437606024592}
        ,
        'stream250': {"file_name": "stream250", "mean": 3279301.066678744, "std": 88087550.26655602},
        'stream500':  {"file_name": "stream500", "mean": 2355273.207031221,  "std": 42371255.252134465},
        "tisr": {
            "file_name": "toa_incident_solar_radiation",
            "mean": 1074504.8,
            "std": 1439846.4
        },
        "orography": {
            "file_name": "constants",
            "mean": 379.4976,
            "std": 859.87225
        },
        "lsm": {
            "file_name": "constants",
            "mean": 0,  # Do not normalize since it is in [0, 1] already
            "std": 1
        },
        "lat2d": {
            "file_name": "constants",
            "mean": 0,
            "std": 51.936146191742026
        },
        "lon2d": {
            "file_name": "constants",
            "mean": 177.1875,
            "std": 103.9103617607503
        }
    }

    def __init__(
            self,
            data_path: str,
            prognostic_variable_names_and_levels: dict,
            prescribed_variable_names: list = [],
            constant_names: list = None,
            start_date: np.datetime64 = np.datetime64("1979-01-01"),
            stop_date: np.datetime64 = np.datetime64("2014-12-31"),
            timedelta: int = 6,
            init_dates: np.array = None,
            sequence_length: int = 15,
            noise: float = 0.0,
            normalize: bool = False,
            downscale_factor: int = 1,
            context_size: int = 2,
            engine: str = "netcdf4",
            height: int = 32,
            width: int = 64,
            statistics = None,
            **kwargs
        ):
        """
        Constructor of a pytorch dataset module.

        :param data_name: The name of the data
        :param data_start_date: The first time step of the data on the disk
        :param data_stop_date: The last time step of the data on the disk
        :param used_start_date: The first time step to be considered by the dataloader
        :param used_stop_date: The last time step to be considered by the dataloader
        :param data_src_path: The source path of the data
        :param sequence_length: The number of time steps used for training

        HPX 8 - Full Train Set
        tisr
        "mean": 0.25008788148804867,
        "std": 0.32267114861296226
        msl
        "mean": 101142.46702547799,
        "std": 1044.7123165632315

        stream250
        "mean": -18528.25642087052,
        "std": 66413973.460995585

        stream500
        "mean": -11990.837318404307,
        "std": 30880461.37769371
        """

        full_manual = {
        "tisr":
        {"file_name": "toa_incident_solar_radiation",
        "mean":  0.25008788148804867,
        "std": 0.32267114861296226},
        "lsm": 
            {"file_name": "lsm", # do not normalize the constant
            "mean": 0,
            "std": 1},
        "lat2d": 
            {"file_name": "lat2d",
            "mean": 0,
            "std": 51.936146191742026},
        "lon2d": 
            {"file_name": "lon2d",
            "mean": 177.1875,
            "std": 103.9103617607503},
        "orography": 
            {"file_name": "orography",
            "mean": 379.4976 ,
            "std": 859.87225},
        "msl":
        { "file_name": "msl",
         "mean": 101142.46702547799,
        "std": 1044.7123165632315},

        "stream250":
        {"file_name": "stream250", 
         "mean": -18528.25642087052,
        "std": 66413973.460995585},

        "stream500":
        {"file_name": "stream500",
         "mean": -11990.837318404307,
        "std": 30880461.37769371}
        }

        half_manual = {"tisr": 
            {"file_name": "toa_incident_solar_radiation",
            "mean": 0.2500256896018982,
            "std": 0.3228904902935028},

            "lsm": 
            {"file_name": "lsm", # do not normalize the constant
            "mean": 0,
            "std": 1},

            "lat2d": 
            {"file_name": "lat2d",
            "mean": 0.5,
            "std": 51.96072},

            "lon2d": 
            {"file_name": "lon2d",
            "mean": 179.5,
            "std": 103.922646},

            "orography": 
            {"file_name": "orography",
            "mean": 370.4871669133824 ,
            "std": 840.2511704642723},

            "msl": 
            {"file_name": "msl",
            "mean": 101155.25,
            "std": 1093.2652587890625},

            "stream250": 
            {"file_name": "stream250",
            "mean": -803927.5,
            "std": 66434284.0},

            "stream500": {"file_name": "stream500",
            "mean": -368886.46875,
            "std": 30952428.0}}
        
        half_manual_32 = {"tisr": 
            {"file_name": "toa_incident_solar_radiation",
            "mean": 0.2508913626665014,
            "std": 0.3232416586372475},

            "lsm": 
            {"file_name": "lsm", # do not normalize the constant
            "mean": 0,
            "std": 1},

            # TO DO !
            "geopotential-50": 
            {"file_name": "geopotential-50",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-100": 
            {"file_name": "geopotential-100",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-150": 
            {"file_name": "geopotential-150",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-200": 
            {"file_name": "geopotential-200",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-250": 
            {"file_name": "geopotential-250",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-300": 
            {"file_name": "geopotential-300",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-400": 
            {"file_name": "geopotential-400",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-500": 
            {"file_name": "geopotential-500",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-600": 
            {"file_name": "geopotential-600",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-700": 
            {"file_name": "geopotential-700",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-850": 
            {"file_name": "geopotential-850",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "geopotential-925": 
            {"file_name": "geopotential-925",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},
            
            "geopotential-1000": 
            {"file_name": "geopotential-1000",
            "mean": 2.684756054804893e-06,
            "std": 5.950411718913529e-07},

            "lat2d": 
            {"file_name": "lat2d",
            "mean": 0.5,
            "std": 51.96072},

            "lon2d": 
            {"file_name": "lon2d",
            "mean": 179.5,
            "std": 103.922646},

            "orography": 
            {"file_name": "orography",
            "mean": 370.4871669133824 ,
            "std": 840.2511704642723},

            "msl": 
            {"file_name": "msl",
            "mean": 101141.42129534102,
            "std": 1091.1232502005057},

            "stream250": 
            {"file_name": "stream250",
            "mean": -749880.9128180227,
            "std": 66771987.291932374},

            "stream500": {"file_name": "stream500",
            "mean": -48478.47349296967,
            "std": 31118257.308523186}}
   
        try:
            if datetime.strptime(statistics, "%Y-%m-%d").year == 1940: 
                stats = full_manual
            else:
                stats = half_manual
        except:
            if datetime.strptime(start_date, "%Y-%m-%d").year == 1940: 
                stats = full_manual
            else:
                print('using half?')
                stats = half_manual

     
        self.stats = half_manual_32 #half_manual #
        print("DATA IS INITIALIZED AT FULL TRAIN SET")
        self.prognostic_variable_names_and_levels = prognostic_variable_names_and_levels
        self.prescribed_variable_names = prescribed_variable_names
        self.constant_names = constant_names

        self.timedelta = timedelta
        self.sequence_length = sequence_length
        self.noise = noise
        self.normalize = normalize
        self.downscale_factor = downscale_factor
        self.context_size = context_size
        self.init_dates = init_dates

        # Get paths to all (yearly) netcdf/zarr files
        fpaths = []
        for p in prognostic_variable_names_and_levels:
            fpaths += glob.glob(os.path.join(data_path, self.stats[p]["file_name"], "*"))
        for p in prescribed_variable_names:
            fpaths += glob.glob(os.path.join(data_path, self.stats[p]["file_name"], "*"))
        if constant_names: 
            fpaths += glob.glob(os.path.join(data_path, "constants", "*"))

        print(f"\tLoading dataset from {start_date} to {stop_date} into RAM...", sep=" ", end=" ", flush=True)
        a = time.time()
        # Load the data as xarray dataset
        self.ds = xr.open_mfdataset(fpaths, engine=engine).sel(time=slice(start_date, stop_date, timedelta))
        
        # Chunk and load dataset to memory (distinguish between HEALPix, i.e., when "face" in coords, and LatLon mesh)
        if "face" in self.ds.coords:
            chunkdict = dict(time=self.sequence_length+1, face=12, height=height, width=width)
        else:
            chunkdict = dict(time=self.sequence_length+1, lat=height, lon=width)
        self.ds = self.ds.chunk(chunkdict) #.load()
        print(f"took {time.time() - a} seconds")
        
        # Downscale dataset if desired
        if downscale_factor > 1:
            assert "face" not in self.ds.coords, "Downscaling only supported with LatLon and not with HEALPix data."
            self.ds = self.ds.coarsen(lat=downscale_factor, lon=downscale_factor).mean()

        # Prepare the constants to shape [#constants, lat, lon]
        if constant_names:
            constants = []
            for c in constant_names:
                
                lazy_data = self.ds[c]
                if self.normalize: lazy_data = (lazy_data-self.stats[c]["mean"])/self.stats[c]["std"]
                constants.append(lazy_data.compute())
            self.constants = np.expand_dims(np.float32(np.stack(constants)), axis=0)
        else:
            self.constants = th.nan  # Dummy tensor is returned if no constants are used

        print('context size', self.context_size)
        print('statistics on dataset')

        self.statistics = self.compute_statistics()
        
    def __len__(self):
        if self.init_dates is None:
            # Randomly sample initialization dates from the dataset
            return (self.ds.sizes["time"]-self.sequence_length)//self.sequence_length
        else:
            return len(self.init_dates)

    def __getitem__(self, item):
        """
        return: Four arrays of shape [batch, time, dim, (face), height, width], where face is optionally added when
            operating on the HEALPix mesh.
        """

        item = item*self.sequence_length if self.init_dates is None else item

        # Load the (normalized) prescribed variables of shape [time, #prescribed_vars, lat, lon] into memory
        if self.prescribed_variable_names:
            prescribed = []
            for p in self.prescribed_variable_names:
                manual_tisr = False
                if self.init_dates is None:
                    lazy_data = self.ds[p].isel(time=slice(item, item+self.sequence_length))
                    
                else:
                    lazy_data = self.ds[p].sel(
                        time=slice(self.init_dates[item],
                                   self.init_dates[item]+pd.Timedelta(f"{self.sequence_length*self.timedelta}h"))
                    
                    )
               
                if self.init_dates is not None and self.sequence_length > len(lazy_data.time):
                    # Augment TISR with values from 2017 when exceeding the date of the stored data
                    manual_tisr = True 
                    diff = self.sequence_length - len(lazy_data.time)
                    start_date = self.init_dates[item] + pd.Timedelta("11h")
                    stop_date = self.init_dates[item] + pd.Timedelta(f"{self.sequence_length*self.timedelta*24}h")
                    dates = pd.date_range(start=start_date, end=stop_date, freq=f"{self.timedelta*24}h")
                    lazy_data = lazy_data.values
                   
                    tmp = list()
                    # Overide year with 2017 under consideration of leap years
                    year_rep = 2022 # 2017
                    for date in dates[-diff:]:
                        date = date.replace(year=year_rep, day=28) if date.month == 2 and date.day > 28 else date.replace(year=year_rep)
                        tmp.append(self.ds.tisr.sel(time=date).values)
                    lazy_data = np.concatenate((lazy_data, np.array(tmp)))
                    #print(lazy_data)
                if self.normalize: lazy_data = (lazy_data-self.stats[p]["mean"])/self.stats[p]["std"]
                prescribed.append(lazy_data.compute() if not manual_tisr else lazy_data)  # Loads data into memory
            prescribed = np.float32(np.stack(prescribed, axis=1))
           
        else:
            prescribed = th.nan  # Dummy tensor returned if no prescribed variables are used

        # Load the (normalized) prognostic variables of shape [time, #prognostic_vars, lat, lon] into memory
        prognostic = []
        for p in self.prognostic_variable_names_and_levels:
            if self.init_dates is None:
                lazy_data = self.ds[p].isel(time=slice(item, item+self.sequence_length+1)) # loads one more time step (output)
                # print("lazy data time (prognostic) - input vector from dataloader")   
                # print(lazy_data.time)
            else:
                lazy_data = self.ds[p].sel(
                    time=slice(self.init_dates[item],
                               self.init_dates[item]+pd.Timedelta(f"{(self.sequence_length+1)*self.timedelta*24}h"))
                )
            # Load the data to memory
            if "level" in lazy_data.coords:
                for l in self.prognostic_variable_names_and_levels[p]:
                    lazy_data_l = lazy_data.sel(level=l)
                    if self.normalize:
                        lazy_data_l = (lazy_data_l-self.stats[p]["level"][l]["mean"])/self.stats[p]["level"][l]["std"]
                        lazy_data_l = lazy_data_l.fillna(0)
                    prognostic.append(lazy_data_l.compute())
            else:
                if self.normalize: lazy_data = (lazy_data-self.stats[p]["mean"])/self.stats[p]["std"]
                lazy_data = lazy_data.fillna(0)
                prognostic.append(lazy_data.compute())
        prognostic = np.float32(np.stack(prognostic, axis=1)) # stack along the time dimension
        # Append zeros to the prog. vars when exceeding the date of the stored data (required for long rollouts)
        if len(prognostic) < self.sequence_length:
            print("adding zero's to the target?")
            diff = self.sequence_length - len(prognostic)
            fill = np.zeros((diff, *prognostic.shape[1:]), dtype=np.float32)
            prognostic = np.concatenate((prognostic, fill), axis=0)

        
        # Separate prognostic variables into inputs and targets
        target = prognostic[1:]
        prognostic = prognostic[:-1] + np.float32(np.random.randn(*prognostic[:-1].shape)*self.noise)

        return self.constants, prescribed, prognostic, target[self.context_size:]

    
    def compute_statistics(self):
        """
        Computes the statistics of the given prognostic variables and prints mean and standard deviation to console
        """
        statistics = {}
        
        # Constants
        for c in self.constant_names:
            print(c)
            lazy_data = self.ds[c]
            mean, std = lazy_data.mean().values, lazy_data.std().values
            statistics[c] = {"file_name": f"{c}", "mean": float(mean), "std": float(std)}
        # Prescribed variables
        for p in self.prescribed_variable_names:
            print(p)
            lazy_data = self.ds[p]
            mean, std = lazy_data.mean().values, lazy_data.std().values
            print(f'"mean": {mean},\n"std": {std}')
            statistics[p] = {"file_name": f"toa_incident_solar_radiation", "mean": float(mean), "std": float(std)}
        # Prognostic variables (optionally with levels)
        for p in self.prognostic_variable_names_and_levels:
            print(p)
            lazy_data = self.ds[p]
            if "level" in lazy_data.coords:
                for l in lazy_data.level.values:
                    mean, std = lazy_data.sel(level=l).mean().values, lazy_data.sel(level=l).std().values
                    print(f'{l}: {{"mean": {mean}, "std": {std}}},')
                    statistics[p][float(l)] = {"file_name": f"{p}", "mean": float(mean), "std": float(std)}
            else:
                mean, std = lazy_data.mean().values, lazy_data.std().values
                print(f'"mean": {mean},\n"std": {std}')
                statistics[p] = {"file_name": f"{p}", "mean": float(mean), "std": float(std)}
            
            print()
        return statistics


if __name__ == "__main__":
    def make_biweekly_inits(
        start: str = "2017-01-01T00:00:00.000000000",
        end: str = "2017-10-20T00:00:00.000000000",
        sequence_length: int = 15,
        timedelta: int = 1 
    ):
        # Convert start and end to pandas Timestamp objects with UTC timezone
        start_date = pd.Timestamp(start, tz='UTC')  #+ pd.Timedelta(hours=sequence_length*timedelta*24)
        end_date = pd.Timestamp(end, tz='UTC') - pd.Timedelta(hours=sequence_length*timedelta*24)
        
        # Generate date range for Mondays at 11:00 UTC
        mondays = pd.date_range(start=start_date, end=end_date, freq='W-MON', tz='UTC').map(lambda x: x.replace(hour=11, minute=0, second=0, microsecond=0))
        
        # Generate date range for Thursdays at 11:00 UTC
        thursdays = pd.date_range(start=start_date, end=end_date, freq='W-THU', tz='UTC').map(lambda x: x.replace(hour=11, minute=0, second=0, microsecond=0))
        
        # Combine Mondays and Thursdays
        all_dates = mondays.union(thursdays).sort_values()

        naive_timestamp = all_dates.tz_localize(None)
        print(naive_timestamp)

        return naive_timestamp.to_numpy()


    init_dates = make_biweekly_inits(
            #start=cfg.data.test_start_date,
            #end=cfg.data.test_stop_date,
            sequence_length=14,
            timedelta=1
    )

    # Example of creating a WeatherBench dataset for the PyTorch DataLoader
    dataset = WeatherBenchDataset(
        data_path="data/netcdf/weatherbench/",
        prognostic_variable_names_and_levels={
            #"u10": [],   # 10m_u_component_of_wind
            #"v10": [],   # 10m_v_component_of_wind
            #"t2m": [],   # 2m_temperature
            #"z": [500],     # geopotential
            #"z500",  # geopotential_500
            #"pv": [500],    # potential_vorticity
            #"r": [500, 700],     # relative_humidity
            #"q": [500],     # specific_humidity
            #"t": [50, 500, 850],     # temperature
            #"tcc": [],   # total_cloud_cover
            #"u": [500],     # u_component_of_wind
            #"v": [500],     # v_component_of_wind
            #"vo": [500]     # vorticity
            'sst': []
        },
        prescribed_variable_names=[
            "tisr"   # top of atmosphere incoming solar radiation
        ],
        constant_names=[
            "orography",
            "lsm",   # land-sea mask
            #"slt",   # soil type
            "lat2d",
            "lon2d"
        ],
        sequence_length=15,
        noise=0.0,
        normalize=True,
        downscale_factor=1,
        init_dates=init_dates,
        start_date='2017-01-01',
        stop_date='2018-12-31',
     
    )
    print(f"Dataset length: {len(dataset)}")

    train_dataloader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    # HERE I GET AN ERROR
    for constants, prescribed, prognostic, target in train_dataloader:
        print(constants.shape, prescribed.shape, prognostic.shape, target.shape)
        print(target)
        
        break

    print(dataset)
