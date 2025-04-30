import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import sys

base_path= '/home/adboer/dlwp-benchmark/src/dlwpbench/'
sys.path.append(base_path)

# internal imports
from scripts.evaluate import make_rmses, make_accs
from scripts.helper_scripts.evaluation_helper import get_adjusting_weights


def loop_over_combinations(paths, ds_targets, lat_weights, ds_climatology, N = 5):
    """
    Loop over all combinations of 5 models and calculate RMSE for each combination.

    Parameters:
        paths (list): List of paths to model outputs.
        calculate_rmse_function (function): Function to calculate RMSE for a given combination.
    
    Returns:
        dict: A dictionary where keys are combinations of paths and values are RMSEs.
    """
    # Generate all combinations of 5 models
    combinations = list(itertools.combinations(paths, N))

    rmse_results = {}
    accs_results = {}
    
    for idx, combo in enumerate(combinations[:1]):
    
        print(combo)
        # Call the RMSE calculation function for the current combination
        # Open all datasets

        #datasets = [xr.open_dataset(os.path.join(path, 'outputs.nc')) for path in combo]
        # Open all files as Dask-backed datasets with chunking
        file_paths = [os.path.join(path, 'outputs.nc') for path in combo]
        ds = xr.open_mfdataset(file_paths, 
                            combine='nested', 
                            concat_dim='ensemble',
                            chunks={'time': 10, 'sample': 100})  # Adjust chunk sizes
                            # Perform operations lazily
        combined_outputs = ds.mean(dim='ensemble').isel(time=slice(0,45), sample=slice(0,194))
        combined_outputs.to_netcdf(f"outputs_ensemble{N}.nc")
        # Combine datasets along a new 'ensemble' dimension
        #combined = xr.concat(datasets, dim='ensemble')
        # Take the mean along the 'ensemble' dimension
        #combined_outputs = combined.mean(dim='ensemble').isel(time=slice(0, 45)).isel(sample=slice(0, 194))
        print('making rmses.. ')
        #combined_outputs.to_netcdf(f"{idx}_combination{N}.nc")
        targets_ds = ds_targets.isel(time=slice(0, 45)).isel(sample=slice(0, 194))
        rmses = make_rmses(targets_ds, combined_outputs, lat_weights)
        rmse_results[idx] = rmses
        rmses.to_netcdf(f'rmses_ensemble{N}.nc')
        del rmses

        lat_weights_np = lat_weights.cpu().detach().numpy()
        accs = make_accs(ds_targets, combined_outputs, ds_climatology, lat_weights_np)
        accs_results[idx] = accs
        accs.to_netcdf(f'accs_ensemble{N}.nc')

        del combined_outputs,  accs
    
    return rmse_results, accs_results


def make_plot(paths, ensemble_3, ds_targets, lat_weights, climatology_path, persistence_path, ensemble_5 = None, output_path="output_plot.png"):

    variables = ['msl', 'z-850', 'z-500', 'z-250']

    big3ensemble = xr.open_dataset(ensemble_3)

    if ensemble_5:
        big_5_ensemble = xr.open_dataset(ensemble_5)
    else: 
        big_5_ensemble = None

    climatology_ds_outputs = xr.open_dataset(f"{climatology_path}outputs.nc")

    # paths to multiple models
    ensembles_of_3, accs_of_3 = loop_over_combinations(paths, ds_targets, lat_weights, climatology_ds_outputs, N = 3)
    ensembles_of_5, accs_of_5 = loop_over_combinations(paths, ds_targets, lat_weights, climatology_ds_outputs, N = 5)
    ensembles_of_7, accs_of_7 = loop_over_combinations(paths, ds_targets, lat_weights, climatology_ds_outputs, N = 7)

    # Extract RMSE values from the dictionary
    acc_values_list = list(accs_of_5.values())  # Shape:THIS IS A LIST OF XARRATY FRAMES
    acc_values_list_7 = list(accs_of_7.values()) 
    acc_values_list_3 = list(accs_of_3.values())

    rmse_values_list = list(ensembles_of_5.values())  # Shape:THIS IS A LIST OF XARRATY FRAMES
    rmse_values_list_7 = list(ensembles_of_7.values()) 
    rmse_values_list_3 = list(ensembles_of_3.values())
    
    # Create subplots
    fig, axs = plt.subplots(2, 4, figsize=(20, 12))
    #fig.suptitle('RMSE and ACC Evolution Across Time Steps for Different Variables', fontsize=16)
    length = 11

    for idx, var in enumerate(variables):
        for metric_idx, metric in enumerate(['rmses', 'accs']):
            ax = axs[metric_idx, idx]

            # Load data and stack into array
            data = []
            for p in paths:
                ds = xr.open_dataset(f"{p}{metric}.nc")
                data.append(ds[var].values[:length])

            data = np.array(data)

            time_steps = np.arange(length)
            mean_metric = np.nanmean(data, axis=0)
            std_metric = np.nanstd(data, axis=0)

            for i, p in enumerate(paths):
                
                ax.plot(time_steps, data[i], linestyle='--', alpha=0.7, label=f'Seed {i+1}')

            # Plot mean with error bands
            ax.plot(time_steps, mean_metric, color='black', linewidth=2, label='Mean')
            ax.fill_between(time_steps, mean_metric - std_metric, mean_metric + std_metric, color='gray', alpha=0.3, label='±1 STD')


            # Plot other models
            if metric == 'accs':

                # 5 MODEL ENSEMBLE (ACCS)
                rmse_values = np.array([frame[var].values[:length] for frame in acc_values_list])
                mean_rmse = np.mean(rmse_values, axis=0)
                std_rmse = np.std(rmse_values, axis=0)
                
                # Plot individual RMSE curves for each ensemble combination
                for i, rmse_ds in enumerate(rmse_values):
                    ax.plot(time_steps, rmse_ds, color='orange', alpha=0.5, label=f'5 Model Ensemble')
                
                # Plot mean RMSE with error bands
                ax.plot(time_steps, mean_rmse, color='orange', linewidth=2) # label='Mean ACC'
                ax.fill_between(time_steps, mean_rmse - std_rmse, mean_rmse + std_rmse,
                                color='gray', alpha=0.3) # label='±1 STD'


                #7 MODEL ENSEMBLE
                rmse_values = np.array([frame[var].values[:length] for frame in acc_values_list_7])
                mean_rmse = np.mean(rmse_values, axis=0)
                std_rmse = np.std(rmse_values, axis=0)
                
                # Plot individual RMSE curves for each ensemble combination
                for i, rmse_ds in enumerate(rmse_values):
                    ax.plot(time_steps, rmse_ds, color='green', alpha=0.5, label=f'7 Model Ensemble')
                
                # Plot mean RMSE with error bands
                ax.plot(time_steps, mean_rmse, color='green', linewidth=2)
                ax.fill_between(time_steps, mean_rmse - std_rmse, mean_rmse + std_rmse,
                                color='gray', alpha=0.3) # label='±1 STD'


                # 3 MODEL ENSEMBLE
                rmse_values = np.array([frame[var].values[:length] for frame in acc_values_list_3])
                mean_rmse = np.mean(rmse_values, axis=0)
                std_rmse = np.std(rmse_values, axis=0)
                
                # Plot individual RMSE curves for each ensemble combination
                for i, rmse_ds in enumerate(rmse_values):
                    ax.plot(time_steps, rmse_ds, color='yellow', alpha=0.5, label=f'3 Model Ensemble')
                
                # Plot mean RMSE with error bands
                ax.plot(time_steps, mean_rmse, color='yellow', linewidth=2)
                ax.fill_between(time_steps, mean_rmse - std_rmse, mean_rmse + std_rmse,
                                color='gray', alpha=0.3) # label='±1 STD'

                

            if metric == 'rmses':
                #climatology_ds = xr.open_dataset(f"/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDEref/evaluation/{metric}_climatology.nc")
            
                persistence_ds = xr.open_dataset(f"{persistence_path}{metric}_persistence.nc")
                climatology_ds  = xr.open_dataset(f"{persistence_path}{metric}_climatology.nc")

                # 5 MODEL ENSEMBLE
                rmse_values = np.array([frame[var].values[:length] for frame in rmse_values_list])
                mean_rmse = np.mean(rmse_values, axis=0)
                std_rmse = np.std(rmse_values, axis=0)
                
                # Plot individual RMSE curves for each ensemble combination
                for i, rmse_ds in enumerate(rmse_values):
                    ax.plot(time_steps, rmse_ds, color='orange', alpha=0.5, label=f'5 Model Ensemble')
                
                # Plot mean RMSE with error bands
                ax.plot(time_steps, mean_rmse, color='orange', linewidth=2) # label='Mean RMSE'
                ax.fill_between(time_steps, mean_rmse - std_rmse, mean_rmse + std_rmse,
                                color='gray', alpha=0.3) # label='±1 STD'


                # 7 MODEL ENSEMBLE
                rmse_values = np.array([frame[var].values[:length] for frame in rmse_values_list_7])
                mean_rmse = np.mean(rmse_values, axis=0)
                std_rmse = np.std(rmse_values, axis=0)
                
                # Plot individual RMSE curves for each ensemble combination
                for i, rmse_ds in enumerate(rmse_values):
                    ax.plot(time_steps, rmse_ds, color='green',alpha=0.5, label=f'7 Model Ensemble')
                
                # Plot mean RMSE with error bands
                ax.plot(time_steps, mean_rmse, color='green', linewidth=2)
                ax.fill_between(time_steps, mean_rmse - std_rmse, mean_rmse + std_rmse,
                                color='gray', alpha=0.3)


                # 3 MODEL ENSEMBLE
                rmse_values = np.array([frame[var].values[:length] for frame in rmse_values_list_3])
                mean_rmse = np.mean(rmse_values, axis=0)
                std_rmse = np.std(rmse_values, axis=0)
                
                # Plot individual RMSE curves for each ensemble combination
                for i, rmse_ds in enumerate(rmse_values):
                    ax.plot(time_steps, rmse_ds, color='yellow', alpha=0.5, label=f'3 Model Ensemble')
                
                # Plot mean RMSE with error bands
                ax.plot(time_steps, mean_rmse, color='yellow', linewidth=2)
                ax.fill_between(time_steps, mean_rmse - std_rmse, mean_rmse + std_rmse,
                                color='gray', alpha=0.3)


                ax.plot(time_steps, climatology_ds[var].values[:length], color='red', linewidth=2, label='Climatology')
                ax.plot(time_steps, persistence_ds[var].values[1:length+1], color='brown', linewidth=2, label='Persistence')
                #ax.plot(time_steps, ensemble_ds[var].values[:10], color='pink', linewidth=2, label='Small 3 Model Ensemble')
                
                #ax.plot(time_steps, big3ensemble[var].values[:length], color='green', linewidth=2, label='3 Model Ensemble')
                if big_5_ensemble:
                    ax.plot(time_steps, big_5_ensemble[var].values[:length], color='purple', linewidth=2, label='5 Model Ensemble')

            
            #ax.plot(time_steps, big_ds[var].values[:10], color='purple', linewidth=2, label='Big UNet')
            
            # Formatting
            if metric_idx > 0:
                ax.set_xlabel('Lead day', fontsize=10)
            if idx == 0:
                ax.set_ylabel(f'{"RMSE" if metric == "rmses" else "ACC"} ', fontsize=10)
            if metric_idx == 0:
                ax.set_title(f'{var}', fontsize=12)
            ax.set_xticks(np.arange(0, 10, 1)) # label the x-axis from 1 to 10
            ax.set_xticklabels(range(1, 11))
            ax.set_xlim(0, 9)
            ax.grid(True, linestyle='--', alpha=0.7)
            if idx == 3:  # Only show legend for the rightmost plots
                ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    # Save the plot to the specified output path
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

    return None


def main():
    # THIS IS ON 5.625 RESOLUTION

    big_5_ensemble = '/home/adboer/dlwp-benchmark/src/dlwpbench/big_ensemble_5items_rmses.nc'
    #small_ensemble = '/projects/prjs1254/outputs/PDEref/evaluation/small_ensemble.nc'
    #big_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDEref/evaluation/'
    climatology_path = '/projects/prjs1254/climatology/'
    #climatology_path = '/projects/prjs1254/climatology1D_32/'

    persistence_path = '/projects/prjs1254/outputs/big_seed3000/evaluation/'
    #persistence_path = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321238/evaluation/'
    
    #ensemble_3 = '/projects/prjs1254/outputs/PDEref/evaluation/big_ensemble.nc'


    #ensemble_ds = xr.open_dataset(small_ensemble)

    paths = [
    '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefsmall_seed2/evaluation/',
    '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefsmall_seed20/evaluation/',
    '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefsmall_seed1234/evaluation/'
    ]
    
    # 5.6 degrees!
    paths = ['/projects/prjs1254/outputs/big_seed100/evaluation/',
    '/projects/prjs1254/outputs/big_seed3000/evaluation/',
    '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDEref/evaluation/',
    '/projects/prjs1254/outputs/big_seed1235/evaluation/',
    '/projects/prjs1254/outputs/big_seed1236/evaluation/',
    '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDEref1237/evaluation/',
    '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDEref1238/evaluation/'
    ]


    #paths = ['/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321237/evaluation/','/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321238/evaluation/', '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321239/evaluation/', '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321240/evaluation/','/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321241/evaluation/']

    # make a function that loops over all possible combinations of 5 models
    # for each of these combinations take the mean over their outputs
    # e.g. evaluation/outputs.nc
    # this is the ensemble mean
    # then calculate the rmses for this new ensemble mean

    ds_outputs = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDEref1238/evaluation/outputs.nc'
    #ds_outputs = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321237/evaluation/outputs.nc'

    ds_outputs = xr.open_dataset(ds_outputs).isel(time=slice(0, 45)).isel(sample=slice(0, 194))

    ds_targets = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDEref1238/evaluation/targets.nc'
    #ds_targets = '/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PDErefhpx321237/evaluation/targets.nc'
    ds_targets = xr.open_dataset(ds_targets).isel(time=slice(0, 45)).isel(sample=slice(0, 194))

    T = ds_outputs.sizes["time"]  # Number of time steps
    S = ds_outputs.sizes["sample"]
    deg = 5.625
    lat_weights = get_adjusting_weights(deg).unsqueeze(0).unsqueeze(0).expand(S,T,ds_outputs.sizes["lat"],ds_outputs.sizes["lon"]) 

    # do this for each combination of 5

    make_plot(paths = paths, ensemble_3 = big_5_ensemble, ds_targets = ds_targets, lat_weights = lat_weights, climatology_path = climatology_path, persistence_path=persistence_path)


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
