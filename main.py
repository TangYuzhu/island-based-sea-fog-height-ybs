"""
Fog Top Height Retrieval Main Program - Modular Version

This is the modularized main program entry point, integrating functions from various modules.

Features include:
1. Data reading and preprocessing
2. Regression and classification modeling
3. Result statistics and performance evaluation
4. Visualization output

Author: [plyang]
Version: 2.0 (Modular Version)
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import modules
from config import *
from data_loader import read_key_islands, read_modis_cth, read_calipso_sfth, pick_radiance, read_csv_to_list
from models import *
from visualization import *
from utils import calc_metrics, extract_date_from_filename, collect_track_data


def resolve_modis_image_path(radiance_path):
    """Return the MODIS preview image matching a B03 radiance file.

    Both dataset naming conventions are supported:
    ``*_B03.npy`` (legacy) and ``*.B03.npy`` (current).  Image products may be
    stored as PNG or JPEG files.
    """
    filename = Path(radiance_path).name
    b03_suffixes = ('.B03.npy', '_B03.npy')
    image_suffixes = ('.png', '.jpg', '.jpeg')

    for b03_suffix in b03_suffixes:
        if filename.lower().endswith(b03_suffix.lower()):
            image_stem = filename[:-len(b03_suffix)]
            candidates = [modis_img_path / f'{image_stem}{suffix}'
                          for suffix in image_suffixes]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate

            candidate_text = ', '.join(str(path) for path in candidates)
            raise FileNotFoundError(
                f'No matching MODIS image for {filename}. Tried: {candidate_text}'
            )

    raise ValueError(
        f'Unsupported B03 filename {filename!r}; expected a name ending in '
        "'.B03.npy' or '_B03.npy'"
    )


def process_single_sample(p, island_heights, island_lats, island_lons, island_areas):
    """
    Complete processing pipeline for a single sample
    
    Args:
        p (Path): Sample file path
        island_heights (np.array): Island height array
        island_lats (np.array): Island latitude array  
        island_lons (np.array): Island longitude array
        island_areas (np.array): Island area array
        
    Returns:
        dict: Dictionary containing processing results, returns None if processing fails
    """
    try:
        # Import necessary module functions
        from models import logistic_regression, svm_regression
        import matplotlib.pyplot as plt
        
        # Extract date information
        date_doy, datetime_str, date_str = extract_date_from_filename(p)
        
        # Load and process data
        radiance = np.load(p)
        img = plt.imread(resolve_modis_image_path(p))
        
        # Read various data
        modis_cth = read_modis_cth(p)
        calipso_sfth = read_calipso_sfth(p)
        id, category = read_key_islands(p)
        island_h = island_heights[id-1]
        island_refs = pick_radiance(id, radiance, category, island_lats, island_lons, island_areas)
        
        # Regression modeling
        a2, b2 = logistic_regression(island_refs, island_h, category)
        y2_est = a2 * np.arange(0, 1, 0.1) + b2
        height_LR = a2 * radiance + b2
        
        a3, b3 = svm_regression(island_refs, island_h, category)
        y3_est = a3 * np.arange(0, 1, 0.1) + b3
        height_SVM = a3 * radiance + b3
        
        # Result stacking
        predicted_sfth = np.dstack((height_LR, height_SVM))
        
        # Extract track point heights
        calipso_lats = calipso_sfth[:, 0]
        calipso_lons = calipso_sfth[:, 1]
        modis_row = ((max_lat - calipso_lats) / res_modis).astype(int)
        modis_col = ((calipso_lons - min_lon) / res_modis).astype(int)
        radiance_row = ((max_lat - calipso_lats) / res_radiance).astype(int)
        radiance_col = ((calipso_lons - min_lon) / res_radiance).astype(int)
        modis_cth_track = modis_cth[modis_row, modis_col]
        track_refs = radiance[radiance_row, radiance_col]

        predicted_sfth_track = predicted_sfth[radiance_row, radiance_col]

        # Data cleaning
        modis_cth_track[modis_cth_track>1000] = 1000
        modis_cth_track[calipso_sfth[:,2]==0] = 0
        predicted_sfth_track[calipso_sfth[:,2]==0] = 0
        predicted_sfth_track[predicted_sfth_track<0] = 0
        modis_cth_track[modis_cth_track<0] = 0
        predicted_sfth_track[predicted_sfth_track>1000] = 1000
        
        calip_mod_pred_sfth = np.hstack((calipso_sfth[:,2,np.newaxis], modis_cth_track[:,np.newaxis], predicted_sfth_track, track_refs[:,np.newaxis]))
        
        # Metric statistics
        mae, mape, rmse, HR = calc_metrics(calip_mod_pred_sfth[calip_mod_pred_sfth[:,0]>0,:])
        
        return {
            'date_doy': date_doy,
            'datetime_str': datetime_str,
            'date_str': date_str,
            'data': calip_mod_pred_sfth,
            'modis_cth': modis_cth,
            'predicted_cths': predicted_sfth,
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'HR': HR,
            'island_refs': island_refs,
            'island_h': island_h,
            'category': category,
            'calipso_cth': calipso_sfth,
            'track_refs': track_refs,
            'y2_est': y2_est,
            'y3_est': y3_est,
            'img': img,
            'island_lats': island_lats[id-1],
            'island_lons': island_lons[id-1]
        }
        
    except Exception as e:
        print(f"Error processing file {p.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main program entry point
    """
    print("=" * 60)
    print("Fog Top Height Retrieval")
    print("=" * 60)
    
    # Initialize statistical variables
    data_all = None  # For recording prediction results of each case
    is_plot = True   # Whether to perform visualization

    # Read island information
    try:
        islands_info = pd.read_csv('island_info_nasadem.csv', sep=',')
        island_lons = np.asarray(islands_info['lon'])
        island_lats = np.asarray(islands_info['lat'])
        island_heights = np.asarray(islands_info['pe'])
        island_areas = np.asarray(islands_info['area'])
        print("Island information read successfully")
    except Exception as e:
        print(f"Failed to read island information: {e}")
        return

    # Plot 3D island height map
    if is_plot:
        plot_island_height(island_lons, island_lats, island_heights)
        print("3D island height map plotted successfully")



    # Read sample list
    try:
        filename = 'filelist_49cases.csv'
        sample_list = read_csv_to_list(filename)
        sample_list.sort()
        print(f"Sample list read successfully, total {len(sample_list)} samples")
    except Exception as e:
        print(f"Failed to read sample list: {e}")
        return
    
    # Output table header
    print('\n' + '=' * 100)
    print('\tMAE\t\t\tMAPE\t\t\tRMSE\t\t\tHR\t\t')
    print('id\tmodis\tLR\tSVM\tmodis\tLR\tSVM\tmodis\tLR\tSVM\tmodis\tLR\tSVM')
    print('=' * 100)

    # Define metric collection lists before main loop
    case_metrics_list = []
    all_case_scatter_data = []  # For collecting all case data for combined plotting
    all_track_data = []         # For collecting all case track data for combined plotting
    bias_data = []              # For unified collection of error data

    # Main loop: process samples one by one
    successful_cases = 0
    for i, sample_path in enumerate(sample_list):
        p = Path(sample_path)        
        try:
            # Process single sample
            result = process_single_sample(p, island_heights, island_lats, island_lons, island_areas)
            
            if result is None:
                print(f"  Sample {p.name} processing failed, skipping")
                continue
                
            successful_cases += 1
            
            # Output results
            print(f"{result['date_doy']}\t"
                  f"{result['mae'][0]:.2f}\t{result['mae'][1]:.2f}\t{result['mae'][2]:.2f}\t"
                  f"{result['mape'][0]:.2f}\t{result['mape'][1]:.2f}\t{result['mape'][2]:.2f}\t"
                  f"{result['rmse'][0]:.2f}\t{result['rmse'][1]:.2f}\t{result['rmse'][2]:.2f}\t"
                  f"{result['HR'][0]:.2f}\t{result['HR'][1]:.2f}\t{result['HR'][2]:.2f}\t")
            
            # Visualization output
            if is_plot:
                # Plot track map
                plot_track(footprint_path.joinpath(f'{result["date_doy"]}.pdf'), 
                          result['island_lats'], result['island_lons'], 
                          result['category'], result['calipso_cth'], result['img'])
                
                # Plot scatter plot
                plot_island_scatter(
                    result['island_refs'], result['island_h'], result['category'], 
                    result['y2_est'], result['y3_est'],
                    scatter_path.joinpath(f'{result["date_doy"]}.png'), result["date_str"]
                )
                
                # Plot height comparison chart
                plot_height_compare(
                    result['calipso_cth'], result['data'][:,1], result['data'][:,2:4],
                    height_cmp_path.joinpath(f'{result["date_doy"]}.pdf'), result["date_str"]
                )
                
                # Plot height images
                plot_height_image(result['predicted_cths'][:,:,0], 
                                height_image_path.joinpath(f'{result["date_doy"]}_LR.pdf'), result["date_str"],
                                'Fog-top height (m)')
                plot_height_image(result['predicted_cths'][:,:,1], 
                                height_image_path.joinpath(f'{result["date_doy"]}_SVM.pdf'), result["date_str"],
                                'Fog-top height (m)')
                plot_height_image(result['modis_cth'], 
                                height_image_path.joinpath(f'{result["date_doy"]}_MODIS.pdf'), result["date_str"],
                                'Cloud-top height (m)')
            
            # Update global statistics
            data_all = result['data'] if data_all is None else np.vstack((data_all, result['data']))
            
            # Error statistics - only count CALIPSO heights greater than 0
            valid_mask = result['data'][:,0] > 0
            if np.any(valid_mask):
                valid_data = result['data'][valid_mask]
                bias = valid_data[:,1:-1] - valid_data[:,0,np.newaxis]
                bias = bias.T.ravel()
                
                # Unified collection of error data into list
                n_points_per_method = len(bias) // 3
                
                for j, error_val in enumerate(bias):
                    method_idx = j // n_points_per_method
                    if method_idx < 3:  # Ensure index does not go out of bounds
                        bias_data.append({
                            'Date': result['datetime_str'],
                            'Method': method_idx,
                            'Bias(m)': error_val
                        })

            # Collect metric data
            case_metrics_list.append([
                result['date_str'],
                result['mae'].tolist(),
                result['mape'].tolist(),
                result['rmse'].tolist(),
                result['HR'].tolist()
            ])

            # Collect data for combined plotting
            all_case_scatter_data.append((
                result['date_str'], result['island_refs'], result['island_h'], 
                result['category'], result['y2_est'], result['y3_est']
            ))

            # Collect track data for combined plotting
            track_data = collect_track_data(
                result['island_lats'], result['island_lons'], result['category'], 
                result['calipso_cth'], result['img'], result['date_str']
            )
            all_track_data.append(track_data)
            
        except Exception as e:
            print(f"  Error processing sample {p.name}: {e}")
            continue

    # Calculate overall metrics
    if data_all is not None and len(data_all) > 0:
        data_all = data_all[data_all[:,0] > 0, :]
        mae, mape, rmse, HR = calc_metrics(data_all)
        
        # Insert overall AVG metrics into case_metrics_list
        case_metrics_list.append([
            'Average',
            mae.tolist(),
            mape.tolist(),
            rmse.tolist(),
            HR.tolist(),
        ])

        print('\n' + '=' * 100)
        print(f'Average\t{mae[0]:.2f}\t{mae[1]:.2f}\t{mae[2]:.2f}\t'
              f'{mape[0]:.2f}\t{mape[1]:.2f}\t{mape[2]:.2f}\t'
              f'{rmse[0]:.2f}\t{rmse[1]:.2f}\t{rmse[2]:.2f}\t'
              f'{HR[0]:.2f}\t{HR[1]:.2f}\t{HR[2]:.2f}')
        print('=' * 100)
        
        print(f"\nProcessing completed! Successfully processed {successful_cases}/{len(sample_list)} samples")

        plot_metrics_heatmap(case_metrics_list)

        # Plot combined charts
        if is_plot:
            print("\nStarting to plot combined charts...")
            
            # Plot metrics heatmap
            plot_metrics_heatmap(case_metrics_list)
            print("  Metrics heatmap plotted successfully")
            
            # Plot bias violin plots
            if bias_data:
                df = pd.DataFrame(bias_data)
                plot_bias_violin(df, method_colors)
                plot_biase_violin_by_date(df, method_colors)
                print("  Bias violin plots plotted successfully")
            
            # Plot combined scatter plots for all cases
            if len(all_case_scatter_data) > 0:
                plot_all_island_scatters(all_case_scatter_data, 
                                       scatter_path.joinpath('all_cases_scatter_combined.pdf'))
                print("  Combined scatter plots plotted successfully")
            
            # Plot combined track plots for all cases
            if len(all_track_data) > 0:
                plot_all_tracks_subplots(all_track_data, 
                                      footprint_path.joinpath('all_cases_tracks_combined.pdf'), 
                                      island_lons, island_lats, island_heights)
                print("  Combined track plots plotted successfully")
            
            print("All visualization charts plotted successfully!")
    
    else:
        print("\nWarning: No sample data was successfully processed")
    
    print("\n" + "=" * 60)
    print("Program execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
