"""
Fog Top Height Retrieval Data Loader Module
Data Loading and Preprocessing Functions

Contains:
- Various data file reading functions
- Data preprocessing and transformation functions
- Metric calculation functions
"""

import numpy as np
import pandas as pd
import csv
from datetime import datetime, timedelta
from pathlib import Path
from skimage.filters import threshold_otsu

# Import configuration
from config import is_path, modis_cth_path, calipso_path, modis_img_path
from config import max_lat, min_lat, max_lon, min_lon, res_radiance, res_modis

# Import utility functions
from utils import calc_metrics


def read_key_islands(path):
    """
    Read key island information
    
    Args: 
        path (Path): File path object
        
    Returns: 
        tuple: (id, category) - Island ID array and category array
        
    Raises:
        FileNotFoundError: If CSV file does not exist
        ValueError: If file format is incorrect
    """
    try:
        idx = path.name.index('20')
        key_island_path = path.name[idx:idx+7] + '.csv'
        csv_path = is_path.joinpath(key_island_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Island information file does not exist: {csv_path}")
            
        key_islands_info = pd.read_csv(csv_path, sep=',')
        
        if 'id' not in key_islands_info.columns or 'category' not in key_islands_info.columns:
            raise ValueError(f"CSV file missing required columns: {csv_path}")
            
        id = np.asarray(key_islands_info['id'])
        category = np.asarray(key_islands_info['category'])
        
        return id, category
        
    except ValueError as e:
        if "substring not found" in str(e):
            raise ValueError(f"Date identifier not found in filename: {path.name}") from e
        raise


def read_modis_cth(path):
    """
    Read MODIS cloud top height product
    Args: path (Path)
    Returns: modis_cth (np.array)
    """
    idx = path.name.index('20')
    modis_filename = list(modis_cth_path.glob(f'MYD06*{path.name[idx:idx+7]}*.npy'))[0]
    modis_cth = np.load(modis_filename)
    modis_cth[np.isnan(modis_cth)] = 0
    return modis_cth


def read_calipso_sfth(path):
    """
    Read CALIPSO fog top height
    Args: path (Path)
    Returns: calipso_cth (np.array)
    """
    idx = path.name.index('20')
    date_str = path.name[idx:idx+7]
    calipso_filename = date_str + '_calipso_result.csv'
    calipso_cth = np.loadtxt(calipso_path.joinpath(calipso_filename), delimiter=',')
    return calipso_cth


def pick_radiance(id, radiance, category, island_lats, island_lons, island_areas):
    """
    Get reflectance corresponding to key islands
    
    Args: 
        id (np.array): Island ID array
        radiance (np.array): Radiance data
        category (np.array): Island category array
        island_lats (np.array): Island latitude array
        island_lons (np.array): Island longitude array
        island_areas (np.array): Island area array
        
    Returns: 
        refs (np.array): Processed reflectance values
    """
    refs = np.zeros(id.shape)
    
    # Precompute coordinates for all islands
    island_indices = id - 1  # Convert to 0-based index
    lats = island_lats[island_indices]
    lons = island_lons[island_indices]
    areas = island_areas[island_indices]
    
    # Calculate row and column coordinates for all islands
    rows = ((max_lat - lats) / res_radiance).astype(int)
    cols = ((lons - min_lon) / res_radiance).astype(int)
    
    # Batch process islands of the same category
    for cat in [1, 2]:
        mask = category == cat
        if not np.any(mask):
            continue
            
        for i in np.where(mask)[0]:
            length = int(np.sqrt(areas[i]) / res_radiance / 100 * 3)
            row, col = rows[i], cols[i]
            
            # Ensure indices are within boundaries
            row_start = max(0, row - length)
            row_end = min(radiance.shape[0], row + length + 1)
            col_start = max(0, col - length)
            col_end = min(radiance.shape[1], col + length + 1)
            
            sub_im = radiance[row_start:row_end, col_start:col_end]
            valid_values = sub_im[np.isfinite(sub_im)]

            if valid_values.size == 0:
                raise ValueError(
                    f'No valid radiance values in the sampling window for '
                    f'island ID {id[i]}'
                )
            
            if cat == 2:
                refs[i] = np.mean(valid_values)
            else:
                thres = threshold_otsu(valid_values)
                values_above_threshold = valid_values[valid_values > thres]
                refs[i] = (np.mean(values_above_threshold)
                           if values_above_threshold.size > 0
                           else np.mean(valid_values))
    
    return refs


def read_csv_to_list(filename):
    """
    Read CSV file and return a list of the first element of each row.

    Args:
        filename (str): Path to the CSV file to read.

    Returns:
        list: List of strings containing the content of the first column of each row.

    Raises:
        FileNotFoundError: Raised when the file does not exist.
        UnicodeDecodeError: Raised when the file cannot be decoded with UTF-8.
    """
    data_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_list.append(row[0])
    return data_list
