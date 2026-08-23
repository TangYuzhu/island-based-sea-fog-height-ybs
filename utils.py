"""
Fog Top Height Retrieval Utils Module
Utility Functions

"""

import numpy as np
import pandas as pd


def collect_track_data(island_lats, island_lons, category, calipso_cth, img, date_str):
    """
    Collect trajectory and island status data for a single case
    
    Parameters:
        island_lats: Array of island latitudes
        island_lons: Array of island longitudes
        category: Array of island status classifications
        calipso_cth: CALIPSO trajectory data
        img: Image data
        date_str: Date string
    
    Returns:
        dict: Dictionary containing case data
    """
    return {
        'date_str': date_str,
        'island_lats': island_lats,
        'island_lons': island_lons,
        'category': category,
        'calipso_cth': calipso_cth,
        'img': img
    }


def load_island_info():
    """
    Load island information
    
    Returns:
        tuple: (island_lons, island_lats, island_heights, island_areas)
    """
    try:
        islands_info = pd.read_csv('island_info.csv', sep=',')
        island_lons = np.asarray(islands_info['lon'])
        island_lats = np.asarray(islands_info['lat'])
        island_heights = np.asarray(islands_info['pe'])
        island_areas = np.asarray(islands_info['area'])
        return island_lons, island_lats, island_heights, island_areas
    except Exception as e:
        print(f"Failed to load island information: {e}")
        raise


def process_sample_list():
    """
    Process sample list
    
    Returns:
        list: Sorted sample list
    """
    from data_loader import read_csv_to_list
    filename = 'filelist.csv'
    sample_list = read_csv_to_list(filename)
    sample_list.sort()
    return sample_list


def calc_metrics(data):
    '''
    Calculate various metrics for predicted values
    
    Parameters:
        data: Data array containing predicted values
        
    Returns:
        tuple: (mae, mape, rmse, HR) - Mean Absolute Error, Mean Absolute Percentage Error, Root Mean Square Error, Hit Rate
    '''
    errors = data[:,1:-1]-data[:,0,np.newaxis]
    abs_errors = np.abs(errors)
    is_Hit = abs_errors<=30
    HR = np.sum(is_Hit,0) / is_Hit.shape[0] * 100
    mae = np.mean(abs_errors,0)
    mape = np.mean(np.abs(errors/data[:,0,np.newaxis]),0) * 100
    rmse = np.sqrt(np.mean(errors**2,0))
    means = np.mean(data[:,0],0)
    SSE = np.sum(errors**2,0)
    SST = np.sum((data[:,0]-means)**2,0)
    R2 = 1 - (SSE / SST)
    return mae, mape, rmse, HR


def extract_date_from_filename(filename):
    """
    Extract date information from filename
    
    Parameters:
        filename (str or Path): Filename or path
        
    Returns:
        tuple: (date_doy, datetime_str, date_str)
    """
    from datetime import datetime, timedelta
    
    def doy_to_date(doy_str):
        """
        Convert 'YYYYDDD' format date string to standard 'YYYY-MM-DD' date string
        
        Parameters:
            doy_str (str): String representing year and day of year, e.g., '2024150' represents the 150th day of 2024
            
        Returns:
            int: Integer representation of the corresponding date in YYYYMMDD format
        """
        year = int(doy_str[:4])
        doy = int(doy_str[4:])
        date = datetime(year, 1, 1) + timedelta(days=doy-1)
        return int(date.strftime('%Y%m%d'))
    
    if hasattr(filename, 'name'):
        filename = filename.name
        
    idx = filename.index('20')
    date_doy = filename[idx:idx+7]
    date_ymd = doy_to_date(date_doy)
    date_str = datetime.strptime(str(int(date_ymd)), '%Y%m%d').strftime('%Y-%m-%d')
    # Use date-only labels consistently.  New daily filenames do not contain
    # an acquisition time, so interpreting the characters after YYYYDDD as
    # HHMM produced labels such as ``da:il(UTC)``.
    datetime_str = date_str

    return date_doy, datetime_str, date_str


def create_bias_dataframe(bias_data):
    """
    Create DataFrame for bias data
    
    Parameters:
        bias_data: List of bias data
        
    Returns:
        pd.DataFrame: DataFrame containing bias data
    """
    df = pd.DataFrame(bias_data)
    # Convert method index to method name
    method_map = {0: 'Modis', 1: 'LR', 2: 'SVM'}
    df['Method'] = df['Method'].map(method_map)
    return df
