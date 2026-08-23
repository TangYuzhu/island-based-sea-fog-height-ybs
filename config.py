"""
Fog Top Height Retrieval Configuration Module
Global Configuration and Parameter Settings

Contains:
- Global parameters and path settings
- Research area extent parameters
- Path initialization
"""

from pathlib import Path

# Font and plotting settings
plt_rcParams = {
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix'  # Set math formula font to Times New Roman style
}

# Path settings
is_path = Path('./islands_categories')
modis_cth_path = Path('./modis_cths')
calipso_path = Path('./calipso_sfths')
modis_img_path = Path('./modis_imgs')
modis_b03_path = Path('./modis_b03')

# Output path initialization
scatter_path = Path('scatter_figs'); scatter_path.mkdir(exist_ok=True)
height_cmp_path = Path('height_compare'); height_cmp_path.mkdir(exist_ok=True)
height_image_path = Path('height_image'); height_image_path.mkdir(exist_ok=True)
footprint_path = Path('calipso_footprints'); footprint_path.mkdir(exist_ok=True)

# Research area extent parameters
max_lat = 42
min_lat = 30
max_lon = 129
min_lon = 117
extent = [min_lon, max_lon, min_lat, max_lat]  # Area extent

# Resolution parameters
res_modis = 0.05
res_radiance = 0.005

# Method names and color configuration
method_names = ['Modis', 'LR', 'SVM']
method_colors = ['#1f77b4', '#2ca02c', '#d62728']
