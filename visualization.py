"""
Fog Top Height Retrieval Visualization Module
Visualization functions

Contains:
- Various plotting functions
- Combined plotting functionality
- Statistical chart plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import datetime

# Import configuration
from config import extent, min_lon, max_lon, min_lat, max_lat
from config import plt_rcParams

def get_slice_xticks(lon, lat, ntick, decimals=2, lon_formatter=None, lat_formatter=None):
    '''Set ntick equally spaced ticks on the slice, return x values and tick positions and labels.'''
    # xticks are the logical positions of ticks, corresponding longitude and latitude values are obtained through linear interpolation.
    npt = len(lon)
    x = np.arange(npt)
    xticks = np.linspace(0, npt - 1, ntick)
    lons = np.interp(xticks, x, lon).round(decimals)
    lats = np.interp(xticks, x, lat).round(decimals)

    # Get tick labels in string format.
    xticklabels = []
    if lon_formatter is None:
        lon_formatter = LongitudeFormatter()
    if lat_formatter is None:
        lat_formatter = LatitudeFormatter()
    for i in range(ntick):
        lon_str = lon_formatter(lons[i])
        lat_str = lat_formatter(lats[i])
        xticklabels.append(lon_str + '\n' + lat_str)

    return x, xticks, xticklabels


def region_mask(lon, lat, extent):
    '''Mark data that falls within the longitude-latitude bounding box.'''
    lonmin, lonmax, latmin, latmax = extent
    return (
        (lon >= lonmin) & (lon <= lonmax) &
        (lat >= latmin) & (lat <= latmax)
    )


def plot_island_height(lons, lats, heights):
    """
    Plot 3D island height visualization
    """
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set coordinate range to research area extent
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # Completely remove height axis
    ax.set_zticks([])
    ax.set_zticklabels([])
    ax.zaxis.line.set_visible(False)
    ax.set_zlabel('')

    # Remove all grid lines (including vertical planes)
    ax.grid(False)

    # Modify axis labels, add degree units
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')

    # Remove side planes (xz and yz planes)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Set xz plane transparent
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Set yz plane transparent

    # Add coastline (precisely clipped to research area)
    from shapely.geometry import Polygon, LineString
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    geoms = coastline.geometries()

    # Create polygon boundary for research area
    study_area = Polygon([
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, max_lat),
        (max_lon, min_lat),
        (min_lon, min_lat)
    ])

    coastline_lines = []
    for geom in geoms:
        # Clip geometry to research area
        if geom.intersects(study_area):
            intersection = geom.intersection(study_area)
            
            # Handle different geometry types
            if intersection.geom_type == 'LineString':
                if not intersection.is_empty:
                    coastline_lines.append(np.array(intersection.coords))
            elif intersection.geom_type == 'MultiLineString':
                for line in intersection.geoms:
                    coastline_lines.append(np.array(line.coords))

    # Draw clipped coastline
    for line in coastline_lines:
        xs, ys = line[:, 0], line[:, 1]
        zs = np.zeros_like(xs)  # Draw at z=0 plane
        ax.plot(xs, ys, zs, color='black', linewidth=0.5)

    # Compress height direction (z-axis compressed to 1/3)
    ax.set_box_aspect([1, 1, 0.2])

    # Keep only longitude-latitude axes (hide height axis)
    ax.set_zticks([])  # Hide z-axis ticks
    ax.set_zticklabels([])  # Hide z-axis tick labels
    ax.zaxis.line.set_visible(False)  # Hide z-axis line
    ax.set_zlabel('')  # Hide z-axis label

    # Draw 3D island bar chart
    for i in range(len(lons)):
        ax.bar3d(lons[i], lats[i], 0, 0.1, 0.1, heights[i], 
                color=plt.cm.jet(heights[i]/max(heights)), 
                alpha=0.7, edgecolor='gray', linewidth=0.1)

    # Add four borders of the base plane
    rect_x = [min_lon, max_lon, max_lon, min_lon, min_lon]
    rect_y = [min_lat, min_lat, max_lat, max_lat, min_lat]
    rect_z = [0, 0, 0, 0, 0]
    ax.plot(rect_x, rect_y, rect_z, color='black', linewidth=1)

    # Precisely set axis range to match base plane
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_zlim(0, max(heights)*1.1)  # z-axis range includes all bars
    ax.set_box_aspect([1, 1, 0.2])  # Reapply compression ratio

    # Add horizontal colorbar, placed inside bottom of base plane
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=max(heights)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='bottom', fraction=0.04, pad=-0.25)
    cbar.set_label('Height (m)')

    # Set viewing angle
    ax.view_init(elev=30, azim=-90)

    # Manually adjust layout to avoid tight_layout error
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)  # Increase bottom space

    # Add transparent point to solve bounding box issue (not displayed but ensures correct calculation)
    ax.scatter([min_lon], [min_lat], [0], s=0, alpha=0)
    # Save as PDF format
    plt.savefig('3d_island_height.pdf', format='pdf')
    plt.close(fig)


def plot_track(save_path, island_lats, island_lons, category, calipso_cth, img):
    """
    Plot track visualization
    """
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    proj = ccrs.PlateCarree()  # Projection type
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=proj)
    ax.imshow(img, extent=extent, origin='upper')
    ax.coastlines(color='wheat', lw=0.1)
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([117, 119, 121, 123, 125, 127])
    gl.ylocator = mticker.FixedLocator([32,34, 36, 38, 40, 42, 44])
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 16} 
    gl.ylabel_style = {'size': 16} 

    ax.plot(calipso_cth[:,1],calipso_cth[:,0], color='gold', linewidth=3) # CALIPSO track

    fog_points = calipso_cth[calipso_cth[:,2] > 0, :]
    ax.plot(fog_points[:,1],fog_points[:,0],'.', color='#afbac1', markersize=2.1)

    # Island status
    ax.scatter(island_lons[category==1], island_lats[category==1], s=100, c='#1f77b4', marker='o', edgecolors='k', linewidths=0.5, label='Visible')  
    ax.scatter(island_lons[category==2], island_lats[category==2], s=100, c='#ff7f0e', marker='^', edgecolors='k', linewidths=0.5, label='Obscured')
    
    # Create unified legend including CALIPSO track and fog points
    legend_elements = [
        Line2D([0], [0], color='gold', lw=2, label='Footprints'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='#afbac1', markersize=15, label='Fog points'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markeredgecolor='k', markersize=8, label='Visible'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', markeredgecolor='k', markersize=8, label='Obscured')
    ]
    ax.legend(handles=legend_elements, loc='lower left', facecolor='white',
              edgecolor='black', framealpha=1.0, fontsize=16)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_height_image(heights, path, date_str, height_label='Fog-top height (m)'):
    """
    Plot a height image with a method-appropriate colorbar label.
    """
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    proj = ccrs.PlateCarree()  # Projection type

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=proj)

    # Use nonlinear color table to highlight 0-500m contrast, weaken 500-1000m contrast
    from matplotlib.colors import LinearSegmentedColormap
    # Improved color mapping: more distinct blue to white gradient
    colors = [
        '#001f4d',  # Very dark blue - 0m
        '#002c66',  # Dark blue
        '#003a80',  # Medium dark blue
        '#004799',  # Medium blue
        '#0055b3',  # Bright blue
        '#0066cc',  # Brighter blue
        '#1a75e6',  # Light blue
        '#3385ff',  # Pale blue
        '#4d94ff',  # Very pale blue
        '#66a3ff',  # Near-white blue 1
        '#80b3ff',  # Near-white blue 2
        '#99c2ff',  # Near-white blue 3
        '#b3d1ff',  # Near-white blue 4
        '#cce0ff',  # Near-white blue 5
        '#e6f0ff',  # Near-white blue 6
        '#ffffff',  # White - middle color
        '#ffebeb',  # Very pale pink
        '#ffd6d6',  # Pale pink
        '#ffc2c2',  # Light pink
        '#ffadad',  # Pink
        '#ff9999',  # Bright pink
        '#ff8585',  # Brighter red
        '#ff7070',  # Red
        '#ff5c5c',  # Dark red
        '#ff4747',  # Darker red
        '#ff3333',  # Very dark red
        '#e60000',  # Dark red
        '#cc0000',  # Dark deep red
        '#b30000',  # Dark darker red
        '#990000',  # Dark very dark red
        '#800000'   # Deep red - 1000m
    ]

    
    # Create discrete color table - each color corresponds to a height interval
    # Divide 0-1000m height range into 20 equally spaced intervals, each interval corresponds to a color
    bounds = np.linspace(0, 620, len(colors) + 1)
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(heights, cmap=cmap, norm=norm, extent=extent, origin='upper')
    ax.coastlines(color='white', lw=1)

    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([117, 119, 121, 123, 125, 127])
    gl.ylocator = mticker.FixedLocator([32,34, 36, 38, 40, 42, 44])
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 16} 
    gl.ylabel_style = {'size': 16} 
    
    # Add colorbar for continuous color scale
    cbar = fig.colorbar(im, fraction=0.045)
    cbar.set_label(height_label, fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    # Set tick labels, emphasize low height regions
    tick_labels = ['0', '60', '120', '180', '240', '300', '360', '420', '480', '540', '600']
    cbar.set_ticks(np.arange(0, 601, 60))
    cbar.set_ticklabels(tick_labels)
    cbar.set_ticklabels(tick_labels, fontsize=14)

    ax.text(0.97, 0.97, f'{date_str}', transform=ax.transAxes, fontsize=16,
        va='top', ha='right', bbox=dict(boxstyle="square,pad=0.3", facecolor="w", edgecolor='none'))

    
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_island_scatter(island_refs, island_h, category, y2_est, y3_est, save_path, date_str):
    """
    Plot enhanced scatter plot of island reflectance-height relationship with fitting curves, suitable for academic papers.
    Args:
        island_refs: Island reflectance array
        island_h: Island height array
        category: Island category array (1 unobscured, 2 obscured)
        y2_est, y3_est: Fitted heights for each method
        save_path: Image save path
        date_str: date string
    """
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)
    # Scatter style
    ax.scatter(island_refs[category==1], island_h[category==1], s=60, c='#1f77b4', marker='o', edgecolors='k', linewidths=0.5, label='Visible')
    ax.scatter(island_refs[category==2], island_h[category==2], s=60, c='#ff7f0e', marker='^', edgecolors='k', linewidths=0.5, label='Obscured')
    # Fitting curve style
    ax.plot(np.arange(0, 1, 0.1), y2_est, color='#2ca02c', linestyle='-', linewidth=2, label='LR')
    ax.plot(np.arange(0, 1, 0.1), y3_est, color='#d62728', linestyle='-', linewidth=2, label='SVM')
    # Axes and labels
    ax.set_xlabel('Reflectance', fontsize=14)
    ax.set_ylabel('Height (m)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 650)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    # Legend enhancement
    legend = ax.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=False, borderpad=1)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.9)
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.97, 0.97, f'{date_str}', transform=ax.transAxes, fontsize=14,
        va='top', ha='right', bbox=dict(boxstyle="square,pad=0.3", facecolor="w", edgecolor='none'))

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_all_island_scatters(all_case_data, save_path):
    """
    Plot the 49 case scatter plots in a fixed 7 x 7 layout.
    
    Args:
        all_case_data: List containing all case data, each element is a tuple
            (date_str, island_refs, island_h
        save_path: Image save path
    """
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    n_cases = len(all_case_data)
    
    # The selected validation set contains exactly 49 cases.
    nrows = 7
    ncols = 7    
    if n_cases > nrows * ncols:
        raise ValueError(
            f'The 7 x 7 layout supports at most 49 cases, got {n_cases}.'
        )
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.7, 6.2))
    axes = axes.flatten()

    # Draw one subplot for each case; no overall-regression panel is included.
    for i, (date_str, island_refs, island_h, category, y2_est, y3_est) in enumerate(all_case_data):
        subplot_idx = i
        if subplot_idx >= len(axes):
            break
            
        ax = axes[subplot_idx]
        
        # Scatter style - sample 10% of CALIPSO points for plotting
        ax.scatter(island_refs[category==1], island_h[category==1], 
                  s=4, c='#1f77b4', marker='o', edgecolors='k', linewidths=0.3, label='Visible islands')
        ax.scatter(island_refs[category==2], island_h[category==2], 
                  s=4, c='#ff7f0e', marker='^', edgecolors='k', linewidths=0.3, label='Obscured islands')
        
        # Fitting curve style
        ax.plot(np.arange(0, 1, 0.1), y2_est, color='#2ca02c', linestyle='-', linewidth=1.0, label='LR')
        ax.plot(np.arange(0, 1, 0.1), y3_est, color='#d62728', linestyle='-', linewidth=1.0, label='SVM')
        
        # Set axis range
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 650)
        
        # Set xtick to 0.2, 0.4, 0.6, 0.8
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        
        # Add title to inner top-right corner
        ax.text(0.97, 0.97, f'{date_str}', transform=ax.transAxes, fontsize=7,
                va='top', ha='right', bbox=dict(boxstyle="square,pad=0.3", facecolor="none", edgecolor='none'))
        
        # Only add labels to first column and bottom row
        if subplot_idx % ncols == 0:
            ax.set_ylabel('Height (m)', fontsize=7)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        if subplot_idx >= (nrows-1)*ncols:
            ax.set_xlabel('Reflectance', fontsize=7)
        else:
            ax.set_xticklabels([])
        
        # Reduce tick label and tick line size
        ax.tick_params(axis='both', which='major', labelsize=7, length=1, width=0.5)
        
        # Set border line width
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
    # Hide extra subplots
    for i in range(n_cases, len(axes)):
        axes[i].set_visible(False)
    
    # Create unified legend
    from matplotlib.lines import Line2D
    
    # Create legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markeredgecolor='k', markersize=4, label='Visible islands'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', markeredgecolor='k', markersize=4, label='Obscured islands'),
        Line2D([0], [0], color='#2ca02c', lw=1, label='LR'),
        Line2D([0], [0], color='#d62728', lw=1, label='SVM')
    ]
    
    # Add unified legend (display in one row, centered position)
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.025), ncol=4, fontsize=7, 
              frameon=True, fancybox=False, shadow=False)
    
    # Adjust layout to make space for bottom legend and compress image height by 30%
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.15, wspace=0.15)
    
    # Save figure
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_height_compare(calipso_sfth, modis_cth_track, predicted_sfth_track, save_path, date_str, step=20):
    """
    Plot comparison of fog top heights from different methods.
    
    Args:
        calipso_sfth: CALIPSO data, shape=(n,3)
        modis_cth_track: MODIS height, shape=(n,)
        predicted_sfth_track: Other method heights, shape=(n,3)
        save_path: Image save path
        step: Sampling step, default 20
        date_str: date string

    """
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    x = np.arange(0, len(calipso_sfth), step)
    ax.plot(x, calipso_sfth[::step,2], color='k', linewidth=1.5, label='Calipso')
    ax.plot(x, modis_cth_track[::step], color='#1f77b4', linewidth=1.5, label='Modis', alpha=0.8)
    ax.plot(x, predicted_sfth_track[::step, 0], color='#2ca02c', linewidth=1.5, label='LR', alpha=0.8)
    ax.plot(x, predicted_sfth_track[::step, 1], color='#d62728', linewidth=1.5, label='SVM', alpha=0.8)
    ax.legend(loc='upper left', fontsize=14, frameon=True)
    # Get slice ticks
    _, xticks, xticklabels = get_slice_xticks(
        calipso_sfth[:, 1], calipso_sfth[:, 0], ntick=5, decimals=1
    )
    ax.set_xlim([len(calipso_sfth), 0])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_ylabel('SFTH (m)', fontsize=14)
    ax.set_ylim([0, 1000])
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.97, 0.97, f'{date_str}', transform=ax.transAxes, fontsize=16,
        va='top', ha='right', bbox=dict(boxstyle="square,pad=0.3", facecolor="w", edgecolor='none'))

    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_metrics_heatmap(
    case_metrics_list,
    save_path="method_all_metrics_heatmap_grouped.png",
):
    """
    Plot metrics heatmap (function copied from main program)
    """
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    method_names = ['Modis', 'LR', 'SVM']
    case_dates = [c[0] for c in case_metrics_list]
    metric_arr = np.array([np.array(c[1:]).astype(float) for c in case_metrics_list])
    metric_arr = np.transpose(metric_arr, (1,2,0))

    def create_grouped_data(metric_index, metric_name):
        group_data = []
        group_labels = []
        for j, method in enumerate(method_names):
            group_data.append(metric_arr[metric_index][j])
            group_labels.append(f"{method}")
        return group_data, [metric_name], group_labels

    mae_data, mae_header, mae_labels = create_grouped_data(0, "MAE")
    rmse_data, rmse_header, rmse_labels = create_grouped_data(2, "RMSE")
    hr_data, hr_header, hr_labels = create_grouped_data(3, "HR")

    all_data = mae_data + rmse_data
    all_labels = mae_labels + rmse_labels
    all_headers = mae_header + rmse_header
    percent_data = hr_data
    percent_labels = hr_labels
    percent_headers = hr_header

    df_m = pd.DataFrame(all_data, index=all_labels, columns=[str(date) for date in case_dates])
    df_percent = pd.DataFrame(percent_data, index=percent_labels, columns=[str(date) for date in case_dates])

    nrows1 = 6
    nrows2 = 3

    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        # Keep each heatmap cell close to square when dozens of cases are
        # displayed across the fixed journal-column width.
        figsize=(6.7, 1.2),
        dpi=600,
        sharex=True,
        gridspec_kw={'height_ratios': [nrows1, nrows2]}
    )

    # Upper subplot: metric indicators
    for i, header in enumerate(all_headers):
        group_start = i * len(method_names)
        group_center = group_start + len(method_names) / 2
        ax1.text(-1.5, group_center, header,
                ha='right', va='center', fontsize=5, rotation=90, fontweight='bold')

    custom_cmap1 = sns.light_palette("#1f77b4", as_cmap=True, reverse=False)
    sns.heatmap(
        df_m, 
        annot=True, 
        fmt=".0f", 
        cmap=custom_cmap1, 
        ax=ax1, 
        vmin=0, 
        vmax=400,
        cbar=False,
        annot_kws={'fontsize': 3, 'color': 'black'},
        yticklabels=True
    )
    cbar1 = fig.colorbar(ax1.collections[0], ax=ax1, aspect=40, pad=0.01)
    cbar1.set_ticks([0, 100, 200, 300, 400])
    cbar1.set_ticklabels(["0m", "100m", "200m", "300m", "400m"])
    cbar1.ax.tick_params(labelsize=4, width=0.5, length=1.0, pad=0.5)
    cbar1.outline.set_linewidth(0.5)
    ax1.set_xticks([])
    ax1.set_xlabel('')
    ax1.spines['bottom'].set_visible(False)
    ax1.hlines([3, 6], *ax1.get_xlim(), color='black', linewidth=0.5)
    # Improve y-axis label display: reduce font size, increase spacing, method labels not rotated, metric labels rotated 90 degrees, method labels aligned to the right
    ax1.tick_params(axis='y', rotation=45, labelsize=4, pad=1, width=0.5, length=1.0)
    ax1.tick_params(axis='x', width=0, length=1.0, pad=0.5)
    # Set y-axis label top alignment and bold font
    plt.setp(ax1.get_yticklabels(), va='top')
    # Ensure all y-axis ticks and ticklabels are displayed, place yticks in the middle of each color block
    ax1.set_yticks([i + 0.5 for i in range(len(all_labels))])
    ax1.set_yticklabels(all_labels)

    # Lower subplot: percentage indicators
    ax2.text(-1.5, 1.5, 'HR', ha='right', va='center', fontsize=5, rotation=90, fontweight='bold') 

    custom_cmap2 = sns.light_palette("#1f77b4", as_cmap=True, reverse=True)
    sns.heatmap(
        df_percent, 
        annot=True, 
        fmt=".0f", 
        cmap=custom_cmap2, 
        ax=ax2, 
        vmin=0, 
        vmax=100,
        cbar=False,
        annot_kws={'fontsize': 3, 'color': 'black'},
        yticklabels=True
    )
    cbar2 = fig.colorbar(ax2.collections[0], ax=ax2, aspect=40*nrows2/nrows1, pad=0.01)
    cbar2.set_ticks([0, 25, 50, 75, 100])
    cbar2.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar2.ax.tick_params(labelsize=4, width=0.5, length=1.0, pad=0.5)
    cbar2.outline.set_linewidth(0.5)
    ax2.hlines([3], *ax2.get_xlim(), color='black', linewidth=0.5)
    # Improve y-axis label display: reduce font size, increase spacing, method labels not rotated
    ax2.tick_params(axis='y', rotation=45, labelsize=4, pad=1, width=0.5, length=1.0)
    # Display dense case labels vertically so adjacent dates do not overlap.
    # Seaborn uses ``auto`` label density by default and may hide every other
    # date when many cases are present.  Set every cell centre and label
    # explicitly so each case is identified.
    ax2.set_xticks(np.arange(len(case_dates)) + 0.5)
    ax2.set_xticklabels([str(date) for date in case_dates])
    plt.setp(
        ax2.get_xticklabels(),
        rotation=90,
        ha='center',
        va='top',
        fontsize=4,
    )
    ax2.tick_params(axis='x', width=0.5, length=1.0, pad=0.5)
    # Set y-axis label top alignment
    plt.setp(ax2.get_yticklabels(), va='top')
    # Ensure all y-axis ticks and ticklabels are displayed, place yticks in the middle of each color block
    ax2.set_yticks([i + 0.5 for i in range(len(percent_labels))])
    ax2.set_yticklabels(percent_labels)
    
    # Adjust overall layout to leave more space for y-axis labels
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.05, top=0.99, hspace=0.1)
    
    plt.savefig(save_path, format="png", bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.close()


def plot_bias_violin(df, method_colors):
    """
    Plot bias violin plot.

    Args:
    df (pandas.DataFrame): DataFrame containing data, must include "Method" and "Bias(m)" columns.
    method_colors (list): A list of colors used to distinguish violin plots for different methods.

    Returns:
    None
    """      
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    plt.figure(figsize=(3.35, 2.5), dpi=300)
    sns.violinplot(x="Method", y="Bias(m)", data=df, hue="Method", palette=method_colors, inner="box", linewidth=1, legend=False)
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    plt.xticks(np.arange(3), ['Modis', 'LR', 'SVM'], fontsize=8)
    plt.yticks(np.arange(-1000, 1001, 200), fontsize=8)
    plt.xlabel('Method', fontsize=8, fontweight='bold')
    plt.ylabel('Bias (m)', fontsize=8, fontweight='bold')
    plt.ylim([-1001, 1001])
    plt.tight_layout()
    plt.savefig('bias_violin.pdf')
    plt.close()


def plot_biase_violin_by_date(
    df,
    method_colors,
    save_path='bias_violin_by_date.pdf',
):
    """
    Plot bias violin plots by date.

    Args:
    df (pandas.DataFrame): DataFrame containing data, must include "Date", "Method" and "Bias(m)" columns.
    method_colors (list): A list of colors used to distinguish violin plots for different methods.

    Returns:
    None
    """  
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    # Plot violin plots for individual cases, and add overall error distribution in the first subplot    
    # df['Date'] = df['Date'].apply(lambda x: datetime.strptime(str(int(x)), '%Y%m%d').strftime('%Y-%m-%d'))
    df_total = df.copy()
    df_total['Date'] = 'Total'
    df_combined = pd.concat([df_total, df], ignore_index=True)
    
    # Get all unique dates
    unique_dates = df_combined['Date'].unique()
    
    # Use a fixed 5 x 10 journal layout: 49 cases plus the overall panel.
    n_dates = len(unique_dates)
    n_cols = 10
    n_rows = 5
    if n_dates > n_rows * n_cols:
        raise ValueError(
            f'The 5 x 10 layout supports at most 50 panels, got {n_dates}.'
        )
    
    # Create figure and subplots
    height = 4.5
    width = 6.7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), sharey=True)
    axes = axes.flatten() if n_dates > 1 else [axes]  
    
    # Create subplot for each date
    for i, date in enumerate(unique_dates):
        ax = axes[i]
        
        # Filter data for current date
        date_data = df_combined[df_combined['Date'] == date]
        
        # Draw violin plot, showing median and quartile statistics
        sns.violinplot(data=date_data, x="Method", y="Bias(m)", ax=ax, hue="Method", palette=method_colors, 
                      width=0.9, linewidth=0.3, inner="box", legend=False)
        
        
        # Add reference line
        ax.axhline(y=0, linestyle=":", linewidth=0.5, color='gray')
        
        # Set axes
        ax.set_xlim(-.5, 2.5)  
        ax.set_ylim(-500, 500)
        ax.set_xticks(np.arange(3))  
        ax.set_yticks(np.arange(-500, 501, 200))
        
        # Set title (date)
        ax.set_title(date, fontsize=6, pad=3)
        # ax.text(0.97, 0.97, date, transform=ax.transAxes, fontsize=6,
        #         va='top', ha='right', bbox=dict(boxstyle="square,pad=0.3", facecolor="none", edgecolor='none'))
        
        # Set x and y axis tick line width to 0.5, label font size to 6
        ax.tick_params(axis='x', width=0.5, labelsize=6, rotation=45, length=1.5)
        ax.tick_params(axis='y', width=0.5, labelsize=6, length=1.5)
        
        # Handle y-axis labels: only show y-axis labels in first column
        if i % n_cols == 0:  # First column
            ax.set_ylabel('Bias (m)', fontsize=6)
            ax.set_yticklabels(['-500', '-300', '-100', '100', '300', '500'])
        else:  # Not first column
            ax.set_ylabel('')
            # Don't set yticklabels to empty, but hide labels
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_visible(False)
        
        # Handle x-axis labels: only show x-axis labels in last row, but don't show "Method" label
        if i >= n_dates - n_cols:  # Last row
            ax.set_xticklabels(['Modis', 'LR', 'SVM'], fontsize=6, ha='right')  # Change to 3 method labels
            # Don't show xlabel
            ax.set_xlabel('')
        else:  # Not last row
            ax.set_xlabel('')
            ax.set_xticklabels([])
            # Ensure no x-axis tick labels are displayed
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
        
        # Ensure all four borders of subplots are displayed
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        # Set border line width
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    
    # Hide extra subplots
    for i in range(n_dates, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout to make horizontal and vertical spacing equal
    plt.subplots_adjust(wspace=0.1, hspace=0.2*width/height, left=0.06, right=0.99, top=0.96, bottom=0.08)
    plt.savefig(save_path)
    plt.close()


def plot_all_tracks_subplots(all_track_data, save_path, island_lons, island_lats, island_heights):
    """
    Plot all case track diagrams as subplots in the same figure
    
    Parameters:
        all_track_data: List containing all case data, each element is a dictionary returned by collect_track_data
        save_path: Image save path
    """
    # Apply global font settings
    plt.rcParams.update(plt_rcParams)
    
    n_cases = len(all_track_data)
    
    # Calculate subplot grid layout (5x7 suitable for 35 subplots, including island distribution plot)
    nrows = 7
    ncols = 7
    if n_cases > nrows * ncols:
        raise ValueError(
            f'The 7 x 7 layout supports at most 49 cases, got {n_cases}.'
        )
    
    # Create large figure, adjusted to size suitable for academic papers
    fig = plt.figure(figsize=(6.7, 6.7))
    
    # Draw all island geographic distribution and height information in the first row and first column
    extent = [min_lon, max_lon, min_lat, max_lat]
    proj = ccrs.PlateCarree()
    ax = plt.subplot(nrows, ncols, 1, projection=proj)
    ax.set_extent(extent, crs=proj)
    
    # Add coastline and geographic features
    ax.coastlines(color='black', lw=0.5)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m"),
                   ec="black", fc="lightgray", lw=0.3)
    
    # Add grid lines
    
    # Set correct longitude and latitude tick values
    ax.set_xticks([119, 121, 123, 125, 127])
    ax.set_yticks([32, 34, 36, 38, 40])
    # First subplot does not show X-axis labels and ticks
    ax.set_xticklabels(['', '', '', '', ''])  # Do not display X-axis labels
    ax.set_yticklabels(['32°', '34°', '36°', '38°', '40°'], fontsize=8)
    # Set tick length shorter
    ax.tick_params(axis='both', which='major', length=2)

    
    # Draw all islands, using color to represent height
    scatter = ax.scatter(island_lons, island_lats, c=island_heights, 
                        s=20, cmap='jet', edgecolors='black', linewidths=0.5, 
                        vmin=0, vmax=max(island_heights))
    
    # Add title
    ax.text(0.97, 0.97, 'Islands Distribution', transform=ax.transAxes, fontsize=8, 
            va='top', ha='right', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'))

    # The distribution panel is no longer part of the combined figure.
    fig.delaxes(ax)

    
    # Adjust first subplot size to be consistent with other subplots - remove colorbar
    # Colorbar will be displayed side by side in the legend area
    
    # Draw one subplot for each case, starting from the first grid cell.
    for i, track_data in enumerate(all_track_data):
        subplot_idx = i
        if subplot_idx >= nrows * ncols:
            break
            
        date_str = track_data['date_str']
        island_lats = track_data['island_lats']
        island_lons = track_data['island_lons']
        category = track_data['category']
        calipso_cth = track_data['calipso_cth']
        img = track_data['img']
        
        # Set projection and extent
        extent = [min_lon, max_lon, min_lat, max_lat]
        proj = ccrs.PlateCarree()
        ax = plt.subplot(nrows, ncols, subplot_idx+1, projection=proj)
        ax.set_extent(extent, crs=proj)
        
        # Display image
        ax.imshow(img, extent=extent, origin='upper')
        
        # Completely disable Cartopy's automatic grid lines
        gl = ax.gridlines(draw_labels=False, color='none', alpha=0)
        gl.xlines = False
        gl.ylines = False
        
        # Use matplotlib's default tick labels instead of cartopy's gridlines
        # Set correct longitude and latitude tick values to avoid incorrect labels like 0, 0.5, 0.1
        ax.set_xticks([119, 121, 123, 125, 127])
        ax.set_yticks([32, 34, 36, 38, 40])
        
        # Only show Y-axis ticks and labels in the first column, only show X-axis ticks and labels in the last row
        if (subplot_idx % ncols) == 0:  # First column
            ax.set_yticklabels(['32°', '34°', '36°', '38°', '40°'], fontsize=8)
        else:
            ax.set_yticks([])  # Other columns show neither Y ticks nor labels

        # On the left column, display every other latitude tick.
        if (subplot_idx % ncols) == 0:
            ax.set_yticks([32, 36, 40])
            ax.set_yticklabels(['32°', '36°', '40°'], fontsize=8)
        
        if subplot_idx >= (nrows-1)*ncols:  # Last row
            ax.set_xticklabels(['119°', '121°', '123°', '125°', '127°'], fontsize=8)
        else:
            ax.set_xticks([])  # Other rows show neither X ticks nor labels

        # On the bottom row, display every other longitude tick.
        if subplot_idx >= (nrows-1)*ncols:
            ax.set_xticks([119, 123, 127])
            ax.set_xticklabels(['119°', '123°', '127°'], fontsize=8)
            
        # Set tick length shorter
        ax.tick_params(axis='both', which='major', length=2)
        
        # Draw CALIPSO track
        ax.plot(calipso_cth[:,1], calipso_cth[:,0], color='gold', linewidth=1.0)
        
        # Draw fog points
        fog_points = calipso_cth[calipso_cth[:,2] > 0, :]
        ax.plot(fog_points[:,1], fog_points[:,0], '.', color='#afbac1', markersize=1.0)
        
        # Draw island status
        ax.scatter(island_lons[category==1], island_lats[category==1], 
                  s=12, c='#1f77b4', marker='o', edgecolors='k', linewidths=0.2, label='Visible')
        ax.scatter(island_lons[category==2], island_lats[category==2], 
                  s=12, c='#ff7f0e', marker='^', edgecolors='k', linewidths=0.2, label='Obscured')
        
        # Put the date in the lower-left corner to avoid the upper track area.
        ax.text(0.03, 0.03, f'{date_str}', transform=ax.transAxes, fontsize=8,
                va='bottom', ha='left',
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor='none'))
    
    # Hide extra subplots
    for subplot_idx in range(n_cases, nrows * ncols):
        empty_ax = fig.add_subplot(nrows, ncols, subplot_idx + 1)
        empty_ax.set_visible(False)
    
    # Create unified legend and colorbar, placed in lower left and lower right corners respectively
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='gold', lw=2, label='CALIPSO footprints'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='#afbac1', markersize=10, label='Fog points'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markeredgecolor='k', markersize=5, label='Visible islands'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', markeredgecolor='k', markersize=5, label='Obscured islands')
    ]

    # Create colorbar element
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=max(island_heights)))
    sm.set_array([])
    
    # Add colorbar to the inner top of the first subplot (island distribution plot)
    # Get position information of the first subplot (island distribution plot)
    island_ax = plt.gcf().get_axes()[0]  # Get the first created cartopy subplot
    island_ax_pos = island_ax.get_position()
    # Adjust colorbar position and size to make it more slender and adjust to the left
    cbar_ax = fig.add_axes([island_ax_pos.x0 - 0.048, island_ax_pos.y0 + island_ax_pos.height - 0.015, island_ax_pos.width * 1.0, 0.01])  # Inner top of first subplot, more slender, adjusted upward and to the left
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # Remove title
    # cbar.set_label('Height (m)', fontsize=6)
    # Place colorbar ticks and labels on top
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=5)
    # Adjust colorbar border and tick width, and label distance
    cbar.outline.set_linewidth(0.5)  # Border width
    cbar.ax.tick_params(width=0.5, length=1)   # Tick line width and length
    cbar.ax.tick_params(pad=1)  # Distance between tick labels and axis
    # Add unit 'm' to each tick label, use fixed ticks to avoid warnings
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{int(tick)}m" for tick in ticks])  # Then set labels
    # Add background to the entire colorbar to avoid interference from the background
    cbar_ax.set_facecolor('white')        # Background color white
    cbar_ax.patch.set_edgecolor('black')  # Border color
    cbar_ax.patch.set_linewidth(0.5)      # Border width
    cbar_ax.patch.set_alpha(1.0)          # Set opacity to 1 to ensure background is fully visible
    # Add a rectangular background box
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 1, 1, transform=cbar_ax.transAxes, facecolor='white', edgecolor='black', linewidth=0.5, alpha=1.0, zorder=-1)
    cbar_ax.add_patch(rect)
    # The height colorbar belonged to the removed island-distribution panel.
    cbar_ax.set_visible(False)
    
    # Add legend to center position
    fig.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=8,
              frameon=True, fancybox=True, shadow=False)

    # Adjust layout, increase bottom space to accommodate legend and colorbar, make subplot spacing more uniform
    plt.subplots_adjust(bottom=0.085, hspace=0.05, wspace=0.05)
    
    # Save figure
    fig.savefig(save_path, bbox_inches='tight', dpi=450)
    plt.close(fig)
