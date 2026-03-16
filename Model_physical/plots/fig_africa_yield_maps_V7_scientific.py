import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import scienceplots
from statsmodels.nonparametric.kernel_regression import KernelReg
from matplotlib.patches import Rectangle

# Set global scientific style
plt.style.use(['science', 'ieee'])

PROJECT_ROOT = os.getcwd()

def smooth_data_kernel_regression(arr: np.array, years: np.array) -> np.ndarray:
    """
    Smooth the data using kernel regression with a bandwidth of 3.
    Matches the logic in verify_fao.py
    """
    if len(arr) < 3:
        return arr
    try:
        kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[3])
        smoothed_data = kr.fit()[0]
        return smoothed_data
    except:
        return arr

def calculate_detrended_values(df):
    """
    Calculate detrended values for each location using kernel regression
    Returns (yield - trend) / trend for each location and year
    """
    detrended_data = []
    
    unique_pcodes = df['PCODE'].unique()
    for pcode in unique_pcodes:
        location_data = df[df['PCODE'] == pcode].sort_values('Year').copy()
        
        if len(location_data) < 3:
            continue
            
        years = location_data['Year'].values
        yields = location_data['Yield_Estimated_t_ha'].values
        
        # Kernel trend
        trend_values = smooth_data_kernel_regression(yields, years)
        
        # Calculate detrended values: (yield - trend) / trend
        detrended = (yields - trend_values) / trend_values
        
        location_detrended = location_data.copy()
        location_detrended['detrended_yield'] = detrended
        detrended_data.append(location_detrended)
    
    return pd.concat(detrended_data, ignore_index=True)

def main():
    # 1. Load Data
    data_path = os.path.join(PROJECT_ROOT, 'Model_physical', 'Results', 'V7_model', 'Global_Maize_Yield_V7.csv')
    df = pd.read_csv(data_path)
    
    # Filter for season 1
    df = df[df['Season'] == 1].copy()
    
    print("Calculating kernel detrended values...")
    detrended_df = calculate_detrended_values(df)
    
    # 2. Load Shapefiles
    admin2_gdf = gpd.read_file(os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_2_processed.shp'))
    admin1_gdf = gpd.read_file(os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_1_processed.shp'))
    
    # Define sets for merging
    admin1_pcodes = set(admin1_gdf['FNID'])
    admin2_pcodes = set(admin2_gdf['FNID'])
    
    # Background for missing data
    africa_outline = admin1_gdf.dissolve()

    # 3. Figure Setup (IEEE Double Column Width)
    width = 7.0
    height = 7.0  # Square-ish for 2x2 maps
    fig, axes = plt.subplots(2, 2, figsize=(width, height), sharex=True, sharey=True)
    
    target_years = [2005, 2010, 2015, 2020]
    
    # Define Colormap (Divergent for anomalies)
    cmap = plt.cm.RdYlGn  # Red (low) to Green (high)
    
    # Global scale across selected years
    subset_years = detrended_df[detrended_df['Year'].isin(target_years)]
    vmax = subset_years['detrended_yield'].abs().quantile(0.98)
    vmin = -vmax

    for i, year in enumerate(target_years):
        ax = axes.flatten()[i]
        
        year_data = detrended_df[detrended_df['Year'] == year].copy()
        
        # Split by admin level for mapping
        data_admin1 = year_data[year_data['PCODE'].isin(admin1_pcodes)]
        data_admin2 = year_data[year_data['PCODE'].isin(admin2_pcodes)]
        
        # Plot background
        africa_outline.plot(ax=ax, color='#f0f0f0', edgecolor='none')
        admin2_gdf.plot(ax=ax, color='#d3d3d3', edgecolor='white', linewidth=0.05)
        
        # Plot data
        if not data_admin2.empty:
            merged2 = admin2_gdf.merge(data_admin2, left_on='FNID', right_on='PCODE')
            merged2.plot(ax=ax, column='detrended_yield', cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
            
        if not data_admin1.empty:
            merged1 = admin1_gdf.merge(data_admin1, left_on='FNID', right_on='PCODE')
            merged1.plot(ax=ax, column='detrended_yield', cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
        
        # 1. Spines and Frame
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')

        # 2. Coordinates (Latitude and Longitude)
        ax.set_xlim(-20, 55)
        ax.set_ylim(-35, 40)
        
        # Define tick intervals
        lon_ticks = np.arange(-20, 61, 20)
        lat_ticks = np.arange(-30, 41, 15)
        
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        
        from matplotlib.ticker import FuncFormatter

        def format_lon(x, pos):
            if x > 0: return f'{x:g}°E'
            if x < 0: return f'{abs(x):g}°W'
            return f'{x:g}°'

        def format_lat(y, pos):
            if y > 0: return f'{y:g}°N'
            if y < 0: return f'{abs(y):g}°S'
            return f'{y:g}°'

        ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
        
        # Show ticks on specific sides according to user request
        # 0,0: Top & Left | 0,1: Top | 1,0: Left | 1,1: None
        has_top = (i < 2)
        has_left = (i % 2 == 0)
        
        ax.tick_params(axis='both', which='major', labelsize=8, direction='in', 
                       length=3, width=0.8, colors='black',
                       top=True, right=True, bottom=True, left=True,
                       labeltop=has_top, labelleft=has_left, 
                       labelbottom=False, labelright=False)
        
        # 3. Gridlines
        ax.grid(True, linestyle='--', linewidth=0.3, color='#cccccc', alpha=0.5, zorder=0)

        # 4. Title/Year label below map
        ax.text(0.5, -0.1, f'{year} Detrended Yield Anomaly', transform=ax.transAxes, 
                ha='center', va='top', fontsize=9, fontweight='bold')

    # Colorbar
    cax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Yield Anomaly (Fractional Change)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(wspace=0.1, hspace=0.2, bottom=0.15)
    
    # Save
    out_dir = os.path.join(PROJECT_ROOT, 'Model_physical','plots', 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    save_path_pdf = os.path.join(out_dir, 'africa_yield_maps_V7_scientific.pdf')
    save_path_png = os.path.join(out_dir, 'africa_yield_maps_V7_scientific.png')
    
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Figures saved to {out_dir}")

if __name__ == "__main__":
    main()
