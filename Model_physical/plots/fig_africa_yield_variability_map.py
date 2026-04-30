import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import scienceplots
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter
from statsmodels.nonparametric.kernel_regression import KernelReg

# Set global scientific style
plt.style.use(['science', 'ieee'])

PROJECT_ROOT = os.getcwd()

def smooth_data_kernel_regression(arr: np.array, years: np.array) -> np.ndarray:
    if len(arr) < 3: return arr
    try:
        kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[3])
        return kr.fit()[0]
    except:
        return arr

def calculate_detrended_values(df):
    detrended_data = []
    for pcode in df['PCODE'].unique():
        loc_data = df[df['PCODE'] == pcode].sort_values('Year').copy()
        if len(loc_data) < 3: continue
        years = loc_data['Year'].values
        yields = loc_data['Yield_Estimated_t_ha'].values
        trend_values = smooth_data_kernel_regression(yields, years)
        detrended = (yields - trend_values) / trend_values
        loc_data['detrended_yield'] = detrended
        detrended_data.append(loc_data)
    return pd.concat(detrended_data, ignore_index=True)

def format_lon(x, pos):
    if x > 0: return f'{x:g}°E'
    if x < 0: return f'{abs(x):g}°W'
    return f'{x:g}°'

def format_lat(y, pos):
    if y > 0: return f'{y:g}°N'
    if y < 0: return f'{abs(y):g}°S'
    return f'{y:g}°'

def main():
    # 1. Load RAW Data
    data_path = os.path.join(PROJECT_ROOT, 'Model_physical', 'Results', 'V7_model', 'Global_Maize_Yield_V7.csv')
    df = pd.read_csv(data_path)
    
    if 'Season' in df.columns:
        df = df[df['Season'] == 1].copy()
        
    # Remove first and last year from the dataset before detrending
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df = df[~df['Year'].isin([min_year, max_year])].copy()
        
    print("Calculating kernel detrended values on cleaned data...")
    df = calculate_detrended_values(df)

    print("Calculating yield variability (standard deviation of detrended yield)...")
    variability_df = df.groupby('PCODE')['detrended_yield'].std().reset_index()
    variability_df.rename(columns={'detrended_yield': 'yield_variability'}, inplace=True)
    variability_df['yield_variability'] = variability_df['yield_variability'].fillna(0)
    variability_df = variability_df[variability_df['yield_variability'] > 0]
    
    # 2. Load Shapefiles
    admin2_path = os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_2_processed.shp')
    admin1_path = os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_1_processed.shp')
    
    print("Loading shapefiles...")
    admin2_gdf = gpd.read_file(admin2_path)
    admin1_gdf = gpd.read_file(admin1_path)
    
    admin1_pcodes = set(admin1_gdf['FNID'])
    admin2_pcodes = set(admin2_gdf['FNID'])
    africa_outline = admin1_gdf.dissolve()

    data_admin1 = variability_df[variability_df['PCODE'].isin(admin1_pcodes)]
    data_admin2 = variability_df[variability_df['PCODE'].isin(admin2_pcodes)]

    # 3. Figure Setup
    width = 7.0
    height = width * 1.0 
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Define Colormap (Blue to Red)
    cmap = plt.cm.OrRd 
    
    vmax = variability_df['yield_variability'].quantile(0.98)
    vmin = 0.0
    if np.isnan(vmax) or vmax == 0: vmax = 1.0

    print("Plotting map...")
    africa_outline.plot(ax=ax, color='#f0f0f0', edgecolor='none')
    admin2_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.05)
    
    if not data_admin2.empty:
        merged2 = admin2_gdf.merge(data_admin2, left_on='FNID', right_on='PCODE')
        merged2.plot(ax=ax, column='yield_variability', cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
        
    if not data_admin1.empty:
        merged1 = admin1_gdf.merge(data_admin1, left_on='FNID', right_on='PCODE')
        merged1.plot(ax=ax, column='yield_variability', cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor('black')

    ax.set_xlim(-20, 55)
    ax.set_ylim(-35, 40)
    
    lon_ticks = np.arange(-20, 61, 20)
    lat_ticks = np.arange(-30, 41, 15)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    
    ax.tick_params(axis='both', which='major', labelsize=10, direction='in', 
                   length=4, width=0.8, colors='black',
                   top=True, right=True, bottom=True, left=True,
                   labeltop=False, labelleft=True, 
                   labelbottom=True, labelright=False)
    
    ax.grid(True, linestyle='--', linewidth=0.3, color='#cccccc', alpha=0.5, zorder=0)
    ax.set_title('Maize Yield Variability', fontsize=12, fontweight='bold', pad=15)

    cax = fig.add_axes([0.25, 0.08, 0.5, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    
    cbar.ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    cbar.set_label('Yield Variability (Std. Dev. of Relative Anomaly)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.subplots_adjust(bottom=0.15)
    
    out_dir = os.path.join(PROJECT_ROOT, 'Model_physical', 'plots', 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    save_path_pdf = os.path.join(out_dir, 'africa_yield_variability_map.pdf')
    save_path_png = os.path.join(out_dir, 'africa_yield_variability_map.png')
    
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Figures saved to {out_dir}")

if __name__ == "__main__":
    main()
