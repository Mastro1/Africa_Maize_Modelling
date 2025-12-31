import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import imageio.v2 as imageio

# --- BRANDING CONFIGURATION ---
ASR_NAVY = '#262262'   
ASR_GOLD = '#FDB913'   
ASR_GREY = '#E0E0E0'   
FONT_FAMILY = 'sans-serif' 

# Set global plot styles
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['text.color'] = ASR_NAVY
plt.rcParams['axes.labelcolor'] = ASR_NAVY
plt.rcParams['xtick.color'] = ASR_NAVY
plt.rcParams['ytick.color'] = ASR_NAVY

PROJECT_ROOT = os.getcwd()

# Read the predictions data
predictions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Model', 'africa_results', 'all_africa_maize_yield_predictions.csv'))
predictions_df = predictions_df[predictions_df['season_index'] == 1].copy()

# Read shapefiles
admin2_gdf = gpd.read_file(os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_2_processed.shp'))
admin1_gdf = gpd.read_file(os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_1_processed.shp'))

# Find countries that are in admin1 but not in admin2
admin1_countries = set(admin1_gdf['ADMIN0'].unique())
admin2_countries = set(admin2_gdf['ADMIN0'].unique())
admin1_only_countries = admin1_countries - admin2_countries
background_admin1_only_gdf = admin1_gdf[admin1_gdf['ADMIN0'].isin(admin1_only_countries)]

# Function to calculate detrended values
def calculate_detrended_values(df):
    detrended_data = []
    for pcode in df['PCODE'].unique():
        location_data = df[df['PCODE'] == pcode].copy()
        if len(location_data) < 3:
            continue
        X = location_data['year'].values.reshape(-1, 1)
        y = location_data['pred_yield'].values
        reg = LinearRegression().fit(X, y)
        trend_values = reg.predict(X)
        detrended = (y - trend_values) / trend_values
        location_detrended = location_data.copy()
        location_detrended['detrended_yield'] = detrended
        detrended_data.append(location_detrended)
    return pd.concat(detrended_data, ignore_index=True)

detrended_df = calculate_detrended_values(predictions_df)

admin1_pcodes = set(admin1_gdf['FNID'])
admin2_pcodes = set(admin2_gdf['FNID'])

# --- COLORMAP ---
detrend_colors = ['#8b0000', '#d73027', '#f46d43', '#fdae61', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006400']
detrend_cmap = LinearSegmentedColormap.from_list('redgreen', detrend_colors, N=256)

detrend_values = detrended_df['detrended_yield'].dropna().values
detrend_vmin, detrend_vmax = np.percentile(detrend_values, [5, 95])
detrend_abs_max = max(abs(detrend_vmin), abs(detrend_vmax))
detrend_vmin, detrend_vmax = -detrend_abs_max, detrend_abs_max

out_dir = os.path.join(PROJECT_ROOT, 'Plots', 'output')
gif_frames_dir = os.path.join(out_dir, 'gif_frames')
os.makedirs(gif_frames_dir, exist_ok=True)

# --- PLOTTING FUNCTION ---
def plot_single_year(year, data, value_column, cmap, vmin, vmax):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10)) 
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 1. Background Map 
    admin2_gdf.plot(ax=ax, color=ASR_GREY, edgecolor='none')
    if not background_admin1_only_gdf.empty:
        background_admin1_only_gdf.plot(ax=ax, color=ASR_GREY, edgecolor='none')

    # 2. Data Plotting
    plot_params = {
        'edgecolor': 'none', 
        'cmap': cmap
    }

    data_admin1 = data[data['PCODE'].isin(admin1_pcodes)]
    data_admin2 = data[data['PCODE'].isin(admin2_pcodes)]

    merged_admin1 = admin1_gdf.merge(data_admin1, left_on='FNID', right_on='PCODE', how='inner')
    merged_admin2 = admin2_gdf.merge(data_admin2, left_on='FNID', right_on='PCODE', how='inner')

    if not merged_admin2.empty:
        merged_admin2.plot(ax=ax, column=value_column, vmin=vmin, vmax=vmax, **plot_params)
    if not merged_admin1.empty:
        merged_admin1.plot(ax=ax, column=value_column, vmin=vmin, vmax=vmax, **plot_params)

    # 3. BOUNDARIES OVERLAY (UPDATED)
    # Changed linewidth from 0.6 to 0.25 for a finer look
    admin1_gdf.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=0.25)

    # 4. Titles and Branding
    plt.text(-18, 42, "Maize Yield Anomalies", 
             fontsize=24, fontweight='bold', color=ASR_NAVY, ha='left')
    plt.text(-18, 39, "Model Results", 
             fontsize=14, color=ASR_NAVY, alpha=0.7, ha='left')
    plt.text(52, 38, f"{year}", 
             fontsize=45, fontweight='bold', color=ASR_GOLD, ha='right')

    # UPDATED LIMITS: Lowered bottom limit from -35 to -38 to show South Africa fully
    ax.set_xlim(-20, 55)
    ax.set_ylim(-38, 45) 
    ax.set_aspect('equal')
    ax.axis('off')

    # 5. Colorbar
    cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.set_label('Yield Anomaly (Detrended)', fontsize=12, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=10, size=0)

    return fig

# --- EXECUTION ---
available_years = sorted(detrended_df['year'].unique())
frame_files = []

print(f"Generating frames for {len(available_years)} years...")

for year in available_years:
    year_data = detrended_df[detrended_df['year'] == year][['PCODE', 'country', 'detrended_yield']].copy()
    if year_data.empty: continue

    fig = plot_single_year(year, year_data, 'detrended_yield', detrend_cmap, detrend_vmin, detrend_vmax)
    
    frame_filename = os.path.join(gif_frames_dir, f'frame_{year}.png')
    fig.savefig(frame_filename, dpi=150, bbox_inches='tight', facecolor='white')
    frame_files.append(frame_filename)
    plt.close(fig)

print("Creating GIF...")
gif_filename = os.path.join(out_dir, 'asr_branded_yield_animation_v3.gif')

frames = [imageio.imread(f) for f in frame_files]
imageio.mimsave(gif_filename, frames, duration=1000, loop=0)

print(f"Done! GIF saved to: {gif_filename}")