import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import imageio.v2 as imageio
import glob

PROJECT_ROOT = os.getcwd()

# Read the predictions data
predictions_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'Model', 'africa_results', 'all_africa_maize_yield_predictions.csv'))

# Filter for season1 only
predictions_df = predictions_df[predictions_df['season_index'] == 1].copy()

# Read shapefiles
admin2_gdf = gpd.read_file(os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_2_processed.shp'))
admin1_gdf = gpd.read_file(os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_1_processed.shp'))

# Find countries that are in admin1 but not in admin2 to ensure a complete background
admin1_countries = set(admin1_gdf['ADMIN0'].unique())
admin2_countries = set(admin2_gdf['ADMIN0'].unique())
admin1_only_countries = admin1_countries - admin2_countries
background_admin1_only_gdf = admin1_gdf[admin1_gdf['ADMIN0'].isin(admin1_only_countries)]

# Function to calculate detrended values
def calculate_detrended_values(df):
    """
    Calculate detrended values for each location using linear regression
    Returns (yield - trend) / trend for each location and year
    """
    detrended_data = []

    for pcode in df['PCODE'].unique():
        location_data = df[df['PCODE'] == pcode].copy()

        if len(location_data) < 3:  # Need at least 3 points for meaningful trend
            continue

        # Fit linear regression
        X = location_data['year'].values.reshape(-1, 1)
        y = location_data['pred_yield'].values

        reg = LinearRegression().fit(X, y)
        trend_values = reg.predict(X)

        # Calculate detrended values: (yield - trend) / trend
        detrended = (y - trend_values) / trend_values

        # Add to results
        location_detrended = location_data.copy()
        location_detrended['detrended_yield'] = detrended
        location_detrended['trend_yield'] = trend_values
        detrended_data.append(location_detrended)

    return pd.concat(detrended_data, ignore_index=True)

# Calculate detrended values
detrended_df = calculate_detrended_values(predictions_df)

# Define which countries use admin1 vs admin2
ADMIN1_ONLY_COUNTRIES = ['Lesotho', 'Nigeria']

# Get sets of PCODEs for each admin level from the shapefiles
admin1_pcodes = set(admin1_gdf['FNID'])
admin2_pcodes = set(admin2_gdf['FNID'])

# Define colormaps
# Red to green colormap for detrended values - more saturated for small ranges
detrend_colors = ['#8b0000', '#d73027', '#f46d43', '#fdae61', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006400']
detrend_cmap = LinearSegmentedColormap.from_list('redgreen', detrend_colors, N=256)

# Calculate color scale ranges for all detrended values
detrend_values = detrended_df['detrended_yield'].dropna().values
detrend_vmin, detrend_vmax = np.percentile(detrend_values, [5, 95])  # Tighter range for better contrast
detrend_abs_max = max(abs(detrend_vmin), abs(detrend_vmax))
detrend_vmin, detrend_vmax = -detrend_abs_max, detrend_abs_max

# Create output directory
out_dir = os.path.join(PROJECT_ROOT, 'Plots', 'output')
gif_frames_dir = os.path.join(out_dir, 'gif_frames')
os.makedirs(gif_frames_dir, exist_ok=True)

# Function to plot single year data
def plot_single_year(year, data, value_column, cmap, vmin, vmax):
    """
    Create a single plot for one year
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f'Maize Yield Anomalies (Model Generated) - {year}', fontsize=16, fontname='Times New Roman')

    # Common plotting parameters
    plot_params = {
        'edgecolor': 'white',
        'linewidth': 0.1,
        'cmap': cmap
    }

    # 1. Plot the entire Admin-2 shapefile as a grey background
    admin2_gdf.plot(ax=ax, color='#d3d3d3', edgecolor='white', linewidth=0.1)

    # 1b. Plot the missing admin-1 countries for a complete background
    if not background_admin1_only_gdf.empty:
        background_admin1_only_gdf.plot(ax=ax, color='#d3d3d3', edgecolor='white', linewidth=0.1)

    # 2. Separate data into admin levels and merge with corresponding shapefiles
    data_admin1 = data[data['PCODE'].isin(admin1_pcodes)]
    data_admin2 = data[data['PCODE'].isin(admin2_pcodes)]

    merged_admin1 = admin1_gdf.merge(data_admin1, left_on='FNID', right_on='PCODE', how='inner')
    merged_admin2 = admin2_gdf.merge(data_admin2, left_on='FNID', right_on='PCODE', how='inner')

    # 3. Plot Admin-2 data on top of the background
    if not merged_admin2.empty:
        merged_admin2.plot(ax=ax, column=value_column, vmin=vmin, vmax=vmax, **plot_params)

    # 4. Plot Admin-1 data on top of everything
    if not merged_admin1.empty:
        merged_admin1.plot(ax=ax, column=value_column, vmin=vmin, vmax=vmax, **plot_params)

    #ax.set_title(f'{year} Yield Anomaly', fontsize=14, fontname='Times New Roman', pad=20)
    ax.set_xlim(-20, 55)
    ax.set_ylim(-35, 40)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add a square border around the subplot
    from matplotlib.patches import Rectangle
    border = Rectangle((-20, -35), 75, 75, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(border)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Yield Anomaly (Detrended)', fontsize=12, fontname='Times New Roman')
    cbar.ax.tick_params(labelsize=10)

    # Adjust layout
    plt.subplots_adjust(top=0.9, bottom=0.2)

    return fig

# Get all available years
available_years = sorted(detrended_df['year'].unique())
print(f"Creating GIF with {len(available_years)} frames for years: {available_years}")

# Create individual frames for each year
frame_files = []
for year in available_years:
    print(f"Processing year {year}...")

    # Filter data for this year
    year_data = detrended_df[detrended_df['year'] == year][['PCODE', 'country', 'detrended_yield']].copy()

    if year_data.empty:
        print(f"No data for year {year}, skipping...")
        continue

    # Create the plot
    fig = plot_single_year(year, year_data, 'detrended_yield', detrend_cmap, detrend_vmin, detrend_vmax)

    # Save the frame
    frame_filename = os.path.join(gif_frames_dir, f'frame_{year}.png')
    fig.savefig(frame_filename, dpi=150, bbox_inches='tight', facecolor='white')
    frame_files.append(frame_filename)

    # Close the figure to free memory
    plt.close(fig)

print(f"Created {len(frame_files)} frame images")

# Create GIF
print("Creating GIF animation...")
gif_filename = os.path.join(out_dir, 'africa_yield_anomalies_animation.gif')

# Read all frames
frames = []
for frame_file in frame_files:
    frames.append(imageio.imread(frame_file))

# Save as GIF with 1 second per frame (1000 ms)
imageio.mimsave(gif_filename, frames, duration=1000, loop=0)

print(f"GIF animation saved as: {gif_filename}")
print(f"Individual frames saved in: {gif_frames_dir}")

print("\nData summary:")
print(f"- Total predictions: {len(predictions_df)}")
print(f"- Unique locations: {predictions_df['PCODE'].nunique()}")
print(f"- Year range: {predictions_df['year'].min()} - {predictions_df['year'].max()}")
print(f"- Yield range: {predictions_df['pred_yield'].min():.3f} - {predictions_df['pred_yield'].max():.3f}")
print(f"- GIF frames: {len(frame_files)}")
print(f"- Duration: {len(frame_files)} seconds")
