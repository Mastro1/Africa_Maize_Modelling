import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.linear_model import LinearRegression
import os


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

# Prepare data for visualization
# Filter detrended data for specific years (include country column)
data_2005_detrended = detrended_df[detrended_df['year'] == 2005][['PCODE', 'country', 'detrended_yield']].copy()
data_2010_detrended = detrended_df[detrended_df['year'] == 2010][['PCODE', 'country', 'detrended_yield']].copy()
data_2015_detrended = detrended_df[detrended_df['year'] == 2015][['PCODE', 'country', 'detrended_yield']].copy()
data_2020_detrended = detrended_df[detrended_df['year'] == 2020][['PCODE', 'country', 'detrended_yield']].copy()

# Define which countries use admin1 vs admin2
ADMIN1_ONLY_COUNTRIES = ['Lesotho', 'Nigeria']

# Function to merge data with shapefiles properly
def merge_with_shapefiles(data, value_column):
    # This function is now more flexible.
    # It attempts to merge all data with admin2 first.
    # Any data that fails to merge (based on country) is then attempted with admin1.
    
    # Merge all data with admin2 shapefile initially
    admin2_merged = admin2_gdf.merge(data, left_on='FNID', right_on='PCODE', how='left')
    
    # Identify which countries from the input data were successfully merged at admin2 level
    all_countries_in_data = set(data['country'].unique())
    # Use the 'country' column from the merged data, which comes from the input CSV
    countries_merged_at_admin2 = set(admin2_merged.dropna(subset=[value_column])['country'].unique())

    # Countries that need to be attempted at admin1 level
    countries_for_admin1_merge = list(all_countries_in_data - countries_merged_at_admin2)
    
    # Add the explicitly defined ADMIN1_ONLY_COUNTRIES to this list, ensuring no duplicates
    for country in ADMIN1_ONLY_COUNTRIES:
        if country not in countries_for_admin1_merge:
            countries_for_admin1_merge.append(country)

    # Filter data for these countries and merge with admin1 shapefile
    admin1_countries_data = data[data['country'].isin(countries_for_admin1_merge)]
    
    admin1_merged = None
    if not admin1_countries_data.empty:
        admin1_merged = admin1_gdf.merge(admin1_countries_data, left_on='FNID', right_on='PCODE', how='left')
    
    return admin2_merged, admin1_merged


# Create the visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Maize Yield Anomalies (Model Generated)', fontsize=16, fontname='Times New Roman')

# Read full Africa outline for background
africa_outline = gpd.read_file(os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_0_processed.shp')) if 'gadm41_AFR_0_processed.shp' in str(admin2_gdf) else None
if africa_outline is None:
    # Create Africa outline by dissolving all countries from admin1 data
    africa_outline = admin1_gdf.dissolve()

# Define colormaps
# Red to green colormap for detrended values - more saturated for small ranges
detrend_colors = ['#8b0000', '#d73027', '#f46d43', '#fdae61', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006400']
detrend_cmap = LinearSegmentedColormap.from_list('redgreen', detrend_colors, N=256)

# Blue colormap for average yields
avg_colors = ['#f7f7f7', '#24245c']
avg_cmap = LinearSegmentedColormap.from_list('custom', avg_colors, N=20)

# Get sets of PCODEs for each admin level from the shapefiles
admin1_pcodes = set(admin1_gdf['FNID'])
admin2_pcodes = set(admin2_gdf['FNID'])

# Function to plot data on axis
def plot_on_axis(ax, data, value_column, title, cmap, vmin=None, vmax=None, center_zero=False):
    # Common plotting parameters
    plot_params = {
        'edgecolor': 'white',
        'linewidth': 0.1,
        'cmap': cmap
    }
    
    # Adjust vmin/vmax for centered colormap if needed
    if center_zero and vmin is not None and vmax is not None:
        # Center colormap around zero
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

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
    
    ax.set_title(title, fontsize=14, fontname='Times New Roman', pad=20)
    ax.set_xlim(-20, 55)
    ax.set_ylim(-35, 40)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add a square border around each subplot
    from matplotlib.patches import Rectangle
    border = Rectangle((-20, -35), 75, 75, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(border)

# Calculate color scale ranges
# For detrended values (centered around 0) - now including all four years
detrend_values = []
for data in [data_2005_detrended, data_2010_detrended, data_2015_detrended, data_2020_detrended]:
    if 'detrended_yield' in data.columns:
        detrend_values.extend(data['detrended_yield'].dropna().values)

detrend_vmin, detrend_vmax = np.percentile(detrend_values, [5, 95])  # Tighter range for better contrast
detrend_abs_max = max(abs(detrend_vmin), abs(detrend_vmax))
detrend_vmin, detrend_vmax = -detrend_abs_max, detrend_abs_max

# Plot 2005 detrended data
plot_on_axis(axes[0, 0], data_2005_detrended, 'detrended_yield', 
            '2005 Detrended Yield Anomaly', detrend_cmap, detrend_vmin, detrend_vmax, center_zero=True)

# Plot 2010 detrended data
plot_on_axis(axes[0, 1], data_2010_detrended, 'detrended_yield', 
            '2010 Detrended Yield Anomaly', detrend_cmap, detrend_vmin, detrend_vmax, center_zero=True)

# Plot 2015 detrended data
plot_on_axis(axes[1, 0], data_2015_detrended, 'detrended_yield', 
            '2015 Detrended Yield Anomaly', detrend_cmap, detrend_vmin, detrend_vmax, center_zero=True)

# Plot 2020 detrended data
plot_on_axis(axes[1, 1], data_2020_detrended, 'detrended_yield', 
            '2020 Detrended Yield Anomaly', detrend_cmap, detrend_vmin, detrend_vmax, center_zero=True)

# Add single colorbar for all detrended values
sm_detrend = plt.cm.ScalarMappable(cmap=detrend_cmap, norm=plt.Normalize(vmin=detrend_vmin, vmax=detrend_vmax))
sm_detrend.set_array([])
cbar_det_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03])
cbar_det = fig.colorbar(sm_detrend, cax=cbar_det_ax, orientation='horizontal')
cbar_det.set_label('Yield Anomaly (Detrended)', fontsize=12, fontname='Times New Roman')
cbar_det.ax.tick_params(labelsize=10)

# Adjust layout with reduced spacing
plt.subplots_adjust(top=0.9, bottom=0.2, wspace=0, hspace=0.2)

out_dir = os.path.join(PROJECT_ROOT, 'Plots', 'output')
os.makedirs(out_dir, exist_ok=True)

# Save the figure
plt.savefig(os.path.join(out_dir, 'africa_yield_maps_4panel.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Map visualization saved as africa_yield_maps_4panel.png")
print(f"Data summary:")
print(f"- Total predictions: {len(predictions_df)}")
print(f"- Unique locations: {predictions_df['PCODE'].nunique()}")
print(f"- Year range: {predictions_df['year'].min()} - {predictions_df['year'].max()}")
print(f"- Yield range: {predictions_df['pred_yield'].min():.3f} - {predictions_df['pred_yield'].max():.3f}")