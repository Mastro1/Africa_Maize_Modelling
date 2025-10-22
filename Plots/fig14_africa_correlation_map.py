import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
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
print("Calculating detrended values...")
detrended_df = calculate_detrended_values(predictions_df)

# Create correlation matrix
print("Creating correlation matrix...")
def create_correlation_matrix(df):
    """
    Create correlation matrix between all locations using detrended yield values
    """
    # Pivot data to have years as rows and locations as columns
    pivot_df = df.pivot(index='year', columns='PCODE', values='detrended_yield')
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()
    
    return corr_matrix, pivot_df

corr_matrix, pivot_df = create_correlation_matrix(detrended_df)

# Choose reference location (let's pick a location with good data coverage)
# Find location with most years of data
location_coverage = pivot_df.count().sort_values(ascending=False)
reference_location = location_coverage.index[0]  # Location with most data points

print(f"Reference location selected: {reference_location}")
print(f"Reference location has {location_coverage.iloc[0]} years of data")

# Get correlations for the reference location
reference_correlations = corr_matrix[reference_location].dropna()

# Create correlation data for mapping
correlation_data = []
for pcode, correlation in reference_correlations.items():
    if pcode != reference_location:  # Exclude self-correlation
        correlation_data.append({
            'PCODE': pcode,
            'correlation': correlation
        })

correlation_df = pd.DataFrame(correlation_data)

# Get reference location info for title
ref_location_info = detrended_df[detrended_df['PCODE'] == reference_location].iloc[0]
ref_country = ref_location_info['country']

# Define which countries use admin1 vs admin2
ADMIN1_ONLY_COUNTRIES = ['Lesotho', 'Nigeria']

# Get sets of PCODEs for each admin level from the shapefiles
admin1_pcodes = set(admin1_gdf['FNID'])
admin2_pcodes = set(admin2_gdf['FNID'])

# Get reference location name from shapefiles
ref_location_name = None
if reference_location in admin2_pcodes:
    ref_name_row = admin2_gdf[admin2_gdf['FNID'] == reference_location]
    if not ref_name_row.empty and 'ADMIN2' in ref_name_row.columns:
        ref_location_name = ref_name_row['ADMIN2'].iloc[0]
elif reference_location in admin1_pcodes:
    ref_name_row = admin1_gdf[admin1_gdf['FNID'] == reference_location]
    if not ref_name_row.empty and 'ADMIN1' in ref_name_row.columns:
        ref_location_name = ref_name_row['ADMIN1'].iloc[0]

# Use name if available, otherwise use PCODE
ref_display_name = ref_location_name if ref_location_name else reference_location

# Create the visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
fig.suptitle(f'Yield Correlation with Reference Location ({ref_country})', 
            fontsize=12, fontname='Times New Roman', y=0.95)

# Define correlation colormap (blue to red, with white at 0)
corr_colors = ['#24245c', '#5a7fb8', '#a8c5e8', '#ffffff', '#f7a8a8', '#d73027', '#8b0000']
corr_cmap = LinearSegmentedColormap.from_list('correlation', corr_colors, N=256)

# Plot the entire Admin-2 shapefile as a grey background
admin2_gdf.plot(ax=ax, color='#d3d3d3', edgecolor='white', linewidth=0.1)

# Plot the missing admin-1 countries for a complete background
if not background_admin1_only_gdf.empty:
    background_admin1_only_gdf.plot(ax=ax, color='#d3d3d3', edgecolor='white', linewidth=0.1)

# Separate correlation data into admin levels (exclude reference location from correlation plotting)
corr_admin1 = correlation_df[correlation_df['PCODE'].isin(admin1_pcodes) & (correlation_df['PCODE'] != reference_location)]
corr_admin2 = correlation_df[correlation_df['PCODE'].isin(admin2_pcodes) & (correlation_df['PCODE'] != reference_location)]

# Merge with shapefiles
merged_admin1 = admin1_gdf.merge(corr_admin1, left_on='FNID', right_on='PCODE', how='inner')
merged_admin2 = admin2_gdf.merge(corr_admin2, left_on='FNID', right_on='PCODE', how='inner')

# Set correlation range (-1 to 1)
vmin, vmax = -1, 1

# Common plotting parameters
plot_params = {
    'edgecolor': 'white',
    'linewidth': 0.1,
    'cmap': corr_cmap,
    'vmin': vmin,
    'vmax': vmax
}

# Plot Admin-2 correlation data
if not merged_admin2.empty:
    merged_admin2.plot(ax=ax, column='correlation', **plot_params)

# Plot Admin-1 correlation data
if not merged_admin1.empty:
    merged_admin1.plot(ax=ax, column='correlation', **plot_params)

# Highlight reference location in yellow (plot AFTER correlation data to ensure it's on top)
ref_admin1 = admin1_gdf[admin1_gdf['FNID'] == reference_location]
ref_admin2 = admin2_gdf[admin2_gdf['FNID'] == reference_location]

if not ref_admin2.empty:
    ref_admin2.plot(ax=ax, color='#fccd32', edgecolor='black', linewidth=2)
elif not ref_admin1.empty:
    ref_admin1.plot(ax=ax, color='#fccd32', edgecolor='black', linewidth=2)

# Set map properties
ax.set_xlim(-20, 55)
ax.set_ylim(-35, 40)
ax.set_aspect('equal')
ax.axis('off')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=corr_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.6, pad=0.1)
cbar.set_label('Correlation Coefficient', fontsize=12, fontname='Times New Roman')
cbar.ax.tick_params(labelsize=10)

# Add text box with reference location info
textstr = f'Reference Location: {ref_display_name}\n{ref_country}\nYears of data: {location_coverage.iloc[0]}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontname='Times New Roman')

out_dir = os.path.join(PROJECT_ROOT, 'Plots', 'output')
os.makedirs(out_dir, exist_ok=True)

# Save the figure
plt.savefig(os.path.join(out_dir, 'africa_correlation_map.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Correlation map saved as africa_correlation_map.png")
print(f"\nCorrelation Analysis Summary:")
print(f"- Reference location: {ref_display_name} ({ref_country})")
print(f"- Total locations in correlation matrix: {len(corr_matrix)}")
print(f"- Locations plotted on map: {len(correlation_df)}")
print(f"- Correlation range: {correlation_df['correlation'].min():.3f} to {correlation_df['correlation'].max():.3f}")
print(f"- Mean correlation: {correlation_df['correlation'].mean():.3f}")
print(f"- Locations with high correlation (>0.5): {len(correlation_df[correlation_df['correlation'] > 0.5])}")
print(f"- Locations with low correlation (<0.2): {len(correlation_df[correlation_df['correlation'] < 0.2])}")