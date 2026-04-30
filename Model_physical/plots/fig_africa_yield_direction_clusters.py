import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import ListedColormap, to_rgba
import numpy as np
import os
import scienceplots
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from statsmodels.nonparametric.kernel_regression import KernelReg
import warnings

# Set global scientific style
plt.style.use(['science', 'ieee'])

PROJECT_ROOT = os.getcwd()

# ------ PARAMETERS ------
NUM_CLUSTERS = 5
SPATIAL_WEIGHT = 2.5  # Weight of geographical coordinates (0 means pure temporal clustering)
# ------------------------

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
        # Avoid division by zero
        trend_values[trend_values == 0] = np.nan
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
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if 'Season' in df.columns:
        df = df[df['Season'] == 1].copy()
        
    print("Removing boundary years to avoid detrending artifacts...")
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    print(f"Original years: {min_year} - {max_year}")
    df = df[~df['Year'].isin([min_year, max_year])].copy()
    
    print("Calculating kernel detrended values on cleaned data...")
    df = calculate_detrended_values(df)

    df = df.groupby(['PCODE', 'Year'])['detrended_yield'].mean().reset_index()
    
    # 2. Pivot to Time-Series Matrix (N locations x T years)
    print("Pivoting data to location-year matrix...")
    pivot_df = df.pivot(index='PCODE', columns='Year', values='detrended_yield')
    years = pivot_df.columns.tolist()
    
    # Missing values mean no specific anomaly recorded, fill with 0 (no deviation from trend)
    pivot_df = pivot_df.fillna(0)
    pivot_df.reset_index(inplace=True)
    
    # 3. Load Shapefiles to get centroids
    admin2_path = os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_2_processed.shp')
    admin1_path = os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_1_processed.shp')
    
    print("Loading shapefiles...")
    admin2_gdf = gpd.read_file(admin2_path)
    admin1_gdf = gpd.read_file(admin1_path)
    
    import shapely
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        admin1_gdf['centroid_x'] = admin1_gdf.geometry.centroid.x
        admin1_gdf['centroid_y'] = admin1_gdf.geometry.centroid.y
        admin2_gdf['centroid_x'] = admin2_gdf.geometry.centroid.x
        admin2_gdf['centroid_y'] = admin2_gdf.geometry.centroid.y
    
    centroids_df1 = admin1_gdf[['FNID', 'centroid_x', 'centroid_y']].rename(columns={'FNID': 'PCODE'})
    centroids_df2 = admin2_gdf[['FNID', 'centroid_x', 'centroid_y']].rename(columns={'FNID': 'PCODE'})
    centroids_df = pd.concat([centroids_df1, centroids_df2]).drop_duplicates(subset=['PCODE'])
    
    # Merge pivoted time series with centroids
    analysis_df = pivot_df.merge(centroids_df, on='PCODE', how='inner')
    
    print(f"Clustering into {NUM_CLUSTERS} directional time-series zones (Spatial Weight: {SPATIAL_WEIGHT})...")
    
    # temporal Features: Standardize each location's time series relative to itself (Z-score row-wise)
    # This transforms the features so Euclidean distance = Pearson Correlation distance
    ts_features_raw = analysis_df[years].values
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # Apply zscore row-wise
        ts_features = zscore(ts_features_raw, axis=1)
        # Locations with completely flat 0 anomalies will result in NaNs, fill with 0
        ts_features = np.nan_to_num(ts_features, nan=0.0)
    
    # Spatial Features
    spatial_scaler = StandardScaler()
    spatial_features = spatial_scaler.fit_transform(analysis_df[['centroid_x', 'centroid_y']]) * SPATIAL_WEIGHT
    
    # Combine Features
    X = np.hstack((ts_features, spatial_features))
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    analysis_df['Cluster_raw'] = kmeans.fit_predict(X)
    
    # Let's sort cluster IDs somewhat arbitrarily but deterministically for colors (e.g. by overall count)
    cluster_counts = analysis_df['Cluster_raw'].value_counts()
    sorted_idx = cluster_counts.index
    mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted_idx)}
    analysis_df['Cluster'] = analysis_df['Cluster_raw'].map(mapping)
    
    print("\nCluster Characteristics:")
    for cluster_id in range(1, NUM_CLUSTERS + 1):
        subset = analysis_df[analysis_df['Cluster'] == cluster_id]
        print(f"Cluster {cluster_id}: {len(subset)} locations")
    
    # 4. Figure Setup
    admin1_pcodes = set(admin1_gdf['FNID'])
    admin2_pcodes = set(admin2_gdf['FNID'])
    africa_outline = admin1_gdf.dissolve()

    data_admin1 = analysis_df[analysis_df['PCODE'].isin(admin1_pcodes)]
    data_admin2 = analysis_df[analysis_df['PCODE'].isin(admin2_pcodes)]
    
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1.2, 1], 
                          wspace=0.15, hspace=0.3)
    
    ax_map = fig.add_subplot(gs[:, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[1, 1])
    
    # Distinct qualitative colors from the loaded science style
    with plt.style.context('science'):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    alpha = 0.85   # 0.8 ~ 0.9 usually looks best for slight transparency

    # One-liner to add alpha
    cmap_colors = [to_rgba(colors[i % len(colors)], alpha=alpha) 
               for i in range(NUM_CLUSTERS)]

    cmap = ListedColormap(cmap_colors)
        
    vmin = 0.5
    vmax = NUM_CLUSTERS + 0.5

    # ======== MAP PLOT ========
    print("\nPlotting map...")
    africa_outline.plot(ax=ax_map, color='#f0f0f0', edgecolor='none')
    admin2_gdf.plot(ax=ax_map, color='#f0f0f0', edgecolor='white', linewidth=0.05)
    
    if not data_admin2.empty:
        merged2 = admin2_gdf.merge(data_admin2, left_on='FNID', right_on='PCODE')
        merged2.plot(ax=ax_map, column='Cluster', cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
        
    if not data_admin1.empty:
        merged1 = admin1_gdf.merge(data_admin1, left_on='FNID', right_on='PCODE')
        merged1.plot(ax=ax_map, column='Cluster', cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
    
    for spine in ax_map.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor('black')

    ax_map.set_xlim(-20, 55)
    ax_map.set_ylim(-35, 40)
    
    lon_ticks = np.arange(-20, 61, 20)
    lat_ticks = np.arange(-30, 41, 15)
    ax_map.set_xticks(lon_ticks)
    ax_map.set_yticks(lat_ticks)
    
    ax_map.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax_map.yaxis.set_major_formatter(FuncFormatter(format_lat))
    
    ax_map.tick_params(axis='both', which='major', labelsize=9, direction='in', 
                   length=4, width=0.8, colors='black',
                   top=True, right=True, bottom=True, left=True,
                   labeltop=False, labelleft=True, 
                   labelbottom=True, labelright=False)
    
    ax_map.grid(True, linestyle='--', linewidth=0.3, color='#cccccc', alpha=0.5, zorder=0)
    ax_map.set_title('Directional Yield Temporal Clusters', fontsize=11, fontweight='bold', pad=10)

    # Adding a simple colorbar/legend to Map
    cax = fig.add_axes([0.15, 0.08, 0.35, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.arange(1, NUM_CLUSTERS + 1))
    cbar.ax.set_xticklabels([f'Group {i}' for i in range(1, NUM_CLUSTERS + 1)])
    cbar.ax.tick_params(labelsize=8)

    # ======== TIME SERIES PROFILES (Bar Chart) ========
    print("Plotting time-series cluster bar charts...")
    year_indices = np.arange(len(years))
    bar_width = 0.8 / NUM_CLUSTERS
    
    cluster_means_dict = {}
    
    for cluster_id in range(1, NUM_CLUSTERS + 1):
        cluster_data = analysis_df[analysis_df['Cluster'] == cluster_id]
        
        # Calculate mean and std raw detrended anomaly for this cluster across years
        mean_ts = cluster_data[years].mean(axis=0).values * 100 # Convert to %
        std_ts = cluster_data[years].std(axis=0).values * 100
        cluster_means_dict[f'Group {cluster_id}'] = mean_ts
        
        cluster_color = cmap(cluster_id - 1)
        offset = ((cluster_id - 1) - (NUM_CLUSTERS - 1) / 2.0) * bar_width
        ax_ts.bar(year_indices + offset, mean_ts, width=bar_width, 
                  label=f'Group {cluster_id} (n={len(cluster_data)})', color=cluster_color)
        
    ax_ts.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    # Formatting Time-Series Axes
    ax_ts.set_title('Average Yield Anomaly per Cluster', fontsize=11, fontweight='bold', pad=10)
    ax_ts.set_xlabel('Year', fontsize=10)
    ax_ts.set_ylabel('Mean Yield Anomaly (%)', fontsize=10)
    ax_ts.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    
    ax_ts.set_xticks(year_indices)
    ax_ts.set_xticklabels(years)
    ax_ts.tick_params(axis='x', rotation=45, labelsize=8)
    ax_ts.tick_params(axis='y', labelsize=8)
    
    ax_ts.grid(True, linestyle=':', linewidth=0.5, alpha=0.4)
    ax_ts.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black', ncol=2)
    
    for spine in ax_ts.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor('black')
        
    # ======== CORRELATION MATRIX ========
    print("Plotting cluster correlation matrix...")
    corr_df = pd.DataFrame(cluster_means_dict).corr()
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    # Plot lower triangle using imshow
    import matplotlib.colors as mcolors
    try:
        corr_cmap = plt.cm.get_cmap('RdBu_r')
    except AttributeError:
        corr_cmap = plt.get_cmap('RdBu_r')
        
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    
    corr_masked = np.where(mask, np.nan, corr_df.values)
    im = ax_corr.imshow(corr_masked, cmap=corr_cmap, norm=norm, aspect='auto')
    
    # Add text annotations
    for i in range(NUM_CLUSTERS):
        for j in range(NUM_CLUSTERS):
            if not mask[i, j]:
                val = corr_masked[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)
                
    ax_corr.set_xticks(np.arange(NUM_CLUSTERS))
    ax_corr.set_yticks(np.arange(NUM_CLUSTERS))
    ax_corr.set_xticklabels(corr_df.columns)
    ax_corr.set_yticklabels(corr_df.index)
    
    ax_corr.tick_params(axis='both', which='major', labelsize=9)
    # Hide top and right spines
    ax_corr.spines['top'].set_visible(False)
    ax_corr.spines['right'].set_visible(False)
    ax_corr.spines['bottom'].set_linewidth(0.8)
    ax_corr.spines['left'].set_linewidth(0.8)
    
    ax_corr.set_title('Cluster Trajectory Correlation', fontsize=11, fontweight='bold', pad=10)
    
    cbar_corr = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar_corr.set_label('Pearson Correlation', fontsize=9)
    
    out_dir = os.path.join(PROJECT_ROOT, 'Model_physical', 'plots', 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    save_path_pdf = os.path.join(out_dir, 'africa_yield_direction_clusters.pdf')
    save_path_png = os.path.join(out_dir, 'africa_yield_direction_clusters.png')
    
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Figures successfully saved to {out_dir}")

if __name__ == "__main__":
    main()
