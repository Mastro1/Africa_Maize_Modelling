import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import os
import scienceplots
from scipy.stats import linregress
from matplotlib.ticker import FuncFormatter

# Set global scientific style
plt.style.use(['science', 'ieee'])

PROJECT_ROOT = os.getcwd()
RESULTS_V7_GLOBAL = os.path.join(PROJECT_ROOT, 'Model_physical', 'Results', 'V7_model', 'Global_Maize_Yield_V7.csv')
RESULTS_V7_OPTIMIZED_DIR = os.path.join(PROJECT_ROOT, 'Model_physical', 'Results', 'V7_model_optimized')
SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_0_processed.shp')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Model_physical', 'plots', 'output')

def main():
    # 1. Load Data and Calculate Correlations
    if not os.path.exists(RESULTS_V7_GLOBAL):
        print(f"Error: Global results file not found at {RESULTS_V7_GLOBAL}")
        return

    df_global = pd.read_csv(RESULTS_V7_GLOBAL)
    countries = df_global['Country'].unique()
    
    country_correlations = {}
    all_model_dev = []
    all_fao_dev = []

    print(f"Processing {len(countries)} countries...")

    for country in countries:
        # Normalize name for filename (Burkina Faso -> Burkina Faso_fao_v7_optimized_table.csv)
        # Assuming the filenames use spaces as in the user example
        filename = f"{country}_fao_v7_optimized_table.csv"
        file_path = os.path.join(RESULTS_V7_OPTIMIZED_DIR, filename)
        
        if not os.path.exists(file_path):
            # Try replacing underscores with spaces if underscores were used in global file
            filename_space = f"{country.replace('_', ' ')}_fao_v7_optimized_table.csv"
            file_path = os.path.join(RESULTS_V7_OPTIMIZED_DIR, filename_space)
            if not os.path.exists(file_path):
                print(f"  Warning: No optimized table for {country}")
                continue

        df_country = pd.read_csv(file_path)
        
        # Ensure required columns exist
        if 'Model_Dev_Kernel' in df_country.columns and 'FAO_Dev_Kernel' in df_country.columns:
            # Drop NaNs for correlation calculation
            df_valid = df_country[['Model_Dev_Kernel', 'FAO_Dev_Kernel']].dropna()
            
            if len(df_valid) >= 3:
                corr = df_valid['Model_Dev_Kernel'].corr(df_valid['FAO_Dev_Kernel'])
                country_correlations[country] = corr
                
                all_model_dev.extend(df_valid['Model_Dev_Kernel'].values)
                all_fao_dev.extend(df_valid['FAO_Dev_Kernel'].values)
            else:
                print(f"  Skipping {country}: Not enough data points ({len(df_valid)})")

    if not country_correlations:
        print("Error: No correlation data collected.")
        return

    # 2. Load Shapefile
    gdf_africa = gpd.read_file(SHAPEFILE_PATH)
    
    # 3. Create Figure (Two Panels)
    width = 7.0 # IEEE double column
    height = width * 0.45 
    # USER TIP: 
    # - 'width_ratios' controls how much horizontal space each panel gets. 
    # - 'wspace' controls the empty distance between the two panels. Decrease it to bring them closer.
    fig, (ax_map, ax_scatter) = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.11})

    # --- Panel A: Map ---
    # Merge correlations into GeoDataFrame
    # Normalize names for joining: shapefile uses 'ADMIN0'
    mapped_corrs = pd.DataFrame(list(country_correlations.items()), columns=['Country', 'Correlation'])
    
    # Simple normalization: lower case and replace _ with spaces for names like "South_Africa"
    mapped_corrs['norm_name'] = mapped_corrs['Country'].str.lower().str.replace('_', ' ')
    gdf_africa['norm_name'] = gdf_africa['ADMIN0'].str.lower()
    
    gdf_merged = gdf_africa.merge(mapped_corrs, on='norm_name', how='left')
    
    # Plot background (all Africa)
    gdf_africa.plot(ax=ax_map, color='#eeeeee', edgecolor='white', linewidth=0.2)
    
    # Plot correlations
    cmap = plt.cm.RdYlGn
    
    # Create an inset axis for the colorbar to decouple it from the main map layout.
    # [x0, y0, width, height] relative to ax_map
    cax = ax_map.inset_axes([0.1, -0.1, 0.8, 0.05])
    
    gdf_merged.dropna(subset=['Correlation']).plot(
        ax=ax_map, column='Correlation', cmap=cmap, 
        vmin=-1, vmax=1, edgecolor='black', linewidth=0.1,
        cax=cax, legend=True, 
        legend_kwds={'label': "Correlation (r)", 'orientation': "horizontal"}
    )

    
    # 1. Spines and Frame
    for spine in ax_map.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.2) # Thinner spine
        spine.set_edgecolor('black')

    # 2. Coordinates (Latitude and Longitude)
    ax_map.set_xlim(-20, 55)
    ax_map.set_ylim(-35, 40)
    
    # Define tick intervals
    lon_ticks = np.arange(-20, 61, 20)
    lat_ticks = np.arange(-30, 41, 15)
    
    ax_map.set_xticks(lon_ticks)
    ax_map.set_yticks(lat_ticks)
    
    from matplotlib.ticker import FuncFormatter

    def format_lon(x, pos):
        if x > 0: return f'{x:g}°E'
        if x < 0: return f'{abs(x):g}°W'
        return f'{x:g}°'

    def format_lat(y, pos):
        if y > 0: return f'{y:g}°N'
        if y < 0: return f'{abs(y):g}°S'
        return f'{y:g}°'

    ax_map.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax_map.yaxis.set_major_formatter(FuncFormatter(format_lat))
    
    # Show ticks on specific sides according to user request (Top & Left)
    ax_map.tick_params(axis='both', which='major', labelsize=7, direction='in', 
                       length=2, width=0.4, colors='black',
                       top=True, right=True, bottom=True, left=True,
                       labeltop=True, labelleft=True, 
                       labelbottom=False, labelright=False)
    
    # 3. Gridlines
    ax_map.grid(True, linestyle='--', linewidth=0.2, color='#cccccc', alpha=0.5, zorder=0)

    # Removed Title as requested

    # --- Panel B: Scatter Plot ---
    all_model_dev = np.array(all_model_dev)
    all_fao_dev = np.array(all_fao_dev)
    
    # Filter out extreme outliers for better visualization if any (keeping it consistent with model logic)
    ax_scatter.scatter(all_fao_dev, all_model_dev, color='black', s=2, alpha=0.5, facecolors='none', edgecolors='black', linewidth=0.3)
    
    # Linear Regression (y = ax + b)
    slope, intercept, r_value, p_value, std_err = linregress(all_fao_dev, all_model_dev)
    x_range = np.array([all_fao_dev.min(), all_fao_dev.max()])
    y_fit = slope * x_range + intercept
    
    ax_scatter.plot(x_range, y_fit, color='black', linestyle='-', linewidth=1)
    
    # 1:1 line for reference
    lims = [
        min(ax_scatter.get_xlim()[0], ax_scatter.get_ylim()[0]),
        max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1])
    ]
    ax_scatter.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=0.8)
    
    # Text Annotation
    sign = "+" if intercept >= 0 else "-"
    eq_text = f"$y = {slope:.2f}x {sign} {abs(intercept):.2f}$\n$R^2 = {r_value**2:.2f}$"
    ax_scatter.text(0.95, 0.05, eq_text, transform=ax_scatter.transAxes, 
                    ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax_scatter.set_xlabel("Actual Trend distance (FAO Dev Kernel)", fontsize=9)
    ax_scatter.set_ylabel("Modeled Trend Distance (Model Dev Kernel)", fontsize=9)
    # Move y-axis label and ticks to the right
    ax_scatter.yaxis.tick_right()
    ax_scatter.yaxis.set_label_position("right")
    ax_scatter.tick_params(axis='both', which='major', labelsize=8)
    ax_scatter.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    
    # USER TIP: 
    # The map naturally draws as a square because of geography (lon 75 range vs lat 75 range).
    # This command forces the scatter plot to also draw as a perfect square matching the map.
    ax_scatter.set_box_aspect(1)

    plt.tight_layout()
    
    # Save Outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path_pdf = os.path.join(OUTPUT_DIR, 'fig_correlation_with_fao_V7.pdf')
    save_path_png = os.path.join(OUTPUT_DIR, 'fig_correlation_with_fao_V7.png')
    
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Figures saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
