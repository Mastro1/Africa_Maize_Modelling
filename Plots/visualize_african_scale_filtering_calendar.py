import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import numpy as np
import rasterio
from rasterio.mask import mask
from matplotlib import font_manager

def visualize_africa_filtering():
    """
    Creates a three-panel visualization for crop area filtering on a continental African scale.

    The first panel shows a crop map overlay for the entire continent.
    The second panel shows the administrative regions remaining after crop area filtering for the year 2019.
    The third panel shows the final filtered regions after maize crop calendar validation.
    """
    # 1. --- DEFINE PATHS ---
    admin1_shp_path = os.path.join('GADM', 'gadm41_AFR_shp', 'gadm41_AFR_1_processed.shp')
    admin2_shp_path = os.path.join('GADM', 'gadm41_AFR_shp', 'gadm41_AFR_2_processed.shp')
    filtered_data_path = os.path.join('GADM', 'crop_areas', 'africa_crop_areas_glad_filtered.csv')
    crop_map_path = os.path.join('GLAD', 'Global_cropland_3km_2019.tif')
    maize_calendar_path = os.path.join('GADM', 'crop_calendar', 'maize_crop_calendar_extraction.csv')
    output_dir = os.path.join('Plots', 'output')
    output_path = os.path.join(output_dir, 'africa_scale_filtering_comparison_with_calendar.png')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. --- DATA LOADING ---
    print("Loading shapefiles and data...")
    admin1_gdf = gpd.read_file(admin1_shp_path)
    admin2_gdf = gpd.read_file(admin2_shp_path)
    filtered_df = pd.read_csv(filtered_data_path)
    maize_calendar_df = pd.read_csv(maize_calendar_path)
    
    # Filter for the year 2019
    print("Filtering data for the year 2019...")
    filtered_df = filtered_df[filtered_df['year'] == 2019].copy()

    # 3. --- PLOTTING SETUP ---
    print("Creating visualization...")
    # Set Times New Roman as default font
    font_path = font_manager.findfont('Times New Roman')
    if font_path:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Crop Area Filtering on the African Scale (2019)', fontsize=20, y=0.98, ha='center')

    # Use a dissolved boundary of Admin1 for masking the raster, ensuring full continental coverage
    africa_mask_boundary = admin1_gdf.dissolve()

    # --- Panel 1: Crop Map Overlay ---
    axes[0].set_title('African Crop Map Overlay', fontsize=16)
    admin1_gdf.plot(ax=axes[0], color='lightgrey', edgecolor='white', linewidth=0.5, zorder=1)
    admin2_gdf.plot(ax=axes[0], color='lightgrey', edgecolor='white', linewidth=0.5, zorder=1)

    print("Loading and processing crop map for Panel 1...")
    try:
        with rasterio.open(crop_map_path) as src:
            out_image, out_transform = mask(src, africa_mask_boundary.geometry, crop=True, filled=False)
            
            # Calculate the extent of the cropped raster for correct plotting
            height, width = out_image.shape[1], out_image.shape[2]
            extent = [out_transform.c, out_transform.c + out_transform.a * width,
                      out_transform.f + out_transform.e * height, out_transform.f]

            # Threshold the crop data to create a binary mask of crop areas
            crop_data = out_image[0].filled(0)
            crop_binary = (crop_data > 0.2).astype(np.uint8)  # Using a 20% threshold for continental scale
            
            # Mask out non-crop areas so they are transparent
            crop_masked = np.ma.masked_where(crop_binary == 0, crop_binary)
            
            axes[0].imshow(crop_masked, cmap='Reds', extent=extent, alpha=0.8, interpolation='nearest', zorder=2, vmin=0, vmax=1)
            print(f"Found and plotted {np.sum(crop_binary)} crop pixels.")

    except Exception as e:
        print(f"Could not process or plot crop map: {e}")

    # --- Panel 2: Filtered Administrative Regions ---
    axes[1].set_title('Filtered Administrative Regions (2019)', fontsize=16)
    admin1_gdf.plot(ax=axes[1], color='lightgrey', edgecolor='white', linewidth=0.5)
    admin2_gdf.plot(ax=axes[1], color='lightgrey', edgecolor='white', linewidth=0.5)

    print("Processing filtered regions for Panel 2...")
    # Separate PCODEs for admin1 and admin2 levels from the filtered data
    filtered_df['admin_2'] = filtered_df['admin_2'].replace('', np.nan)
    admin1_pcodes = filtered_df[filtered_df['admin_2'].isna()]['PCODE'].unique()
    admin2_pcodes = filtered_df[filtered_df['admin_2'].notna()]['PCODE'].unique()
    
    # Select the geometries for the filtered regions
    filtered_admin1_gdf = admin1_gdf[admin1_gdf['FNID'].isin(admin1_pcodes)]
    filtered_admin2_gdf = admin2_gdf[admin2_gdf['FNID'].isin(admin2_pcodes)]

    # Plot the filtered regions
    if not filtered_admin1_gdf.empty:
        filtered_admin1_gdf.plot(ax=axes[1], color='#251E5D', edgecolor='gray', linewidth=0.2)
    if not filtered_admin2_gdf.empty:
        filtered_admin2_gdf.plot(ax=axes[1], color='#251E5D', edgecolor='gray', linewidth=0.2)
    
    print(f"Plotted {len(filtered_admin1_gdf)} Admin1 regions and {len(filtered_admin2_gdf)} Admin2 regions.")

    # --- Panel 3: Final Calendar-Based Filtering ---
    axes[2].set_title('Final Calendar-Based Filtering (2019)', fontsize=16)
    admin1_gdf.plot(ax=axes[2], color='lightgrey', edgecolor='white', linewidth=0.5)
    admin2_gdf.plot(ax=axes[2], color='lightgrey', edgecolor='white', linewidth=0.5)

    print("Processing calendar-filtered regions for Panel 3...")
    # Get FNIDs that have valid maize calendar data
    calendar_fnids = maize_calendar_df['FNID'].unique()

    # Separate PCODEs for admin1 and admin2 levels that have calendar data
    # Note: We need to intersect with the previously filtered regions
    filtered_df['admin_2'] = filtered_df['admin_2'].replace('', np.nan)
    calendar_filtered_df = filtered_df[filtered_df['PCODE'].isin(calendar_fnids)].copy()

    admin1_calendar_pcodes = calendar_filtered_df[calendar_filtered_df['admin_2'].isna()]['PCODE'].unique()
    admin2_calendar_pcodes = calendar_filtered_df[calendar_filtered_df['admin_2'].notna()]['PCODE'].unique()

    # Select the geometries for the calendar-filtered regions
    calendar_admin1_gdf = admin1_gdf[admin1_gdf['FNID'].isin(admin1_calendar_pcodes)]
    calendar_admin2_gdf = admin2_gdf[admin2_gdf['FNID'].isin(admin2_calendar_pcodes)]

    # Plot the calendar-filtered regions
    if not calendar_admin1_gdf.empty:
        calendar_admin1_gdf.plot(ax=axes[2], color='#2E8B57', edgecolor='lightgrey', linewidth=0.2)
    if not calendar_admin2_gdf.empty:
        calendar_admin2_gdf.plot(ax=axes[2], color='#2E8B57', edgecolor='lightgrey', linewidth=0.2)

    print(f"Plotted {len(calendar_admin1_gdf)} Admin1 regions and {len(calendar_admin2_gdf)} Admin2 regions with valid calendar data.")

    # 4. --- FINAL STYLING AND SAVING ---
    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Manual control of subplot spacing
    plt.subplots_adjust(wspace=-0.25, hspace=0) 

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    plt.show()

    print("-"*100)
    print("NOTICE: This script uses the 3km resolution GLAD crop map for the seek of visualization. Hence, the real number of kept locations is slightly different from the actual one.")
    print("-"*100)

if __name__ == "__main__":
    visualize_africa_filtering() 