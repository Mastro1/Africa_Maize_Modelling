"""
This script visualizes GDHY maize yield data from NetCDF files overlaid on the African continent.
Handles 0-360 longitude format by shifting to -180-180.
"""

import rasterio
from rasterio.transform import Affine
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import argparse
from pathlib import Path

# Configuration
DATA_DIR = os.path.join("ResultsComparison", "GDHY", "data")
SHAPEFILE_DIR = os.path.join("GADM", "gadm41_AFR_shp", "gadm41_AFR_0_processed.shp")
DEFAULT_YEAR = 2000

# Color scheme for yield visualization
YIELD_COLORS = ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679',
                '#41ab5d', '#238443', '#006837', '#004529']

def load_gdhy_data(year):
    """Load the NetCDF data for a specific year and correct 0-360 longitude."""
    filename = f"yield_{year}.nc4"
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading GDHY data from: {file_path}")

    with rasterio.open(file_path) as src:
        data = src.read(1)
        bounds = src.bounds
        nodata = src.nodata
        profile = src.profile

        print(f"Original Bounds: {bounds}")
        print(f"Original Shape: {data.shape}")

        # Check if we need to shift from 0-360 to -180-180
        # If left is 0 and right is 360
        if np.isclose(bounds.left, 0) and np.isclose(bounds.right, 360):
            print("Detected 0-360 longitude. shifting to -180-180...")
            
            # Shift data
            # Assumes full global coverage width=360 degrees
            width = data.shape[1]
            shift = width // 2
            data = np.roll(data, shift, axis=1)
            
            # Update transform
            # New west is -180
            new_transform = Affine(src.transform.a, src.transform.b, -180.0,
                                   src.transform.d, src.transform.e, src.transform.f)
            
            # Update bounds manually for plotting logic
            # (left, bottom, right, top)
            new_extent = [-180.0, bounds.bottom, 180.0, bounds.top]
            
        else:
            new_extent = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            new_transform = src.transform

        # Mask nodata
        if nodata is not None:
            # Create a masked array
            data = np.ma.masked_equal(data, nodata)
            # Also mask values that might be implicitly nodata (like very small negatives if not expected)
            # But here nodata is explicit.
        
        return data, new_extent, new_transform

def load_africa_shapefile():
    """Load the Africa shapefile at country level (admin 0)."""
    if not os.path.exists(SHAPEFILE_DIR):
        raise FileNotFoundError(f"Africa shapefile not found: {SHAPEFILE_DIR}")

    print(f"Loading Africa shapefile: {SHAPEFILE_DIR}")
    gdf = gpd.read_file(SHAPEFILE_DIR)
    print(f"Loaded {len(gdf)} countries")
    return gdf

def create_map(year, data, extent, gdf, output_dir):
    """Create and save the map."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Create custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("yield_cmap", YIELD_COLORS)

    # Plot raster
    # extent is [left, bottom, right, top]
    # imshow expects [left, right, bottom, top]
    imshow_extent = [extent[0], extent[2], extent[1], extent[3]]
    
    im = ax.imshow(data, extent=imshow_extent, cmap=cmap, origin='upper')

    # Plot Africa
    gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, alpha=0.7)

    # Focus on Africa (roughly)
    # Longitude: -25 to 60
    # Latitude: -40 to 40
    ax.set_xlim(-25, 60)
    ax.set_ylim(-40, 40)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, aspect=30, pad=0.05)
    cbar.set_label('Maize Yield (tons/ha)', fontsize=12)

    ax.set_title(f'GDHY Maize Yield - {year}\n(Data shifted to -180/180)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"gdhy_yield_map_{year}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Map saved to: {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=DEFAULT_YEAR)
    args = parser.parse_args()

    try:
        data, extent, transform = load_gdhy_data(args.year)
        gdf = load_africa_shapefile()
        create_map(args.year, data, extent, gdf, DATA_DIR)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
