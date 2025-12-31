"""
This script visualizes SPAM maize yield data from GeoTIFF files overlaid on the African continent.
Shows the full raster data for a specific year without administrative aggregation.
"""

import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import os
import argparse
from pathlib import Path


# Configuration
DATA_DIR = "ResultsComparison/SPAM/data"
SHAPEFILE_DIR = "GADM/gadm41_AFR_shp/gadm41_AFR_0_processed.shp"
CROP_TYPE = "SPAM_Maize"
DEFAULT_YEAR = 2000

# Color scheme for yield visualization (same as aggregated map for consistency)
YIELD_COLORS = ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679',
                '#41ab5d', '#238443', '#006837', '#004529']


def load_raster_data(year):
    """Load the GeoTIFF raster data for a specific year."""
    raster_filename = f"spam_maiz_{year}.tif"
    raster_path = os.path.join(DATA_DIR, raster_filename)

    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    print(f"Loading SPAM maize raster data from: {raster_path}")

    with rasterio.open(raster_path) as src:
        # Read the data
        data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # Get the bounds for plotting
        bounds = src.bounds

        print(f"Raster shape: {data.shape}")
        print(f"Bounds: {bounds}")
        print(f"CRS: {crs}")
        print(f"NoData value: {nodata}")

        # Mask nodata values
        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)
        else:
            # Common nodata values for yield data
            data = np.ma.masked_where(data <= 0, data)

        # Convert from kg/ha to tonnes/ha (SPAM default is kg/ha, except 2020 which is already in tonnes/ha)
        if year != 2020:
            data = data / 1000.0

        return data, transform, bounds, crs


def load_africa_shapefile():
    """Load the Africa shapefile at country level (admin 0)."""
    if not os.path.exists(SHAPEFILE_DIR):
        raise FileNotFoundError(f"Africa shapefile not found: {SHAPEFILE_DIR}")

    print(f"Loading Africa shapefile: {SHAPEFILE_DIR}")
    gdf = gpd.read_file(SHAPEFILE_DIR)
    print(f"Loaded {len(gdf)} countries")

    return gdf


def create_raster_map(year, data, bounds, gdf, output_dir="ResultsComparison/SPAM/data"):
    """Create a map visualization of the raw raster data overlaid on Africa."""
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Create custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("yield_cmap", YIELD_COLORS)

    # Calculate extent from bounds (left, bottom, right, top)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # Plot the raster data
    im = ax.imshow(data, extent=extent, cmap=cmap, origin='upper')

    # Plot Africa boundaries on top
    gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8, alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, aspect=30, pad=0.05)
    cbar.set_label('SPAM Maize Yield (tons/ha)', fontsize=12)

    # Customize the plot
    ax.set_title(f'SPAM Maize Yield Distribution - {year}\nRaw GeoTIFF Data Overlaid on Africa',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text box
    valid_data = data.compressed()  # Remove masked values
    if len(valid_data) > 0:
        stats_text = f'Mean: {np.mean(valid_data):.2f} t/ha\nMax: {np.max(valid_data):.2f} t/ha\nMin: {np.min(valid_data):.2f} t/ha'
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                fontsize=10, verticalalignment='bottom')

    # Adjust layout
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_filename = f'spam_maize_yield_raster_{year}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to: {output_path}")

    # Also save as PDF for better quality
    pdf_filename = f'spam_maize_yield_raster_{year}.pdf'
    pdf_path = os.path.join(output_dir, pdf_filename)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Map also saved as PDF: {pdf_path}")

    return fig, ax


def main(year=None):
    """Main function to create the raster yield map."""
    if year is None:
        year = DEFAULT_YEAR

    try:
        print(f"Creating SPAM maize yield map for year {year}...")

        # Load data
        data, transform, bounds, crs = load_raster_data(year)
        gdf = load_africa_shapefile()

        # Create and save the map
        fig, ax = create_raster_map(year, data, bounds, gdf)

        # Show the plot
        plt.show()

        print(f"\nSPAM maize yield raster map for {year} created successfully!")

    except Exception as e:
        print(f"Error creating SPAM maize yield raster map: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot SPAM maize yield raster data for a specific year')
    parser.add_argument('--year', type=int, default=DEFAULT_YEAR,
                       help=f'Year to plot (default: {DEFAULT_YEAR})')

    args = parser.parse_args()
    main(year=args.year)