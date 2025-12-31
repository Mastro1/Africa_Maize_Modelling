"""
This script creates a map visualization of average maize yields at admin2 level
using the aggregated yield data.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os
import warnings
warnings.filterwarnings('ignore')


# Configuration
INPUT_DIR = "ResultsComparison/cropyield5min/aggregate_admin2"
SHAPEFILE_DIR = "GADM/gadm41_AFR_shp/gadm41_AFR_final.shp"
COUNTRY_SHAPEFILE_DIR = "GADM/gadm41_AFR_shp/gadm41_AFR_0_processed.shp"
CROP_TYPE = "Maize"
START_YEAR = 1982
END_YEAR = 2015

# Color scheme for yield visualization
YIELD_COLORS = ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679',
                '#41ab5d', '#238443', '#006837', '#004529']


def load_yield_data():
    """Load the aggregated yield data."""
    csv_file = f"{CROP_TYPE}_yield_admin2_{START_YEAR}_{END_YEAR}.csv"
    csv_path = os.path.join(INPUT_DIR, csv_file)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Yield data file not found: {csv_path}")

    print(f"Loading yield data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Calculate average yield per location
    avg_yield = df.groupby('FNID')['yield'].mean().reset_index()
    avg_yield.columns = ['FNID', 'avg_yield']

    print(f"Calculated average yields for {len(avg_yield)} locations")
    print(f"Average yield range: {avg_yield['avg_yield'].min():.2f} - {avg_yield['avg_yield'].max():.2f} tons/ha")

    return avg_yield


def load_shapefile():
    """Load the shapefile with administrative boundaries."""
    if not os.path.exists(SHAPEFILE_DIR):
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_DIR}")

    print(f"Loading admin2 shapefile: {SHAPEFILE_DIR}")
    gdf = gpd.read_file(SHAPEFILE_DIR)

    # Ensure we have the FNID column
    if 'FNID' not in gdf.columns:
        raise ValueError("Shapefile must contain 'FNID' column")

    print(f"Loaded {len(gdf)} administrative units from shapefile")
    return gdf


def load_country_shapefile():
    """Load the country-level shapefile for boundary overlay."""
    if not os.path.exists(COUNTRY_SHAPEFILE_DIR):
        print(f"Warning: Country shapefile not found: {COUNTRY_SHAPEFILE_DIR}")
        print("Proceeding without country boundaries...")
        return None

    print(f"Loading country shapefile: {COUNTRY_SHAPEFILE_DIR}")
    gdf_country = gpd.read_file(COUNTRY_SHAPEFILE_DIR)
    print(f"Loaded {len(gdf_country)} countries for boundary overlay")
    return gdf_country


def create_yield_map(yield_data, gdf, gdf_country=None):
    """Create a map visualization of average yields."""
    # Merge yield data with shapefile
    merged_gdf = gdf.merge(yield_data, on='FNID', how='left')

    # Filter out locations without yield data
    merged_gdf = merged_gdf.dropna(subset=['avg_yield'])

    print(f"Creating map for {len(merged_gdf)} locations with yield data")

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    # Create custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("yield_cmap", YIELD_COLORS)

    # Plot the map
    merged_gdf.plot(
        column='avg_yield',
        cmap=cmap,
        linewidth=0.3,
        edgecolor='black',
        ax=ax,
        legend=True,
        legend_kwds={
            'label': 'Average Maize Yield (tons/ha)',
            'orientation': 'horizontal',
            'shrink': 0.8,
            'aspect': 30
        }
    )

    # Overlay country boundaries if available
    if gdf_country is not None:
        gdf_country.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Customize the plot
    ax.set_title(f'Average Maize Yield by Administrative Unit\n({START_YEAR}-{END_YEAR})',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Create custom legend with yield ranges
    yield_min = merged_gdf['avg_yield'].min()
    yield_max = merged_gdf['avg_yield'].max()
    yield_range = yield_max - yield_min

    # Create legend elements
    legend_elements = []
    for i, color in enumerate(YIELD_COLORS):
        lower = yield_min + (i * yield_range / len(YIELD_COLORS))
        upper = yield_min + ((i + 1) * yield_range / len(YIELD_COLORS))
        if i == len(YIELD_COLORS) - 1:
            label = '.2f'
        else:
            label = '.2f'
        legend_elements.append(Patch(facecolor=color, edgecolor='black', linewidth=0.5,
                                   label=label))

    # Add statistics text box
    stats_text = '.2f'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=10, verticalalignment='bottom')

    # Adjust layout and save
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(INPUT_DIR, f'{CROP_TYPE}_yield_map_{START_YEAR}_{END_YEAR}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to: {output_path}")

    # Also save as PDF for better quality
    pdf_path = os.path.join(INPUT_DIR, f'{CROP_TYPE}_yield_map_{START_YEAR}_{END_YEAR}.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Map also saved as PDF: {pdf_path}")

    return fig, ax


def main():
    """Main function to create the yield map."""
    try:
        # Load data
        yield_data = load_yield_data()
        gdf = load_shapefile()
        gdf_country = load_country_shapefile()

        # Create and save the map
        fig, ax = create_yield_map(yield_data, gdf, gdf_country)

        # Show the plot
        plt.show()

        print("\nMap creation completed successfully!")

    except Exception as e:
        print(f"Error creating yield map: {e}")
        raise


if __name__ == "__main__":
    main()
