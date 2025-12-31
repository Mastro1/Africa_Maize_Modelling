"""
This script aggregates the SPAM maize yield data at the admin 2 level.
It extracts average yield values for each administrative unit (FNID) from SPAM GeoTIFF files.
"""

import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


INPUT_DIR = "ResultsComparison/SPAM/data"
OUTPUT_DIR = "ResultsComparison/SPAM/aggregate_admin2"
SHAPEFILE_DIR = "GADM/gadm41_AFR_shp/gadm41_AFR_final.shp"
YEARS = [2000, 2005, 2010, 2017, 2020]  # Available SPAM years
CROP_TYPE = "SPAM_Maize"


def load_shapefile(shapefile_path):
    """Load the shapefile with administrative boundaries."""
    print(f"Loading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)

    # Ensure the shapefile has the correct CRS (same as the raster data)
    if gdf.crs is None:
        print("Warning: Shapefile has no CRS. Setting to EPSG:4326 (WGS84)")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs != "EPSG:4326":
        print(f"Warning: Shapefile CRS {gdf.crs} differs from raster CRS EPSG:4326. Reprojecting...")
        gdf = gdf.to_crs("EPSG:4326")

    print(f"Loaded {len(gdf)} administrative units with CRS: {gdf.crs}")
    return gdf


def extract_yield_for_year(year, raster_path, gdf):
    """Extract average yield for each administrative unit from a single year's raster."""
    print(f"Processing year {year}...")

    # First, get the actual nodata value from the raster
    with rasterio.open(raster_path) as src:
        actual_nodata = src.nodata
        print(f"Using nodata value: {actual_nodata}")

    # Calculate zonal statistics (mean) for each polygon
    stats = zonal_stats(
        gdf,
        raster_path,
        stats=['mean', 'count'],
        nodata=actual_nodata  # Use the actual nodata value from raster
    )

    # Extract mean values and create results list
    year_results = []
    valid_count = 0
    for idx, stat in enumerate(stats):
        fnid = gdf.iloc[idx]['FNID']
        mean_yield = stat['mean']
        pixel_count = stat.get('count', 0)

        # Skip if no data, invalid values, or no valid pixels
        if (mean_yield is not None and
            not pd.isna(mean_yield) and
            not np.isinf(mean_yield) and
            pixel_count > 0 and
            mean_yield >= 0):  # Allow zero yields but not negative

            # Convert from kg/ha to tonnes/ha (SPAM default is kg/ha, except 2020 which is already in tonnes/ha)
            if year == 2020:
                mean_yield_tonnes = mean_yield  # 2020 is already in tonnes/ha
            else:
                mean_yield_tonnes = mean_yield / 1000.0  # Other years are in kg/ha

            year_results.append({
                'year': year,
                'FNID': fnid,
                'yield': round(mean_yield_tonnes, 4)
            })
            valid_count += 1

    print(f"Extracted {valid_count} valid yield values for {year}")
    return year_results


def process_all_years(gdf):
    """Process all available SPAM years."""
    all_results = []

    for year in tqdm(YEARS):
        raster_filename = f"spam_maiz_{year}.tif"
        raster_path = os.path.join(INPUT_DIR, raster_filename)

        if not os.path.exists(raster_path):
            print(f"Warning: Raster file not found: {raster_path}")
            continue

        try:
            year_results = extract_yield_for_year(year, raster_path, gdf)
            all_results.extend(year_results)
        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue

    return all_results


def main():
    """Main function to orchestrate the aggregation process."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load shapefile
    gdf = load_shapefile(SHAPEFILE_DIR)

    # Check required columns exist
    if 'FNID' not in gdf.columns:
        raise ValueError("Shapefile must contain 'FNID' column for administrative unit identification")

    # Process all years
    print("Starting SPAM maize yield aggregation process...")
    results = process_all_years(gdf)

    # Convert to DataFrame and save
    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(OUTPUT_DIR, f"spam_maize_yield_admin2_{YEARS[0]}_{YEARS[-1]}.csv")
        df.to_csv(output_path, index=False)

        print("\nAggregation complete!")
        print(f"Processed {len(df)} records")
        print(f"Years: {df['year'].min()} - {df['year'].max()}")
        print(f"Unique locations: {df['FNID'].nunique()}")
        print(f"Output saved to: {output_path}")

        # Summary statistics
        print("\nYield statistics (tonnes/ha):")
        print(df['yield'].describe())
    else:
        print("No valid data found to process!")


if __name__ == "__main__":
    main()