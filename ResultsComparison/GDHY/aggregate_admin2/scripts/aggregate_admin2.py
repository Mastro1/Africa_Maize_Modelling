"""
This script aggregates the GDHY maize yield data at the admin 2 level.
It extracts average yield values for each administrative unit (FNID) from NetCDF files.
Handles 0-360 longitude format by shifting to -180-180 before aggregation.
"""

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from rasterstats import zonal_stats
import numpy as np
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
INPUT_DIR = os.path.join("ResultsComparison", "GDHY", "data")
OUTPUT_DIR = os.path.join("ResultsComparison", "GDHY", "aggregate_admin2")
SHAPEFILE_DIR = os.path.join("GADM", "gadm41_AFR_shp", "gadm41_AFR_final.shp")
START_YEAR = 1981
END_YEAR = 2016
CROP_TYPE = "Maize"


def load_shapefile(shapefile_path):
    """Load the shapefile with administrative boundaries."""
    print(f"Loading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)

    # Ensure the shapefile has the correct CRS (EPSG:4326 for global lat/lon)
    if gdf.crs is None:
        print("Warning: Shapefile has no CRS. Setting to EPSG:4326 (WGS84)")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs != "EPSG:4326":
        print(f"Warning: Shapefile CRS {gdf.crs} differs from raster CRS EPSG:4326. Reprojecting...")
        gdf = gdf.to_crs("EPSG:4326")

    print(f"Loaded {len(gdf)} administrative units with CRS: {gdf.crs}")
    return gdf


def load_and_correct_raster(raster_path):
    """
    Load raster data and correct 0-360 longitude to -180-180.
    Returns the corrected data array and affine transform.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        bounds = src.bounds
        nodata = src.nodata
        
        # Check if we need to shift from 0-360 to -180-180
        if np.isclose(bounds.left, 0) and np.isclose(bounds.right, 360):
            # Shift data
            width = data.shape[1]
            shift = width // 2
            data = np.roll(data, shift, axis=1)
            
            # Update transform: New west is -180
            transform = Affine(src.transform.a, src.transform.b, -180.0,
                               src.transform.d, src.transform.e, src.transform.f)
        else:
            transform = src.transform

        # Handle nodata
        if nodata is not None:
            # We want masked values to be ignored by zonal_stats
            # zonal_stats expects a numpy array and a nodata value, or a masked array.
            # Here we ensure the data is consistent with the nodata value.
            # If the data was shifted, the nodata values shifted with it.
            pass
            
        return data, transform, nodata


def extract_yield_for_year(year, raster_path, gdf):
    """Extract average yield for each administrative unit from a single year's raster."""
    
    try:
        data, transform, nodata = load_and_correct_raster(raster_path)
    except Exception as e:
        print(f"Error loading raster for {year}: {e}")
        return []

    # Calculate zonal statistics
    # We pass the loaded array and transform directly
    stats = zonal_stats(
        gdf,
        data,
        affine=transform,
        stats=['mean', 'count'],
        nodata=nodata
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
            year_results.append({
                'year': year,
                'FNID': fnid,
                'yield': round(mean_yield, 4)
            })
            valid_count += 1

    # print(f"Extracted {valid_count} valid yield values for {year}")
    return year_results


def process_all_years(gdf):
    """Process all years from START_YEAR to END_YEAR."""
    all_results = []

    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Processing years"):
        raster_filename = f"yield_{year}.nc4"
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
    print("Starting yield aggregation process...")
    results = process_all_years(gdf)

    # Convert to DataFrame and save
    if results:
        df = pd.DataFrame(results)
        output_filename = f"{CROP_TYPE.lower()}_yield_admin2_{START_YEAR}_{END_YEAR}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        df.to_csv(output_path, index=False)

        print("\nAggregation complete!")
        print(f"Processed {len(df)} records")
        print(f"Years: {df['year'].min()} - {df['year'].max()}")
        print(f"Unique locations: {df['FNID'].nunique()}")
        print(f"Output saved to: {output_path}")

        # Summary statistics
        print("\nYield statistics:")
        print(df['yield'].describe())
    else:
        print("No valid data found to process!")


if __name__ == "__main__":
    main()
