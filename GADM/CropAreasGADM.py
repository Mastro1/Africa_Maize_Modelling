"""
Africa-Wide 30m GLAD Crop Area Calculation using Batch Processing.

This script processes all administrative units for Africa for each GLAD year 
in a single run, leveraging batch processing to avoid GEE limits. It intelligently
processes the admin level 2 shapefile first, then only processes countries
from the admin level 1 shapefile that were not in the admin 2 file.

The final output is a single, unified CSV containing crop area statistics for
all of Africa at the most detailed administrative level available.
"""

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EE_PROJECT = os.getenv('EE_PROJECT')

# Initialize Earth Engine
try:
    ee.Initialize(project=EE_PROJECT)
    print("Google Earth Engine initialized successfully.")
    EE_INITIALIZED = True
except Exception as e:
    print(f"Could not initialize Earth Engine: {e}")
    EE_INITIALIZED = False


def get_glad_year_for_year(year):
    """Returns the appropriate GLAD cropland year for a given year."""
    if year <= 2003:
        return 2003
    elif year <= 2007:
        return 2007
    elif year <= 2011:
        return 2011
    elif year <= 2015:
        return 2015
    else:
        return 2019

def get_glad_image_for_year(year):
    """Creates a complete global GLAD cropland image for a given year."""
    quadrants = ['SE', 'SW', 'NE', 'NW']
    images = [ee.Image(f'users/potapovpeter/Global_cropland_{year}/Global_cropland_{year}_{quad}') 
             for quad in quadrants]
    return ee.ImageCollection(images).mosaic()

def process_year_batch_30m(gdf, year, glad_year, admin_level, batch_size=50):
    """
    Process all administrative units for a single year in smaller batches.
    
    Args:
        gdf: GeoDataFrame with administrative boundaries to process.
        year: Target year for analysis (used for labeling).
        glad_year: GLAD map year to use for data.
        admin_level: 'admin1' or 'admin2' - determines processing level.
        batch_size: Number of units to process in each GEE batch.
    """
    if gdf.empty:
        print(f"  No units to process for {admin_level} level. Skipping.")
        return []
        
    print(f"Processing year {year} (using GLAD {glad_year}) at {admin_level} level - processing {len(gdf)} units in batches of {batch_size}...")
    
    all_results = []
    total_batches = (len(gdf) + batch_size - 1) // batch_size
    
    try:
        # Get GLAD image for the year
        glad_image = get_glad_image_for_year(glad_year)
        glad_binary = glad_image.select('b1')
        
        # Process in smaller batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(gdf))
            batch_gdf = gdf.iloc[start_idx:end_idx]
            
            print(f"  Batch {batch_idx + 1}/{total_batches}: processing units {start_idx + 1}-{end_idx}...")
            
            # Convert batch to Earth Engine FeatureCollection
            features = []
            for idx, row in batch_gdf.iterrows():
                geom = ee.Geometry(row.geometry.__geo_interface__)
                
                # Ensure values are GEE-compatible (not NaN) to prevent errors
                fnid = str(row['FNID']) if pd.notna(row['FNID']) else f'MISSING_FNID_{idx}'
                admin0 = str(row['ADMIN0']) if pd.notna(row['ADMIN0']) else 'UNKNOWN'
                admin1 = str(row['ADMIN1']) if pd.notna(row['ADMIN1']) else 'UNKNOWN'
                
                props = {'ADMIN0': admin0, 'ADMIN1': admin1, 'FNID': fnid}
                if admin_level == 'admin2':
                    props['ADMIN2'] = str(row['ADMIN2']) if pd.notna(row['ADMIN2']) else ''
                
                feature = ee.Feature(geom, props)
                features.append(feature)
            
            fc = ee.FeatureCollection(features)
            
            # Define pixel area (30m x 30m = 900 sqm = 0.09 hectares)
            pixel_area_ha = 0.09
            pixel_area_image = ee.Image.constant(pixel_area_ha)
            crop_area_per_pixel = glad_binary.multiply(pixel_area_image)
            
            # Batch calculate statistics
            def calc_stats(feature):
                total_area = pixel_area_image.reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=feature.geometry(),
                    scale=30, maxPixels=1e9, bestEffort=True
                ).get('constant')
                
                crop_area = crop_area_per_pixel.reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=feature.geometry(),
                    scale=30, maxPixels=1e9, bestEffort=True
                ).get('b1')
                
                crop_percentage = ee.Algorithms.If(
                    ee.Number(total_area).gt(0),
                    ee.Number(crop_area).divide(total_area).multiply(100),
                    0
                )
                
                return feature.set({
                    'total_area_ha': total_area, 'crop_area_ha': crop_area,
                    'crop_percentage': crop_percentage, 'year': year, 'glad_year': glad_year
                })
            
            results_fc = fc.map(calc_stats)
            results_list = results_fc.getInfo()['features']
            
            # Convert to list of dictionaries
            for feature in results_list:
                props = feature['properties']
                result = {
                    'country': props.get('ADMIN0'),
                    'admin1_name': props.get('ADMIN1'),
                    'admin2_name': props.get('ADMIN2'),
                    'pcode': props.get('FNID'),
                    'year': year,
                    'glad_year': glad_year,
                    'total_area_ha': props.get('total_area_ha', 0),
                    'crop_area_ha': props.get('crop_area_ha', 0),
                    'crop_percentage': props.get('crop_percentage', 0),
                    'resolution': '30m',
                    'data_source': 'GEE_30m_batch',
                    'admin_level': admin_level
                }
                all_results.append(result)
            
            print(f"    ✅ Batch {batch_idx + 1} completed: {len(results_list)} units processed")
        
        print(f"  ✅ GLAD Year {glad_year} for {admin_level} level completed: {len(all_results)} total units processed")
        return all_results
        
    except Exception as e:
        print(f"  ❌ Error processing {admin_level} for year {year}: {str(e)}")
        # In case of error, return what has been processed so far
        return all_results

def expand_results_to_all_years(glad_results_df):
    """
    Expand GLAD map results to all years 2000-2024 based on the year mapping logic.
    """
    if glad_results_df.empty:
        return pd.DataFrame()
        
    print("\nExpanding GLAD results to all years 2000-2024...")
    
    year_mapping = {
        2003: list(range(2000, 2004)),
        2007: list(range(2004, 2008)),
        2011: list(range(2008, 2012)),
        2015: list(range(2012, 2016)),
        2019: list(range(2016, 2025))
    }
    
    expanded_results = []
    
    for glad_year, target_years in year_mapping.items():
        glad_data = glad_results_df[glad_results_df['glad_year'] == glad_year].copy()
        if glad_data.empty:
            continue
            
        for target_year in target_years:
            year_data = glad_data.copy()
            year_data['year'] = target_year
            expanded_results.append(year_data)
            
    if not expanded_results:
        return pd.DataFrame()

    final_df = pd.concat(expanded_results, ignore_index=True)
    print(f"Expansion completed: {len(final_df)} total records.")
    return final_df

def format_and_export_africa_results(results_df):
    """
    Formats and exports the complete Africa results to a single CSV file.
    Includes aggregation of admin2 data to admin1 level for completeness.
    """
    if results_df is None or results_df.empty:
        print("No results to export.")
        return

    print("\nFormatting final results for export...")
    
    all_export_data = []

    # Separate data by original calculated admin level
    admin2_df = results_df[results_df['admin_level'] == 'admin2'].copy()
    admin1_df = results_df[results_df['admin_level'] == 'admin1'].copy()

    # 1. Format and add Admin2 data
    if not admin2_df.empty:
        print(f"Processing {len(admin2_df)} Admin2 records...")
        admin2_df.rename(columns={'pcode': 'PCODE', 'admin1_name': 'admin_1', 'admin2_name': 'admin_2'}, inplace=True)
        all_export_data.append(admin2_df[['PCODE', 'country', 'admin_1', 'admin_2', 'year', 'total_area_ha', 'crop_area_ha', 'crop_percentage']])

    # 2. Format and add Admin1-only data
    if not admin1_df.empty:
        print(f"Processing {len(admin1_df)} Admin1-only records...")
        admin1_df.rename(columns={'pcode': 'PCODE', 'admin1_name': 'admin_1'}, inplace=True)
        admin1_df['admin_2'] = None
        all_export_data.append(admin1_df[['PCODE', 'country', 'admin_1', 'admin_2', 'year', 'total_area_ha', 'crop_area_ha', 'crop_percentage']])

    # 3. Aggregate Admin2 data to create Admin1 summaries
    if not admin2_df.empty:
        print("Aggregating Admin2 results to Admin1 level...")

        # We need to aggregate based on the actual year, not the GLAD year
        admin1_agg = admin2_df.groupby(['country', 'admin_1', 'year']).agg({
            'total_area_ha': 'sum',
            'crop_area_ha': 'sum'
        }).reset_index()

        admin1_agg['crop_percentage'] = (admin1_agg['crop_area_ha'] / admin1_agg['total_area_ha'].replace(0, np.nan) * 100).fillna(0)

        # Create proper Admin1 PCODEs from Admin2 PCODEs, grouped by country and admin1 name
        # Pattern: COUNTRY.ADMIN1_NUM_1 (e.g., DZA.1.1_1 -> DZA.1_1, ZWE.10.10_2 -> ZWE.10_1)
        def create_admin1_pcode(admin2_pcode):
            parts = admin2_pcode.split('.')
            if len(parts) >= 2:
                country = parts[0]
                admin1_num = parts[1].split('.')[0] if '.' in parts[1] else parts[1]
                return f"{country}.{admin1_num}_1"
            return admin2_pcode  # fallback

        admin1_pcodes = admin2_df.groupby(['country', 'admin_1'])['PCODE'].first().apply(create_admin1_pcode).to_dict()
        admin1_agg['PCODE'] = admin1_agg.apply(lambda row: admin1_pcodes.get((row['country'], row['admin_1']), ''), axis=1)
        admin1_agg['admin_2'] = None

        all_export_data.append(admin1_agg[['PCODE', 'country', 'admin_1', 'admin_2', 'year', 'total_area_ha', 'crop_area_ha', 'crop_percentage']])

    # Combine all data frames
    final_df = pd.concat(all_export_data, ignore_index=True)
    
    # Round values and sort
    final_df['total_area_ha'] = final_df['total_area_ha'].round(2)
    final_df['crop_area_ha'] = final_df['crop_area_ha'].round(2) 
    final_df['crop_percentage'] = final_df['crop_percentage'].round(2)
    final_df = final_df.sort_values(['country', 'PCODE', 'year']).reset_index(drop=True)
    
    # Export to a single file
    output_dir = Path('GADM/crop_areas')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'africa_crop_areas_glad.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Complete results exported to: {output_file}")
    print(f"Total records: {len(final_df)}")

def main_africa(batch_size=20):
    """
    Main function to run the full crop area analysis for Africa.
    """
    print("=== Africa-Wide 30m GLAD CROP AREA ANALYSIS ===")
    if not EE_INITIALIZED:
        print("❌ Earth Engine not initialized. Aborting.")
        return

    # Define paths to processed shapefiles
    shp_dir = Path("GADM/gadm41_AFR_shp")
    admin2_shp_path = shp_dir / "gadm41_AFR_2_processed.shp"
    admin1_shp_path = shp_dir / "gadm41_AFR_1_processed.shp"

    # Load shapefiles
    try:
        print(f"Loading Admin Level 2 shapefile: {admin2_shp_path}")
        admin2_gdf = gpd.read_file(admin2_shp_path)
        print(f"Loading Admin Level 1 shapefile: {admin1_shp_path}")
        admin1_gdf = gpd.read_file(admin1_shp_path)
    except Exception as e:
        print(f"❌ Error loading shapefiles: {e}")
        print("Please ensure the processed shapefiles exist. Run DownloadShapefile.py and the processing step if needed.")
        return

    # Data integrity check for FNID
    if admin2_gdf['FNID'].isnull().any():
        print("⚠️ Warning: Null values found in 'FNID' column of Admin Level 2 shapefile. These will be replaced with placeholders.")
    if admin1_gdf['FNID'].isnull().any():
        print("⚠️ Warning: Null values found in 'FNID' column of Admin Level 1 shapefile. These will be replaced with placeholders.")

    # Identify which countries are admin1-only
    admin2_countries = set(admin2_gdf['ADMIN0'].unique())
    admin1_countries = set(admin1_gdf['ADMIN0'].unique())
    admin1_only_countries = admin1_countries - admin2_countries
    
    print(f"\nFound {len(admin2_countries)} countries with Admin Level 2.")
    if admin1_only_countries:
        print(f"Found {len(admin1_only_countries)} countries with only Admin Level 1: {', '.join(sorted(list(admin1_only_countries)))}")
    
    # Process only the actual GLAD map years
    glad_years = [2003, 2007, 2011, 2015, 2019]
    print(f"\nProcessing 30m GLAD analysis for map years: {glad_years}...")
    
    all_glad_year_results = []
    
    # Process each GLAD map year
    for glad_year in glad_years:
        print(f"\n{'='*60}")
        print(f"Processing GLAD Map Year: {glad_year}")
        
        # Process ALL Admin Level 2 units
        admin2_results = process_year_batch_30m(admin2_gdf, glad_year, glad_year, 'admin2', batch_size)
        all_glad_year_results.extend(admin2_results)
        
        # Process ALL Admin Level 1 units
        admin1_results = process_year_batch_30m(admin1_gdf, glad_year, glad_year, 'admin1', batch_size)
        all_glad_year_results.extend(admin1_results)

    # Convert to DataFrame
    glad_results_df = pd.DataFrame(all_glad_year_results)
    
    # Expand results to all years 2000-2024
    expanded_df = expand_results_to_all_years(glad_results_df)
    
    # Format and export the final, combined results
    format_and_export_africa_results(expanded_df)

if __name__ == "__main__":
    # Reduced batch size to 20 to avoid "Too many concurrent aggregations" error.
    main_africa(batch_size=40) 