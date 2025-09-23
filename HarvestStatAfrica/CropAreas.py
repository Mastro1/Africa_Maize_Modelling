"""
Optimized 30m GLAD crop area calculation using batch processing.
This version processes all admin units for each year in a single API call,
dramatically reducing processing time.
Supports both admin1-only and admin1+admin2 administrative levels.
"""

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
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


def detect_admin_level(gdf):
    """
    Detect the available administrative level for the country.
    Returns 'admin1' if only admin1 is available, 'admin2' if both admin1 and admin2 are available.
    """
    # Check if ADMIN2 column exists and has meaningful data
    if 'ADMIN2' not in gdf.columns:
        return 'admin1'
    
    # Check if ADMIN2 has meaningful values (not all null/empty/same as ADMIN1)
    admin2_values = gdf['ADMIN2'].dropna()
    if len(admin2_values) == 0:
        return 'admin1'
    
    # Check if ADMIN2 values are different from ADMIN1 (indicating real admin2 level)
    admin1_values = gdf['ADMIN1'].dropna()
    if len(admin2_values) > 0 and len(set(admin2_values) - set(admin1_values)) > 0:
        return 'admin2'
    else:
        return 'admin1'

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

def process_year_batch_30m(gdf, year, glad_year, admin_level, batch_size=10):
    """
    Process all administrative units for a single year in smaller batches.
    This avoids the "Too many concurrent aggregations" error from GEE.
    
    Args:
        gdf: GeoDataFrame with administrative boundaries
        year: Target year for analysis
        glad_year: GLAD map year to use
        admin_level: 'admin1' or 'admin2' - determines processing level
        batch_size: Number of units to process in each batch
    """
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
                
                # Create feature properties based on admin level
                if admin_level == 'admin2':
                    feature = ee.Feature(geom, {
                        'admin1': row['ADMIN1'],
                        'admin2': row['ADMIN2'], 
                        'fnid': row['FNID'],
                        'index': idx
                    })
                else:  # admin1 only
                    feature = ee.Feature(geom, {
                        'admin1': row['ADMIN1'],
                        'fnid': row['FNID'],
                        'index': idx
                    })
                features.append(feature)
            
            fc = ee.FeatureCollection(features)
            
            # Define pixel area (30m = 0.09 hectares)
            pixel_area_ha = 0.09
            pixel_area_image = ee.Image.constant(pixel_area_ha)
            crop_area_per_pixel = glad_binary.multiply(pixel_area_image)
            
            # Batch calculate statistics for this batch
            def calc_stats(feature):
                # Total area
                total_area = pixel_area_image.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=feature.geometry(),
                    scale=30,
                    maxPixels=1e9,
                    bestEffort=True
                ).get('constant')
                
                # Crop area  
                crop_area = crop_area_per_pixel.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=feature.geometry(),
                    scale=30,
                    maxPixels=1e9,
                    bestEffort=True
                ).get('b1')
                
                # Calculate percentage
                crop_percentage = ee.Algorithms.If(
                    ee.Number(total_area).gt(0),
                    ee.Number(crop_area).divide(total_area).multiply(100),
                    0
                )
                
                return feature.set({
                    'total_area_ha': total_area,
                    'crop_area_ha': crop_area,
                    'crop_percentage': crop_percentage,
                    'year': year,
                    'glad_year': glad_year
                })
            
            # Apply calculation to all features in this batch
            results_fc = fc.map(calc_stats)
            
            # Extract results for this batch
            results_list = results_fc.getInfo()['features']
            
            # Convert to list of dictionaries
            for feature in results_list:
                props = feature['properties']
                
                # Create result record based on admin level
                if admin_level == 'admin2':
                    result = {
                        'admin1_name': props['admin1'],
                        'admin2_name': props['admin2'],
                        'pcode': props['fnid'],
                        'year': year,
                        'glad_year': glad_year,
                        'total_area_ha': props.get('total_area_ha', 0),
                        'crop_area_ha': props.get('crop_area_ha', 0),
                        'crop_percentage': props.get('crop_percentage', 0),
                        'resolution': '30m',
                        'data_source': 'GEE_30m_batch',
                        'admin_level': 'admin2'
                    }
                else:  # admin1 only
                    result = {
                        'admin1_name': props['admin1'],
                        'admin2_name': None,
                        'pcode': props['fnid'],
                        'year': year,
                        'glad_year': glad_year,
                        'total_area_ha': props.get('total_area_ha', 0),
                        'crop_area_ha': props.get('crop_area_ha', 0),
                        'crop_percentage': props.get('crop_percentage', 0),
                        'resolution': '30m',
                        'data_source': 'GEE_30m_batch',
                        'admin_level': 'admin1'
                    }
                
                all_results.append(result)
            
            print(f"    ✅ Batch {batch_idx + 1} completed: {len(results_list)} units processed")
        
        print(f"  ✅ Year {year} completed: {len(all_results)} total units processed")
        return all_results
        
    except Exception as e:
        print(f"  ❌ Error processing year {year}: {str(e)}")
        return []

def expand_results_to_all_years(glad_results_df):
    """
    Expand GLAD map results to all years 2000-2024 based on the year mapping logic.
    Each GLAD map applies to multiple years, so we duplicate the results accordingly.
    
    Args:
        glad_results_df: DataFrame with results for GLAD map years (2003, 2007, 2011, 2015, 2019)
    
    Returns:
        DataFrame with results for all years 2000-2024
    """
    print("Expanding GLAD results to all years 2000-2024...")
    
    # Define the year mapping (same as get_glad_year_for_year function)
    year_mapping = {
        2003: list(range(2000, 2004)),  # 2000-2003
        2007: list(range(2004, 2008)),  # 2004-2007  
        2011: list(range(2008, 2012)),  # 2008-2011
        2015: list(range(2012, 2016)),  # 2012-2015
        2019: list(range(2016, 2025))   # 2016-2024
    }
    
    expanded_results = []
    
    for glad_year, target_years in year_mapping.items():
        # Get results for this GLAD year
        glad_data = glad_results_df[glad_results_df['glad_year'] == glad_year].copy()
        
        if len(glad_data) == 0:
            print(f"  ⚠️ No data found for GLAD year {glad_year}")
            continue
            
        # Duplicate results for each target year
        for target_year in target_years:
            year_data = glad_data.copy()
            year_data['year'] = target_year  # Update the year column
            expanded_results.append(year_data)
            
        print(f"  ✅ GLAD {glad_year} → Years {target_years[0]}-{target_years[-1]} ({len(target_years)} years)")
    
    # Combine all results
    final_df = pd.concat(expanded_results, ignore_index=True)
    
    print(f"Expansion completed: {len(final_df)} total records ({len(final_df) // len(glad_results_df)} years per GLAD map)")
    return final_df

def process_optimized_30m_smart(country_name, batch_size=10):
    """
    Smart processing: Only process actual GLAD map years, then expand to all years.
    Automatically detects admin level and processes accordingly.
    """

    COUNTRY_NAME = country_name

    if not EE_INITIALIZED:
        print("Earth Engine not initialized.")
        return None, None
    
    # Load administrative boundaries
    gdf = gpd.read_file('HarvestStatAfrica/data/hvstat_africa_boundary_v1.0.gpkg')
    country_gdf = gdf[gdf['ADMIN0'] == COUNTRY_NAME].copy()
    
    # Detect administrative level
    admin_level = detect_admin_level(country_gdf)
    print(f"Detected administrative level: {admin_level}")
    print(f"Loaded {len(country_gdf)} administrative units")
    
    # Process only the actual GLAD map years
    glad_years = [2003, 2007, 2011, 2015, 2019]
    print(f"Processing 30m GLAD analysis for GLAD map years: {glad_years}")
    print(f"Using batch size: {batch_size} admin units per batch")
    
    all_results = []
    
    # Process each GLAD map year
    for glad_year in glad_years:
        print(f"\n{'='*50}")
        print(f"Processing GLAD map year: {glad_year}")
        year_results = process_year_batch_30m(country_gdf, glad_year, glad_year, admin_level, batch_size)
        all_results.extend(year_results)
    
    # Convert to DataFrame
    glad_results_df = pd.DataFrame(all_results)
    
    # Expand results to all years 2000-2024
    print(f"\n{'='*50}")
    expanded_results_df = expand_results_to_all_years(glad_results_df)
    
    return expanded_results_df, admin_level

def format_and_export_30m_complete(results_df, admin_level, country_name, include_admin1_aggregation=True):
    """
    Format and export complete results based on detected admin level.
    
    Args:
        results_df: DataFrame with processing results
        admin_level: 'admin1' or 'admin2' - determines export format
        include_admin1_aggregation: If True and admin_level is 'admin2', also create admin1 aggregations
    """

    COUNTRY_NAME = country_name

    if results_df is None or len(results_df) == 0:
        print("No results to export")
        return
    
    print(f"Formatting results for export (admin level: {admin_level})...")
    
    all_export_data = []
    
    if admin_level == 'admin1':
        # Process Admin1-only data
        print("Processing Admin1 level data...")
        admin1_formatted = results_df.copy()
        admin1_formatted['country'] = COUNTRY_NAME
        # admin1_formatted['country_code'] = COUNTRY_CODE 
        admin1_formatted['admin_1'] = admin1_formatted['admin1_name']
        admin1_formatted['admin_2'] = None  # No admin2 level
        admin1_formatted['PCODE'] = admin1_formatted['pcode']
        
        # Select and order columns for admin1
        admin1_export = admin1_formatted[['PCODE', 'country', 'admin_1', 'admin_2', 'year', 
                                         'total_area_ha', 'crop_area_ha', 'crop_percentage']].copy()
        all_export_data.append(admin1_export)
        
    else:  # admin_level == 'admin2'
        # Process Admin2 level data (what we calculated)
        print("Processing Admin2 level data...")
        admin2_formatted = results_df.copy()
        admin2_formatted['country'] = COUNTRY_NAME
        # admin2_formatted['country_code'] = COUNTRY_CODE 
        admin2_formatted['admin_1'] = admin2_formatted['admin1_name']
        admin2_formatted['admin_2'] = admin2_formatted['admin2_name']
        admin2_formatted['PCODE'] = admin2_formatted['pcode']
        
        # Select and order columns for admin2
        admin2_export = admin2_formatted[['PCODE', 'country', 'admin_1', 'admin_2', 'year', 
                                         'total_area_ha', 'crop_area_ha', 'crop_percentage']].copy()
        all_export_data.append(admin2_export)
        
        # Create Admin1 level data by aggregating Admin2 results
        if include_admin1_aggregation:
            print("Aggregating Admin2 results to Admin1 level...")
            
            admin1_agg = admin2_formatted.groupby(['admin1_name', 'year', 'glad_year']).agg({
                'total_area_ha': 'sum',
                'crop_area_ha': 'sum'
            }).reset_index()
            
            # Calculate crop percentage for admin1
            admin1_agg['crop_percentage'] = (admin1_agg['crop_area_ha'] / admin1_agg['total_area_ha'] * 100)
            
            # Create admin1 PCODEs (take first admin2 PCODE and modify last digit to 0)
            admin1_pcodes = {}
            for admin1_name in admin1_agg['admin1_name'].unique():
                first_admin2_pcode = admin2_formatted[admin2_formatted['admin1_name'] == admin1_name]['pcode'].iloc[0]
                admin1_pcode = first_admin2_pcode[:-1] + '0'
                admin1_pcodes[admin1_name] = admin1_pcode
            
            admin1_agg['PCODE'] = admin1_agg['admin1_name'].map(admin1_pcodes)
            admin1_agg['country'] = COUNTRY_NAME
            # admin1_agg['country_code'] = COUNTRY_CODE
            admin1_agg['admin_1'] = admin1_agg['admin1_name']
            admin1_agg['admin_2'] = None  # Admin1 level doesn't have admin2
            
            # Select and order columns for admin1
            admin1_export = admin1_agg[['PCODE', 'country', 'admin_1', 'admin_2', 'year', 
                                       'total_area_ha', 'crop_area_ha', 'crop_percentage']].copy()
            all_export_data.append(admin1_export)
    
    # Combine all data
    final_df = pd.concat(all_export_data, ignore_index=True)
    
    # Round values
    final_df['total_area_ha'] = final_df['total_area_ha'].round(2)
    final_df['crop_area_ha'] = final_df['crop_area_ha'].round(2) 
    final_df['crop_percentage'] = final_df['crop_percentage'].round(2)
    
    # Sort by PCODE and year
    final_df = final_df.sort_values(['PCODE', 'year']).reset_index(drop=True)
    
    # Export
    output_file = f'HarvestStatAfrica/crop_areas/{COUNTRY_NAME}_crop_areas_glad.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"Complete results exported to: {output_file}")
    print(f"Total records: {len(final_df)}")
    print(f"Years covered: {sorted(final_df['year'].unique())}")
    
    if admin_level == 'admin1':
        print(f"Admin1 records: {len(final_df)}")
    else:
        print(f"Admin1 records: {len(final_df[final_df['admin_2'].isna()])}")
        print(f"Admin2 records: {len(final_df[final_df['admin_2'].notna()])}")
    
    return output_file

def main():
    """Main function for smart optimized processing."""
    print("=== SMART 30m GLAD CROP AREA ANALYSIS ===")
    print("Processing only GLAD map years, then expanding to all years 2000-2024")
    print("Auto-detecting administrative level (admin1 or admin2)")
    
    # Process GLAD map years and expand to all years
    results, admin_level = process_optimized_30m_smart(country_name="Zambia", batch_size=10)
    
    if results is not None and len(results) > 0:
        # Export complete results based on detected admin level
        include_admin1_agg = (admin_level == 'admin2')  # Only aggregate if we have admin2 data
        output_file = format_and_export_30m_complete(results, admin_level, "Zambia", include_admin1_agg)
        
        # Show summary
        print(f"\n=== SUMMARY ===")
        print(f"Successfully processed {len(results)} total records")
        print(f"Administrative level: {admin_level}")
        print(f"GLAD map years processed: [2003, 2007, 2011, 2015, 2019]")
        print(f"Expanded to years: 2000-2024 ({len(results['year'].unique())} years)")
        print(f"Admin units: {len(results[results['year'] == 2019])} per year")
        print(f"Output file: {output_file}")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main() 