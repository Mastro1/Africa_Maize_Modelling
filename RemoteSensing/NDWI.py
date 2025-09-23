"""
Extracts NDWI (Normalized Difference Water Index) from MODISfor a given country and date range, applying the GLAD cropland mask.

Author: Massimo Poretti
Date: 2025-06-06
"""





import ee
import geopandas as gpd
import matplotlib.pyplot as plt
# pandas is implicitly used by geopandas and for DataFrame creation if needed client-side
# os module is removed as progressive local saving is removed
from datetime import datetime # Retained for potential use, though dates are mainly strings now
from unidecode import unidecode
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EE_PROJECT = os.getenv('EE_PROJECT')
EE_OPT_URL = os.getenv('EE_OPT_URL')
GD_FOLDER = os.getenv('GD_FOLDER')
# Initialize Earth Engine globally for the module
# It's generally recommended to initialize EE once.
# If this module is imported elsewhere, this will run upon import.
EE_INITIALIZED = False
EE_INIT_ERROR = None
try:
    # ee.Authenticate() # User should have authenticated via CLI or previous runs
    # User's project ID from previous context
    ee.Initialize(project=EE_PROJECT, opt_url=EE_OPT_URL)
    print("Google Earth Engine initialized successfully for the module.")
    EE_INITIALIZED = True
except Exception as e:
    EE_INIT_ERROR = e
    print(f"Could not initialize Earth Engine for the module: {e}")
    print("Please ensure you have authenticated Earth Engine CLI and the project ID is correct.")

def get_glad_year_for_date(date_ee):
    """
    Returns the appropriate GLAD cropland year for a given date.
    GLAD data is available in four-year intervals:
    - 2000-2003: use 2003 map
    - 2004-2007: use 2007 map
    - 2008-2011: use 2011 map
    - 2012-2015: use 2015 map
    - 2016-2019: use 2019 map
    For years after 2019, use 2019 map
    """
    year = date_ee.get('year')
    return ee.Number(ee.Algorithms.If(
        year.lt(2004),  # 2000-2003
        2003,
        ee.Algorithms.If(
            year.lt(2008),  # 2004-2007
            2007,
            ee.Algorithms.If(
                year.lt(2012),  # 2008-2011
                2011,
                ee.Algorithms.If(
                    year.lt(2016),  # 2012-2015
                    2015,
                    ee.Algorithms.If(
                        year.lt(2020),  # 2016-2019
                        2019,
                        2019  # 2020 and beyond
                    )
                )
            )
        )
    ))


def get_glad_image_for_year(year):
    """
    Creates a complete global GLAD cropland image for a given year by mosaicking all four quadrants.
    
    Args:
        year (int): The year to get GLAD data for (2003, 2007, 2011, 2015, or 2019)
    
    Returns:
        ee.Image: A complete global GLAD cropland image
    """
    quadrants = ['SE', 'SW', 'NE', 'NW']
    images = [ee.Image(f'users/potapovpeter/Global_cropland_{year}/Global_cropland_{year}_{quad}') 
             for quad in quadrants]
    return ee.ImageCollection(images).mosaic()

def extractNDWI(country_name: str, start_date_str: str, end_date_str: str, 
              gpkg_path: str, output_filename_prefix: str, admin_level: int = 2):
    """
    Extracts mean, median, and mode of NDVI and EVI for administrative regions of a specified country 
    within a given date range, applying the GLAD cropland mask. Results are exported to Google Drive.

    Args:
        country_name (str): The name of the country (e.g., 'Zambia').
        start_date_str (str): Start date in 'YYYY-MM-DD' format.
        end_date_str (str): End date in 'YYYY-MM-DD' format.
        gpkg_path (str): Path to the GeoPackage file with administrative boundaries.
        output_filename_prefix (str): Prefix for the output CSV filename in Google Drive.
        admin_level (int): Administrative level to use (0, 1, or 2). 0 = country level. Defaults to 2.
    Returns:
        ee.batch.Task or None: The initiated Earth Engine task object, or None if an error occurred.
    """
    if not EE_INITIALIZED:
        print(f"Earth Engine not initialized. Error: {EE_INIT_ERROR}")
        return None

    if admin_level not in [0, 1, 2]:
        print(f"Invalid admin level: {admin_level}. Must be 0, 1, or 2.")
        return None

    print(f"Starting VI extraction for {country_name} from {start_date_str} to {end_date_str} at Admin level {admin_level}.")

    # 1. Load and filter administrative boundaries
    try:
        print(f"Loading administrative boundaries from: {gpkg_path}")
        world_boundaries = gpd.read_file(gpkg_path)
        country_boundaries_gdf = world_boundaries[world_boundaries['ADMIN0'] == country_name]
        if country_boundaries_gdf.empty:
            print(f"No data found for country: {country_name} in column 'ADMIN0'. Available (sample): {world_boundaries['ADMIN0'].unique()[:10]}")
            return None
        
        if admin_level == 0:
            # For admin level 0, create a single country boundary by dissolving all geometries
            # PERFORMANCE NOTE: This dissolves all admin 2 boundaries into one complex geometry.
            # The geometry is simplified both client-side and server-side to improve performance.
            print(f"Creating country-level boundary for {country_name}.")
            
            # Dissolve all geometries into a single country boundary and simplify for performance
            # Use unary_union with buffer for cleaner geometry without internal artifacts
            from shapely.ops import unary_union
            dissolved_geom = unary_union(country_boundaries_gdf.geometry).buffer(0)
            
            # Simplify the geometry to reduce complexity and improve Earth Engine performance
            # Use a tolerance that maintains the general shape but reduces vertex count
            simplified_geom = dissolved_geom.simplify(tolerance=0.01, preserve_topology=True)
            
            print(f"Original dissolved geometry vertices: {len(dissolved_geom.exterior.coords) if hasattr(dissolved_geom, 'exterior') else 'N/A'}")
            print(f"Simplified geometry vertices: {len(simplified_geom.exterior.coords) if hasattr(simplified_geom, 'exterior') else 'N/A'}")
            
            # Create a single feature for the entire country
            geojson_geom = gpd.GeoSeries([simplified_geom]).__geo_interface__['features'][0]['geometry']
            
            # Create ee.Feature with country-level properties
            country_feature = ee.Feature(ee.Geometry(geojson_geom, opt_evenOdd=True), 
                                       {'ADMIN_NAME': country_name, 'PCODE': f'{country_name}_COUNTRY'})
            
            # Further simplify the geometry in Earth Engine for optimal performance
            # This reduces the geometry complexity on the server side
            simplified_feature = country_feature.simplify(maxError=1000)  # 1000m tolerance
            country_admin_fc = ee.FeatureCollection([simplified_feature])
            
            print(f"Created and server-side simplified country boundary for {country_name}.")
            
        else:
            # For admin level 1 or 2, use existing logic
            admin_column = f'ADMIN{admin_level}'
            if admin_column not in country_boundaries_gdf.columns:
                print(f"Column {admin_column} not found in the GeoPackage. Available columns: {country_boundaries_gdf.columns.tolist()}")
                return None
                
            print(f"Found {len(country_boundaries_gdf)} Admin {admin_level} regions for {country_name}.")
            
            # Convert selected geometries to an ee.FeatureCollection
            features = []
            for index, row in country_boundaries_gdf.iterrows():
                shapely_geom = row['geometry']
                geojson_geom = gpd.GeoSeries([shapely_geom]).__geo_interface__['features'][0]['geometry']
                # Create ee.Feature with geometry and properties
                feature = ee.Feature(ee.Geometry(geojson_geom, opt_evenOdd=True), 
                                   {'ADMIN_NAME': row[admin_column], 'PCODE': row['FNID']})
                features.append(feature)
            country_admin_fc = ee.FeatureCollection(features)
        
        if country_admin_fc.size().getInfo() == 0:
            print("Failed to create Earth Engine FeatureCollection for the country's admin regions.")
            return None

    except Exception as e:
        print(f"Error loading or processing administrative boundaries: {e}")
        return None

    # 2. Define Image Collections with MONTHLY AGGREGATION APPROACH
    print("Setting up NDWI collection for monthly aggregation...")
    ndwi_collection = ee.ImageCollection("MODIS/MCD43A4_006_NDWI") \
        .filterDate(start_date_str, end_date_str) \
        .filterBounds(country_admin_fc.geometry())  # Filter by bounds first
    
    # DIAGNOSTIC: Check collection size
    collection_size = ndwi_collection.size()
    print(f"NDWI collection size after filtering: {collection_size.getInfo()} images")
    
    # Check if collection is empty
    if collection_size.getInfo() == 0:
        print(f"ERROR: No NDWI images found for {country_name} between {start_date_str} and {end_date_str}")
        print("Try expanding your date range or check if the country geometry is correct.")
        return None
    
    # DIAGNOSTIC: Check band names and properties of first image
    first_image = ee.Image(ndwi_collection.first())
    band_names = first_image.bandNames()
    print(f"NDWI image band names: {band_names.getInfo()}")
    
    # DIAGNOSTIC: Get basic info about the NDWI collection
    first_image_props = first_image.propertyNames()
    print(f"First NDWI image properties: {first_image_props.getInfo()}")
    
    # DIAGNOSTIC: Check NDWI value range in a small area to see if data exists
    test_region = country_admin_fc.geometry().centroid().buffer(10000)  # 10km buffer around centroid
    
    test_stats = first_image.select('NDWI').reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=test_region,
        scale=463.313,
        maxPixels=1e6
    )
    print(f"NDWI test stats in small region: {test_stats.getInfo()}")
    
    # PERFORMANCE FIX: Pre-load GLAD images ONCE instead of for every image
    print("Pre-loading GLAD cropland images (this is much more efficient)...")
    glad_images_dict = {
        2003: get_glad_image_for_year(2003),
        2007: get_glad_image_for_year(2007), 
        2011: get_glad_image_for_year(2011),
        2015: get_glad_image_for_year(2015),
        2019: get_glad_image_for_year(2019)
    }
    print("GLAD images loaded successfully.")
    
    # CREATE MONTHLY COMPOSITES
    print("Creating monthly NDWI composites...")
    def create_monthly_composite(start_date, end_date):
        """Create a monthly composite from MODIS NDWI data"""
        monthly_collection = ndwi_collection.filterDate(start_date, end_date)
        
        # Use median composite to reduce noise and fill gaps
        monthly_composite = monthly_collection.median().set({
            'system:time_start': ee.Date(start_date).millis(),
            'month': ee.Date(start_date).format('YYYY-MM'),
            'year': ee.Date(start_date).get('year')
        })
        
        return monthly_composite

    # Generate list of months between start and end dates
    start_date_ee = ee.Date(start_date_str)
    end_date_ee = ee.Date(end_date_str)
    
    # Create monthly composites
    monthly_composites = []
    current_date = start_date_ee
    
    while current_date.millis().lt(end_date_ee.millis()).getInfo():
        month_start = current_date.update(day=1)  # First day of month
        month_end = month_start.advance(1, 'month').advance(-1, 'day')  # Last day of month
        
        # Ensure we don't go beyond end date
        month_end = ee.Date(ee.Algorithms.If(
            month_end.millis().gt(end_date_ee.millis()),
            end_date_ee,
            month_end
        ))
        
        monthly_composite = create_monthly_composite(month_start, month_end)
        monthly_composites.append(monthly_composite)
        
        # Move to next month
        current_date = month_start.advance(1, 'month')
    
    # Convert to ImageCollection
    monthly_ndwi_collection = ee.ImageCollection(monthly_composites)
    monthly_collection_size = monthly_ndwi_collection.size()
    print(f"Created {monthly_collection_size.getInfo()} monthly NDWI composites")
    
    # 4. SIMPLIFIED Server-side processing function 
    def process_monthly_composite_for_zonal_stats(monthly_composite):
        year = ee.Number(monthly_composite.get('year'))
        month = monthly_composite.get('month')
        
        # EFFICIENT: Select GLAD image using server-side logic
        glad_image = ee.Algorithms.If(
            year.lt(2004), glad_images_dict[2003],
            ee.Algorithms.If(
                year.lt(2008), glad_images_dict[2007],
                ee.Algorithms.If(
                    year.lt(2012), glad_images_dict[2011],
                    ee.Algorithms.If(
                        year.lt(2016), glad_images_dict[2015],
                        glad_images_dict[2019]
                    )
                )
            )
        )
        
        glad_image = ee.Image(glad_image)
        
        # PERFORMANCE: Clip GLAD to region of interest
        glad_clipped = glad_image.clip(country_admin_fc.geometry())
        crop_mask = glad_clipped.eq(1)

        # PERFORMANCE: Clip NDWI to region and apply mask
        ndwi_clipped = monthly_composite.select('NDWI').clip(country_admin_fc.geometry())
        ndwi_masked = ndwi_clipped.updateMask(crop_mask)
        
        # Count pixels
        pixel_count = ndwi_masked.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=country_admin_fc.geometry(),
            scale=463.313,
            maxPixels=1e9
        )
        
        # EXACT COPY of what worked in test - most basic reduceRegions
        result = ndwi_masked.reduceRegions(
            collection=country_admin_fc,
            reducer=ee.Reducer.mean(),
            scale=463.313,
            tileScale=8
        )
        
        # Calculate GLAD year  
        glad_year = ee.Number(ee.Algorithms.If(
            year.lt(2004), 2003,
            ee.Algorithms.If(
                year.lt(2008), 2007,
                ee.Algorithms.If(
                    year.lt(2012), 2011,
                    ee.Algorithms.If(
                        year.lt(2016), 2015,
                        2019
                    )
                )
            )
        ))
        
        # Add metadata to each feature - EXACTLY as before
        def add_metadata(feature):
            return feature.set({
                'month': month,
                'year': year,
                'GLAD_year': glad_year,
                'pixel_count': pixel_count.get('NDWI')
            })

        return result.map(add_metadata)

    # Map over the monthly collection and flatten the results
    # Results will be a FeatureCollection where each feature is a region with stats for a particular date
    print("Mapping statistics calculation over monthly NDWI collection...")
    results_fc = monthly_ndwi_collection.map(process_monthly_composite_for_zonal_stats).flatten()

    # Instead of using select() to rename properties (which is failing),
    # let's map a function over results_fc that creates new features with the desired property names.
    def rename_properties(feature):
        """Creates a new feature with renamed statistical properties and no geometry."""
        new_props = {
            'month': feature.get('month'),
            'year': feature.get('year'), 
            'GLAD_year': feature.get('GLAD_year'),
            'ADMIN_NAME': feature.get('ADMIN_NAME'),
            'PCODE': feature.get('PCODE'),
            'NDWI_mean': feature.get('mean'),  # CORRECT: Use 'mean' property from reduceRegions
            'pixel_count': feature.get('pixel_count')
        }
        # Set geometry to None to avoid exporting .geo
        return ee.Feature(None, new_props)

    # Apply the rename_properties function to each feature in results_fc.
    renamed_fc = results_fc.map(rename_properties)

    # Export the renamed FeatureCollection directly.
    # removed: Select and rename columns for the final export
    # removed: original_selectors, new_names, final_fc_to_export = results_fc.select(...)

    # 5. Export to Google Drive
    country_name_description = unidecode(country_name).replace("'", "")
    task_description = f"{output_filename_prefix}_{country_name_description}_{start_date_str}_to_{end_date_str}_monthly"
    # Ensure filename doesn't have problematic characters if prefix comes from user input
    safe_filename_prefix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in output_filename_prefix)
    safe_country_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in country_name)
    
    filename = f"{safe_country_name}_admin{admin_level}_{safe_filename_prefix}_monthly"

    print(f"Initiating export task: {task_description}")
    task = ee.batch.Export.table.toDrive(
        collection=renamed_fc,  # Use the renamed collection directly
        description=task_description,
        folder=GD_FOLDER, 
        fileNamePrefix=filename,
        fileFormat='CSV',
        selectors=[
            'month', 'year', 'GLAD_year', 'ADMIN_NAME', 'PCODE', 
            'NDWI_mean', 'pixel_count'
        ]  # Explicitly specify columns to export, excluding geometries
    )
    try:
        task.start()
        print(f"Task '{task_description}' started successfully. Task ID: {task.id}")
        print(f"Monitor task progress in the Earth Engine Code Editor 'Tasks' tab or your Google Drive folder '{GD_FOLDER}'.")
        return task
    except Exception as e:
        print(f"Could not start export task: {e}")
        if hasattr(task, 'status') and task.status():
             print(f"Task status: {task.status()}")
        return None

if __name__ == "__main__":
    if not EE_INITIALIZED:
        print("Earth Engine failed to initialize. Exiting test run.")
    else:
        # Test parameters - FIXED DATE RANGE
        test_country = "Ghana"
        test_start_date = "2001-02-01"
        test_end_date = "2023-02-01"
        shapefile_path = 'Training\HarvestStatsAfrica\data\hvstat_africa_boundary_v1.0.gpkg' 
        output_prefix = f"NDWI_timeseries_GLAD"
        admin_level = 1  # Test with Admin 0 to see geometry comparison plot

        print(f"--- Starting NDWI extraction with GLAD cropland mask --- ")
        print(f"Country: {test_country}, Period: {test_start_date} to {test_end_date}, Admin Level: {admin_level}")
        
        export_task = extractNDWI(test_country, test_start_date, test_end_date, 
                              shapefile_path, output_prefix, admin_level)
        
        if export_task:
            print(f"Test run initiated export task. Task ID: {export_task.id}")
        else:
            print("Test run failed to initiate export task. Check errors above.")
        print("--- NDWI extraction sent to Earth Engine --- ")
