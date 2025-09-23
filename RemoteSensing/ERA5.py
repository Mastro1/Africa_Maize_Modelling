import ee
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
from unidecode import unidecode
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EE_PROJECT = os.getenv('EE_PROJECT')
EE_OPT_URL = os.getenv('EE_OPT_URL')
GD_FOLDER = os.getenv('GD_FOLDER')

# Initialize Earth Engine globally for the module
EE_INITIALIZED = True
EE_INIT_ERROR = None
try:
    ee.Initialize(project=EE_PROJECT, opt_url=EE_OPT_URL)
    print("Google Earth Engine initialized successfully for the module.")
    EE_INITIALIZED = True
except Exception as e:
    EE_INIT_ERROR = e
    print(f"Could not initialize Earth Engine for the module: {e}")
    print("Please ensure you have authenticated Earth Engine CLI and the project ID is correct.")


def extractERA5(country_name: str, start_date_str: str, end_date_str: str, 
               gpkg_path: str, output_filename_prefix: str, admin_level: int = 2):
    """
    Extracts mean values of ERA5-Land variables for administrative regions of a specified country 
    within a given date range. Results are exported to Google Drive.

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

    print(f"Starting ERA5 extraction for {country_name} from {start_date_str} to {end_date_str} at Admin level {admin_level}.")

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
            
            # Create a visualization of the simplified geometry
            # plot_simplified_geometry(simplified_geom, country_name)
            
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
            
            features = []
            for index, row in country_boundaries_gdf.iterrows():
                shapely_geom = row['geometry']
                geojson_geom = gpd.GeoSeries([shapely_geom]).__geo_interface__['features'][0]['geometry']
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

    # 2. Define ERA5-Land Image Collection
    era5_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(start_date_str, end_date_str) \
        .select([
            'temperature_2m',
            'volumetric_soil_water_layer_1',
            'total_precipitation_sum',
            'temperature_2m_min',
            'temperature_2m_max'
        ])
    
    # Add debugging information
    collection_size = era5_collection.size().getInfo()
    print(f"Number of images in collection: {collection_size}")
    
    if collection_size == 0:
        print("No images found in the collection. Checking date range...")
        # Get the first and last available dates
        first_date = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").first().date().format('YYYY-MM-dd').getInfo()
        last_date = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").sort('system:time_start', False).first().date().format('YYYY-MM-dd').getInfo()
        print(f"Available date range: {first_date} to {last_date}")
        return None

    # Print the first image date for verification
    first_image = era5_collection.first()
    first_image_date = first_image.date().format('YYYY-MM-dd').getInfo()
    print(f"First image date: {first_image_date}")

    # 3. Server-side processing function
    def process_image_for_zonal_stats(era5_image):
        image_date_ee = era5_image.date()
        
        # Calculate mean for each band
        stats_for_image_date = era5_image.reduceRegions(
            collection=country_admin_fc,
            reducer=ee.Reducer.mean(),
            scale=11132,  # ERA5-Land resolution is 11.132 km
            tileScale=4
        )
        
        def sanitize_feature_and_add_date(feature_from_reduce_regions):
            geom = feature_from_reduce_regions.geometry()
            
            f = ee.Feature(geom, {
                'ADMIN_NAME': feature_from_reduce_regions.get('ADMIN_NAME'),
                'PCODE': feature_from_reduce_regions.get('PCODE'),
                'date': image_date_ee.format('YYYY-MM-dd')
            })

            # Add mean values for each variable
            variables = [
                'temperature_2m',
                'volumetric_soil_water_layer_1',
                'total_precipitation_sum',
                'temperature_2m_min',
                'temperature_2m_max'
            ]
            
            for var in variables:
                f = f.set(var, feature_from_reduce_regions.get(var))
            
            return f

        return stats_for_image_date.map(sanitize_feature_and_add_date)

    # Map over the ERA5 collection and flatten the results
    print("Mapping statistics calculation over ERA5 image collection...")
    results_fc = era5_collection.map(process_image_for_zonal_stats).flatten()

    # Rename properties for export
    def rename_properties(feature):
        new_props = {
            'date': feature.get('date'),
            'ADMIN_NAME': feature.get('ADMIN_NAME'),
            'PCODE': feature.get('PCODE'),
            'temperature_2m': feature.get('temperature_2m'),
            'volumetric_soil_water_layer_1': feature.get('volumetric_soil_water_layer_1'),
            'total_precipitation_sum': feature.get('total_precipitation_sum'),
            'temperature_2m_min': feature.get('temperature_2m_min'),
            'temperature_2m_max': feature.get('temperature_2m_max')
        }
        return ee.Feature(None, new_props)

    renamed_fc = results_fc.map(rename_properties)

    # Export to Google Drive
    country_name_description = unidecode(country_name).replace("'", "")
    task_description = f"{output_filename_prefix}_{country_name_description}_{start_date_str}_to_{end_date_str}"
    safe_filename_prefix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in output_filename_prefix)
    safe_country_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in country_name)
    
    filename = f"{safe_country_name}_admin{admin_level}_{safe_filename_prefix}"

    print(f"Initiating export task: {task_description}")
    task = ee.batch.Export.table.toDrive(
        collection=renamed_fc,
        description=task_description,
        folder=GD_FOLDER, 
        fileNamePrefix=filename,
        fileFormat='CSV',
        selectors=[
            'date', 'ADMIN_NAME', 'PCODE',
            'temperature_2m',
            'volumetric_soil_water_layer_1',
            'total_precipitation_sum',
            'temperature_2m_min',
            'temperature_2m_max'
        ]
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
        # Test parameters
        test_country = "Kenya"
        test_start_date = "1990-01-01" 
        test_end_date = "2023-12-31"
        shapefile_path = 'Training\HarvestStatsAfrica\data\hvstat_africa_boundary_v1.0.gpkg' 
        output_prefix = f"ERA5_timeseries"
        admin_level = 1  # Test with Admin 0 to see geometry comparison plot

        print(f"--- Starting ERA5 extraction --- ")
        print(f"Country: {test_country}, Period: {test_start_date} to {test_end_date}, Admin Level: {admin_level}")
        
        export_task = extractERA5(test_country, test_start_date, test_end_date, 
                                shapefile_path, output_prefix, admin_level)
        
        if export_task:
            print(f"Test run initiated export task. Task ID: {export_task.id}")
        else:
            print("Test run failed to initiate export task. Check errors above.")
        print("--- ERA5 extraction sent to Earth Engine --- ")
