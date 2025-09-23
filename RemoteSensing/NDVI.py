import ee
import geopandas as gpd
import matplotlib.pyplot as plt
from unidecode import unidecode
# pandas is implicitly used by geopandas and for DataFrame creation if needed client-side
# os module is removed as progressive local saving is removed
from datetime import datetime # Retained for potential use, though dates are mainly strings now
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

def plot_simplified_geometry(simplified_geom, country_name, save_plot=True):
    """
    Creates a plot of the simplified country geometry showing only the external boundary.
    
    Args:
        simplified_geom: The simplified geometry
        country_name (str): Name of the country for the plot title
        save_plot (bool): Whether to save the plot to file
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create a clean GeoDataFrame with only the external boundary
        # This ensures we only plot the outer boundary without any internal artifacts
        clean_gdf = gpd.GeoDataFrame([1], geometry=[simplified_geom])
        
        # Plot with no edge color to avoid internal line artifacts, then add clean boundary
        clean_gdf.plot(ax=ax, color='lightcoral', edgecolor='none')
        
        # Plot just the exterior boundary as a clean line
        if hasattr(simplified_geom, 'exterior'):
            x, y = simplified_geom.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, alpha=0.8)
            simplified_vertices = len(simplified_geom.exterior.coords)
        else:
            simplified_vertices = 'N/A'
        
        ax.set_title(f'{country_name} - Simplified Country Boundary\n{simplified_vertices} vertices', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Remove axis ticks for cleaner look
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = f'{country_name}_simplified_boundary.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Simplified boundary plot saved as: {plot_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Could not create simplified geometry plot: {e}")

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

def extractVI(country_name: str, start_date_str: str, end_date_str: str, 
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

    # 2. Define Image Collections
    vi_collection = ee.ImageCollection("MODIS/061/MOD13A1") \
        .filterDate(start_date_str, end_date_str)
    
    # 3. Define Combined Reducer for mean, median, mode
    combined_reducer = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.median(), sharedInputs=True
    ).combine(
        reducer2=ee.Reducer.mode(), sharedInputs=True
    )

    # 4. Server-side processing function (to be mapped over vi_collection)
    def process_image_for_zonal_stats(vi_image):
        image_date_ee = vi_image.date()
        glad_year = get_glad_year_for_date(image_date_ee)
        
        # Create a dictionary of GLAD images and select based on year
        glad_images = ee.Dictionary({
            '2003': get_glad_image_for_year(2003),
            '2007': get_glad_image_for_year(2007),
            '2011': get_glad_image_for_year(2011),
            '2015': get_glad_image_for_year(2015),
            '2019': get_glad_image_for_year(2019)
        })
        
        # Select the appropriate GLAD image using the year
        glad_image = ee.Image(glad_images.get(glad_year.format('%d')))
        
        # GLAD data is binary: 1 for cropland, 0 for non-cropland
        crop_mask = glad_image.eq(1)

        scale_factor = 0.0001
        ndvi_masked = vi_image.select('NDVI').multiply(scale_factor).updateMask(crop_mask)
        evi_masked = vi_image.select('EVI').multiply(scale_factor).updateMask(crop_mask)
        
        bands_to_reduce = ndvi_masked.rename('NDVI_val').addBands(evi_masked.rename('EVI_val'))

        stats_for_image_date = bands_to_reduce.reduceRegions(
            collection=country_admin_fc,
            reducer=combined_reducer,
            scale=500,
            tileScale=4
        )
        
        # Ensure all expected stat properties are present, setting to null if missing.
        # Also add the date here.
        def sanitize_feature_and_add_date(feature_from_reduce_regions):
            geom = feature_from_reduce_regions.geometry()

            # Create a base feature with guaranteed non-statistical properties.
            # .get() will return server-side null if ADMIN_NAME or PCODE were unexpectedly missing.
            f = ee.Feature(geom, {
                'ADMIN_NAME': feature_from_reduce_regions.get('ADMIN_NAME'),
                'PCODE': feature_from_reduce_regions.get('PCODE'),
                'date': image_date_ee.format('YYYY-MM-dd'),
                'GLAD_year': glad_year
            })

            # Explicitly set each statistical property on the feature 'f'.
            # If feature_from_reduce_regions.get(stat_name) returns null (property absent or null),
            # then f will have that property set to server-side null.
            f = f.set('NDVI_val_mean', feature_from_reduce_regions.get('NDVI_val_mean'))
            f = f.set('NDVI_val_median', feature_from_reduce_regions.get('NDVI_val_median'))
            f = f.set('NDVI_val_mode', feature_from_reduce_regions.get('NDVI_val_mode'))
            f = f.set('EVI_val_mean', feature_from_reduce_regions.get('EVI_val_mean'))
            f = f.set('EVI_val_median', feature_from_reduce_regions.get('EVI_val_median'))
            f = f.set('EVI_val_mode', feature_from_reduce_regions.get('EVI_val_mode'))
            
            return f

        return stats_for_image_date.map(sanitize_feature_and_add_date)

    # Map over the VI collection and flatten the results
    # Results will be a FeatureCollection where each feature is a region with stats for a particular date
    print("Mapping statistics calculation over VI image collection...")
    results_fc = vi_collection.map(process_image_for_zonal_stats).flatten()

    # Instead of using select() to rename properties (which is failing),
    # let's map a function over results_fc that creates new features with the desired property names.
    def rename_properties(feature):
        """Creates a new feature with renamed statistical properties and no geometry."""
        new_props = {
            'date': feature.get('date'),
            'GLAD_year': feature.get('GLAD_year'),
            'ADMIN_NAME': feature.get('ADMIN_NAME'),
            'PCODE': feature.get('PCODE'),
            'NDVI_mean': feature.get('NDVI_val_mean'),
            'NDVI_median': feature.get('NDVI_val_median'),
            'NDVI_mode': feature.get('NDVI_val_mode'),
            'EVI_mean': feature.get('EVI_val_mean'),
            'EVI_median': feature.get('EVI_val_median'),
            'EVI_mode': feature.get('EVI_val_mode')
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
    task_description = f"{output_filename_prefix}_{country_name_description}_{start_date_str}_to_{end_date_str}"
    # Ensure filename doesn't have problematic characters if prefix comes from user input
    safe_filename_prefix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in output_filename_prefix)
    safe_country_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in country_name)
    
    filename = f"{safe_country_name}_admin{admin_level}_{safe_filename_prefix}"

    print(f"Initiating export task: {task_description}")
    task = ee.batch.Export.table.toDrive(
        collection=renamed_fc,  # Use the renamed collection directly
        description=task_description,
        folder=GD_FOLDER, 
        fileNamePrefix=filename,
        fileFormat='CSV',
        selectors=[
            'date', 'GLAD_year', 'ADMIN_NAME', 'PCODE', 
            'NDVI_mean', 'NDVI_median', 'NDVI_mode', 
            'EVI_mean', 'EVI_median', 'EVI_mode'
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
        # Test parameters
        test_country = "Côte d'Ivoire"
        test_start_date = "2000-01-01" 
        test_end_date = "2023-12-31"
        shapefile_path = 'Training\HarvestStatsAfrica\data\hvstat_africa_boundary_v1.0.gpkg'
        shapefile_path = 'ASR/GADM/gadm41_AFR_shp/gadm41_AFR_2_processed.csv'
        output_prefix = f"VI_timeseries_GLAD"
        admin_level = 0  # Test with Admin 0 to see geometry comparison plot

        print(f"--- Starting VI extraction with GLAD cropland mask --- ")
        print(f"Country: {test_country}, Period: {test_start_date} to {test_end_date}, Admin Level: {admin_level}")
        
        export_task = extractVI(test_country, test_start_date, test_end_date, 
                              shapefile_path, output_prefix, admin_level)
        
        if export_task:
            print(f"Test run initiated export task. Task ID: {export_task.id}")
        else:
            print("Test run failed to initiate export task. Check errors above.")
        print("--- VI extraction sent to Earth Engine --- ")
