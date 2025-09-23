import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

def extract_crop_calendar():
    """
    Extracts crop calendar data for administrative regions in Africa for Maize crops.

    This script reads administrative boundaries and GEOGLAM crop calendar data.
    For each administrative region, it finds the corresponding crop calendar entry
    for 'Maize 1' and 'Maize 2' based on the region's centroid.

    The script then saves the merged data to a CSV file.
    """
    # Load the shapefiles
    print("Loading shapefiles...")
    try:
        admin_boundaries_path = 'HarvestStatAfrica/data/hvstat_africa_boundary_v1.0.gpkg'
        calendar_path = 'GEOGLAM/GEOGLAM_CM4EW_Calendars_V1.0.shp'
        
        admin_gdf = gpd.read_file(admin_boundaries_path)
        calendar_gdf = gpd.read_file(calendar_path)
        print("Shapefiles loaded successfully.")
    except Exception as e:
        print(f"Error loading shapefiles: {e}")
        return

    # Filter for Maize crops
    maize_calendar = calendar_gdf[calendar_gdf['crop'].isin(['Maize 1', 'Maize 2'])].copy()

    # Set the CRS for the admin boundaries as it's not being detected automatically.
    # WGS84 (EPSG:4326) is the standard for GADM data.
    if admin_gdf.crs is None:
        admin_gdf.set_crs(epsg=4326, inplace=True)

    # Reproject if CRS are different
    if admin_gdf.crs != maize_calendar.crs:
        print("Reprojecting admin boundaries to match calendar CRS...")
        admin_gdf = admin_gdf.to_crs(maize_calendar.crs)

    results = []

    print("Processing administrative boundaries...")
    for idx, admin_row in tqdm(admin_gdf.iterrows(), total=admin_gdf.shape[0]):
        centroid = admin_row.geometry.centroid
        
        # Base result row
        result_row = {
            'FNID': admin_row['FNID'],
            'ADMIN0': admin_row['ADMIN0'],
            'ADMIN1': admin_row['ADMIN1'],
            'ADMIN2': admin_row['ADMIN2'],
        }

        # Find which calendar geometry contains the centroid
        containing_polygons = maize_calendar[maize_calendar.geometry.contains(centroid)]
        
        # Process Maize 1 and Maize 2 separately
        for crop_type in ['Maize 1', 'Maize 2']:
            crop_data = containing_polygons[containing_polygons['crop'] == crop_type]
            
            # Sanitize crop_type for column names
            col_prefix = crop_type.replace(' ', '_')

            if not crop_data.empty:
                if len(crop_data) > 1:
                    print(f"Warning: Centroid for {admin_row['ADMIN2']} has multiple '{crop_type}' geometries. Using first.")
                
                calendar_data = crop_data.iloc[0]
                result_row[f'{col_prefix}_calendar_country'] = calendar_data['country']
                result_row[f'{col_prefix}_calendar_region'] = calendar_data['region']
                result_row[f'{col_prefix}_planting'] = calendar_data['planting']
                result_row[f'{col_prefix}_vegetative'] = calendar_data['vegetative']
                result_row[f'{col_prefix}_harvest'] = calendar_data['harvest']
                result_row[f'{col_prefix}_endofseaso'] = calendar_data['endofseaso']
                result_row[f'{col_prefix}_outofseaso'] = calendar_data['outofseaso']
            else:
                # Add null columns if crop type not found
                result_row[f'{col_prefix}_calendar_country'] = None
                result_row[f'{col_prefix}_calendar_region'] = None
                result_row[f'{col_prefix}_planting'] = None
                result_row[f'{col_prefix}_vegetative'] = None
                result_row[f'{col_prefix}_harvest'] = None
                result_row[f'{col_prefix}_endofseaso'] = None
                result_row[f'{col_prefix}_outofseaso'] = None
        
        results.append(result_row)

    results_df = pd.DataFrame(results)
    
    output_path = 'HarvestStatAfrica/crop_calendar/maize_crop_calendar_extraction.csv'
    print(f"Saving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print("Processing complete.")

if __name__ == '__main__':
    extract_crop_calendar() 