import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

def extract_crop_calendar():
    """
    Extracts crop calendar data for administrative regions in Africa for Maize crops.

    This script reads administrative boundaries (both admin1 and admin2 levels) and 
    GEOGLAM crop calendar data. For each administrative region, it finds the 
    corresponding crop calendar entry for 'Maize 1' and 'Maize 2' based on the 
    region's centroid.

    The script then saves the merged data to a CSV file.
    """
    # Load the shapefiles
    print("Loading shapefiles...")
    try:
        admin1_path = 'GADM/gadm41_AFR_shp/gadm41_AFR_1_processed.shp'
        admin2_path = 'GADM/gadm41_AFR_shp/gadm41_AFR_2_processed.shp'
        calendar_path = 'GEOGLAM/GEOGLAM_CM4EW_Calendars_V1.0.shp'
        
        admin1_gdf = gpd.read_file(admin1_path)
        admin2_gdf = gpd.read_file(admin2_path)
        calendar_gdf = gpd.read_file(calendar_path)
        print("Shapefiles loaded successfully.")
    except Exception as e:
        print(f"Error loading shapefiles: {e}")
        return

    # Identify countries that are only in admin1 (missing from admin2)
    admin1_countries = set(admin1_gdf['ADMIN0'].unique())
    admin2_countries = set(admin2_gdf['ADMIN0'].unique())
    admin1_only_countries = admin1_countries - admin2_countries
    
    print(f"Found {len(admin1_only_countries)} countries only available at admin1 level:")
    for country in sorted(admin1_only_countries):
        print(f"  {country}")
    
    # Get admin1 records for countries not in admin2
    admin1_missing = admin1_gdf[admin1_gdf['ADMIN0'].isin(admin1_only_countries)].copy()
    # Add empty ADMIN2 column to match admin2 structure
    admin1_missing['ADMIN2'] = None
    
    # Combine admin2 data with admin1-only data
    admin_gdf = pd.concat([admin2_gdf, admin1_missing], ignore_index=True)
    print(f"Combined dataset: {len(admin2_gdf)} admin2 + {len(admin1_missing)} admin1 = {len(admin_gdf)} total regions")

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
            'ADMIN2': admin_row['ADMIN2'] if pd.notna(admin_row['ADMIN2']) else None,
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
                    region_name = admin_row['ADMIN2'] if pd.notna(admin_row['ADMIN2']) else admin_row['ADMIN1']
                    print(f"Warning: Centroid for {region_name} has multiple '{crop_type}' geometries. Using first.")
                
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

    results_df.dropna(subset=['Maize_1_calendar_country'], inplace=True)
    
    output_path = 'GADM/crop_calendar/maize_crop_calendar_extraction.csv'
    print(f"Saving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print("Processing complete.")

    print(f"\n📊 SUMMARY:")
    print(f"   • Total regions kept: {len(results_df)}")
    print(f"   • Regions with Maize 1 data: {results_df['Maize_1_calendar_country'].notna().sum()}")
    print(f"   • Regions with Maize 2 data: {results_df['Maize_2_calendar_country'].notna().sum()}")
    print(f"   • Regions with both Maize types: {(results_df['Maize_1_calendar_country'].notna() & results_df['Maize_2_calendar_country'].notna()).sum()}")


if __name__ == '__main__':
    extract_crop_calendar() 