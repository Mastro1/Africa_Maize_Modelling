
import geopandas as gpd
import pandas as pd
import os

def generate_lat_lon_mapping():
    # Define paths
    base_dir = r"GADM/gadm41_AFR_shp"
    output_path = r"Model_physical/Input/pcode_lat_lon_mapping.csv"
    
    shapefiles = [
        ("Admin1", os.path.join(base_dir, "gadm41_AFR_1_processed.shp")),
        ("Admin2", os.path.join(base_dir, "gadm41_AFR_2_processed.shp"))
    ]
    
    all_data = []
    
    for level_name, shp_path in shapefiles:
        if not os.path.exists(shp_path):
            print(f"Warning: Shapefile not found at {shp_path}")
            continue
            
        print(f"Processing {level_name} shapefile: {shp_path}")
        try:
            gdf = gpd.read_file(shp_path)
            
            # Check for PCODE column (usually FNID or similar)
            # Based on previous file viewing, PCODEs are likely in FNID
            if 'FNID' not in gdf.columns:
                print(f"Error: 'FNID' column not found in {level_name}. Columns: {gdf.columns}")
                continue
                
            # Calculate centroid
            # Warning: Centroid of a geographic CRS (lat/lon) is largely okay for this purpose (radiation based on latitude)
            # but ideally should be done in projected CRS. For simplicity and since we need Lat/Lon, we do it directly.
            centroids = gdf.geometry.centroid
            
            for idx, row in gdf.iterrows():
                pcode = row['FNID']
                lat = centroids[idx].y
                lon = centroids[idx].x
                
                # Try to get name if available
                name = ""
                if 'NAME_1' in row: name = row['NAME_1'] # Admin 1 name?
                if level_name == "Admin2" and 'NAME_2' in row: name = row['NAME_2'] # Overwrite with Admin 2 name
                
                # Check for alternative name columns if standard ones fail
                if not name and 'ADMIN1' in row: name = row['ADMIN1']
                if not name and 'ADMIN2' in row: name = row['ADMIN2']
                
                all_data.append({
                    'PCODE': pcode,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Name': name,
                    'Level': level_name
                })
                
        except Exception as e:
            print(f"Error processing {level_name}: {e}")

    if not all_data:
        print("No data extracted.")
        return

    # Create DataFrame
    df_out = pd.DataFrame(all_data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Successfully saved mapping to {output_path}")
    print(f"Total entries: {len(df_out)}")
    print(df_out.head())

if __name__ == "__main__":
    generate_lat_lon_mapping()
