import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import pyodbc
import os
from rasterstats import zonal_stats
from tqdm import tqdm

def extract_soil_properties_all_locations(
    gpkg_path: str = "public/hvstat_africa_boundary_v1.0.gpkg",
    admin_level: int = 2,
    raster_path: str = "ASR/soil/HWSD2.bil",
    mdb_path: str = "ASR/soil/HWSD2.mdb",
    output_csv: str = "ASR/soil/Africa_Admin2_Soil_Stats.csv",
    topsoil_max_depth: int = 30
):
    """
    Extracts soil properties for all administrative regions in Africa.

    This script reads administrative boundaries and HWSD2 soil data.
    For each administrative region, it extracts soil properties including:
    - Clay content (%)
    - Sand content (%)
    - Silt content (%)
    - Organic carbon (%)
    - pH (water)
    - Reference bulk density (g/cm³)
    - Coarse fragments (%)

    Args:
        gpkg_path (str): Path to the administrative boundaries GeoPackage file
        admin_level (int): Administrative level to process (default: 2)
        raster_path (str): Path to the HWSD2 raster file
        mdb_path (str): Path to the HWSD2 database file
        output_csv (str): Output CSV file path
        topsoil_max_depth (int): Maximum depth for topsoil (cm, default: 30)
    """
    
    print("Loading administrative boundaries...")
    try:
        gdf = gpd.read_file(gpkg_path)
        admin_column = f'ADMIN{admin_level}'
        
        if admin_column not in gdf.columns:
            raise ValueError(f"Column {admin_column} not found in admin boundaries.")

        # Ensure CRS is set (likely WGS84 for GADM data)
        if gdf.crs is None:
            print("No CRS found for admin boundaries, setting to EPSG:4326 (WGS84)")
            gdf = gdf.set_crs('EPSG:4326')
        
        print(f"Found {len(gdf)} admin {admin_level} units across all countries")
        
    except Exception as e:
        print(f"Error loading administrative boundaries: {e}")
        return

    # Load HWSD2 raster and get projection info
    print("Loading HWSD2 raster...")
    try:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            nodata = src.nodata
            print(f"Raster CRS: {raster_crs}")
            print(f"Raster NoData value: {nodata}")
        
        # Reproject admin boundaries to match raster CRS
        if gdf.crs != raster_crs:
            print("Reprojecting admin boundaries to match raster CRS...")
            gdf = gdf.to_crs(raster_crs)
            
    except Exception as e:
        print(f"Error loading raster: {e}")
        return

    # Load HWSD2 database
    print("Loading HWSD2 database...")
    try:
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            f'DBQ={os.path.abspath(mdb_path)};'
        )
        conn = pyodbc.connect(conn_str)
        layers_df = pd.read_sql("SELECT * FROM HWSD2_LAYERS", conn)
        conn.close()
        
        # Filter for topsoil layers only
        layers_df = layers_df[layers_df['TOPDEP'] < topsoil_max_depth]
        print(f"Loaded {len(layers_df)} soil layer records for topsoil analysis")
        
    except Exception as e:
        print(f"Error loading HWSD2 database: {e}")
        return

    # Define soil properties to extract
    properties = [
        ('CLAY', 'T_CLAY'),
        ('SAND', 'T_SAND'),
        ('SILT', 'T_SILT'),
        ('ORG_CARBON', 'T_OC'),
        ('PH_WATER', 'T_PH_H2O'),
        ('REF_BULK', 'T_REF_BULK'),
        ('COARSE', 'T_GRAVEL'),
    ]

    # Extract soil statistics for all administrative units
    print("Extracting soil properties for all administrative units...")
    
    # Get pixel counts for each polygon using zonal stats
    print("Computing zonal statistics...")
    zs = zonal_stats(
        gdf['geometry'], 
        raster_path, 
        stats=None, 
        categorical=True, 
        nodata=nodata, 
        all_touched=True, 
        raster_out=False
    )
    
    results = []
    
    print("Processing administrative units...")
    for idx, (row, smu_counts) in enumerate(tqdm(zip(gdf.itertuples(), zs), total=len(gdf))):
        
        # Skip if no valid soil data
        if not smu_counts or sum(smu_counts.values()) == 0:
            # Still add the row with NaN values
            result = {
                'FNID': getattr(row, 'FNID'),
                'ADMIN0': getattr(row, 'ADMIN0'),
                'ADMIN1': getattr(row, 'ADMIN1'),
                'ADMIN2': getattr(row, 'ADMIN2'),
            }
            # Add NaN values for all soil properties
            for _, out_name in properties:
                result[out_name + '_mean'] = np.nan
            results.append(result)
            continue
        
        # Initialize property value containers
        prop_vals = {out_name: [] for _, out_name in properties}
        
        # Process each soil mapping unit (SMU) in the polygon
        for smu_id, count in smu_counts.items():
            if smu_id == nodata:
                continue
                
            # Get soil component data for this SMU
            comp_rows = layers_df[layers_df['HWSD2_SMU_ID'] == smu_id]
            if comp_rows.empty:
                continue
            
            # Calculate weighted average for each property
            total_share = comp_rows['SHARE'].sum()
            for prop, out_name in properties:
                if total_share > 0:
                    weighted = np.average(comp_rows[prop], weights=comp_rows['SHARE'])
                else:
                    weighted = np.nan
                
                # Add this weighted value for each pixel with this SMU
                prop_vals[out_name].extend([weighted] * count)
        
        # Create result record
        result = {
            'FNID': getattr(row, 'FNID'),
            'ADMIN0': getattr(row, 'ADMIN0'),
            'ADMIN1': getattr(row, 'ADMIN1'),
            'ADMIN2': getattr(row, 'ADMIN2'),
        }
        
        # Calculate mean values for each property
        for _, out_name in properties:
            vals = np.array(prop_vals[out_name])
            if len(vals) > 0:
                result[out_name + '_mean'] = np.nanmean(vals)
            else:
                result[out_name + '_mean'] = np.nan
        
        results.append(result)

    # Save results to CSV
    print("Saving results...")
    df_out = pd.DataFrame(results)
    output_csv = os.path.join(os.path.dirname(output_csv), os.path.basename(output_csv))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    
    print(f"Soil property statistics saved to {output_csv}")
    print(f"Processed {len(results)} administrative units")
    
    # Print summary statistics
    print("\nSummary statistics:")
    for _, out_name in properties:
        col_name = out_name + '_mean'
        valid_count = df_out[col_name].notna().sum()
        mean_val = df_out[col_name].mean()
        print(f"{out_name}: {valid_count}/{len(df_out)} units with data, mean = {mean_val:.2f}")

if __name__ == "__main__":
    extract_soil_properties_all_locations(
        gpkg_path="HarvestStatAfrica/data/hvstat_africa_boundary_v1.0.gpkg",
        admin_level=2,
        raster_path="HWSD/HWSD2.bil",
        mdb_path="HWSD/HWSD2.mdb",
        output_csv="HarvestStatAfrica/HWSD/HSAfrica_Soil_Stats.csv",
        topsoil_max_depth=30
    )

    extract_soil_properties_all_locations(
        gpkg_path="GADM/gadm41_AFR_shp/gadm41_AFR_2_processed.shp",
        admin_level=2,
        raster_path="HWSD/HWSD2.bil",
        mdb_path="HWSD/HWSD2.mdb",
        output_csv="GADM/HWSD/Africa_Admin2_Soil_Stats.csv",
        topsoil_max_depth=30
    )

    extract_soil_properties_all_locations(
        gpkg_path="GADM/gadm41_AFR_shp/gadm41_AFR_1_processed.shp",
        admin_level=1,
        raster_path="HWSD/HWSD2.bil",
        mdb_path="HWSD/HWSD2.mdb",
        output_csv="GADM/HWSD/Africa_Admin1_Soil_Stats.csv",
        topsoil_max_depth=30
    )

    df_admin1 = pd.read_csv("GADM/HWSD/Africa_Admin1_Soil_Stats.csv")
    df_admin2 = pd.read_csv("GADM/HWSD/Africa_Admin2_Soil_Stats.csv")
    df_merged = pd.concat([df_admin1, df_admin2])
    df_merged.to_csv("GADM/HWSD/Africa_GADM_Soil_Stats.csv", index=False)

