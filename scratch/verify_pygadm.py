import sys
import os

# Add project root to path if needed
sys.path.append(os.getcwd())

from RemoteSensing.FPAR import get_boundary_gdf
import ee

# Initialize EE for pygadm if it needs it (usually it doesn't for the GDF part, but good to have)
# Actually, FPAR.py already initializes it on import if env vars are set.

try:
    print("Testing get_boundary_gdf for 'Italy' (missing from local Africa shapefile)...")
    gdf = get_boundary_gdf(
        country_name="Italy", 
        admin_level=0, 
        gpkg_path='GADM/gadm41_AFR_shp/gadm41_AFR_final.shp', 
        tolerance=0.01
    )
    
    if gdf is not None:
        print(f"Success! Fetched {len(gdf)} regions.")
        print(f"Columns: {gdf.columns.tolist()}")
        print(f"Sample data:\n{gdf.head(1)}")
    else:
        print("Failed to fetch boundary.")
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
