import geopandas as gpd
import pandas as pd
import os
import time
import warnings
from shapely.validation import make_valid

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
# Input Files (Relative to Project Root)
HSA_SHAPEFILE = 'HarvestStatAfrica/data/hvstat_africa_boundary_v1.0.gpkg'
GADM_SHAPEFILE_ADMIN1 = 'GADM/gadm41_AFR_shp/gadm41_AFR_1.shp'
GADM_SHAPEFILE_ADMIN2 = 'GADM/gadm41_AFR_shp/gadm41_AFR_2.shp'

# Output File (Relative to Project Root)
OUTPUT_CSV = 'HarvestStatAfrica/GADM_matching/HSA_to_GADM_mapping.csv'

# Parameters
SPATIAL_THRESHOLD = 75.0  # Percentage

# --- Manual Mappings ---
# Map HSA country names (normalized) to GADM country names (normalized)
COUNTRY_MAPPING = {
    "tanzania, united republic of": "tanzania",
    "drc": "democratic republic of the congo",
    "cote d'ivoire": "côte d'ivoire"
}

def normalize_text(text):
    """Normalize text for comparison."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.replace("-", " ").replace("_", " ")
    return text

def get_mapped_country(country_norm):
    """Apply manual country mapping."""
    return COUNTRY_MAPPING.get(country_norm, country_norm)

def get_hsa_target_level(row):
    """Determine matching level (1 or 2) based on HSA columns."""
    if pd.notna(row.get('ADMIN2')):
        return 2
    return 1

def match_by_spatial(hsa_row, gadm_subset, target_level):
    """
    Find best spatial match for HSA polygon in GADM subset.
    Returns: (Matched GID, matched_name, overlap_pct)
    """
    hsa_geom = hsa_row.geometry
    if hsa_geom is None or hsa_geom.is_empty:
        return None, None, 0.0
        
    gadm_gid_col = f"GID_{target_level}"
    gadm_name_col = f"NAME_{target_level}"

    try:
        # Calculate intersection areas
        # Note: aligning crs is handled in main
        intersections = gadm_subset.geometry.intersection(hsa_geom).area
        
        valid_intersections = intersections[intersections > 0]
        
        if valid_intersections.empty:
            return None, None, 0.0
            
        # Best match
        best_idx = valid_intersections.idxmax()
        best_area = valid_intersections.loc[best_idx]
        
        # Calculate overlap percentage relative to HSA area
        hsa_area = hsa_geom.area
        overlap_pct = (best_area / hsa_area) * 100
        
        matched_gid = gadm_subset.loc[best_idx, gadm_gid_col]
        matched_name = gadm_subset.loc[best_idx, gadm_name_col]
        
        return matched_gid, matched_name, overlap_pct

    except Exception as e:
        print(f"    Error in spatial matching: {e}")
        return None, None, 0.0

def main():
    start_time = time.time()
    print("=== Starting HSA to GADM Matching Process ===")
    print(f"Threshold: {SPATIAL_THRESHOLD}%")
    print("-" * 50)

    # 1. Load Data
    print(f"Loading HSA Data: {HSA_SHAPEFILE}")
    if not os.path.exists(HSA_SHAPEFILE):
        print(f"CRITICAL ERROR: File not found: {HSA_SHAPEFILE}")
        return
    hsa_gdf = gpd.read_file(HSA_SHAPEFILE)
    print(f"  > Loaded {len(hsa_gdf)} records.")

    print(f"Loading GADM Admin 1: {GADM_SHAPEFILE_ADMIN1}")
    if not os.path.exists(GADM_SHAPEFILE_ADMIN1):
        print(f"CRITICAL ERROR: File not found: {GADM_SHAPEFILE_ADMIN1}")
        return
    gadm1_gdf = gpd.read_file(GADM_SHAPEFILE_ADMIN1)

    print(f"Loading GADM Admin 2: {GADM_SHAPEFILE_ADMIN2}")
    if not os.path.exists(GADM_SHAPEFILE_ADMIN2):
        print(f"CRITICAL ERROR: File not found: {GADM_SHAPEFILE_ADMIN2}")
        return
    gadm2_gdf = gpd.read_file(GADM_SHAPEFILE_ADMIN2)

    # 2. CRS Alignment
    target_crs = "EPSG:4326"
    print(f"Ensuring CRS alignment to {target_crs}...")
    if hsa_gdf.crs != target_crs:
        hsa_gdf = hsa_gdf.to_crs(target_crs)
    if gadm1_gdf.crs != target_crs:
        gadm1_gdf = gadm1_gdf.to_crs(target_crs)
    if gadm2_gdf.crs != target_crs:
        gadm2_gdf = gadm2_gdf.to_crs(target_crs)

    # 3. Pre-processing
    print("Normalizing country names...")
    hsa_gdf['ADMIN0_norm'] = hsa_gdf['ADMIN0'].apply(normalize_text)
    gadm1_gdf['NAME_0_norm'] = gadm1_gdf['NAME_0'].apply(normalize_text)
    gadm2_gdf['NAME_0_norm'] = gadm2_gdf['NAME_0'].apply(normalize_text)

    countries = hsa_gdf['ADMIN0_norm'].unique()
    final_results = []
    
    print("-" * 50)
    print("Running Country-by-Country Matching...")

    for country_norm in countries:
        if not country_norm: continue
        
        # Determine GADM Country Name
        gadm_country_norm = get_mapped_country(country_norm)
        original_country_name = hsa_gdf[hsa_gdf['ADMIN0_norm'] == country_norm].iloc[0]['ADMIN0']
        
        print(f"Processing: {original_country_name} (GADM: {gadm_country_norm})")
        
        # Subsets
        hsa_subset = hsa_gdf[hsa_gdf['ADMIN0_norm'] == country_norm]
        gadm1_subset = gadm1_gdf[gadm1_gdf['NAME_0_norm'] == gadm_country_norm]
        gadm2_subset = gadm2_gdf[gadm2_gdf['NAME_0_norm'] == gadm_country_norm]
        
        if gadm1_subset.empty and gadm2_subset.empty:
            print(f"  WARNING: No GADM data found for '{gadm_country_norm}'. Checks mapping?")
            # We still keep the HSA records, but GID will be None
        
        matches_found = 0
        
        for idx, row in hsa_subset.iterrows():
            target_level = get_hsa_target_level(row)
            
            # Select relevant GADM level
            gadm_target_subset = gadm2_subset if target_level == 2 else gadm1_subset
            
            # Perform Spatial Match
            matched_gid, matched_name, overlap_pct = match_by_spatial(row, gadm_target_subset, target_level)
            
            status = "No Match"
            final_gid = None
            final_gadm_name = None
            
            if matched_gid:
                if overlap_pct >= SPATIAL_THRESHOLD:
                    status = "Matched"
                    final_gid = matched_gid
                    final_gadm_name = matched_name
                    matches_found += 1
                else:
                    status = f"Below Threshold ({overlap_pct:.1f}%)"
                    # User requested: "I want to consider it a match only if... shares at least X%"
                    # We will NOT assign the GID if it fails threshold to keep the file clean.
                    # Or should we?
                    # "It is important that the final matching file contains the right codes to match"
                    # If we omit it, it won't match.
                    # I will leave GID empty here as per "only if" requirement.
                    final_gid = None 
                    final_gadm_name = None
            
            final_results.append({
                'FNID': row['FNID'],
                'HSA_ADMIN0': row['ADMIN0'],
                'HSA_ADMIN1': row['ADMIN1'],
                'HSA_ADMIN2': row.get('ADMIN2', None),
                'match_status': status,
                'overlap_pct': overlap_pct,
                'gadm_id': final_gid,        # The key column requested (GID_1 or GID_2)
                'gadm_name': final_gadm_name,
                'gadm_level': target_level if final_gid else None
            })
            
        print(f"  > {matches_found}/{len(hsa_subset)} records matched above {SPATIAL_THRESHOLD}%.")

    # 4. Save Results
    print("-" * 50)
    df_final = pd.DataFrame(final_results)
    
    # Report
    total_recs = len(df_final)
    matched_recs = df_final['gadm_id'].notna().sum()
    print(f"Total HSA Records Processed: {total_recs}")
    print(f"Total Successfully Matched:  {matched_recs} ({matched_recs/total_recs*100:.1f}%)")
    print(f"Dropped (Below {SPATIAL_THRESHOLD}%): {total_recs - matched_recs}")
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Final Mapping File Saved: {OUTPUT_CSV}")
    
    elapsed = time.time() - start_time
    print(f"Process completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
