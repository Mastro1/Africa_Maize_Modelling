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

# Output File (Relative to Project Root, inside analysis folder)
OUTPUT_CSV = 'HarvestStatAfrica/GADM_matching/analysis/HSA_GADM_comparison.csv'

SPATIAL_THRESHOLD = 75.0 # Percentage

# --- Manual Mappings ---
# Map HSA country names (normalized) to GADM country names (normalized)
COUNTRY_MAPPING = {
    "tanzania, united republic of": "tanzania",
    "drc": "democratic republic of the congo",
    "cote d'ivoire": "côte d'ivoire" # Just in case
}

# --- Helper Functions ---

def normalize_text(text):
    """Normalize text for comparison (lower case, strip, simple replacements)."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Normalize internal spaces/hyphens
    text = text.replace("-", " ").replace("_", " ")
    return text

def get_mapped_country(country_norm):
    """Apply manual country mapping."""
    return COUNTRY_MAPPING.get(country_norm, country_norm)

def get_hsa_target_level(row):
    """Determine if we should match against GADM Level 1 or Level 2."""
    if pd.notna(row.get('ADMIN2')):
        return 2
    return 1

def match_by_text(hsa_row, gadm_subset, target_level):
    """
    Attempt to match HSA row to GADM subset using Name.
    Returns: (Matched GID, matched_method_details) or (None, reason)
    """
    hsa_name = ""
    # GADM Column to check
    gadm_col = f"NAME_{target_level}"
    gadm_gid_col = f"GID_{target_level}"
    
    if target_level == 2:
        hsa_name = normalize_text(hsa_row['ADMIN2'])
    else:
        hsa_name = normalize_text(hsa_row['ADMIN1'])
    
    if not hsa_name:
        return None, "No HSA Name"

    # Exact Match (Normalized)
    # gadm_subset already has 'norm_name' calculated in the main loop for efficiency
    
    match = gadm_subset[gadm_subset['norm_name'] == hsa_name]
    
    if len(match) == 1:
        return match.iloc[0][gadm_gid_col], "Exact Match"
    elif len(match) > 1:
        return match.iloc[0][gadm_gid_col], "Ambiguous (picked first)" 
    
    return None, "No Text Match"

def match_by_spatial(hsa_row, gadm_subset, target_level):
    """
    Attempt to match HSA row to GADM subset using Spatial Intersection.
    Returns: (Matched GID, matched_method_details, overlap_pct)
    """
    hsa_geom = hsa_row.geometry
    if hsa_geom is None or hsa_geom.is_empty:
        return None, "Empty HSA Geometry", 0.0
        
    gadm_gid_col = f"GID_{target_level}"

    try:
        # Calculate intersection areas
        intersections = gadm_subset.geometry.intersection(hsa_geom).area
        
        # Filter for non-zero intersection
        valid_intersections = intersections[intersections > 0]
        
        if valid_intersections.empty:
            return None, "No Overlap", 0.0
            
        # Find candidate with max intersection area
        best_idx = valid_intersections.idxmax()
        best_area = valid_intersections.loc[best_idx]
        
        # Check if the overlap is significant relative to HSA area
        hsa_area = hsa_geom.area
        overlap_pct = (best_area / hsa_area) * 100
        
        matched_gid = gadm_subset.loc[best_idx, gadm_gid_col]
        return matched_gid, f"Spatial Overlap ({overlap_pct:.1f}%)", overlap_pct

    except Exception as e:
        return None, f"Spatial Error: {e}", 0.0


# --- Main Logic ---

def main():
    print(f"Loading HSA data from {HSA_SHAPEFILE}...")
    hsa_gdf = gpd.read_file(HSA_SHAPEFILE)
    print(f"Loaded {len(hsa_gdf)} HSA records.")

    print(f"Loading GADM Admin 1 from {GADM_SHAPEFILE_ADMIN1}...")
    gadm1_gdf = gpd.read_file(GADM_SHAPEFILE_ADMIN1)
    
    print(f"Loading GADM Admin 2 from {GADM_SHAPEFILE_ADMIN2}...")
    gadm2_gdf = gpd.read_file(GADM_SHAPEFILE_ADMIN2)

    # Ensure CRS Match
    target_crs = "EPSG:4326"
    if hsa_gdf.crs != target_crs:
        hsa_gdf = hsa_gdf.to_crs(target_crs)
    if gadm1_gdf.crs != target_crs:
        gadm1_gdf = gadm1_gdf.to_crs(target_crs)
    if gadm2_gdf.crs != target_crs:
        gadm2_gdf = gadm2_gdf.to_crs(target_crs)

    results = []
    
    # Pre-calculate normalized names for GADM
    gadm1_gdf['NAME_0_norm'] = gadm1_gdf['NAME_0'].apply(normalize_text)
    gadm1_gdf['norm_name'] = gadm1_gdf['NAME_1'].apply(normalize_text) # For Text Match
    
    gadm2_gdf['NAME_0_norm'] = gadm2_gdf['NAME_0'].apply(normalize_text)
    gadm2_gdf['norm_name'] = gadm2_gdf['NAME_2'].apply(normalize_text) # For Text Match

    # HSA Countries
    hsa_gdf['ADMIN0_norm'] = hsa_gdf['ADMIN0'].apply(normalize_text)
    countries = hsa_gdf['ADMIN0_norm'].unique()
    
    total_start_time = time.time()
    
    for country_norm in countries:
        if not country_norm: continue
        
        # Apply Mapping logic
        gadm_country_norm = get_mapped_country(country_norm)
        
        original_country_name = hsa_gdf[hsa_gdf['ADMIN0_norm'] == country_norm].iloc[0]['ADMIN0']
        print(f"\nProcessing Country: {original_country_name} (Mapped to GADM: '{gadm_country_norm}')")
        
        # Filter HSA for this country
        hsa_country_subset = hsa_gdf[hsa_gdf['ADMIN0_norm'] == country_norm]
        
        # Filter GADM for this country using the MAPPED name
        gadm1_country_subset = gadm1_gdf[gadm1_gdf['NAME_0_norm'] == gadm_country_norm]
        gadm2_country_subset = gadm2_gdf[gadm2_gdf['NAME_0_norm'] == gadm_country_norm]
        
        if gadm1_country_subset.empty and gadm2_country_subset.empty:
            print(f"  WARNING: Country '{gadm_country_norm}' not found in GADM. Check mapping.")
        
        # Track stats for this country
        country_results = []

        for idx, row in hsa_country_subset.iterrows():
            target_level = get_hsa_target_level(row)
            
            # Select appropriate GADM subset
            gadm_subset = gadm2_country_subset if target_level == 2 else gadm1_country_subset
            
            # Method A: Text Match
            text_gid, text_reason = match_by_text(row, gadm_subset, target_level)
            
            # Method B: Spatial Match
            spatial_gid, spatial_reason, overlap_pct = match_by_spatial(row, gadm_subset, target_level)
            
            # Check Threshold
            is_strong_spatial = (overlap_pct >= SPATIAL_THRESHOLD)
            
            # Compare
            agreement = (text_gid == spatial_gid) if (text_gid and spatial_gid) else False
            
            res = {
                'HSA_Index': idx,
                'Country': row['ADMIN0'],
                'HSA_Admin1': row['ADMIN1'],
                'HSA_Admin2': row.get('ADMIN2', None),
                'Target_Level': target_level,
                'Text_GID': text_gid,
                'Text_Reason': text_reason,
                'Spatial_GID': spatial_gid,
                'Spatial_Reason': spatial_reason,
                'Spatial_Overlap_Pct': overlap_pct,
                'Is_Strong_Spatial': is_strong_spatial,
                'Agreement': agreement
            }
            results.append(res)
            country_results.append(res)
        
        # Print Per-Country Stats
        df_country = pd.DataFrame(country_results)
        n_total = len(df_country)
        n_strong = df_country['Is_Strong_Spatial'].sum()
        
        print(f"  > Records: {n_total}")
        print(f"  > Strong Matches (>{SPATIAL_THRESHOLD}%): {n_strong} ({n_strong/n_total*100:.1f}%)")
        
        # Flag weak potential issues
        n_weak = n_total - n_strong
        if n_weak > 0:
            print(f"  > WARNING: {n_weak} Weak Matches found!")
            
    # Compile Results
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAnalysis Complete. Results saved to {OUTPUT_CSV}")
    
    # Print Summary Report
    print("\n--- GLOBAL SUMMARY REPORT ---")
    print(f"Total Records Processed: {len(results_df)}")
    
    strong_matches = results_df['Is_Strong_Spatial'].sum()
    weak_matches = len(results_df) - strong_matches
    
    print(f"Strong Matches (OVERLAP >= {SPATIAL_THRESHOLD}%): {strong_matches} ({strong_matches/len(results_df)*100:.1f}%)")
    print(f"Weak Matches   (OVERLAP <  {SPATIAL_THRESHOLD}%): {weak_matches} ({weak_matches/len(results_df)*100:.1f}%)")
    
    if weak_matches > 0:
        print("\nSample Weak Matches:")
        print(results_df[results_df['Is_Strong_Spatial'] == False][['Country', 'HSA_Admin1', 'Spatial_Overlap_Pct']].head())

if __name__ == "__main__":
    main()
