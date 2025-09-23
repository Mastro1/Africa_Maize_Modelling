import os
import json
import pandas as pd
import re
from scipy.special import k0
from pathlib import Path


# Directory to store combined outputs
BASE_PATH = Path.cwd()
COMBINED_DIR = os.path.join(BASE_PATH, 'RemoteSensing', 'HarvestStatAfrica', 'preprocessed_concat')
os.makedirs(COMBINED_DIR, exist_ok=True)


# File patterns
ADMIN0_MERGED = "{country}_admin0_merged_data_GLAM.csv"
ADMIN_GT0_MERGED = "{country}_admin{admin_level}_merged_data_GLAM.csv"
ADMIN0_RS = "{country}_admin0_remote_sensing_data_GLAM.csv"
ADMIN_GT0_RS = "{country}_admin{admin_level}_remote_sensing_data_GLAM.csv"

# Use the file pattern to robustly extract country names (handles multi-word countries)
glam_results_dir = os.path.join(BASE_PATH, "RemoteSensing", "HarvestStatAfrica", "preprocessed")
list_of_files = os.listdir(glam_results_dir)

# Regex to match files like "Country Name_adminX_merged_data_GLAM.csv" or similar
country_pattern = re.compile(r"^(?P<country>.+?)_admin\d+_.*\.csv$")
countries_found = set()
for file in list_of_files:
    match = country_pattern.match(file)
    if match:
        countries_found.add(match.group("country"))
COUNTRIES = list(countries_found)

# Output files
OUT_ADMIN0_MERGED = os.path.join(COMBINED_DIR, "all_admin0_merged_data.csv")
OUT_ADMIN_GT0_MERGED = os.path.join(COMBINED_DIR, "all_admin_gt0_merged_data.csv")
OUT_ADMIN0_RS = os.path.join(COMBINED_DIR, "all_admin0_remote_sensing_data.csv")
OUT_ADMIN_GT0_RS = os.path.join(COMBINED_DIR, "all_admin_gt0_remote_sensing_data.csv")

# Helper to get admin level for a country (from DataAvailability.py logic)
def get_admin_gt0_level(country):
    # Try admin2 first, fallback to admin1
    for level in [2, 1]:
        testfile = os.path.join(
            f"RemoteSensing/HarvestStatAfrica/preprocessed",
            f"{country}_admin{level}_merged_data_GLAM.csv"
        )
        if os.path.exists(testfile):
            return level
    return None

def load_single_country_file(country, pattern, admin_level=None, filetype="merged_data"):
    """
    Load a file for a single country (fixes the double loop issue)
    """
    if admin_level is not None:
        filename = pattern.format(country=country, admin_level=admin_level)
    else:
        filename = pattern.format(country=country)
    
    # Try both relative to script and from project root
    rel_path = os.path.join("RemoteSensing", "HarvestStatAfrica", "preprocessed", filename)
    abs_path = os.path.abspath(rel_path)
    
    if not os.path.exists(abs_path):
        # Try with underscores for spaces (legacy)
        alt_filename = filename.replace(" ", "_")
        rel_path_alt = os.path.join("RemoteSensing", "HarvestStatAfrica", "preprocessed", alt_filename)
        abs_path_alt = os.path.abspath(rel_path_alt)
        if os.path.exists(abs_path_alt):
            abs_path = abs_path_alt
        else:
            return None, f"Missing {filetype} for {country} at admin{admin_level if admin_level is not None else 0}: {filename}"
    
    try:
        df = pd.read_csv(abs_path)
        return df, None
    except Exception as e:
        return None, f"Could not read {abs_path}: {e}"

def concat_files(pattern, admin_level=None, filetype="merged_data"):
    """
    Concatenate files for ALL countries (used for admin0 files)
    """
    dfs = []
    summary = {}
    for country in COUNTRIES:
        df, error = load_single_country_file(country, pattern, admin_level, filetype)
        if df is not None:
            dfs.append(df)
            summary[country] = len(df)
        elif error:
            print(f"[WARN] {error}")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
    else:
        combined = pd.DataFrame()
    return combined, summary

def main():
    print("\n=== CONCATENATING ADMIN0 MERGED DATA ===")
    admin0_merged, admin0_merged_summary = concat_files(ADMIN0_MERGED, filetype="merged_data")
    admin0_merged.to_csv(OUT_ADMIN0_MERGED, index=False)
    print(f"Saved: {OUT_ADMIN0_MERGED}")
    print("Rows per country:", admin0_merged_summary)
    print(f"Total rows: {len(admin0_merged)}\n")

    print("=== CONCATENATING ADMIN>0 MERGED DATA ===")
    admin_gt0_dfs = []
    admin_gt0_summary = {}
    for country in COUNTRIES:
        level = get_admin_gt0_level(country)
        if level is None:
            print(f"[WARN] No admin1/admin2 merged data for {country}")
            continue
        
        # FIXED: Load single country file instead of all countries
        df, error = load_single_country_file(country, ADMIN_GT0_MERGED, admin_level=level, filetype="merged_data")
        if df is not None:
            admin_gt0_dfs.append(df)
            admin_gt0_summary[country] = len(df)
            print(f"[INFO] Loaded {len(df)} rows for {country} (admin{level})")
        elif error:
            print(f"[ERROR] {error}")
    
    if admin_gt0_dfs:
        admin_gt0_merged = pd.concat(admin_gt0_dfs, ignore_index=True)
    else:
        admin_gt0_merged = pd.DataFrame()
    admin_gt0_merged.to_csv(OUT_ADMIN_GT0_MERGED, index=False)
    print(f"Saved: {OUT_ADMIN_GT0_MERGED}")
    print("Rows per country:", admin_gt0_summary)
    print(f"Total rows: {len(admin_gt0_merged)}\n")

    print("=== CONCATENATING ADMIN0 REMOTE SENSING DATA ===")
    admin0_rs, admin0_rs_summary = concat_files(ADMIN0_RS, filetype="remote_sensing")
    admin0_rs.to_csv(OUT_ADMIN0_RS, index=False)
    print(f"Saved: {OUT_ADMIN0_RS}")
    print("Rows per country:", admin0_rs_summary)
    print(f"Total rows: {len(admin0_rs)}\n")

    print("=== CONCATENATING ADMIN>0 REMOTE SENSING DATA ===")
    admin_gt0_rs_dfs = []
    admin_gt0_rs_summary = {}
    for country in COUNTRIES:
        level = get_admin_gt0_level(country)
        if level is None:
            print(f"[WARN] No admin1/admin2 remote sensing data for {country}")
            continue
        
        # FIXED: Load single country file instead of all countries
        df, error = load_single_country_file(country, ADMIN_GT0_RS, admin_level=level, filetype="remote_sensing")
        if df is not None:
            admin_gt0_rs_dfs.append(df)
            admin_gt0_rs_summary[country] = len(df)
            print(f"[INFO] Loaded {len(df)} rows for {country} remote sensing (admin{level})")
        elif error:
            print(f"[ERROR] {error}")
    
    if admin_gt0_rs_dfs:
        admin_gt0_rs = pd.concat(admin_gt0_rs_dfs, ignore_index=True)
    else:
        admin_gt0_rs = pd.DataFrame()
    admin_gt0_rs.to_csv(OUT_ADMIN_GT0_RS, index=False)
    print(f"Saved: {OUT_ADMIN_GT0_RS}")
    print("Rows per country:", admin_gt0_rs_summary)
    print(f"Total rows: {len(admin_gt0_rs)}\n")

if __name__ == "__main__":
    main() 