import pandas as pd
import numpy as np
import os
import sys
import glob
import warnings

warnings.filterwarnings("ignore")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from verify_dynamic_calendar_v3_4 import (
    run_country_calendar_v3_4
)

# Configuration & Paths
BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
EXTRACTIONS_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
CALENDAR_PATH = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
CROP_AREA_PATH = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis_V3_4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def run_global_analysis():
    print("--- Scaling Dynamic Calendar V3.4 (Strict Fenced Search - Modular) ---", flush=True)
    
    print("Loading Crop Area Filters...", flush=True)
    allowed_pcodes = None
    if os.path.exists(CROP_AREA_PATH):
        df_crop_area = pd.read_csv(CROP_AREA_PATH)
        allowed_pcodes = set(df_crop_area['PCODE'].unique())
        print(f"  Filtering for {len(allowed_pcodes)} relevant locations.")
    else:
        print(f"  Warning: {CROP_AREA_PATH} not found. Processing all locations.")

    # Files
    vi_files = glob.glob(os.path.join(EXTRACTIONS_DIR, "*_admin*_VI_timeseries_GADM.csv"))
    countries = sorted(list(set([os.path.basename(f).split('_admin')[0] for f in vi_files])))
    
    print(f"Found {len(countries)} countries with Admin 2 VI data.", flush=True)
    for c in countries:
        print(c)
    
    all_country_results = []
    
    for country in countries:
        try:
            # Call the modular country analysis
            df_country = run_country_calendar_v3_4(
                country=country,
                base_dir=BASE_DIR,
                allowed_pcodes=allowed_pcodes,
                save_plots=False # Typically false for global runs
            )
            
            if not df_country.empty:
                print(f"  Generated {len(df_country)} season records.")
                all_country_results.append(df_country)
            else:
                print(f"  No results for {country}.")
                
        except Exception as e:
            print(f"  Error processing {country}: {e}", flush=True)
            continue

    # Combine and Save
    if all_country_results:
        df_final = pd.concat(all_country_results, ignore_index=True)
        final_path = os.path.join(OUTPUT_DIR, "Global_Calendar_V3_4.csv")
        df_final.to_csv(final_path, index=False)
        print(f"Done. Saved {len(df_final)} rows to {final_path}")
    else:
        print("No results generated across all countries.")

if __name__ == "__main__":
    run_global_analysis()
