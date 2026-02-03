
import pandas as pd
import os

# Define paths
base_dir = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
fpar_path = os.path.join(base_dir, "Model_physical", "Kenya_admin2_FPAR_timeseries_GLAD.csv")
era5_new_path = os.path.join(base_dir, "Model_physical", "Kenya_admin2_new_ERA5_timeseries.csv")
era5_gadm_path = os.path.join(base_dir, "Model_physical", "Kenya_admin2_ERA5_timeseries_GADM.csv")
calendar_path = os.path.join(base_dir, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")

def check_file(path, name):
    print(f"\n--- Checking {name} ---")
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return None
    
    try:
        # Read just a few rows first to check columns
        df = pd.read_csv(path, nrows=5)
        print(f"Columns: {list(df.columns)}")
        
        # Read full file (or chunks if too large, but these seem manageable for now to check PCODE)
        # For efficiency, just check if 'PCODE' or 'FNID' is in columns and unique values
        id_col = 'PCODE' if 'PCODE' in df.columns else 'FNID'
        if id_col not in df.columns:
             print(f"WARNING: Could not find ID column (PCODE or FNID). Found: {df.columns}")
             return None

        # Check for specific ID
        target_id = "KEN.22.5_1"
        
        # We need to read the whole file to check for existence of ID, but use usecols to be faster
        df_ids = pd.read_csv(path, usecols=[id_col])
        unique_ids = df_ids[id_col].unique()
        
        if target_id in unique_ids:
            print(f"SUCCESS: Found target ID '{target_id}' in {name}")
        else:
            print(f"WARNING: Target ID '{target_id}' NOT found in {name}")
            print(f"First 5 IDs: {unique_ids[:5]}")
            
        print(f"Total rows: {len(df_ids)}")
        print(f"Unique IDs: {len(unique_ids)}")
        
        return df
        
    except Exception as e:
        print(f"ERROR reading {name}: {e}")
        return None

# Check all files
check_file(fpar_path, "FPAR Data")
check_file(era5_new_path, "ERA5 New Data")
check_file(era5_gadm_path, "ERA5 GADM Data")
check_file(calendar_path, "Crop Calendar")
