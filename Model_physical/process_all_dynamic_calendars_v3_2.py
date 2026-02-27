import pandas as pd
import numpy as np
import os
import sys
import glob
import warnings
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from verify_dynamic_calendar_v3_2 import (
    refine_smoothing, 
    calculate_dates_v3_2,
    SOS_THRESHOLD_PERC,
    EOS_THRESHOLD_PERC,
    MIN_AMPLITUDE_SIGNAL,
    SOS_WINDOW_DAYS,
    MAX_SEASON_LENGTH,
    SILKING_GDD_PERC,
    T_BASE
)

# Configuration & Paths
USE_STSG_SMOOTHING = True    # If True, looks for pre-calculated STSG from CSV, fallback to SG
BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
EXTRACTIONS_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
CALENDAR_PATH = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
CROP_AREA_PATH = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis_V3_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def preprocess_and_merge(df_vi, df_era5):
    valid_pcodes = df_vi['PCODE'].unique()
    df_era5 = df_era5[df_era5['PCODE'].isin(valid_pcodes)]
    
    # Merge and Interpolate (VI data might be 8-day or have gaps)
    # Ensure both have date as datetime
    if df_vi['date'].dtype == 'O': df_vi['date'] = pd.to_datetime(df_vi['date'])
    if df_era5['date'].dtype == 'O': df_era5['date'] = pd.to_datetime(df_era5['date'])

    df_merged = pd.merge(df_era5, df_vi[['date', 'PCODE', 'NDVI_mean']], on=['date', 'PCODE'], how='left')
    
    # Interpolate NDVI per PCODE
    df_merged['NDVI_mean'] = df_merged.groupby('PCODE')['NDVI_mean'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
    
    # K to C
    if not df_merged.empty and 'temperature_2m' in df_merged.columns and df_merged['temperature_2m'].iloc[0] > 100:
        for col in ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max']:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col] - 273.15
                
    # GDD Calculation
    if 'temperature_2m_min' in df_merged.columns and 'temperature_2m_max' in df_merged.columns:
        t_avg = (df_merged['temperature_2m_max'] + df_merged['temperature_2m_min'].clip(lower=10)) / 2
    elif 'temperature_2m' in df_merged.columns:
        t_avg = df_merged['temperature_2m']
    else:
        t_avg = 0
        
    df_merged['GDD_daily'] = (t_avg - T_BASE).clip(lower=0)
    
    return df_merged

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def run_global_analysis():
    print("Loading Global Calendar Data...", flush=True)
    df_cal = pd.read_csv(CALENDAR_PATH)
    pcode_col = 'FNID' if 'FNID' in df_cal.columns else 'PCODE'
    
    print("Loading Filtered Crop Areas...", flush=True)
    if os.path.exists(CROP_AREA_PATH):
        df_crop_area = pd.read_csv(CROP_AREA_PATH)
        allowed_pcodes = set(df_crop_area['PCODE'].unique())
        print(f"  Filtering for {len(allowed_pcodes)} relevant locations.")
    else:
        print(f"  Warning: {CROP_AREA_PATH} not found. Processing all locations.")
        allowed_pcodes = None

    # Files
    vi_files = glob.glob(os.path.join(EXTRACTIONS_DIR, "*_admin2_VI_timeseries_GADM.csv"))
    countries = [os.path.basename(f).split('_admin2')[0] for f in vi_files]
    countries = sorted(list(set(countries)))
    
    print(f"Found {len(countries)} countries with Admin 2 VI data.", flush=True)
    
    all_results = []
    
    for country in countries:
        country_clean = country.replace(' ', '_')
        print(f"Processing: {country}...", flush=True)

        try:
            vi_path = os.path.join(EXTRACTIONS_DIR, f"{country}_admin2_VI_timeseries_GADM.csv")
            era5_path = os.path.join(EXTRACTIONS_DIR, f"{country}_admin2_ERA5_timeseries_GADM.csv")
            
            if not os.path.exists(era5_path):
                print(f"  Skipping {country}: Missing ERA5 file.", flush=True)
                continue
                
            df_vi = pd.read_csv(vi_path)
            df_era5 = pd.read_csv(era5_path)
            
            df_main = preprocess_and_merge(df_vi, df_era5)
            pcodes = df_main['PCODE'].unique()
            
            # --- Filter pcodes by crop area ---
            if allowed_pcodes is not None:
                pcodes = [p for p in pcodes if p in allowed_pcodes]
                if not pcodes:
                    print(f"  No relevant PCODEs for {country} (skipping).")
                    continue
            
            # --- STSG HANDLING ---
            stsg_df = None
            if USE_STSG_SMOOTHING:
                stsg_path = os.path.join(BASE_DIR, "Model_physical", "Results", "STSG", f"{country_clean}_NDVI_STSG.csv")
                if os.path.exists(stsg_path):
                    stsg_df = pd.read_csv(stsg_path, parse_dates=['date'])
                    print(f"  Loaded STSG for {country}")
                else:
                    print(f"  STSG file not found for {country}. Skipping STSG for this country.")

            for pcode in pcodes:
                df_pcode = df_main[df_main['PCODE'] == pcode].copy()
                if df_pcode.empty: continue
                
                # Apply Smoothing
                applied_stsg = False
                if stsg_df is not None:
                     df_pcode_stsg = stsg_df[stsg_df['PCODE'] == pcode]
                     if not df_pcode_stsg.empty:
                         if 'date' in df_pcode.columns and 'date' in df_pcode_stsg.columns:
                            df_pcode = pd.merge(df_pcode, df_pcode_stsg[['date', 'NDVI_STSG']], on='date', how='left')
                            df_pcode['NDVI_smooth'] = df_pcode['NDVI_STSG'].interpolate(method='linear', limit_direction='both')
                            applied_stsg = True

                # Always apply the refinement filter
                input_col_to_smooth = 'NDVI_smooth' if applied_stsg else 'NDVI_mean'
                df_pcode = refine_smoothing(df_pcode, input_col=input_col_to_smooth)

                if 'NDVI_smooth' not in df_pcode.columns:
                     df_pcode['NDVI_smooth'] = df_pcode['NDVI_mean'] # Fallback

                df_pcode = df_pcode.dropna(subset=['date'])
                df_pcode['Year'] = df_pcode['date'].dt.year
                
                # Calendar Lookup
                cal_pcode = df_cal[df_cal[pcode_col] == pcode]
                if cal_pcode.empty: continue
                cal_row = cal_pcode.iloc[0]
                
                seasons_config = []
                if pd.notna(cal_row.get('Maize_1_planting')) and pd.notna(cal_row.get('Maize_1_endofseaso')):
                    seasons_config.append({'index': 1, 'planting': int(cal_row['Maize_1_planting']), 'endofseason': int(cal_row['Maize_1_endofseaso'])})
                if pd.notna(cal_row.get('Maize_2_planting')) and pd.notna(cal_row.get('Maize_2_endofseaso')):
                    seasons_config.append({'index': 2, 'planting': int(cal_row['Maize_2_planting']), 'endofseason': int(cal_row['Maize_2_endofseaso'])})
                
                if not seasons_config: continue
                
                # Process Years
                years = df_pcode['Year'].dropna().unique()
                years = sorted([int(y) for y in years if y > 0])
                
                for year in years:
                    if year < 2000: continue 
                    
                    res_map = calculate_dates_v3_2(df_pcode, year, seasons_config)
                    
                    for s_idx, data in res_map.items():
                        sos, silk, eos = data['dates']
                        static_sos, static_eos = data.get('static_dates', (None, None))
                        gdd = data.get('gdd_total', 0)
                        method = data.get('method', 'Unknown')
                        
                        all_results.append({
                            'Country': country,
                            'PCODE': pcode,
                            'Year': year,
                            'Season': s_idx,
                            'SOS': sos.date() if hasattr(sos, 'date') else sos,
                            'Silking': silk.date() if hasattr(silk, 'date') else silk,
                            'EOS': eos.date() if hasattr(eos, 'date') else eos,
                            'Length_Days': (eos - sos).days if sos and eos else None,
                            'GDD_Total': gdd,
                            'Cold_Days': data.get('cold_days', 0),
                            'Method': method,
                            'Static_SOS': static_sos.date() if hasattr(static_sos, 'date') else static_sos,
                            'Static_EOS': static_eos.date() if hasattr(static_eos, 'date') else static_eos
                        })

        except Exception as e:
            print(f"  Error processing {country}: {e}", flush=True)
            continue

    # Save
    if all_results:
        df_res = pd.DataFrame(all_results)
        final_path = os.path.join(OUTPUT_DIR, "Global_Calendar_V3_2.csv")
        df_res.to_csv(final_path, index=False)
        print(f"Done. Saved {len(df_res)} rows to {final_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_global_analysis()
