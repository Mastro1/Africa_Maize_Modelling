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
from verify_dynamic_calendar_v3_4 import (
    refine_smoothing, 
    generate_season_anchors,
    solve_overlaps_and_fence,
    extract_season_strict,
    USE_STSG_SMOOTHING,
    SOS_THRESHOLD_PERC,
    EOS_THRESHOLD_PERC,
    MIN_AMPLITUDE_SIGNAL,
    SILKING_GDD_PERC,
    T_BASE
)

# Configuration & Paths
BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
EXTRACTIONS_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
CALENDAR_PATH = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
CROP_AREA_PATH = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis_V3_4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def preprocess_and_merge(df_vi, df_era5):
    valid_pcodes = df_vi['PCODE'].unique()
    df_era5 = df_era5[df_era5['PCODE'].isin(valid_pcodes)]
    
    # Merge and Interpolate (VI data might be 8-day or have gaps)
    if 'date' in df_vi.columns and df_vi['date'].dtype == 'O': df_vi['date'] = pd.to_datetime(df_vi['date'])
    if 'date' in df_era5.columns and df_era5['date'].dtype == 'O': df_era5['date'] = pd.to_datetime(df_era5['date'])

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
    print("--- Scaling Dynamic Calendar V3.4 (Strict Fenced Search) ---", flush=True)
    
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
                    print(f"  STSG file not found for {country}. Using refined smoothing on-the-fly.")

            for pcode in pcodes:
                df_pcode = df_main[df_main['PCODE'] == pcode].copy()
                if df_pcode.empty: continue
                
                # Apply Smoothing
                applied_stsg = False
                if stsg_df is not None:
                     df_pcode_stsg = stsg_df[stsg_df['PCODE'] == pcode]
                     if not df_pcode_stsg.empty:
                          df_pcode = pd.merge(df_pcode, df_pcode_stsg[['date', 'NDVI_STSG']], on='date', how='left')
                          if 'NDVI_STSG' in df_pcode.columns:
                              df_pcode['NDVI_smooth'] = df_pcode['NDVI_STSG'].interpolate(method='linear', limit_direction='both')
                              applied_stsg = True

                input_col_to_smooth = 'NDVI_smooth' if applied_stsg else 'NDVI_mean'
                df_pcode = refine_smoothing(df_pcode, input_col=input_col_to_smooth)
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
                
                # V3.4 Strict Fenced Logic Start
                unique_years = sorted(df_pcode['Year'].unique())
                all_anchors = generate_season_anchors(unique_years, seasons_config)
                fenced_anchors = solve_overlaps_and_fence(all_anchors)
                
                for meta in fenced_anchors:
                    if meta['fence_end'] > df_pcode['date'].max() or meta['fence_start'] < df_pcode['date'].min():
                        continue
                        
                    res = extract_season_strict(df_pcode, meta)
                    if res:
                        all_results.append({
                            'Country': country,
                            'PCODE': pcode,
                            'Year': meta['year'],
                            'Season': meta['season_idx'],
                            'SOS': res['sos'].date() if hasattr(res['sos'], 'date') else res['sos'],
                            'Silking': res['silk'].date() if res['silk'] and hasattr(res['silk'], 'date') else res['silk'],
                            'EOS': res['eos'].date() if hasattr(res['eos'], 'date') else res['eos'],
                            'Length_Days': (res['eos'] - res['sos']).days if res['sos'] and res['eos'] else None,
                            'GDD_Total': res['gdd'],
                            'Peak_Date': res['peak'].date() if hasattr(res['peak'], 'date') else res['peak'],
                            'Method': res['method'],
                            'Static_SOS': meta['static_sos'].date() if hasattr(meta['static_sos'], 'date') else meta['static_sos'],
                            'Static_EOS': meta['static_eos'].date() if hasattr(meta['static_eos'], 'date') else meta['static_eos'],
                            'Static_Length': meta['static_len'],
                            'Fence_Start': meta['fence_start'].date() if hasattr(meta['fence_start'], 'date') else meta['fence_start'],
                            'Fence_End': meta['fence_end'].date() if hasattr(meta['fence_end'], 'date') else meta['fence_end']
                        })

        except Exception as e:
            print(f"  Error processing {country}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    # Save
    if all_results:
        df_res = pd.DataFrame(all_results)
        final_path = os.path.join(OUTPUT_DIR, "Global_Calendar_V3_4.csv")
        df_res.to_csv(final_path, index=False)
        print(f"Done. Saved {len(df_res)} rows to {final_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_global_analysis()
