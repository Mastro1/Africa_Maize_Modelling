import pandas as pd
import numpy as np
import os
import sys
import glob
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Consistent with V5)
# ---------------------------------------------------------
T_BASE = 10.0
GDD_SILKING = 800.0
GDD_MATURITY = 1600.0
SOS_WINDOW_DAYS = 30

# Paths
BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
EXTRACTIONS_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
CALENDAR_PATH = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# CORE LOGIC (Borrowed from verify_dynamic_calendar_v5.py)
# ---------------------------------------------------------

def preprocess_and_merge(df_vi, df_era5):
    valid_pcodes = df_vi['PCODE'].unique()
    df_era5 = df_era5[df_era5['PCODE'].isin(valid_pcodes)]
    
    # Merge and Interpolate
    df_merged = pd.merge(df_era5, df_vi[['date', 'PCODE', 'NDVI_mean']], on=['date', 'PCODE'], how='left')
    df_merged['NDVI_mean'] = df_merged.groupby('PCODE')['NDVI_mean'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
    
    # K to C
    if not df_merged.empty and df_merged['temperature_2m'].iloc[0] > 100:
        for col in ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max']:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col] - 273.15
                
    # GDD
    if 'temperature_2m_min' in df_merged.columns and 'temperature_2m_max' in df_merged.columns:
        t_avg = (df_merged['temperature_2m_max'] + df_merged['temperature_2m_min'].clip(lower=10)) / 2
    else:
        t_avg = df_merged['temperature_2m']
    df_merged['GDD_daily'] = (t_avg - T_BASE).clip(lower=0)
    
    return df_merged

def smooth_ndvi(df):
    if len(df) > 31:
        df['NDVI_smooth'] = savgol_filter(df['NDVI_mean'], window_length=31, polyorder=2)
    else:
        df['NDVI_smooth'] = df['NDVI_mean']
    return df

def calculate_dates_v5(df_pcode, year, seasons_config):
    results = {}
    for season in seasons_config:
        s_idx = season['index']
        p_doy = season['planting']
        h_doy = season['harvest']
        
        start_search = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(p_doy - SOS_WINDOW_DAYS - 1))
        if h_doy < p_doy:
            end_search = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=int(h_doy + SOS_WINDOW_DAYS - 1))
        else:
            end_search = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(h_doy + SOS_WINDOW_DAYS - 1))
            
        subset = df_pcode[(df_pcode['date'] >= start_search) & (df_pcode['date'] <= end_search)]
        if subset.empty:
            results[s_idx] = {'dates': (None, None, None), 'cold_days': 0}
            continue

        valid_subset = subset.dropna(subset=['NDVI_smooth'])
        if valid_subset.empty:
            results[s_idx] = {'dates': (None, None, None), 'cold_days': 0}
            continue

        v_max = valid_subset['NDVI_smooth'].max()
        peak_idx = valid_subset['NDVI_smooth'].idxmax()
        peak_date = valid_subset.loc[peak_idx, 'date']
        
        pre_peak = valid_subset[valid_subset['date'] <= peak_date]
        v_min = pre_peak['NDVI_smooth'].min() if not pre_peak.empty else valid_subset['NDVI_smooth'].min()
        
        threshold_sos = v_min + 0.20 * (v_max - v_min)
        
        sos_date = None
        peak_loc = subset.index.get_loc(peak_idx)
        for i in range(peak_loc, -1, -1):
            if subset.iloc[i]['NDVI_smooth'] <= threshold_sos:
                sos_date = subset.iloc[i]['date']
                break
        
        silk_date, eos_date, cold_days = None, None, 0
        if sos_date:
            post_sos = df_pcode[df_pcode['date'] >= sos_date].copy()
            post_sos['Cum_GDD'] = post_sos['GDD_daily'].cumsum()
            
            silk_row = post_sos[post_sos['Cum_GDD'] >= GDD_SILKING].head(1)
            if not silk_row.empty: 
                silk_date = pd.Timestamp(silk_row['date'].values[0])
            
            eos_row = post_sos[post_sos['Cum_GDD'] >= GDD_MATURITY].head(1)
            if not eos_row.empty: 
                eos_date = pd.Timestamp(eos_row['date'].values[0])

            if eos_date:
                df_season = df_pcode[(df_pcode['date'] >= sos_date) & (df_pcode['date'] <= eos_date)]
                t_col = 'temperature_2m_min' if 'temperature_2m_min' in df_season.columns else 'temperature_2m'
                cold_days = (df_season[t_col] < 10).sum()

        results[s_idx] = {'dates': (sos_date, silk_date, eos_date), 'cold_days': cold_days}
    return results

# ---------------------------------------------------------
# GLOBAL RUNNER
# ---------------------------------------------------------

def run_global_analysis():
    print(f"Loading Global Calendar Data...", flush=True)
    df_cal = pd.read_csv(CALENDAR_PATH)
    pcode_col = 'FNID' if 'FNID' in df_cal.columns else 'PCODE'
    
    # Identify countries from the folder
    vi_files = glob.glob(os.path.join(EXTRACTIONS_DIR, "*_admin2_VI_timeseries_GADM.csv"))
    countries = [os.path.basename(f).split('_admin2')[0] for f in vi_files]
    countries = sorted(list(set(countries))) # Ensure unique and sorted
    
    print(f"Found {len(countries)} countries with Admin 2 VI data.", flush=True)
    
    all_results = []
    
    # For testing: process first 5 countries, or all
    processed_count = 0
    for country in countries:
        # if processed_count >= 3: break # Optional: limit for testing
        print(f"\n[{processed_count+1}/{len(countries)}] Processing: {country}...", flush=True)
        try:
            vi_path = os.path.join(EXTRACTIONS_DIR, f"{country}_admin2_VI_timeseries_GADM.csv")
            era5_path = os.path.join(EXTRACTIONS_DIR, f"{country}_admin2_ERA5_timeseries_GADM.csv")
            
            if not os.path.exists(era5_path):
                print(f"  [SKIPPING] Missing ERA5 data for {country}.", flush=True)
                processed_count += 1
                continue
                
            df_vi = pd.read_csv(vi_path, parse_dates=['date'])
            df_era5 = pd.read_csv(era5_path, parse_dates=['date'])
            
            df_main = preprocess_and_merge(df_vi, df_era5)
            pcodes = df_main['PCODE'].unique()
            
            for pcode in pcodes:
                df_pcode = df_main[df_main['PCODE'] == pcode].copy()
                if df_pcode.empty: continue
                
                df_pcode = smooth_ndvi(df_pcode)
                df_pcode['Year'] = df_pcode['date'].dt.year
                
                # Calendar lookup for PCODE
                cal_pcode = df_cal[df_cal[pcode_col] == pcode]
                if cal_pcode.empty: continue
                
                cal_row = cal_pcode.iloc[0] # Take first match
                
                seasons_config = []
                if pd.notna(cal_row.get('Maize_1_planting')):
                    seasons_config.append({'index': 1, 'planting': int(cal_row['Maize_1_planting']), 'harvest': int(cal_row['Maize_1_harvest'])})
                if 'Maize_2_planting' in cal_row and pd.notna(cal_row['Maize_2_planting']):
                    seasons_config.append({'index': 2, 'planting': int(cal_row['Maize_2_planting']), 'harvest': int(cal_row['Maize_2_harvest'])})
                
                if not seasons_config: continue
                
                years = sorted(df_pcode['Year'].unique())
                for year in years:
                    if year < 2018: continue # focus on recent years or all? let's do all valid
                    
                    res = calculate_dates_v5(df_pcode, year, seasons_config)
                    for s_idx, data in res.items():
                        sos, silk, eos = data['dates']
                        cold = data['cold_days']
                        
                        all_results.append({
                            'Country': country,
                            'PCODE': pcode,
                            'Year': year,
                            'Season': s_idx,
                            'Static_Planting': cal_row.get(f'Maize_{s_idx}_planting'),
                            'Static_Harvest': cal_row.get(f'Maize_{s_idx}_harvest'),
                            'Dynamic_SOS': sos.date() if sos else None,
                            'Dynamic_Silking': silk.date() if silk else None,
                            'Dynamic_EOS': eos.date() if eos else None,
                            'Season_Length': (eos - sos).days if sos and eos else None,
                            'Cold_Days': cold
                        })
            
            print(f"  [SUCCESS] {country}: {len(pcodes)} PCODEs processed.", flush=True)
            processed_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {country}: {e}", flush=True)
            continue

    # Save to CSV
    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, "Global_Dynamic_Calendar_Dates_V5.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\n--- ALL DONE ---")
    print(f"Total Extractions: {len(df_results)}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    run_global_analysis()
