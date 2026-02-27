import pandas as pd
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import matplotlib.dates as mdates

# Import shared functions from V5
from verify_dynamic_calendar_v5 import (
    preprocess_and_merge
)

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Dynamic Calendar V3.2)
# ---------------------------------------------------------
USE_STSG_SMOOTHING = True    # If True, looks for pre-calculated STSG from CSV
SOS_THRESHOLD_PERC = 0.20    # SOS: 20% amplitude threshold (rising limb)
EOS_THRESHOLD_PERC = 0.30    # EOS: 30% amplitude drop threshold (falling limb)
MIN_AMPLITUDE_SIGNAL = 0.05  # Minimal NDVI range to consider valid signal
SOS_WINDOW_DAYS = 30         # Search window +/- days around static planting
MAX_SEASON_LENGTH = 300      # Hard limit for season length (days)
SILKING_GDD_PERC = 0.50      # Silking: 50% of total seasonal GDD
T_BASE = 10.0                # Base temperature for GDD calculation

# Smoothing (Iterative Savitzky-Golay - used if STSG is False or missing)
SG_ITERATIONS = 3            # Number of smoothing rounds
SG_WINDOW_LEN = 31           # Polynomial window size
SG_POLY_ORDER = 2            # Polynomial order
# ---------------------------------------------------------

def refine_smoothing(df, iters=SG_ITERATIONS, window=SG_WINDOW_LEN, poly=SG_POLY_ORDER, input_col='NDVI_mean'):
    """
    Applies Savitzky-Golay smoothing iteratively to remove noise.
    """
    print(f"Applying iterative Savitzky-Golay smoothing ({iters} iterations) on '{input_col}'...")
    y = df[input_col].values
    # Ensure window_length is odd and not larger than data
    w = window if len(y) > window else (len(y) // 2 * 2 - 1)
    if w < 5: w = 5 if len(y) >= 5 else 3
    
    for _ in range(iters):
        y = savgol_filter(y, window_length=w, polyorder=poly)
    
    df['NDVI_smooth'] = y
    return df

def plot_season_v3_2(df_pcode, pcode, year, results, output_dir):
    """
    Enhanced Plotting for V3.2: Makes raw NDVI more visible.
    """
    # Determine plot range
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min
    for s_idx, res in results.items():
        sw_start, sw_end = res['window']
        if sw_start < min_date: min_date = sw_start
        if sw_end > max_date: max_date = sw_end
    if min_date == pd.Timestamp.max: return

    plot_start = min_date - pd.Timedelta(days=30)
    plot_end = max_date + pd.Timedelta(days=30)
    subset = df_pcode[(df_pcode['date'] >= plot_start) & (df_pcode['date'] <= plot_end)]
    if subset.empty: return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    # PLOT NDVI - Raw is made MORE VISIBLE (higher alpha, darker lime)
    ax1.plot(subset['date'], subset['NDVI_mean'], color='#32CD32', label='Raw NDVI', alpha=0.8, linewidth=1)
    ax1.plot(subset['date'], subset['NDVI_smooth'], color='darkgreen', linewidth=2.5, label='Smoothed NDVI')
    
    colors = {1: 'blue', 2: 'orange'}
    for s_idx, res in results.items():
        sos, silking, eos = res['dates']
        sw_start, sw_end = res['window']
        c = colors.get(s_idx, 'black')
        ax1.axvspan(sw_start, sw_end, color=c, alpha=0.08, label=f'Season {s_idx} Window')
        if sos:
            ax1.axvline(sos, color=c, linestyle='-', linewidth=2)
            ax1.text(sos, ax1.get_ylim()[1], f'S{s_idx} SOS\n{sos.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c)
        if eos:
            ax1.axvline(eos, color=c, linestyle='-.', linewidth=2)
            ax1.text(eos, ax1.get_ylim()[1], f'S{s_idx} EOS\n{eos.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c)
            if sos and eos:
                s_len = (eos - sos).days
                gdd_total = res.get('gdd_total', 0)
                ax1.text(sos + (eos-sos)/2, ax1.get_ylim()[0] + 0.1, 
                         f'S{s_idx} Length: {s_len} days\nGDD: {gdd_total:.0f}', 
                         color=c, horizontalalignment='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
        silking = res['dates'][1]
        if silking:
            ax1.axvline(silking, color=c, linestyle=':', linewidth=2, alpha=0.7)
            ax1.text(silking, ax1.get_ylim()[1] * 0.9, f'S{s_idx} Silking\n{silking.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c, fontsize=9)

    ax1.set_ylabel('NDVI')
    ax1.set_title(f'Dynamic Season Verification (V3.2) - {pcode} - {year}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{pcode}_{year}_v3.2_dual_season.png")
    plt.savefig(out_file)
    plt.close()

def calculate_dates_v3_2(df_pcode, year, seasons_config):
    """
    New V3.2 Logic with Robust Fallbacks: 
    1. Define window from static planting to static endofseason (+/- delay).
    2. Find Max and Min WITHIN that window (to define this season's amplitude).
    3. Calculate Thresholds: SOS = min + 20% amp, EOS = max - 50% amp.
    4. Search: Look for crossings in the ENTIRE pcode timeseries (searching outwards from the peak).
    5. Fallback: If dynamic detection fails, use the static calendar dates.
    """
    results = {}

    for season in seasons_config:
        season_idx = season['index']
        static_planting_doy = season['planting']
        static_end_doy = season['endofseason']
        
        # Static Dates (for fallbacks and window definition)
        static_sos = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(static_planting_doy - 1))
        if static_end_doy < static_planting_doy:
            static_eos = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=int(static_end_doy - 1))
        else:
            static_eos = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(static_end_doy - 1))

        # 1. Define Search Window (Window only for finding the amplitude of THIS peak)
        start_search = static_sos - pd.Timedelta(days=SOS_WINDOW_DAYS)
        end_search = static_eos + pd.Timedelta(days=SOS_WINDOW_DAYS)
        
        subset = df_pcode[(df_pcode['date'] >= start_search) & (df_pcode['date'] <= end_search)].copy()
        
        result_entry = {
            'dates': (static_sos, None, static_eos), # Default to static
            'window': (start_search, end_search),
            'static_dates': (static_sos, static_eos),
            'method': 'Dynamic'
        }

        if subset.empty:
             result_entry['method'] = 'Static (No Data)'
             results[season_idx] = result_entry
             continue

        # 2. Find Max and Min in this specific window to define the seasonal curve
        v_max = subset['NDVI_smooth'].max()
        peak_idx = subset['NDVI_smooth'].idxmax()
        peak_date = subset.loc[peak_idx, 'date']
        v_min = subset['NDVI_smooth'].min()
        
        amp = v_max - v_min
        if amp <= MIN_AMPLITUDE_SIGNAL: # Minimal signal check
             result_entry['method'] = 'Static (Low Amp)'
             results[season_idx] = result_entry
             continue

        # 3. Define Thresholds
        threshold_sos = v_min + SOS_THRESHOLD_PERC * amp
        threshold_eos = v_max - EOS_THRESHOLD_PERC * amp 
        
        # 4. Search SOS (Backwards from Peak - looking at full df_pcode context)
        sos_date = None
        # Get index of peak date in the full df_pcode
        full_peak_locs = df_pcode.index[df_pcode['date'] == peak_date]
        if len(full_peak_locs) == 0:
            sos_date = static_sos
            result_entry['method'] = 'Static (Peak Loc Fail)'
        else:
            full_peak_loc = full_peak_locs[0]
            peak_pos_in_full = df_pcode.index.get_loc(full_peak_loc)
            
            for i in range(peak_pos_in_full, -1, -1):
                if df_pcode.iloc[i]['NDVI_smooth'] <= threshold_sos:
                    sos_date = df_pcode.iloc[i]['date']
                    break
        
        # 5. Search EOS (Forwards from Peak - looking at full df_pcode context)
        eos_date = None
        if len(full_peak_locs) > 0:
            full_peak_loc = full_peak_locs[0]
            peak_pos_in_full = df_pcode.index.get_loc(full_peak_loc)
            for i in range(peak_pos_in_full, len(df_pcode)):
                if df_pcode.iloc[i]['NDVI_smooth'] <= threshold_eos:
                    eos_date = df_pcode.iloc[i]['date']
                    break

        # 6. Apply Fallbacks if detection failed or is physically impossible
        if sos_date is None:
            sos_date = static_sos
            result_entry['method'] = 'Static (SOS Fail)'
        if eos_date is None:
            eos_date = static_eos
            if result_entry['method'] == 'Dynamic': result_entry['method'] = 'Static (EOS Fail)'

        # ---------------------------------------------------------
        # NEW: Check Season Length Limit
        # ---------------------------------------------------------
        if (eos_date - sos_date).days > MAX_SEASON_LENGTH:
            sos_date = static_sos
            eos_date = static_eos
            result_entry['method'] = 'Static (Max Length)'
            
        # ---------------------------------------------------------
        # NEW: Calculate Silking based on % of seasonal GDD
        # ---------------------------------------------------------
        silking_date = None
        total_gdd = 0
        cold_days = 0 

        if sos_date and eos_date:
            mask_season = (df_pcode['date'] >= sos_date) & (df_pcode['date'] <= eos_date)
            season_df = df_pcode[mask_season]
            if not season_df.empty and 'GDD_daily' in season_df.columns:
                total_gdd = season_df['GDD_daily'].sum()
                target_gdd = total_gdd * SILKING_GDD_PERC
                
                # Accumulate to find silking
                acc_gdd = 0
                for _, row in season_df.iterrows():
                    acc_gdd += row['GDD_daily']
                    if acc_gdd >= target_gdd:
                        silking_date = row['date']
                        break
            
            # Count cold days
            if 'temperature_2m_min' in df_pcode.columns:
                cold_days = (season_df['temperature_2m_min'] < 10).sum()
        
        result_entry['dates'] = (sos_date, silking_date, eos_date)
        result_entry['gdd_total'] = total_gdd
        result_entry['cold_days'] = cold_days
        results[season_idx] = result_entry

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Dynamic Calendar V3.2 (Peak-based Threshold Search)")
    parser.add_argument("--country", type=str, required=True, help="Country name")
    parser.add_argument("--pcode", type=str, required=False, help="Custom PCODE to display (optional)")
    args = parser.parse_args()
    
    country = args.country
    print(f"--- Verify Dynamic Calendar V3.2 (Iterative SG & Peak-Search) : {country} ---")
    
    base_dir = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    vi_dir = os.path.join(base_dir, "RemoteSensing", "GADM", "extractions")
    era5_dir = os.path.join(base_dir, "RemoteSensing", "GADM", "extractions")
    calendar_path = os.path.join(base_dir, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    output_dir = os.path.join(base_dir, "Model_physical", "Results", "V3_2_Verification_Calendar")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    vi_file = os.path.join(vi_dir, f"{country}_admin2_VI_timeseries_GADM.csv")
    era5_file = os.path.join(era5_dir, f"{country}_admin2_ERA5_timeseries_GADM.csv")
    
    if not os.path.exists(vi_file) or not os.path.exists(era5_file):
        print(f"Error: Missing input files for {country}")
        sys.exit(1)
        
    df_vi = pd.read_csv(vi_file, parse_dates=['date'])
    df_era5 = pd.read_csv(era5_file, parse_dates=['date'])
    df_cal = pd.read_csv(calendar_path)
    
    # Preprocess
    df_main = preprocess_and_merge(df_vi, df_era5)
    
    # Selection of Target PCODE
    target_pcode = args.pcode
    
    if target_pcode:
        print(f"Using custom PCODE: {target_pcode}")
        if target_pcode not in df_main['PCODE'].unique():
            print(f"Warning: Custom PCODE {target_pcode} not found in data for {country}.")
            # We don't exit, we let it fail naturally or provide useful error later
    else:
        print(f"Finding district with max crop area for {country}...")
        crop_area_path = os.path.join(base_dir, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
        target_pcode = None
        if os.path.exists(crop_area_path):
            df_area = pd.read_csv(crop_area_path)
            country_match = country.replace("_", " ")
            df_country_area = df_area[df_area['country'] == country_match]
            valid_pcodes = df_main['PCODE'].unique()
            df_country_area = df_country_area[df_country_area['PCODE'].isin(valid_pcodes)]
            if not df_country_area.empty:
                pcode_stats = df_country_area.groupby('PCODE')['crop_area_ha'].mean()
                target_pcode = pcode_stats.idxmax()
                print(f"  - Selected PCODE: {target_pcode} ({pcode_stats.max():.2f} ha)")
                
        if target_pcode is None:
            target_pcode = df_main.groupby('PCODE')['NDVI_mean'].count().idxmax()
            print(f"  - Fallback PCODE: {target_pcode}")

    # Process and Plot
    df_pcode = df_main[df_main['PCODE'] == target_pcode].copy()
    
    # Apply Smoothing logic
    applied_stsg = False
    if USE_STSG_SMOOTHING:
        # Correct path formatting (Matches STSG_smoothing.py output)
        country_clean = country.replace(' ', '_')
        stsg_path = os.path.join(base_dir, "Model_physical", "Results", "STSG", f"{country_clean}_NDVI_STSG.csv")
        
        if not os.path.exists(stsg_path):
            print(f"\n[STSG] Pre-calculated file not found for {country}. Generating it now...")
            try:
                from STSG_smoothing import run_stsg_ndvi
                # Use the clean country name for the function call
                run_stsg_ndvi(country_clean.replace('_', ' ')) 
                print(f"[STSG] Generation complete for {country}.\n")
            except Exception as e:
                print(f"Error during STSG generation: {e}")
        
        # Reload/Load the file
        if os.path.exists(stsg_path):
            print(f"Loading STSG smoothing: {stsg_path}")
            df_stsg = pd.read_csv(stsg_path, parse_dates=['date'])
            # Merge STSG column back
            df_pcode = pd.merge(df_pcode, df_stsg[['date', 'PCODE', 'NDVI_STSG']], on=['date', 'PCODE'], how='left')
            if 'NDVI_STSG' in df_pcode.columns:
                # IMPORTANT: Interpolate the STSG values to daily frequency
                # before filling, otherwise we get spikes from the raw NDVI fill.
                df_pcode['NDVI_smooth'] = df_pcode['NDVI_STSG'].interpolate(method='linear', limit_direction='both')
                applied_stsg = True
            else:
                print("Warning: NDVI_STSG column not found in file. Falling back to SG.")
        else:
            print(f"Warning: STSG file still missing after attempt to generate. Falling back to SG.")

    # Always apply the refinement filter
    # If we have STSG, we smooth the STSG column. If not, we smooth the raw mean.
    input_col_to_smooth = 'NDVI_smooth' if applied_stsg else 'NDVI_mean'
    df_pcode = refine_smoothing(df_pcode, input_col=input_col_to_smooth)

    df_pcode['Year'] = df_pcode['date'].dt.year

    # Extract Seasons Configuration
    pcode_col = 'FNID' if 'FNID' in df_cal.columns else 'PCODE'
    cal_row = df_cal[df_cal[pcode_col] == target_pcode]
    if cal_row.empty: 
        print("No static calendar found. Exit."); sys.exit(1)
    cal_row = cal_row.iloc[0]

    seasons_config = []
    if pd.notna(cal_row['Maize_1_planting']):
        seasons_config.append({'index': 1, 'planting': int(cal_row['Maize_1_planting']), 'endofseason': int(cal_row['Maize_1_endofseaso'])})
    if 'Maize_2_planting' in cal_row and pd.notna(cal_row['Maize_2_planting']):
         seasons_config.append({'index': 2, 'planting': int(cal_row['Maize_2_planting']), 'endofseason': int(cal_row['Maize_2_endofseaso'])})

    years = sorted(df_pcode['Year'].unique())
    for year in years[-6:-1]:
        print(f"Analyzing {year}...")
        results = calculate_dates_v3_2(df_pcode, year, seasons_config)
        
        # Console output
        for s_idx, res in results.items():
            sos, silk, eos = res['dates']
            s_len = (eos - sos).days if (sos and eos) else "N/A"
            print(f"  Season {s_idx}: SOS={sos.date() if sos else 'N/A'}, EOS={eos.date() if eos else 'N/A'} ({s_len} days)")

        # Visualization
        plot_season_v3_2(df_pcode, target_pcode, year, results, output_dir)

    print(f"Done. Results saved in: {output_dir}")
