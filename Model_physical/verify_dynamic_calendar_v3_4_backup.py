import pandas as pd
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import norm

import matplotlib.dates as mdates

# Import shared functions from V5
from verify_dynamic_calendar_v5 import (
    preprocess_and_merge
)

# TODO:
# 1. Implement a maximum season length derived from the fixed (static) calendar (logic TBD).
# 2. Review dual-season methodology to prevent overlapping.
# 3. Detect not growing seasons and exclude them from the analysis (Flat ndvi)

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Strict Fenced V3.4)
# ---------------------------------------------------------
USE_STSG_SMOOTHING = True    
SOS_THRESHOLD_PERC = 0.20    
EOS_THRESHOLD_PERC = 0.30   
MIN_AMPLITUDE_SIGNAL = 0.05  

# Silking Logic
SILKING_GDD_PERC = 0.50      # Target 50% of seasonal GDD for silking
T_BASE = 10.0                # Base temperature for GDD

# "Smart Cap" Logic
MAX_SEASON_LENGTH_BUFFER = 45 # Days to add to static length for max allowed dynamic length
SEARCH_WINDOW_HALF_SIZE = 150 # Days to look before/after the anchor (The Floating Window size)

# Smoothing
SG_ITERATIONS = 3            
SG_WINDOW_LEN = 31           
SG_POLY_ORDER = 2            
# ---------------------------------------------------------

def refine_smoothing(df, iters=SG_ITERATIONS, window=SG_WINDOW_LEN, poly=SG_POLY_ORDER, input_col='NDVI_mean'):
    """Applies Savitzky-Golay smoothing iteratively."""
    y = df[input_col].values
    w = window if len(y) > window else (len(y) // 2 * 2 - 1)
    if w < 5: w = 5 if len(y) >= 5 else 3
    for _ in range(iters):
        y = savgol_filter(y, window_length=w, polyorder=poly)
    df['NDVI_smooth'] = y
    return df

def generate_season_anchors(years, seasons_config):
    """
    Generates a master list of ALL season occurrences across all years.
    Returns a list of dicts: [{'season_idx': 1, 'year': 2000, 'anchor_date': ...}, ...]
    """
    anchors = []
    for year in years:
        for season in seasons_config:
            s_idx = season['index']
            # Calculate Static Dates
            p_doy = season['planting']
            e_doy = season['endofseason']
            
            # Planting Date
            p_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(p_doy - 1))
            
            # End Date (Handle wrap-around logic for the static calendar itself)
            if e_doy < p_doy:
                e_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=int(e_doy - 1))
            else:
                e_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(e_doy - 1))
                
            # The "Anchor" is the middle of the static season
            anchor_date = p_date + (e_date - p_date) / 2
            
            # Static Length (for Smart Cap)
            static_len = (e_date - p_date).days
            
            anchors.append({
                'season_idx': s_idx,
                'year': year, # The "Crop Year" (start year)
                'anchor_date': anchor_date,
                'static_sos': p_date,
                'static_eos': e_date,
                'static_len': static_len
            })
    return anchors

def solve_overlaps_and_fence(anchors):
    """
    Strict Fence Logic:
    If Season 1 of 2000 overlaps with Season 2 of 2000 (or Season 1 of 2001),
    we calculate the midpoint and build a hard wall.
    """
    # Sort chronologically by anchor date
    sorted_anchors = sorted(anchors, key=lambda x: x['anchor_date'])
    
    for i in range(len(sorted_anchors)):
        current = sorted_anchors[i]
        
        # Default Fences (Wide)
        # We start with the anchor and look SEARCH_WINDOW_HALF_SIZE days back/forward
        current['fence_start'] = current['anchor_date'] - pd.Timedelta(days=SEARCH_WINDOW_HALF_SIZE)
        current['fence_end']   = current['anchor_date'] + pd.Timedelta(days=SEARCH_WINDOW_HALF_SIZE)
        
        # 1. Check Backward Overlap (with previous season)
        if i > 0:
            prev = sorted_anchors[i-1]
            # If default fence overlaps with previous static EOS (plus small buffer)
            midpoint = prev['static_eos'] + (current['static_sos'] - prev['static_eos']) / 2
            
            # Hard Fence: Current Start cannot be before the midpoint
            if current['fence_start'] < midpoint:
                current['fence_start'] = midpoint
                
        # 2. Check Forward Overlap (with next season)
        if i < len(sorted_anchors) - 1:
            nxt = sorted_anchors[i+1]
            midpoint = current['static_eos'] + (nxt['static_sos'] - current['static_eos']) / 2
            
            # Hard Fence: Current End cannot be after the midpoint
            if current['fence_end'] > midpoint:
                current['fence_end'] = midpoint
                
    return sorted_anchors

def extract_season_strict(df_pcode, meta):
    """
    The Core "Strict Fenced" Extraction.
    1. Cut the dataframe to the Fence.
    2. Find Max inside Fence.
    3. Search backward/forward, stopping strictly at Fence.
    """
    # 1. Slice Dataframe (The Floating Window)
    subset = df_pcode[(df_pcode['date'] >= meta['fence_start']) & 
                      (df_pcode['date'] <= meta['fence_end'])].copy()
    
    if subset.empty: return None

    # 2. Find Peak (Trapped inside Fence)
    peak_idx = subset['NDVI_smooth'].idxmax()
    if pd.isna(peak_idx): return None

    peak_date = subset.loc[peak_idx, 'date']
    peak_val = subset.loc[peak_idx, 'NDVI_smooth']
    v_min = subset['NDVI_smooth'].min()
    amp = peak_val - v_min
    
    if amp < MIN_AMPLITUDE_SIGNAL: return None # No signal
    
    # Thresholds
    thresh_sos = v_min + SOS_THRESHOLD_PERC * amp
    thresh_eos = peak_val - EOS_THRESHOLD_PERC * amp

    # 3. Search (Trapped)
    # Get location in the FULL dataframe to allow indexing logic, but bounds are strict
    full_idx = df_pcode.index[df_pcode['date'] == peak_date][0]
    peak_pos = df_pcode.index.get_loc(full_idx)
    
    sos_date = meta['fence_start'] # Default to fence
    eos_date = meta['fence_end']   # Default to fence
    
    # Backward Search (SOS)
    # "Stop-Loss": If we hit a valley and go up, stop.
    lowest_seen = peak_val
    for i in range(peak_pos, -1, -1):
        curr_row = df_pcode.iloc[i]
        curr_date = curr_row['date']
        curr_val = curr_row['NDVI_smooth']
        
        # Hit Fence? Stop.
        if curr_date < meta['fence_start']: break
        
        # Stop-Loss (Uphill check) - prevents climbing previous season
        if curr_val < lowest_seen: lowest_seen = curr_val
        if curr_val > lowest_seen + 0.05: # Climbing back up!
            sos_date = df_pcode.iloc[i+1]['date'] # Snap to the bottom
            break

        if curr_val <= thresh_sos:
            sos_date = curr_date
            break

    # Forward Search (EOS)
    for i in range(peak_pos, len(df_pcode)):
        curr_row = df_pcode.iloc[i]
        curr_date = curr_row['date']
        curr_val = curr_row['NDVI_smooth']
        
        # Hit Fence? Stop.
        if curr_date > meta['fence_end']: break
        
        if curr_val <= thresh_eos:
            eos_date = curr_date
            break
            
    # 4. GDD and Silking Logic
    silking_date = None
    gdd_total = 0
    
    # Slice the original PCODE DF for GDD calculation using detected SOS/EOS
    mask_season = (df_pcode['date'] >= sos_date) & (df_pcode['date'] <= eos_date)
    season_df = df_pcode[mask_season]
    
    if not season_df.empty and 'GDD_daily' in season_df.columns:
        gdd_total = season_df['GDD_daily'].sum()
        target_gdd = gdd_total * SILKING_GDD_PERC
        acc_gdd = 0
        for _, row in season_df.iterrows():
            acc_gdd += row['GDD_daily']
            if acc_gdd >= target_gdd:
                silking_date = row['date']
                break

    # 5. Smart Cap Reality Check
    dynamic_len = (eos_date - sos_date).days
    max_allowed = meta['static_len'] + MAX_SEASON_LENGTH_BUFFER
    
    method = "Dynamic_Strict"
    
    # Fallback if too long
    if dynamic_len > max_allowed:
        sos_date = meta['static_sos']
        eos_date = meta['static_eos']
        method = "Static_Fallback_TooLong"
        # Recalculate Silking for Fallback
        mask_fallback = (df_pcode['date'] >= sos_date) & (df_pcode['date'] <= eos_date)
        fallback_df = df_pcode[mask_fallback]
        if not fallback_df.empty and 'GDD_daily' in fallback_df.columns:
            gdd_total = fallback_df['GDD_daily'].sum()
            target_gdd = gdd_total * SILKING_GDD_PERC
            acc_gdd = 0
            for _, row in fallback_df.iterrows():
                acc_gdd += row['GDD_daily']
                if acc_gdd >= target_gdd:
                    silking_date = row['date']
                    break
        
    return {
        'sos': sos_date,
        'silk': silking_date,
        'eos': eos_date,
        'gdd': gdd_total,
        'peak': peak_date,
        'method': method,
        'meta': meta
    }

def plot_season_strict(df_pcode, pcode, year, season_results, output_dir):
    """Plots all seasons associated with a specific Crop Year."""
    if not season_results: return
    
    # Filter results for the specified year
    year_results = [res for res in season_results if res['meta']['year'] == year]
    if not year_results: return

    # Determine plot range
    min_date = min([res['meta']['fence_start'] for res in year_results])
    max_date = max([res['meta']['fence_end'] for res in year_results])
    
    plot_start = min_date - pd.Timedelta(days=30)
    plot_end = max_date + pd.Timedelta(days=30)
    subset = df_pcode[(df_pcode['date'] >= plot_start) & (df_pcode['date'] <= plot_end)]
    if subset.empty: return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(subset['date'], subset['NDVI_mean'], color='#32CD32', label='Raw NDVI', alpha=0.5, linewidth=1)
    ax1.plot(subset['date'], subset['NDVI_smooth'], color='darkgreen', linewidth=2, label='Smoothed NDVI')
    
    colors = {1: 'blue', 2: 'orange'}
    
    for res in year_results:
        meta = res['meta']
        s_idx = meta['season_idx']
        sos, eos, peak, silk = res['sos'], res['eos'], res['peak'], res.get('silk')
        c = colors.get(s_idx, 'black')
        
        # Plot Fence
        ax1.axvspan(meta['fence_start'], meta['fence_end'], color=c, alpha=0.05, label=f'S{s_idx} Fence')
        
        # Plot Static Window (for comparison)
        ax1.axvspan(meta['static_sos'], meta['static_eos'], color=c, alpha=0.1, hatch='//')

        # Plot Peak
        peak_val = subset[subset['date'] == peak]['NDVI_smooth'].values[0]
        ax1.plot(peak, peak_val, marker='*', markersize=12, color=c, label=f'S{s_idx} Peak')
        
        # Plot SOS/EOS
        ax1.axvline(sos, color=c, linestyle='-', linewidth=2)
        ax1.axvline(eos, color=c, linestyle='--', linewidth=2)
        
        # Plot Silking
        if silk:
            ax1.axvline(silk, color=c, linestyle=':', linewidth=2, alpha=0.8)
            ax1.text(silk, ax1.get_ylim()[0], f'Silk\n{silk.strftime("%b-%d")}', 
                     rotation=90, verticalalignment='bottom', color=c, fontsize=9)
        
        # Labeling
        ax1.text(sos, ax1.get_ylim()[1], f'S{s_idx} SOS\n{sos.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c)
        ax1.text(eos, ax1.get_ylim()[1], f'S{s_idx} EOS\n{eos.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c)
        
    ax1.set_ylabel('NDVI')
    ax1.set_title(f'Dynamic Season V3.4 (Strict Fenced) - {pcode} - {year}')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, f"{pcode}_{year}_v3.4_strict.png")
    plt.savefig(out_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Dynamic Calendar V3.4 (Strict Fenced Search)")
    parser.add_argument("--country", type=str, required=True, help="Country name")
    parser.add_argument("--pcode", type=str, required=False, help="Custom PCODE to display (optional)")
    args = parser.parse_args()
    
    country = args.country
    print(f"--- Verify Dynamic Calendar V3.4 (Strict Fenced Search) : {country} ---")
    
    base_dir = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    vi_dir = os.path.join(base_dir, "RemoteSensing", "GADM", "extractions")
    era5_dir = os.path.join(base_dir, "RemoteSensing", "GADM", "extractions")
    calendar_path = os.path.join(base_dir, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    output_dir = os.path.join(base_dir, "Model_physical", "Results", "V3_4_Verification_Calendar")
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
    
    if not target_pcode:
        print(f"Finding district with max crop area for {country}...")
        crop_area_path = os.path.join(base_dir, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
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

    # Process PCODE
    df_pcode = df_main[df_main['PCODE'] == target_pcode].copy()
    
    # Smoothing
    applied_stsg = False
    if USE_STSG_SMOOTHING:
        country_clean = country.replace(' ', '_')
        stsg_path = os.path.join(base_dir, "Model_physical", "Results", "STSG", f"{country_clean}_NDVI_STSG.csv")
        if os.path.exists(stsg_path):
            print(f"Loading STSG smoothing: {stsg_path}")
            df_stsg = pd.read_csv(stsg_path, parse_dates=['date'])
            df_pcode = pd.merge(df_pcode, df_stsg[['date', 'PCODE', 'NDVI_STSG']], on=['date', 'PCODE'], how='left')
            if 'NDVI_STSG' in df_pcode.columns:
                df_pcode['NDVI_smooth'] = df_pcode['NDVI_STSG'].interpolate(method='linear', limit_direction='both')
                applied_stsg = True

    input_col_to_smooth = 'NDVI_smooth' if applied_stsg else 'NDVI_mean'
    df_pcode = refine_smoothing(df_pcode, input_col=input_col_to_smooth)
    df_pcode['Year'] = df_pcode['date'].dt.year

    # Extract Static Calendar
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

    # V4.0 Logic Start
    unique_years = sorted(df_pcode['Year'].unique())
    all_anchors = generate_season_anchors(unique_years, seasons_config)
    fenced_anchors = solve_overlaps_and_fence(all_anchors)
    
    print(f"Generated {len(fenced_anchors)} distinct seasonal search windows.")
    
    final_results = []
    for meta in fenced_anchors:
        if meta['fence_end'] > df_pcode['date'].max() or meta['fence_start'] < df_pcode['date'].min():
            continue
            
        res = extract_season_strict(df_pcode, meta)
        if res:
            final_results.append(res)
            print(f"Season {meta['season_idx']} [{meta['year']}]: {res['method']} "
                  f"({(res['eos'] - res['sos']).days} days) | GDD: {res['gdd']:.0f} | Silking: {res['silk'].date() if res['silk'] else 'N/A'} "
                  f"Window: {meta['fence_start'].date()} to {meta['fence_end'].date()}")

    # Plotting loop (last few years)
    for year in unique_years[-6:-1]:
        print(f"Plotting {year}...")
        plot_season_strict(df_pcode, target_pcode, year, final_results, output_dir)

    print(f"Done. Results saved in: {output_dir}")
