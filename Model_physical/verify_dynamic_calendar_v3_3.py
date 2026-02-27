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
# GLOBAL CONFIGURATION (Dynamic Calendar V3.3)
# ---------------------------------------------------------
USE_STSG_SMOOTHING = True    # If True, looks for pre-calculated STSG from CSV
SOS_THRESHOLD_PERC = 0.20    # SOS: 20% amplitude threshold (rising limb)
EOS_THRESHOLD_PERC = 0.30    # EOS: 30% amplitude drop threshold (falling limb)
MIN_AMPLITUDE_SIGNAL = 0.05  # Minimal NDVI range to consider valid signal

# V3.3 Smart Dynamic Parameters
BUFFER_DAYS = 60             # Phase 1: Buffer days around the Agricultural Cycle
GRAVITY_SIGMA = 45           # Phase 2: Standard Deviation for Gaussian Gravity Wells (days)
DIP_THRESHOLD_PERC = 0.15    # Phase 3: Dynamic dip must be > 15% of lower peak amplitude
DIP_THRESHOLD_ABS = 0.05     # Phase 3: Absolute dip must be > 0.05 NDVI

# Phase 5: Z-Score Clamp Safety Net
ENABLE_Z_SCORE_CLAMP = False # If True, dynamically clamp runaway seasons to historical median
IQR_MULTIPLIER = 1.5         # Phase 5: Z-Score clamp multiplier

MAX_SEASON_LENGTH = 300      # Hard limit for season length (days)
SILKING_GDD_PERC = 0.50      # Silking: 50% of total seasonal GDD
T_BASE = 10.0                # Base temperature for GDD calculation

# Smoothing (Iterative Savitzky-Golay)
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

def plot_season_V3_3(df_pcode, pcode, year, results, output_dir):
    """
    Enhanced Plotting for V3.3: Makes raw NDVI more visible and displays Phase 2/3 features.
    """
    if not results: return
    
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
    # PLOT NDVI
    ax1.plot(subset['date'], subset['NDVI_mean'], color='#32CD32', label='Raw NDVI', alpha=0.8, linewidth=1)
    ax1.plot(subset['date'], subset['NDVI_smooth'], color='darkgreen', linewidth=2.5, label='Smoothed NDVI')
    
    colors = {1: 'blue', 2: 'orange'}
    
    # Track plotted elements to avoid duplicate legend entries
    plotted_valley = False
    
    for s_idx, res in results.items():
        sos, silking, eos = res['dates']
        sw_start, sw_end = res['window']
        c = colors.get(s_idx, 'black')
        
        # 1. Plot Window Background
        ax1.axvspan(sw_start, sw_end, color=c, alpha=0.08, label=f'Season {s_idx} Window')
        
        # 2. Plot Peak
        if 'peak' in res and res['peak']:
            val_df = subset[subset['date']==res['peak']]
            peak_y = val_df['NDVI_smooth'].values[0] if not val_df.empty else ax1.get_ylim()[1]*.8
            ax1.plot(res['peak'], peak_y, marker='*', markersize=15, color=c, label=f'S{s_idx} Peak')

        # 3. Plot Hard Wall (Valley)
        if 'valley' in res and res['valley'] and not plotted_valley:
            ax1.axvline(res['valley'], color='red', linestyle='--', linewidth=2, label='Hard Wall (Valley)')
            plotted_valley = True
            
        # 4. Plot SOS/EOS
        if sos:
            ax1.axvline(sos, color=c, linestyle='-', linewidth=2)
            ax1.text(sos, ax1.get_ylim()[1], f'S{s_idx} SOS\n{sos.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c)
        if eos:
            ax1.axvline(eos, color=c, linestyle='-.', linewidth=2)
            ax1.text(eos, ax1.get_ylim()[1], f'S{s_idx} EOS\n{eos.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c)
            if sos and eos:
                s_len = (eos - sos).days
                gdd_total = res.get('gdd_total', 0)
                m = res.get('method', 'Unk')
                ax1.text(sos + (eos-sos)/2, ax1.get_ylim()[0] + 0.1, 
                         f'S{s_idx} Length: {s_len} days\nGDD: {gdd_total:.0f}\n[{m}]', 
                         color=c, horizontalalignment='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
        if silking:
            ax1.axvline(silking, color=c, linestyle=':', linewidth=2, alpha=0.7)
            ax1.text(silking, ax1.get_ylim()[1] * 0.9, f'S{s_idx} Silking\n{silking.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c, fontsize=9)

    ax1.set_ylabel('NDVI')
    ax1.set_title(f'Dynamic Season Verification (V3.3) - {pcode} - {year}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{pcode}_{year}_v3.3_dual_season.png")
    plt.savefig(out_file)
    plt.close()

def get_cycle_block_dates(year, seasons_config):
    """Phase 1: Determine the start and end of the Agricultural Cycle block + padding."""
    static_dates = []
    for season in seasons_config:
        s_idx = season['index']
        p_doy = season['planting']
        e_doy = season['endofseason']
        
        # Planting date
        p_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(p_doy - 1))
        
        # End date (handle year wrap)
        if e_doy < p_doy:
            e_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=int(e_doy - 1))
        else:
            e_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(e_doy - 1))
            
        static_dates.append({
            'index': s_idx,
            'static_sos': p_date,
            'static_eos': e_date,
            'expected_peak': p_date + (e_date - p_date) / 2
        })
    
    # Global cycle start/end
    cycle_start = min([d['static_sos'] for d in static_dates]) - pd.Timedelta(days=BUFFER_DAYS)
    cycle_end = max([d['static_eos'] for d in static_dates]) + pd.Timedelta(days=BUFFER_DAYS)
    
    return cycle_start, cycle_end, static_dates

def apply_temporal_gravity(subset, expected_peak):
    """Phase 2: Use Temporal Gravity to find biological peak based on expected static peak."""
    if subset.empty: return None, None
    
    dates_num = mdates.date2num(subset['date'])
    expected_num = mdates.date2num(expected_peak)
    
    # Generate Gaussian bell curve (Gravity Well)
    gravity_well = norm.pdf(dates_num, loc=expected_num, scale=GRAVITY_SIGMA)
    # Normalize to max 1.0 so it just acts as a multiplier penalty further away
    gravity_well = gravity_well / gravity_well.max()
    
    # Gravitized NDVI
    gravitized_ndvi = subset['NDVI_smooth'] * gravity_well
    
    # Find biological peak
    peak_idx = gravitized_ndvi.idxmax()
    biological_peak_date = subset.loc[peak_idx, 'date']
    biological_peak_val = subset.loc[peak_idx, 'NDVI_smooth']
    
    return biological_peak_date, biological_peak_val

def dip_test(subset, peak1_date, peak2_date, peak1_val, peak2_val):
    """Phase 3: The Gatekeeper - Test if a true valley exists between peaks."""
    # Ensure sequential
    p1_d, p2_d = min(peak1_date, peak2_date), max(peak1_date, peak2_date)
    
    valley_subset = subset[(subset['date'] >= p1_d) & (subset['date'] <= p2_d)]
    if valley_subset.empty: return False, None
    
    valley_idx = valley_subset['NDVI_smooth'].idxmin()
    valley_date = valley_subset.loc[valley_idx, 'date']
    valley_val = valley_subset.loc[valley_idx, 'NDVI_smooth']
    
    lower_peak_val = min(peak1_val, peak2_val)
    drop = lower_peak_val - valley_val
    
    # Check physiological threshold
    threshold = max(DIP_THRESHOLD_ABS, lower_peak_val * DIP_THRESHOLD_PERC)
    
    if drop > threshold:
        return True, valley_date # Passes Dip Test (Confirmed Bimodal)
    else:
        return False, None       # Fails Dip Test (Treat as Unimodal)

def extract_season(df_pcode, peak_date, limit_start=None, limit_end=None):
    """Phase 4: Thresholding & Boundary Enforcement."""
    # Find peak index
    peak_locs = df_pcode.index[df_pcode['date'] == peak_date]
    if len(peak_locs) == 0: return None, None, None, None, None
    peak_idx = peak_locs[0]
    peak_pos = df_pcode.index.get_loc(peak_idx)
    
    # Define working subset restricted by boundaries
    start_date = limit_start if limit_start else df_pcode['date'].min()
    end_date = limit_end if limit_end else df_pcode['date'].max()
    
    working_subset = df_pcode[(df_pcode['date'] >= start_date) & (df_pcode['date'] <= end_date)]
    if working_subset.empty: return None, None, None, None, None
    
    v_max = df_pcode.loc[peak_idx, 'NDVI_smooth']
    v_min = working_subset['NDVI_smooth'].min()
    amp = v_max - v_min
    
    if amp <= MIN_AMPLITUDE_SIGNAL: return None, None, None, None, None # Signal too weak
    
    threshold_sos = v_min + SOS_THRESHOLD_PERC * amp
    threshold_eos = v_max - EOS_THRESHOLD_PERC * amp
    
    sos_date = None
    eos_date = None

    # Search SOS Backwards
    for i in range(peak_pos, -1, -1):
        curr_date = df_pcode.iloc[i]['date']
        if curr_date < start_date:
            sos_date = start_date # Hit the wall
            break
        if df_pcode.iloc[i]['NDVI_smooth'] <= threshold_sos:
            sos_date = curr_date
            break
            
    # Search EOS Forwards
    for i in range(peak_pos, len(df_pcode)):
        curr_date = df_pcode.iloc[i]['date']
        if curr_date > end_date:
            eos_date = end_date # Hit the wall
            break
        if df_pcode.iloc[i]['NDVI_smooth'] <= threshold_eos:
            eos_date = curr_date
            break
            
    if not sos_date: sos_date = start_date
    if not eos_date: eos_date = end_date
    
    # Calculate Silking & GDD (If we have both SOS/EOS)
    silking_date = None
    total_gdd = 0
    cold_days = 0 
    
    if sos_date and eos_date:
        if (eos_date - sos_date).days > MAX_SEASON_LENGTH:
            # Fallback for runaway seasons
            eos_date = sos_date + pd.Timedelta(days=MAX_SEASON_LENGTH)
            
        mask_season = (df_pcode['date'] >= sos_date) & (df_pcode['date'] <= eos_date)
        season_df = df_pcode[mask_season]
        if not season_df.empty and 'GDD_daily' in season_df.columns:
            total_gdd = season_df['GDD_daily'].sum()
            target_gdd = total_gdd * SILKING_GDD_PERC
            acc_gdd = 0
            for _, row in season_df.iterrows():
                acc_gdd += row['GDD_daily']
                if acc_gdd >= target_gdd:
                    silking_date = row['date']
                    break
        if 'temperature_2m_min' in df_pcode.columns:
            cold_days = (season_df['temperature_2m_min'] < 10).sum()

    return sos_date, silking_date, eos_date, total_gdd, cold_days
    
def calculate_dates_V3_3(df_pcode, year, seasons_config):
    """Smart Dynamic Crop Calendar Core Logic."""
    results = {}
    
    # Phase 1: Cycle Blocks
    cycle_start, cycle_end, static_dates = get_cycle_block_dates(year, seasons_config)
    cycle_subset = df_pcode[(df_pcode['date'] >= cycle_start) & (df_pcode['date'] <= cycle_end)].copy()
    
    if cycle_subset.empty: return results
    
    # Phase 2: Anchor Search (Temporal Gravity)
    peaks = {}
    for d in static_dates:
        peak_date, peak_val = apply_temporal_gravity(cycle_subset, d['expected_peak'])
        if peak_date:
            peaks[d['index']] = {'date': peak_date, 'val': peak_val, 'static': d}
            
    # Phase 3 & 4: Gatekeeper and Boundaries
    is_bimodal = len(seasons_config) > 1 and len(peaks) > 1
    
    if is_bimodal:
        # P1 is the chronologically earlier peak
        p1_idx, p2_idx = sorted(peaks.keys(), key=lambda k: peaks[k]['date'])
        p1, p2 = peaks[p1_idx], peaks[p2_idx]
        
        # Check if they are exactly the same peak (Gravity pulled them to the same point)
        if p1['date'] == p2['date']:
            is_bimodal = False
            
    if is_bimodal:
        # Phase 3: Dip Test
        passed_dip_test, valley_date = dip_test(cycle_subset, p1['date'], p2['date'], p1['val'], p2['val'])
        
        if passed_dip_test:
            # Route 4A: Confirmed Bimodal - Build the Hard Wall
            for idx in [p1_idx, p2_idx]:
                pk = peaks[idx]
                
                # Set Limits based on which season this chronologically is
                limit_start = cycle_start if pk['date'] <= valley_date else valley_date
                limit_end = valley_date if pk['date'] <= valley_date else cycle_end
                
                sos_d, silk_d, eos_d, gdd, cold = extract_season(df_pcode, pk['date'], limit_start, limit_end)
                if sos_d:
                    results[idx] = {
                        'dates': (sos_d, silk_d, eos_d),
                        'gdd_total': gdd,
                        'cold_days': cold,
                        'window': (limit_start, limit_end),
                        'peak': pk['date'],
                        'valley': valley_date,
                        'method': 'Dynamic_Bimodal'
                    }
        else:
            is_bimodal = False # Failed dip test
            
    if not is_bimodal:
        # Route 4B: Unimodal (Or failed bimodal)
        global_peak_idx = cycle_subset['NDVI_smooth'].idxmax()
        global_peak_date = cycle_subset.loc[global_peak_idx, 'date']
        
        # Assign to the first season config index
        s_idx = seasons_config[0]['index']
        
        sos_d, silk_d, eos_d, gdd, cold = extract_season(df_pcode, global_peak_date, cycle_start, cycle_end)
        if sos_d:
            results[s_idx] = {
                'dates': (sos_d, silk_d, eos_d),
                'gdd_total': gdd,
                'cold_days': cold,
                'window': (cycle_start, cycle_end),
                'peak': global_peak_date,
                'valley': None,
                'method': 'Dynamic_Unimodal'
            }

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
    output_dir = os.path.join(base_dir, "Model_physical", "Results", "V3_3_Verification_Calendar")
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

    def apply_z_score_clamp(all_years_results):
        """
        Phase 5: Safety Net (Z-Score Clamp)
        Iterates through all years of extracted seasons. Calculates the IQR for each
        season index (Season 1, Season 2). Clamps outlier season lengths to the
        historical median.
        """
        if not ENABLE_Z_SCORE_CLAMP:
            return all_years_results
            
        lengths = {}
        for yr, res_yr in all_years_results.items():
            for s_idx, res in res_yr.items():
                if res['dates'][0] and res['dates'][2]:
                    l = (res['dates'][2] - res['dates'][0]).days
                    if s_idx not in lengths: lengths[s_idx] = []
                    lengths[s_idx].append({'year': yr, 'len': l})
                    
        clamped_results = all_years_results.copy()
        for s_idx, data in lengths.items():
            if len(data) < 5: continue # Not enough data for stats
            
            arr = np.array([x['len'] for x in data])
            q75, q25 = np.percentile(arr, [75 ,25])
            iqr = q75 - q25
            median_len = np.median(arr)
            
            lower_bound = median_len - IQR_MULTIPLIER * iqr
            upper_bound = median_len + IQR_MULTIPLIER * iqr
            
            for d in data:
                if d['len'] > upper_bound or d['len'] < lower_bound:
                    yr = d['year']
                    print(f"  [Z-Clamp] Year {yr} Season {s_idx} length ({d['len']}d) bound [{lower_bound:.0f}, {upper_bound:.0f}]. Clamping to {median_len:.0f}d")
                    
                    old_sos, old_silk, old_eos = clamped_results[yr][s_idx]['dates']
                    new_eos = old_sos + pd.Timedelta(days=int(median_len))
                    clamped_results[yr][s_idx]['dates'] = (old_sos, old_silk, new_eos)
                    clamped_results[yr][s_idx]['method'] += '_ZClamped'
                    
        return clamped_results

    years = sorted(df_pcode['Year'].unique())
    all_years_results = {}
    
    # Run Algorithm over ALL years
    print(f"Extracting biological seasons for {len(years)} years...")
    for year in years:
        all_years_results[year] = calculate_dates_V3_3(df_pcode, year, seasons_config)
    
    # Phase 5: Z-Score Clamp
    print(f"Applying Phase 5 Z-Score Clamp across all years...")
    clamped_results = apply_z_score_clamp(all_years_results)
    
    # Plotting loop (only the requested subset)
    for year in years[-6:-1]:
        print(f"\nAnalyzing {year}...")
        results = clamped_results.get(year, {})
        
        # Console output
        for s_idx, res in results.items():
            sos, silk, eos = res['dates']
            s_len = (eos - sos).days if (sos and eos) else "N/A"
            m = res.get('method', 'Unknown')
            print(f"  Season {s_idx} [{m}]: SOS={sos.date() if sos else 'N/A'}, EOS={eos.date() if eos else 'N/A'} ({s_len} days)")

        # Visualization
        plot_season_V3_3(df_pcode, target_pcode, year, results, output_dir)

    print(f"Done. Results saved in: {output_dir}")
