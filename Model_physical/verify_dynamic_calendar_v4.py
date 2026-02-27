"""
Script: verify_dynamic_calendar_v4.py
Purpose: Implement and verify GDD-based dynamic calendar logic.
         - SOS: Detected from NDVI (20% amplitude on rising limb).
         - Silking: SOS + 800 GDD.
         - EOS (Maturity): SOS + 1600 GDD.
         - Overlays original static calendar for comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Tweakable)
# ---------------------------------------------------------
T_BASE = 10.0          # Base temperature for Maize (°C)
GDD_SILKING = 800.0    # Accumulated GDD from SOS to Silking
GDD_MATURITY = 1600.0  # Accumulated GDD from SOS to Maturity (EOS)
SOS_WINDOW_DAYS = 30   # Search window +/- days around static planting
# ---------------------------------------------------------

def load_data(country="Zambia"):
    """
    Loads NDVI (VI) and Temperature (ERA5) data.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Pointing to Project Root from Model_physical/
    
    # Paths based on user instructions
    # NDVI: RemoteSensing\GADM\extractions\Zambia_admin2_VI_timeseries_GADM.csv
    vi_path = os.path.join(base_dir, "RemoteSensing", "GADM", "extractions", f"{country}_admin2_VI_timeseries_GADM.csv")
    
    # Temp: RemoteSensing\GADM\extractions\Zambia_admin2_ERA5_timeseries_GADM.csv
    era5_path = os.path.join(base_dir, "RemoteSensing", "GADM", "extractions", f"{country}_admin2_ERA5_timeseries_GADM.csv")
    
    # Crop Calendar (Static) for comparison
    calendar_path = os.path.join(base_dir, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")

    print(f"Loading VI Data: {vi_path}")
    try:
        df_vi = pd.read_csv(vi_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: VI file not found at {vi_path}")
        sys.exit(1)

    print(f"Loading ERA5 Data: {era5_path}")
    try:
        df_era5 = pd.read_csv(era5_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: ERA5 file not found at {era5_path}")
        sys.exit(1)

    print(f"Loading Calendar Data: {calendar_path}")
    try:
        df_calendar = pd.read_csv(calendar_path)
        if 'FNID' in df_calendar.columns:
            df_calendar = df_calendar.rename(columns={'FNID': 'PCODE'})
    except FileNotFoundError:
        print(f"Error: Calendar file not found at {calendar_path}")
        sys.exit(1)

    return df_vi, df_era5, df_calendar

def preprocess_and_merge(df_vi, df_era5):
    """
    Merges VI and ERA5 data on PCODE and date.
    Calculates Daily GDD.
    """
    print("Merging and Preprocessing Data...")
    
    # Filter columns to keep things clean
    # Expecting NDVI_mean in df_vi
    cols_vi = ['date', 'PCODE', 'NDVI_mean']
    # Expecting temperature_2m, temperature_2m_min, temperature_2m_max in df_era5
    # Note: ERA5 data might need Kelvin to Celsius conversion if raw, 
    # but based on previous interactions, it seemed to be in Kelvin (e.g. 291 K).
    # Let's check a sample value or safe convert. 
    # Actually, verify_stress_factors.py subtracted 273.15. So it is likely Kelvin.
    
    cols_era5 = ['date', 'PCODE', 'temperature_2m', 'temperature_2m_min', 'temperature_2m_max']
    
    # Check if min/max exist, otherwise use mean
    era5_cols_avail = df_era5.columns.tolist()
    use_min_max = 'temperature_2m_min' in era5_cols_avail and 'temperature_2m_max' in era5_cols_avail
    
    if not use_min_max:
        print("Warning: Min/Max temperature not found. Using mean temperature for GDD.")
        cols_era5 = ['date', 'PCODE', 'temperature_2m']

    df_vi_sub = df_vi[cols_vi].copy()
    df_era5_sub = df_era5[cols_era5].copy()

    # Merge: Keep all ERA5 dates (Daily) and match VI where available
    # We want to filter ERA5 to only PCODEs that exist in VI data to avoid processing irrelevant regions?
    # Or just assume files match. Let's filter ERA5 by VI PCODEs first to safe memory/time if needed.
    valid_pcodes = df_vi_sub['PCODE'].unique()
    df_era5_sub = df_era5_sub[df_era5_sub['PCODE'].isin(valid_pcodes)]
    
    df_merged = pd.merge(df_era5_sub, df_vi_sub, on=['date', 'PCODE'], how='left')
    
    if not df_merged.empty:
        # Interpolate NDVI to fill daily gaps
        print("Interpolating NDVI for daily values...")
        df_merged['NDVI_mean'] = df_merged.groupby('PCODE')['NDVI_mean'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
        
        # Debug Print
        # print("Sample merged data (before conversion):")
        # print(df_merged[['date', 'temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'NDVI_mean']].head())

    # Convert Kelvin to Celsius if necessary (Assumption: if mean > 100, it's Kelvin)
    # Checking first row
    if not df_merged.empty and df_merged['temperature_2m'].iloc[0] > 100:
        df_merged['temperature_2m'] = df_merged['temperature_2m'] - 273.15
        if use_min_max:
            df_merged['temperature_2m_min'] = df_merged['temperature_2m_min'] - 273.15
            df_merged['temperature_2m_max'] = df_merged['temperature_2m_max'] - 273.15
            
    # Calculate GDD
    # GDD = ((Tmax + Tmin) / 2) - Tbase
    # GDD = max(0, GDD)
    if use_min_max:
        t_avg = (df_merged['temperature_2m_max'] + df_merged['temperature_2m_min']) / 2
    else:
        t_avg = df_merged['temperature_2m']
        
    df_merged['GDD_daily'] = (t_avg - T_BASE).clip(lower=0)
    
    return df_merged

def smooth_ndvi(df):
    """
    Applies Savitzky-Golay smoothing to NDVI.
    """
    # Simple smoothing for V4 verification
    # Using window length 31, polyorder 2
    if len(df) > 31:
        df['NDVI_smooth'] = savgol_filter(df['NDVI_mean'], window_length=31, polyorder=2)
    else:
        df['NDVI_smooth'] = df['NDVI_mean'] # Fallback
    return df

def calculate_dates_v4(df_pcode, year, static_planting_doy, static_harvest_doy):
    """
    Identifies SOS (NDVI logic), Silking (GDD), and EOS (GDD).
    """
    # Wider search window: buffer around the static season
    # Handle Year Crossing (Southern Hemisphere)
    start_search = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(static_planting_doy - SOS_WINDOW_DAYS - 1))
    
    if static_harvest_doy < static_planting_doy:
        # Season ends in the following year
        end_search = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=int(static_harvest_doy + SOS_WINDOW_DAYS - 1))
    else:
        # Season ends in the same year
        end_search = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(static_harvest_doy + SOS_WINDOW_DAYS - 1))
    
    subset = df_pcode[(df_pcode['date'] >= start_search) & (df_pcode['date'] <= end_search)]
    
    if subset.empty:
        # print(f"Warning: Empty search window for {year} ({start_search} to {end_search})")
        return (None, None, None), (start_search, end_search)

    # Find local max and min in this window to define amplitude
    # Drop NaNs for peak detection
    valid_subset = subset.dropna(subset=['NDVI_smooth'])
    if valid_subset.empty:
        return (None, None, None), (start_search, end_search)

    v_max = valid_subset['NDVI_smooth'].max()
    peak_idx = valid_subset['NDVI_smooth'].idxmax()
    
    if pd.isna(peak_idx):
        return (None, None, None), (start_search, end_search)

    peak_date = valid_subset.loc[peak_idx, 'date']

    # We want the threshold based on the rising limb of THIS peak
    # Find the minimum *before* the peak in this window
    pre_peak_subset = valid_subset[valid_subset['date'] <= peak_date]
    if not pre_peak_subset.empty:
        v_min = pre_peak_subset['NDVI_smooth'].min()
    else:
        v_min = valid_subset['NDVI_smooth'].min()

    amp = v_max - v_min
    threshold_sos = v_min + 0.20 * amp
    
    # SOS Logic: Search backwards from peak for SOS
    sos_date = None
    pre_peak = subset.loc[:peak_idx].sort_values('date', ascending=True)
    
    # Iterate backwards from peak
    for i in range(len(pre_peak) - 1, -1, -1):
        row = pre_peak.iloc[i]
        if row['NDVI_smooth'] < threshold_sos:
            sos_date = row['date']
            break
            
    if sos_date is None:
        # Fallback to static if detection fails
        sos_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=static_planting_doy-1)

    # 2. Silking and EOS using GDD accumulation from SOS
    # Filter data starting from SOS
    post_sos = df_pcode[df_pcode['date'] >= sos_date].copy()
    post_sos['Cum_GDD'] = post_sos['GDD_daily'].cumsum()
    
    silking_date = None
    eos_date = None
    
    # Find first date where Cum_GDD >= Threshold
    mask_silk = post_sos['Cum_GDD'] >= GDD_SILKING
    if mask_silk.any():
        silking_date = post_sos[mask_silk].iloc[0]['date']
        
    mask_eos = post_sos['Cum_GDD'] >= GDD_MATURITY
    if mask_eos.any():
        eos_date = post_sos[mask_eos].iloc[0]['date']
        
    return (sos_date, silking_date, eos_date), (start_search, end_search)

def plot_season(df_pcode, pcode, year, result, static_dates, output_dir):
    """
    Plots the season: NDVI curve, GDD-based phases, and static comparison.
    """
    dates, search_window = result
    sos, silking, eos = dates
    sw_start, sw_end = search_window
    static_plant, static_harv = static_dates
    
    if sos is None or eos is None:
        return

    # Define plot range: Search Window +/- 15 days for context
    plot_start = sw_start - pd.Timedelta(days=15)
    plot_end = sw_end + pd.Timedelta(days=15)
    
    subset = df_pcode[(df_pcode['date'] >= plot_start) & (df_pcode['date'] <= plot_end)]
    
    if subset.empty: return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 1. NDVI on Left Axis
    # Original Data (Light Green / Dotted)
    ax1.plot(subset['date'], subset['NDVI_mean'], color='lightgreen', linestyle=':', label='NDVI (Raw)', alpha=0.6)
    # Smoothed Data
    ax1.plot(subset['date'], subset['NDVI_smooth'], color='green', linewidth=2, label='NDVI (Smoothed)')
    ax1.set_ylabel('NDVI', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    
    # 2. V-Stages / GDD (optional, maybe just dates for now)
    
    # 3. Vertical Lines for Dates
    # Dynamic
    ax1.axvline(sos, color='blue', linestyle='-', linewidth=1.5, label='SOS (Dynamic)')
    if silking:
        ax1.axvline(silking, color='gold', linestyle='-', linewidth=1.5, label=f'Silking ({int(GDD_SILKING)} GDD)')
    ax1.axvline(eos, color='red', linestyle='-', linewidth=1.5, label=f'EOS ({int(GDD_MATURITY)} GDD)')

    # Search Window
    ax1.axvspan(sw_start, sw_end, color='gray', alpha=0.1, label='SOS Search Window')
    ax1.axvline(sw_start, color='gray', linestyle=':', alpha=0.3)
    ax1.axvline(sw_end, color='gray', linestyle=':', alpha=0.3)
    
    # Static (Convert DOY to Date for this specific year)
    # Handle year crossing for static dates simply for visualization
    # Static Plant
    curr_year = sos.year
    d_static_plant = pd.Timestamp(year=curr_year, month=1, day=1) + pd.Timedelta(days=static_plant-1)
    # If static plant is way off (e.g. diff year), adjust? 
    # Usually static planting is close to SOS.
    
    d_static_harv = pd.Timestamp(year=curr_year, month=1, day=1) + pd.Timedelta(days=static_harv-1)
    if d_static_harv < d_static_plant:
        d_static_harv = pd.Timestamp(year=curr_year+1, month=1, day=1) + pd.Timedelta(days=static_harv-1)
        
    ax1.axvline(d_static_plant, color='blue', linestyle='--', alpha=0.5, label=f'Static Plant (DOY {static_plant})')
    ax1.axvline(d_static_harv, color='red', linestyle='--', alpha=0.5, label=f'Static Harv (DOY {static_harv})')

    # Shading
    if silking:
        # Vegetative: SOS to Silking
        ax1.axvspan(sos, silking, color='lightgreen', alpha=0.2, label='Vegetative Phase')
        # Reproductive: Silking to EOS
        ax1.axvspan(silking, eos, color='gold', alpha=0.2, label='Grain Filling')
        
    duration = (eos - sos).days
    ax1.set_title(f"Dynamic Calendar V4 (GDD) - {pcode} {year}\nLength: {duration} days | Params: Base={T_BASE}C, Silk={GDD_SILKING}, Mat={GDD_MATURITY}")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Save
    out_name = f"{pcode}_{year}_v4_gdd_calendar.png"
    plt.savefig(os.path.join(output_dir, out_name))
    plt.close()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="Zambia", help="Country name for file paths")
    args = parser.parse_args()
    
    print(f"--- Verify Dynamic Calendar V4 (GDD) : {args.country} ---")
    
    # 1. Load
    df_vi, df_era5, df_cal = load_data(args.country)
    
    # 2. Merge & GDD
    df_main = preprocess_and_merge(df_vi, df_era5)
    
    # 3. Process a specific PCODE (Max Area or First)
    # Select PCODE with the most data and highest average NDVI (likely most agricultural)
    print("Selecting best target PCODE...")
    pcode_stats = df_main.groupby('PCODE')['NDVI_mean'].agg(['count', 'mean']).sort_values(['count', 'mean'], ascending=False)
    
    if pcode_stats.empty:
        print("Error: No data found for any PCODE.")
        sys.exit(1)
        
    target_pcode = pcode_stats.index[0]
    print(f"Processing Target PCODE: {target_pcode} (Count: {pcode_stats.iloc[0]['count']}, Mean NDVI: {pcode_stats.iloc[0]['mean']:.3f})")
    
    df_pcode = df_main[df_main['PCODE'] == target_pcode].copy()
    
    # Smooth NDVI
    df_pcode = smooth_ndvi(df_pcode)
    
    # Get Static Calendar info for this PCODE
    cal_row = df_cal[df_cal['PCODE'] == target_pcode]
    if cal_row.empty:
        print("No static calendar found for PCODE. Using defaults.")
        static_plant = 300 # Nov approx
        static_harv = 120  # April approx
    else:
        static_plant = int(cal_row.iloc[0]['Maize_1_planting'])
        static_harv = int(cal_row.iloc[0]['Maize_1_harvest'])
        
    print(f"Static Dates (DOY): Plant={static_plant}, Harv={static_harv}")
    
    # 4. Loop Years and Calculate
    df_pcode['Year'] = df_pcode['date'].dt.year
    years = sorted(df_pcode['Year'].unique())
    
    output_dir = os.path.join(os.path.dirname(__file__), "Results", "V4_Verification_Calendar")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating plots in: {output_dir}")
    
    # Limit to last 5 years for quick verification
    for year in years[-5:]:
        # Calculate dynamic dates first
        result = calculate_dates_v4(df_pcode, year, static_plant, static_harv)
        dates, search_window = result
        sos, silking, eos = dates
        
        # Count days with Tmin < 10C only between SOS and EOS
        if sos is not None and eos is not None:
            mask_season = (df_pcode['date'] >= sos) & (df_pcode['date'] <= eos)
            df_season = df_pcode[mask_season]
            t_col = 'temperature_2m_min' if 'temperature_2m_min' in df_season.columns else 'temperature_2m'
            cold_days = (df_season[t_col] < 10).sum()
            season_len = (eos - sos).days
            print(f"Analyzing {year}... SOS: {sos.date()}, EOS: {eos.date()} ({season_len} days)")
            print(f"  -> Cold Days during season [Tmin < 10C]: {cold_days}")
        else:
            print(f"Analyzing {year}... (Dynamic dates failed)")
            
        plot_season(df_pcode, target_pcode, year, result, (static_plant, static_harv), output_dir)
        
    print("Done.")

if __name__ == "__main__":
    main()
