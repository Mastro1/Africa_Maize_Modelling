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
import matplotlib.dates as mdates
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
    # Note: ERA5 data need Kelvin to Celsius conversion
    
    cols_era5 = ['date', 'PCODE', 'temperature_2m', 'temperature_2m_min', 'temperature_2m_max']
    
    # Check if min/max exist, otherwise use mean
    era5_cols_avail = df_era5.columns.tolist()
    use_min_max = 'temperature_2m_min' in era5_cols_avail and 'temperature_2m_max' in era5_cols_avail
    
    if not use_min_max:
        print("Error: Min/Max temperature not found.")
        sys.exit(1)

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

    # Convert Kelvin to Celsius
    df_merged['temperature_2m_min'] = df_merged['temperature_2m_min'] - 273.15
    df_merged['temperature_2m_max'] = df_merged['temperature_2m_max'] - 273.15
            
    # Calculate GDD
    # GDD = ((Tmax + Tmin) / 2) - Tbase
    # GDD = max(0, GDD)
    # Clip Tmin at 10 degrees to avoid negative influence due to cold nights
    t_avg = (df_merged['temperature_2m_max'] + df_merged['temperature_2m_min'].clip(lower=10)) / 2
        
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

def calculate_dates_v5(df_pcode, year, seasons_config):
    """
    Identifies SOS (NDVI logic), Silking (GDD), and EOS (GDD) for one or more seasons.
    seasons_config: list of dicts, e.g., [{'planting': 60, 'harvest': 200, 'index': 1}, ...]
    """
    results = {}

    for season in seasons_config:
        season_idx = season['index']
        static_planting_doy = season['planting']
        static_harvest_doy = season['harvest']
        
        # --- SOS Logic (V4) ---
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
             results[season_idx] = {'dates': (None, None, None), 'window': (start_search, end_search), 'cold_days': 0}
             continue

        # Find local max and min in this window to define amplitude
        # Drop NaNs for peak detection
        valid_subset = subset.dropna(subset=['NDVI_smooth'])
        if valid_subset.empty:
             results[season_idx] = {'dates': (None, None, None), 'window': (start_search, end_search), 'cold_days': 0}
             continue

        v_max = valid_subset['NDVI_smooth'].max()
        peak_idx = valid_subset['NDVI_smooth'].idxmax()
        
        if pd.isna(peak_idx):
             results[season_idx] = {'dates': (None, None, None), 'window': (start_search, end_search), 'cold_days': 0}
             continue

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
        
        # SOS Search: Backwards from peak
        sos_date = None
        
        peak_loc_in_subset = subset.index.get_loc(peak_idx)
        
        found_start = False
        # Iterate backwards from peak
        for i in range(peak_loc_in_subset, -1, -1):
            curr_ndvi = subset.iloc[i]['NDVI_smooth']
            curr_date = subset.iloc[i]['date']
            
            if curr_ndvi <= threshold_sos:
                sos_date = curr_date
                found_start = True
                break
        
        if not found_start:
            sos_date = None

        # --- GDD Logic for Silking & EOS ---
        silking_date = None
        eos_date = None
        cold_days = 0
        
        if sos_date:
            # Calculate GDD accumulation from SOS
            post_sos = df_pcode[df_pcode['date'] >= sos_date].copy()
            post_sos['Cum_GDD'] = post_sos['GDD_daily'].cumsum()
            
            # Silking
            silking_row = post_sos[post_sos['Cum_GDD'] >= GDD_SILKING].head(1)
            if not silking_row.empty:
                silking_date = pd.Timestamp(silking_row['date'].values[0])
            
            # EOS
            eos_row = post_sos[post_sos['Cum_GDD'] >= GDD_MATURITY].head(1)
            if not eos_row.empty:
                eos_date = pd.Timestamp(eos_row['date'].values[0])

            # Count Cold Days between SOS and EOS
            if eos_date:
                mask_season = (df_pcode['date'] >= sos_date) & (df_pcode['date'] <= eos_date)
                df_season = df_pcode[mask_season]
                t_col = 'temperature_2m_min' if 'temperature_2m_min' in df_season.columns else 'temperature_2m'
                cold_days = (df_season[t_col] < 10).sum()

        results[season_idx] = {'dates': (sos_date, silking_date, eos_date), 'window': (start_search, end_search), 'cold_days': cold_days}

    return results

def plot_season(df_pcode, pcode, year, results, output_dir):
    """
    Plots NDVI time series with dynamic SOS/EOS markers for one or more seasons.
    results: dict of season_index -> {'dates': (sos, silk, eos), 'window': (start, end), 'cold_days': count}
    """
    # Determine plot range: encompass all search windows
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min
    
    for s_idx, res in results.items():
        sw_start, sw_end = res['window']
        if sw_start < min_date: min_date = sw_start
        if sw_end > max_date: max_date = sw_end
        
    if min_date == pd.Timestamp.max:
        return # No valid seasons

    plot_start = min_date - pd.Timedelta(days=30)
    plot_end = max_date + pd.Timedelta(days=30)
    
    subset = df_pcode[(df_pcode['date'] >= plot_start) & (df_pcode['date'] <= plot_end)]
    
    if subset.empty:
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot NDVI
    ax1.plot(subset['date'], subset['NDVI_mean'], color='lightgreen', label='Raw NDVI', alpha=0.5)
    ax1.plot(subset['date'], subset['NDVI_smooth'], color='green', linewidth=2, label='Smoothed NDVI')
    
    colors = {1: 'blue', 2: 'orange'}
    
    for s_idx, res in results.items():
        sos, silking, eos = res['dates']
        sw_start, sw_end = res['window']
        cold_days = res['cold_days']
        c = colors.get(s_idx, 'black')
        
        # Search Window
        ax1.axvspan(sw_start, sw_end, color=c, alpha=0.1, label=f'Season {s_idx} Search Window')
        
        # Dynamic Markers
        if sos:
            ax1.axvline(sos, color=c, linestyle='-', linewidth=2, label=f'Season {s_idx} SOS')
            ax1.text(sos, ax1.get_ylim()[1], f'S{s_idx} SOS\n{sos.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c)
        if silking:
            ax1.axvline(silking, color=c, linestyle='--', linewidth=1.5, label=f'Season {s_idx} Silking')
        if eos:
            ax1.axvline(eos, color=c, linestyle='-.', linewidth=2, label=f'Season {s_idx} EOS')
            ax1.text(eos, ax1.get_ylim()[1], f'S{s_idx} EOS\n{eos.strftime("%b-%d")}', rotation=90, verticalalignment='top', color=c)
            
            # Annotate Cold Days and Season Length
            if sos and eos:
                season_len = (eos - sos).days
                ax1.text(sos + (eos - sos)/2, ax1.get_ylim()[0] + 0.1, f'S{s_idx} Cold Days: {cold_days}\nS{s_idx} Length: {season_len} days', 
                         color=c, horizontalalignment='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor=c))

    ax1.set_ylabel('NDVI')
    ax1.set_title(f'Dynamic Season Verification (V5) - {pcode} - {year}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"{pcode}_{year}_v5_dual_season.png")
    plt.savefig(out_file)
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Dynamic Calendar V5 (Dual Season)")
    parser.add_argument("--country", type=str, required=True, help="Country name")
    args = parser.parse_args()
    
    country = args.country
    print(f"--- Verify Dynamic Calendar V5 (GDD) : {country} ---")
    
    base_dir = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    vi_dir = os.path.join(base_dir, "RemoteSensing", "GADM", "extractions")
    era5_dir = os.path.join(base_dir, "RemoteSensing", "GADM", "extractions")
    calendar_path = os.path.join(base_dir, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    output_dir = os.path.join(base_dir, "Model_physical", "Results", "V5_Verification_Calendar")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VI
    vi_file = os.path.join(vi_dir, f"{country}_admin2_VI_timeseries_GADM.csv")
    if not os.path.exists(vi_file):
        print(f"Error: VI file not found at {vi_file}")
        sys.exit(1)
    print(f"Loading VI Data: {vi_file}")
    df_vi = pd.read_csv(vi_file, parse_dates=['date']) # Added parse_dates
    
    # Load ERA5
    era5_file = os.path.join(era5_dir, f"{country}_admin2_ERA5_timeseries_GADM.csv")
    if not os.path.exists(era5_file):
        print(f"Error: ERA5 file not found at {era5_file}")
        sys.exit(1)
    print(f"Loading ERA5 Data: {era5_file}")
    df_era5 = pd.read_csv(era5_file, parse_dates=['date']) # Added parse_dates
    
    # Load Calendar
    print(f"Loading Calendar Data: {calendar_path}")
    df_cal = pd.read_csv(calendar_path)
    
    # Preprocess
    print("Merging and Preprocessing Data...")
    df_main = preprocess_and_merge(df_vi, df_era5)
    
    # Select PCODE (Using v2 logic: max crop area)
    print(f"Finding district with max crop area for {country}...")
    crop_area_path = os.path.join(base_dir, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    
    target_pcode = None
    if os.path.exists(crop_area_path):
        df_area = pd.read_csv(crop_area_path)
        # Match country name (handling underscores if present)
        country_match = country.replace("_", " ")
        df_country_area = df_area[df_area['country'] == country_match]
        
        # Ensure selected PCODE has data in df_main
        valid_pcodes = df_main['PCODE'].unique()
        df_country_area = df_country_area[df_country_area['PCODE'].isin(valid_pcodes)]
        
        if not df_country_area.empty:
            pcode_stats = df_country_area.groupby('PCODE')['crop_area_ha'].mean()
            target_pcode = pcode_stats.idxmax()
            print(f"  - Selected PCODE by crop area: {target_pcode} ({pcode_stats.max():.2f} ha)")
        else:
            print("Warning: No matching PCODEs with crop area data found. Falling back to data-quality selection.")
    else:
        print(f"Warning: Crop area file not found at {crop_area_path}. Falling back to data-quality selection.")
        
    if target_pcode is None:
        pcode_stats = df_main.groupby('PCODE')['NDVI_mean'].agg(['count', 'mean']).sort_values(['count', 'mean'], ascending=False)
        if pcode_stats.empty:
            print("Error: No data found for any PCODE.")
            sys.exit(1)
        target_pcode = pcode_stats.index[0]
        print(f"  - Selected PCODE by data quality: {target_pcode}")

    print(f"Processing Target PCODE: {target_pcode}")
    
    df_pcode = df_main[df_main['PCODE'] == target_pcode].copy()
    df_pcode = smooth_ndvi(df_pcode)
    
    # Ensure 'Year' column exists for filtering
    df_pcode['Year'] = df_pcode['date'].dt.year

    # Filter calendar by PCODE (FNID in this file) first, then by country
    pcode_col = 'FNID' if 'FNID' in df_cal.columns else 'PCODE'
    cal_pcode_row = df_cal[df_cal[pcode_col] == target_pcode]
    if cal_pcode_row.empty:
        print(f"No static calendar found for PCODE {target_pcode}. Exiting.")
        sys.exit(1)
    
    # If multiple rows for the same PCODE (e.g., different countries), pick the one matching the country
    cal_row = cal_pcode_row[cal_pcode_row['ADMIN0'] == country.replace('_', ' ')]
    if cal_row.empty:
        # Fallback if country name doesn't match exactly, or if PCODE is unique enough
        cal_row = cal_pcode_row.iloc[0]
    else:
        cal_row = cal_row.iloc[0]

    seasons_config = []
    
    if pd.notna(cal_row['Maize_1_planting']):
        seasons_config.append({'index': 1, 'planting': int(cal_row['Maize_1_planting']), 'harvest': int(cal_row['Maize_1_harvest'])})
        
    if 'Maize_2_planting' in cal_row and pd.notna(cal_row['Maize_2_planting']):
         seasons_config.append({'index': 2, 'planting': int(cal_row['Maize_2_planting']), 'harvest': int(cal_row['Maize_2_harvest'])})
        
    print(f"Detected {len(seasons_config)} season(s).")
    
    years = sorted(df_pcode['Year'].unique())
    for year in years[-6:-1]: # Limit to last 5 years for quick verification
        # Calculate dynamic dates
        results = calculate_dates_v5(df_pcode, year, seasons_config)
        
        print(f"Analyzing {year}...")
        for s_idx, res in results.items():
            sos, silking, eos = res['dates']
            season_len = (eos - sos).days if (sos and eos) else "N/A"
            cold = res['cold_days']
            sos_disp = sos.date() if hasattr(sos, 'date') else sos
            eos_disp = eos.date() if hasattr(eos, 'date') else eos
            print(f"  Season {s_idx}: SOS={sos_disp}, EOS={eos_disp} ({season_len} days), Cold={cold}")

        plot_season(df_pcode, target_pcode, year, results, output_dir)
        
    print("Done.")
