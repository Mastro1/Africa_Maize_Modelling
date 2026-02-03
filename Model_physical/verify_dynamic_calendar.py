
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

def calculate_dynamic_dates(df_vi_pcode, year, fixed_planting_doy, fixed_harvest_doy):
    """
    Calculates dynamic SOS and EOS dates for a given year using fixed calendar as a guide.
    """
    # Step 1: Define "Search Window"
    start_year = year
    if fixed_planting_doy > fixed_harvest_doy:
        # Crosses year. Assuming "Year" argument is the Harvest Year.
        # So planting was in Year - 1.
        start_year = year - 1
        search_start_date = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=fixed_planting_doy - 1 - 30)
        search_end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=fixed_harvest_doy - 1 + 30)
    else:
         # Same year
         search_start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=fixed_planting_doy - 1 - 30)
         search_end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=fixed_harvest_doy - 1 + 30)
         
    # [Debug] Check Window
    win_len = (search_end_date - search_start_date).days
    # print(f"  [Debug Window] Year {year} | Fixed: {fixed_planting_doy}->{fixed_harvest_doy} | Window: {search_start_date.date()} to {search_end_date.date()} ({win_len} days)")

         
    # Define Plotting Window (Centered Context) for Smoothing
    plot_start_date = search_start_date - pd.Timedelta(days=90)
    plot_end_date = search_end_date + pd.Timedelta(days=90)
    
    # Slice Data for Processing
    mask_plot = (df_vi_pcode['date'] >= plot_start_date) & (df_vi_pcode['date'] <= plot_end_date)
    df_plot = df_vi_pcode[mask_plot].copy()
    
    if df_plot.empty:
         return None, None, None, None, None # Return Nones if no data

    # Smooth Data
    df_plot = df_plot.set_index('date').resample('D').interpolate(method='linear')
    
    window_len = 31
    if len(df_plot) < window_len:
        window_len = len(df_plot) // 2 * 2 + 1 
    
    if window_len > 3:
         df_plot['NDVI_smooth'] = savgol_filter(df_plot['NDVI_mean'], window_length=window_len, polyorder=2)
    else:
         df_plot['NDVI_smooth'] = df_plot['NDVI_mean']
         
    # Extract Search Window Subset
    df_search_logic = df_plot.loc[search_start_date:search_end_date].copy()
    
    if df_search_logic.empty:
        return None, None, None, None, None

    # Step 3: Determine Amplitude
    V_min = df_search_logic['NDVI_smooth'].min()
    V_max = df_search_logic['NDVI_smooth'].max()
    peak_idx = df_search_logic['NDVI_smooth'].idxmax()
    Amplitude = V_max - V_min
    
    # Step 4: Thresholds
    threshold = V_min + (0.20 * Amplitude)
    
    # Step 5: Find Intersection (SOS & EOS)
    pre_peak = df_search_logic.loc[:peak_idx]
    post_peak = df_search_logic.loc[peak_idx:]
    
    # SOS Search (Backwards from Peak)
    sos_date = None
    sos_found = False
    for date, row in pre_peak[::-1].iterrows():
        if row['NDVI_smooth'] < threshold:
            sos_date = date + pd.Timedelta(days=1)
            sos_found = True
            break
    if sos_date is None:
        sos_date = search_start_date 
        
    # EOS Search (Forwards from Peak)
    eos_date = None
    eos_found = False
    for date, row in post_peak.iterrows():
        if row['NDVI_smooth'] < threshold:
            eos_date = date - pd.Timedelta(days=1)
            eos_found = True
            break     
    if eos_date is None:
        eos_date = search_end_date
        
    use_dynamic = True
    season_length = (eos_date - sos_date).days
    
    # Step 6: Fail-Safe
    # Relaxed trigger: 180 was too short for Kenya (mean 217). 
    # Using 330 to just catch essentially "Window edge" cases or extremely short seasons.
    if season_length < 45 or season_length > 335 or not sos_found or not eos_found:
        print(f"  [Debug] Fail-Safe Triggered! Length: {season_length}, SOS Found: {sos_found}, EOS Found: {eos_found}")
        
        # 1. Find Min in the Window
        min_idx = df_search_logic['NDVI_smooth'].idxmin()
        v_min_fs = df_search_logic['NDVI_smooth'].min()
        
        # 2. Find Peak *after* Min
        df_after_min = df_search_logic.loc[min_idx:]
        if not df_after_min.empty:
            peak_idx_fs = df_after_min['NDVI_smooth'].idxmax()
            v_max_fs = df_after_min['NDVI_smooth'].max()
        else:
            peak_idx_fs = peak_idx
            v_max_fs = V_max
        
        amp_fs = v_max_fs - v_min_fs
        thresh_soseos = v_min_fs + (0.20 * amp_fs)
        
        # --- SOS: Look for dates AFTER the minimum ---
        sos_fail = None
        df_rising = df_search_logic.loc[min_idx:peak_idx_fs]
        
        found_rising = False
        for date, row in df_rising.iterrows():
            if row['NDVI_smooth'] >= thresh_soseos:
                sos_fail = date
                found_rising = True
                break
        
        if found_rising and sos_fail:
            sos_date = sos_fail
        else:
            # Default to Minimum Date if localized fails
            if min_idx != search_start_date:
                 sos_date = min_idx

        # --- EOS: Look for dates AFTER the peak ---
        eos_fail = None
        df_falling = df_search_logic.loc[peak_idx_fs:]
        
        found_falling = False
        for date, row in df_falling.iterrows():
            if row['NDVI_smooth'] < thresh_soseos:
                eos_fail = date - pd.Timedelta(days=1)
                found_falling = True
                break
        
        if found_falling and eos_fail:
            eos_date = eos_fail
        else:
            eos_date = search_end_date

        # Final Sanity Check for Fail-Safe
        season_length = (eos_date - sos_date).days
        
        if season_length > 300:
             print(f"  [Warning] Extracted season length {season_length} days is quite long (>300).")

        if season_length < 30 or season_length > 365: # Relaxed to 365 to allow GADM max lengths
             print(f"  [Debug] Final sanity check failed ({season_length} days). Window was {win_len} days. Using Fixed Dates.")
             use_dynamic = False
             sos_date = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=fixed_planting_doy - 1)
             if fixed_planting_doy > fixed_harvest_doy:
                  eos_date = pd.Timestamp(year=start_year + 1, month=1, day=1) + pd.Timedelta(days=fixed_harvest_doy - 1)
             else:
                  eos_date = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=fixed_harvest_doy - 1)

    return sos_date, eos_date, use_dynamic, search_start_date, search_end_date

    
def verify_dynamic_calendar(country_name, input_dir=None, pcode=None, years_to_plot=[2005, 2010, 2015, 2020]):
    """
    Verifies dynamic crop calendar logic (SOS/EOS detection) for a specific country and location.
    
    Args:
        country_name (str): Name of the country (e.g., 'Kenya').
        input_dir (str, optional): Directory containing input data. Defaults to 'Model_physical/Input'.
        pcode (str, optional): Specific PCODE to analyze. If None, finds max crop area PCODE.
        years_to_plot (list): List of years to visualize.
    """
    
    # Defaults
    base_dir = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    if input_dir is None:
        input_dir = os.path.join(base_dir, "Model_physical", "Input")
        
    country_file_name = country_name.replace(" ", "_")
    
    # 1. Load Crop Area Data (to find PCODE if needed)
    crop_area_path = os.path.join(base_dir, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    if pcode is None:
         if os.path.exists(crop_area_path):
             print(f"Finding district with max crop area for {country_name}...")
             df_area = pd.read_csv(crop_area_path)
             df_country = df_area[df_area['country'] == country_name]
             df_country = df_country[df_country['admin_2'].notna()]
             # Drop rows where admin_2 is NaN if relevant, or just rely on PCODE
             # But here we just want max area pcode
             if not df_country.empty:
                 pcode_stats = df_country.groupby('PCODE')['crop_area_ha'].mean()
                 pcode = pcode_stats.idxmax()
                 print(f"  - System selected: {pcode} ({pcode_stats.max():.2f} ha)")
             else:
                 print(f"Error: No crop area data found for {country_name}. Please specify PCODE.")
                 return
         else:
             print("Error: Crop area file not found. Please specify PCODE.")
             return

    # 2. Load Calendar Data (Fixed)
    calendar_path = os.path.join(base_dir, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    if not os.path.exists(calendar_path):
        print(f"Error: Calendar file not found at {calendar_path}")
        return

    print("Loading Crop Calendar...")
    df_cal = pd.read_csv(calendar_path)
    if 'FNID' in df_cal.columns:
        df_cal = df_cal.rename(columns={'FNID': 'PCODE'})
    
    cal_row = df_cal[df_cal['PCODE'] == pcode]
    if cal_row.empty:
        print(f"Error: No calendar data found for {pcode}")
        return
        
    fixed_planting_doy = cal_row.iloc[0]['Maize_1_planting']
    fixed_vegetative_doy = cal_row.iloc[0]['Maize_1_vegetative']
    fixed_harvest_doy = cal_row.iloc[0]['Maize_1_harvest']
    fixed_end_doy = cal_row.iloc[0]['Maize_1_endofseaso']
    
    print(f"Fixed Calendar for {pcode}: Plant {fixed_planting_doy}, Veg {fixed_vegetative_doy}, Harv {fixed_harvest_doy}, End {fixed_end_doy}")

    # 3. Load NDVI/VI Data
    vi_path = os.path.join(input_dir, f"{country_file_name}_admin2_VI_timeseries_GADM.csv")
    if not os.path.exists(vi_path):
        print(f"Error: VI data not found at {vi_path}")
        # Try without GADM suffix just in case? Or user specified name exactly
        return

    print(f"Loading VI data from {vi_path}...")
    df_vi = pd.read_csv(vi_path, parse_dates=['date'])
    
    # Filter for PCODE
    df_vi_pcode = df_vi[df_vi['PCODE'] == pcode].copy()
    if df_vi_pcode.empty:
        print(f"Error: No VI data found for {pcode}")
        return
        
    df_vi_pcode = df_vi_pcode.sort_values('date')
    
    # 4. Process for Selected Years
    # We need to construct the logic described in dynamic_crop_calendar.md
    
    fig, axes = plt.subplots(len(years_to_plot), 1, figsize=(12, 4 * len(years_to_plot)), sharex=False)
    if len(years_to_plot) == 1:
        axes = [axes]

    for i, year in enumerate(years_to_plot):
        ax = axes[i]
        print(f"\nProcessing Year {year}...")

        sos_date, eos_date, use_dynamic, search_start_date, search_end_date = calculate_dynamic_dates(
            df_vi_pcode, year, fixed_planting_doy, fixed_end_doy
        )

        if sos_date is None:
            print(f"  - Calculation failed for year {year}")
            continue

        print(f"  - Detected SOS: {sos_date.date()}, EOS: {eos_date.date()}, Length: {(eos_date - sos_date).days}")

        # PLOTTING RE-CREATION (To keep plot working, we need to re-derive some plot objects mostly)
        # Ideally calculate_dynamic_dates could return df_plot to avoid re-doing work, but for now re-calc is cheap for verification.

        # Re-calc plot data just for visualization
        plot_start_date = search_start_date - pd.Timedelta(days=90)
        plot_end_date = search_end_date + pd.Timedelta(days=90)

        # Slice Data for Plotting (Full Context)
        mask_plot = (df_vi_pcode['date'] >= plot_start_date) & (df_vi_pcode['date'] <= plot_end_date)
        df_plot = df_vi_pcode[mask_plot].copy()

        # Smooth Data (Visual Context - Entire Plot Window)
        df_plot = df_plot.set_index('date').resample('D').interpolate(method='linear')

        window_len = 31
        if len(df_plot) < window_len:
            window_len = len(df_plot) // 2 * 2 + 1

        if window_len > 3:
             df_plot['NDVI_smooth'] = savgol_filter(df_plot['NDVI_mean'], window_length=window_len, polyorder=2)
        else:
             df_plot['NDVI_smooth'] = df_plot['NDVI_mean']

        # Extract Search Window logic df just to get threshold for plotting
        df_search_logic = df_plot.loc[search_start_date:search_end_date].copy()
        if not df_search_logic.empty:
             V_min = df_search_logic['NDVI_smooth'].min()
             V_max = df_search_logic['NDVI_smooth'].max()
             Amplitude = V_max - V_min
             threshold = V_min + (0.20 * Amplitude)
        else:
             threshold = 0

        # PLOTTING
        # Plot Plotting Window (Gray context)
        # Raw data (filtered plot mask again if needed, or just iterate df_vi_pcode for points in range)
        # df_plot has 'NDVI_mean' but it's now resampled daily (interpolated).
        # We want to show original points.
        mask_raw = (df_vi_pcode['date'] >= plot_start_date) & (df_vi_pcode['date'] <= plot_end_date)
        df_raw_plot = df_vi_pcode[mask_raw]
        
        ax.plot(df_raw_plot['date'], df_raw_plot['NDVI_mean'], 'o-', color='black', label='Raw NDVI', alpha=0.6, markersize=4, linewidth=1)
        
        # Plot Smoothed Line (Full Context)
        ax.plot(df_plot.index, df_plot['NDVI_smooth'], '-', color='green', label='Smoothed NDVI', linewidth=2)
         
        # Plot Search Window Limits (Shaded region)
        ax.axvspan(search_start_date, search_end_date, color='yellow', alpha=0.1, label='Search Window')
        
        # Plot Threshold line (clipped to search window)
        ax.plot([search_start_date, search_end_date], [threshold, threshold], color='orange', linestyle='--', label='20% Threshold')
        
        # Plot SOS/EOS markers
        ax.axvline(sos_date, color='blue', linestyle='-', linewidth=2, label='Dynamic SOS')
        ax.axvline(eos_date, color='red', linestyle='-', linewidth=2, label='Dynamic EOS')
        
        # Plot FIXED Calendar Dates
        # Determine Start Year similar to calculate_dynamic_dates logic
        start_year = year
        if fixed_planting_doy > fixed_end_doy: # Crosses year
             start_year = year - 1
             
        # Helper to date
        def doy_to_date(y, doy):
            return pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=doy - 1)
            
        date_plant = doy_to_date(start_year, fixed_planting_doy)
        
        # Vegetative: If < Planting, it's next year? Or just later in same year?
        # Usually Plant < Veg < Harv < End.
        # If Plant is late (e.g. 300) and Veg is early (e.g. 30), Veg is next year.
        y_veg = start_year if fixed_vegetative_doy > fixed_planting_doy else start_year + 1
        date_veg = doy_to_date(y_veg, fixed_vegetative_doy)
        
        y_harv = start_year if fixed_harvest_doy > fixed_planting_doy else start_year + 1
        date_harv = doy_to_date(y_harv, fixed_harvest_doy)
        
        y_end = start_year if fixed_end_doy > fixed_planting_doy else start_year + 1
        date_end = doy_to_date(y_end, fixed_end_doy)
        
        # Plot Fixed Lines (Dashed/Dotted)
        ax.axvline(date_plant, color='cyan', linestyle='--', label='Fixed Plant')
        ax.axvline(date_veg, color='lime', linestyle=':', label='Fixed Veg')
        ax.axvline(date_harv, color='orange', linestyle='--', label='Fixed Harv')
        ax.axvline(date_end, color='magenta', linestyle=':', label='Fixed EndSeaso')
        
        # Plot Search Window Limits (Shaded region?)
        ax.axvspan(search_start_date, search_end_date, color='yellow', alpha=0.1, label='Search Window')

        
        title_text = f"Year {year} | PCODE: {pcode}\nSOS: {sos_date.date()} | EOS: {eos_date.date()} | Length: {(eos_date - sos_date).days} days"
        if not use_dynamic:
            title_text += " (FIXED USED)"
        ax.set_title(title_text)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    output_plot = os.path.join(input_dir, "..", "Results", f"{country_name}_dynamic_calendar_verification.png")
    plt.savefig(output_plot)
    print(f"\nVerification Plot saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Dynamic Crop Calendar")
    parser.add_argument("--country", type=str, default="Kenya", help="Country name")
    args = parser.parse_args()
    
    verify_dynamic_calendar(args.country)
