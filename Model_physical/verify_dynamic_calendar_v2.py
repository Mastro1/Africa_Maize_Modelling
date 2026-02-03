
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

def calculate_dynamic_dates_v2(df_vi_pcode, year, fixed_planting_doy, fixed_harvest_doy):
    """
    Calculates dynamic SOS and EOS dates for a given year using derivatives (V2 logic)
    with iterative guardrails for short seasons.
    """
    # Step 1: Define "Search Window"
    start_year = year
    if fixed_planting_doy > fixed_harvest_doy:
        # Crosses year. Assuming "Year" argument is the Harvest Year.
        # So planting was in Year - 1.
        start_year = year - 1
        search_start_date = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=fixed_planting_doy - 1 - 15)
        search_end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=fixed_harvest_doy - 1 + 30)
    else:
         # Same year
         search_start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=fixed_planting_doy - 1 - 15)
         search_end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=fixed_harvest_doy - 1 + 30)
         
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
    df_plot['doy'] = df_plot.index.dayofyear # Add DOY for the snippet logic
    
    window_len = 31
    if len(df_plot) < window_len:
        window_len = len(df_plot) // 2 * 2 + 1 
    
    if window_len > 3:
         df_plot['NDVI_smooth'] = savgol_filter(df_plot['NDVI_mean'], window_length=window_len, polyorder=2)
    else:
         df_plot['NDVI_smooth'] = df_plot['NDVI_mean']
         
    # Extract Search Window Subset
    # We use the search logic from the V2 snippet but adapted to work with dates directly or keeping the subset consistent
    subset = df_plot.loc[search_start_date:search_end_date].copy()
    
    if subset.empty:
        return None, None, None, None, None

    # --- V2 Derivative Logic ---
    
    # 2. Calculate Derivatives
    # 'gradient' calculates central difference
    subset['deriv1'] = np.gradient(subset['NDVI_smooth'])
    subset['deriv2'] = np.gradient(subset['deriv1'])
    
    # 3. Find the Peak (Max NDVI) to split the season in half
    peak_idx = subset['NDVI_smooth'].idxmax()
    # peak_date = peak_idx # Index is Timestamp
    
    # -----------------------------------------
    # 4. Find SOS (Emergence) - Max Acceleration BEFORE Peak
    # -----------------------------------------
    # Look only at days before the peak
    growth_phase = subset.loc[:peak_idx].iloc[:-1] # exclude peak itself to be safe
    
    sos_date = search_start_date # Fallback
    sos_found = False
    
    if not growth_phase.empty:
        # Find index of max 2nd derivative
        sos_idx = growth_phase['deriv2'].idxmax()
        sos_date = sos_idx
        sos_found = True
    
    # -----------------------------------------
    # 5. Find EOS (Maturity) - Max Decay Rate AFTER Peak
    # -----------------------------------------
    # Look only at days after the peak
    decay_phase = subset.loc[peak_idx:].iloc[1:] # exclude peak itself
    
    eos_date = search_end_date # Fallback
    eos_found = False
    
    if not decay_phase.empty:
        # Find index of min 1st derivative (steepest downward slope)
        # We look for the minimum because the slope is negative during senescence
        eos_idx = decay_phase['deriv1'].idxmin()
        eos_date = eos_idx
        eos_found = True

    # --- Iterative Guardrails (< 70 days) ---
    length = (eos_date - sos_date).days
    
    if length < 80 and sos_found and eos_found:
        print(f"  [Iterative] Short season detected ({length} days). Reiterating...")
        
        # Strategy A: Try to move SOS earlier (look for 2nd strongest peak in deriv2)
        # We search in the whole growth_phase again
        
        # Find peaks in 2nd derivative (acceleration peaks)
        # height=0 ensures we only look for positive peaks
        peaks_sos, properties_sos = find_peaks(growth_phase['deriv2'], height=0)
        
        sos_updated = False
        
        if len(peaks_sos) > 1:
            # Sort by height (descending)
            sorted_indices = np.argsort(properties_sos['peak_heights'])[::-1]
            
            # The 1st one (index 0) is likely our current max (or close to it)
            # We want the 2nd one
            second_best_idx = peaks_sos[sorted_indices[1]]
            candidate_sos_date = growth_phase.index[second_best_idx]
            
            # Basic sanity check: is it actually different? 
            # And maybe we prefer earlier? For now just take 2nd strongest.
            if candidate_sos_date != sos_date:
                print(f"  [Iterative] Found 2nd strong SOS candidate at {candidate_sos_date.date()}. Switching SOS.")
                sos_date = candidate_sos_date
                sos_updated = True
        
        # Strategy B: If SOS didn't change (no 2nd peak found), try to move EOS later
        if not sos_updated:
            print("  [Iterative] No alternative SOS found. Trying 2nd strongest EOS...")
            
            # Find valleys in 1st derivative (minima)
            # Equivalent to finding peaks in negative deriv1
            inv_deriv1 = -1 * decay_phase['deriv1']
            peaks_eos, properties_eos = find_peaks(inv_deriv1, height=0)
            
            if len(peaks_eos) > 1:
                # Sort by height (descending) - which is depth of valley
                sorted_indices_eos = np.argsort(properties_eos['peak_heights'])[::-1]
                
                second_best_eos_idx = peaks_eos[sorted_indices_eos[1]]
                candidate_eos_date = decay_phase.index[second_best_eos_idx]
                
                if candidate_eos_date != eos_date:
                    print(f"  [Iterative] Found 2nd strong EOS candidate at {candidate_eos_date.date()}. Switching EOS.")
                    eos_date = candidate_eos_date


    use_dynamic = True
    
    return sos_date, eos_date, use_dynamic, search_start_date, search_end_date

    
def verify_dynamic_calendar_v2(country_name, input_dir=None, pcode=None, years_to_plot=[2005, 2010, 2015, 2020]):
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
    else:
        print(f"Using specified PCODE: {pcode}")

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
        # Try finding a similar one or warn? For now return.
        # Sometimes PCODEs differ slightly, but let's assume valid.
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
    
    fig, axes = plt.subplots(len(years_to_plot), 1, figsize=(12, 4 * len(years_to_plot)), sharex=False)
    if len(years_to_plot) == 1:
        axes = [axes]

    for i, year in enumerate(years_to_plot):
        ax = axes[i]
        print(f"\nProcessing Year {year}...")

        sos_date, eos_date, use_dynamic, search_start_date, search_end_date = calculate_dynamic_dates_v2(
            df_vi_pcode, year, fixed_planting_doy, fixed_end_doy
        )

        if sos_date is None:
            print(f"  - Calculation failed for year {year}")
            continue

        print(f"  - Detected SOS: {sos_date.date()}, EOS: {eos_date.date()}, Length: {(eos_date - sos_date).days}")

        # PLOTTING RE-CREATION
        
        plot_start_date = search_start_date - pd.Timedelta(days=90)
        plot_end_date = search_end_date + pd.Timedelta(days=90)

        # Slice Data for Plotting (Full Context)
        mask_plot = (df_vi_pcode['date'] >= plot_start_date) & (df_vi_pcode['date'] <= plot_end_date)
        df_plot = df_vi_pcode[mask_plot].copy()

        # Smooth Data (Visual Context - Entire Plot Window)
        df_plot = df_plot.set_index('date').resample('D').interpolate(method='linear')

        window_len = 51
        if len(df_plot) < window_len:
            window_len = len(df_plot) // 2 * 2 + 1

        if window_len > 3:
             df_plot['NDVI_smooth'] = savgol_filter(df_plot['NDVI_mean'], window_length=window_len, polyorder=2)
        else:
             df_plot['NDVI_smooth'] = df_plot['NDVI_mean']

        # PLOTTING
        # Plot Plotting Window (Gray context)
        mask_raw = (df_vi_pcode['date'] >= plot_start_date) & (df_vi_pcode['date'] <= plot_end_date)
        df_raw_plot = df_vi_pcode[mask_raw]
        
        ax.plot(df_raw_plot['date'], df_raw_plot['NDVI_mean'], 'o-', color='black', label='Raw NDVI', alpha=0.6, markersize=4, linewidth=1)
        
        # Plot Smoothed Line (Full Context)
        ax.plot(df_plot.index, df_plot['NDVI_smooth'], '-', color='green', label='Smoothed NDVI', linewidth=2)
         
        # Plot Search Window Limits (Shaded region)
        ax.axvspan(search_start_date, search_end_date, color='yellow', alpha=0.1, label='Search Window')
        
        # Plot SOS/EOS markers
        ax.axvline(sos_date, color='blue', linestyle='-', linewidth=2, label='Dynamic SOS (Max Accel)')
        ax.axvline(eos_date, color='red', linestyle='-', linewidth=2, label='Dynamic EOS (Max Decay)')
        
        # Plot FIXED Calendar Dates
        # Determine Start Year similar to calculate_dynamic_dates logic
        start_year = year
        if fixed_planting_doy > fixed_end_doy: # Crosses year
             start_year = year - 1
             
        # Helper to date
        def doy_to_date(y, doy):
            return pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=doy - 1)
            
        date_plant = doy_to_date(start_year, fixed_planting_doy)
        
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
        
        title_text = f"Year {year} | PCODE: {pcode}\nSOS: {sos_date.date()} | EOS: {eos_date.date()} | Length: {(eos_date - sos_date).days} days"
        if not use_dynamic:
            title_text += " (FIXED USED)"
        ax.set_title(title_text)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    output_plot = os.path.join(input_dir, "..", "Results", f"{country_name}_dynamic_calendar_verification_v2.png")
    plt.savefig(output_plot)
    print(f"\nVerification Plot saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Dynamic Crop Calendar V2")
    parser.add_argument("--country", type=str, default="Kenya", help="Country name")
    parser.add_argument("--pcode", type=str, default=None, help="Specific PCODE to analyze (optional)")
    args = parser.parse_args()
    
    verify_dynamic_calendar_v2(args.country, pcode=args.pcode)
