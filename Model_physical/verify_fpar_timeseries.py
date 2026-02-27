
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# MAIN VERIFICATION SCRIPT
# ============================================================

def get_max_area_pcode(country, crop_area_path):
    """
    Finds the PCODE with the highest crop area for the given country.
    """
    if not os.path.exists(crop_area_path):
        print(f"Warning: Crop area file not found at {crop_area_path}")
        return None
    
    print(f"Finding district with max crop area for {country}...")
    df_area = pd.read_csv(crop_area_path)
    df_country = df_area[df_area['country'] == country]
    df_country = df_country.dropna(subset=['admin_2'])
    
    if df_country.empty:
        print(f"Warning: No crop area data found for {country}.")
        return None
    
    # Determine max area PCODE (average across years if multiple records exist per PCODE)
    pcode_stats = df_country.groupby('PCODE')['crop_area_ha'].mean()
    if pcode_stats.empty:
        return None
        
    max_pcode = pcode_stats.idxmax()
    
    print(f"  - Selected PCODE: {max_pcode} ({pcode_stats.max():.2f} ha)")
    return max_pcode

def process_fpar_pcode(df_pcode):
    """
    Applies interpolation and Savitzky-Golay smoothing (Simple Method).
    """
    df_pcode = df_pcode.sort_values(by='date').copy()
    
    # 1. Interpolate (Linear)
    df_pcode['FPAR_interp'] = df_pcode['FPAR_mean'].interpolate(method='linear', limit_direction='both')
    
    # 2. Smooth (Savitzky-Golay)
    if len(df_pcode) > 31:
        series = df_pcode['FPAR_interp'].fillna(method='bfill').fillna(method='ffill').fillna(0)
        try:
            df_pcode['FPAR_smooth'] = savgol_filter(series, window_length=31, polyorder=2)
        except Exception:
            df_pcode['FPAR_smooth'] = series
        df_pcode['FPAR_smooth'] = df_pcode['FPAR_smooth'].clip(0, 1)
    else:
         df_pcode['FPAR_smooth'] = df_pcode['FPAR_interp'].fillna(0)
         
    return df_pcode

def verify_and_plot_fpar(country, target_pcode=None, years_to_plot=[2017, 2018, 2019, 2020]):
    """
    Loads FPAR data, finds the max area PCODE, processes it, and plots specific years.
    """
    base_dir = os.getcwd()
    
    col_path = os.path.join("GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    area_path = os.path.join("GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    
    fpar_filename = f"{country.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv"
    fpar_path = os.path.join("Model_physical", "Input", fpar_filename)
    
    stsg_filename = f"{country.replace(' ', '_')}_FPAR_STSG.csv"
    stsg_path = os.path.join("Model_physical", "Results", "STSG", stsg_filename)
    
    output_dir = os.path.join("Model_physical", "Results", "STSG_visualize")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(fpar_path):
        print(f"Error: FPAR file not found at {fpar_path}")
        return

    # 1. Get Target PCODE
    if target_pcode is None:
        target_pcode = get_max_area_pcode(country, area_path)
    
    if not target_pcode: 
        print(f"Error: Could not determine PCODE for {country}")
        return
    
    print(f"Visualizing FPAR for PCODE: {target_pcode}")

    # 2. Load Calendar
    if not os.path.exists(col_path):
        print(f"Error: Calendar file not found at {col_path}")
        return
    df_cal = pd.read_csv(col_path)
    if 'FNID' in df_cal.columns: df_cal = df_cal.rename(columns={'FNID': 'PCODE'})
    cal_row = df_cal[df_cal['PCODE'] == target_pcode]
    if cal_row.empty:
        print(f"Error: No calendar data for PCODE {target_pcode}")
        return
    f_plant = cal_row.iloc[0]['Maize_1_planting']
    f_end = cal_row.iloc[0]['Maize_1_endofseaso']
    if pd.isna(f_plant) or pd.isna(f_end):
        print("Error: Calendar dates are NaN.")
        return
    f_plant, f_end = int(f_plant), int(f_end)
    print(f"Calendar for {target_pcode}: Plant DOY {f_plant}, End DOY {f_end}")

    # 3. Load Raw Data
    print(f"Loading raw FPAR data from {fpar_path}...")
    df_fpar = pd.read_csv(fpar_path, parse_dates=['date'])
    df_pcode = df_fpar[df_fpar['PCODE'] == target_pcode].copy()
    if df_pcode.empty:
        print(f"Error: No raw FPAR data found for PCODE {target_pcode}")
        return
    
    print("Processing Simple FPAR (Interpolation + Smooth)...")
    df_pcode = process_fpar_pcode(df_pcode)
    
    # 4. Load Pre-computed STSG data
    if os.path.exists(stsg_path):
        print(f"Loading pre-computed STSG data from {stsg_path}...")
        df_stsg_all = pd.read_csv(stsg_path, parse_dates=['date'])
        df_stsg_pcode = df_stsg_all[df_stsg_all['PCODE'] == target_pcode].copy()
        if not df_stsg_pcode.empty:
            df_pcode = pd.merge(df_pcode, df_stsg_pcode[['date', 'FPAR_STSG']], on='date', how='left')
            # Interpolate STSG to daily to match model behavior
            df_pcode['FPAR_STSG'] = df_pcode['FPAR_STSG'].interpolate(method='linear', limit_direction='both')
            
            # --- SECONDARY SMOOTHING (requested by user) ---
            # Apply SavGol filter to the STSG results
            if len(df_pcode) > 31:
                series_stsg = df_pcode['FPAR_STSG'].fillna(method='bfill').fillna(method='ffill').fillna(0)
                try:
                    df_pcode['FPAR_STSG_smooth'] = savgol_filter(series_stsg, window_length=21, polyorder=2)
                except Exception:
                    df_pcode['FPAR_STSG_smooth'] = series_stsg
                df_pcode['FPAR_STSG_smooth'] = df_pcode['FPAR_STSG_smooth'].clip(0, 1)
            else:
                df_pcode['FPAR_STSG_smooth'] = df_pcode['FPAR_STSG']
        else:
            print(f"Warning: No STSG data found for PCODE {target_pcode} in {stsg_path}")
            df_pcode['FPAR_STSG'] = np.nan
            df_pcode['FPAR_STSG_smooth'] = np.nan
    else:
        print(f"Warning: STSG file not found at {stsg_path}. Run STSG_smoothing.py first.")
        df_pcode['FPAR_STSG'] = np.nan
        df_pcode['FPAR_STSG_smooth'] = np.nan

    # Set index for plotting
    df_pcode = df_pcode.set_index('date', drop=False)
    
    # 5. Plotting
    fig, axes = plt.subplots(len(years_to_plot), 1, figsize=(12, 4 * len(years_to_plot)))
    if len(years_to_plot) == 1: axes = [axes]
    
    for i, year in enumerate(years_to_plot):
        ax = axes[i]
        
        # Window logic
        start_year = year
        if f_plant > f_end:
            start_year = year - 1
            
        win_start = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=f_plant - 1 - 30)
        win_end_year = year 
        win_end = pd.Timestamp(year=win_end_year, month=1, day=1) + pd.Timedelta(days=f_end - 1 + 30)
        
        plot_start, plot_end = win_start - pd.Timedelta(days=10), win_end + pd.Timedelta(days=10)
        
        # Safe slicing
        try:
            sub = df_pcode.loc[str(plot_start):str(plot_end)]
        except Exception:
            sub = pd.DataFrame()
        
        if sub.empty:
            ax.text(0.5, 0.5, f"No Data for Year {year}", transform=ax.transAxes, ha='center')
            continue
            
        # Plot
        ax.plot(sub['date'], sub['FPAR_mean'], 'o', color='black', alpha=0.3, label='Raw FPAR', markersize=4)
        ax.plot(sub['date'], sub['FPAR_smooth'], '--', color='green', linewidth=1, label='Simple Smooth (SavGol)', alpha=0.6)
        
        if 'FPAR_STSG' in sub.columns and not sub['FPAR_STSG'].isna().all():
            ax.plot(sub['date'], sub['FPAR_STSG'], '-', color='blue', linewidth=1.5, label='STSG Smooth (Original)', alpha=0.8)
            
        if 'FPAR_STSG_smooth' in sub.columns and not sub['FPAR_STSG_smooth'].isna().all():
            ax.plot(sub['date'], sub['FPAR_STSG_smooth'], '-', color='orange', linewidth=2, label='STSG + SavGol')
        
        cal_plant_date = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=f_plant - 1)
        cal_end_date = pd.Timestamp(year=win_end_year, month=1, day=1) + pd.Timedelta(days=f_end - 1)
        
        ax.axvline(cal_plant_date, color='blue', linestyle='--', alpha=0.5, label='Calendar Plant')
        ax.axvline(cal_end_date, color='red', linestyle='--', alpha=0.5, label='Calendar End')
        ax.axvspan(win_start, win_end, color='yellow', alpha=0.05, label='Buffer Window')
        
        ax.set_title(f"FPAR Timeseries - Year {year} | PCODE: {target_pcode}")
        ax.set_ylabel("FPAR")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper right', ncol=3, fontsize='small')
            
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f"{country}_{target_pcode}_FPAR_STSG_verify.png")
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify FPAR STSG smoothing by comparing with raw and simple SG.")
    parser.add_argument("--country", type=str, default="Zimbabwe")
    parser.add_argument("--pcode", type=str, default=None, help="Specific PCODE to visualize. If None, uses max area PCODE.")
    args = parser.parse_args()
    
    verify_and_plot_fpar(args.country, target_pcode=args.pcode)
