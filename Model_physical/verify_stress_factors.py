"""
Script: verify_stress_factors.py
Purpose: Re-create stress factors (Ts, Ws) as in the model and visualize them.
         Focuses on the Admin-2 unit with the highest crop area in Zimbabwe.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# STRESS FACTOR TUNING PARAMETERS
# ---------------------------------------------------------
T_OPT = 25.0   # Optimal Temperature (°C)
T_SIGMA = 7.0  # Sigma (Width) of the Gaussian Stress Curve
# ---------------------------------------------------------


# Improve plot resolution
plt.rcParams['figure.dpi'] = 150

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import MaizeYieldModelV3 to reuse logic
try:
    from model_v3 import MaizeYieldModelV3
    from verify_dynamic_calendar_v3 import calculate_dynamic_dates_v3, get_search_window
    from STSG_smoothing import get_stsg_path
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

def run_stress_analysis(country="Zimbabwe"):
    # ---------------------------------------------------------
    # 1. Setup & Load Data (Reusing Model V3)
    # ---------------------------------------------------------
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
    CALENDAR_FILE = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results")
    
    print(f"Initializing Model V3 for {country}...")
    model = MaizeYieldModelV3(
        data_dir=DATA_DIR,
        gadm_data_dir=os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions"),
        fpar_file=f"{country.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv",
        era5_new_file=f"{country.replace(' ', '_')}_admin2_new_ERA5_timeseries.csv",
        era5_gadm_file=f"{country.replace(' ', '_')}_admin2_ERA5_timeseries_GADM.csv",
        calendar_file=CALENDAR_FILE,
        output_dir=OUTPUT_DIR,
        country=country,
        crop_area_file=CROP_AREA_FILE
    )

    # Load Data steps from run_full_model, but we stop before calculate_yield
    df_daily = model.load_and_merge_data()
    df_daily = model.preprocess_fpar(df_daily)
    df_daily = model.calculate_biophysical_variables(df_daily)
    df_calendar = model.load_crop_calendar()

    # ---------------------------------------------------------
    # 2. Select Target PCODE
    # ---------------------------------------------------------
    target_pcode = model.get_max_area_pcode(country)
    if not target_pcode:
        print("Could not determine max area PCODE. Using first available.")
        target_pcode = df_daily['PCODE'].unique()[0]
    
    print(f"\nSelected Target PCODE: {target_pcode}")
    
    # ---------------------------------------------------------
    # 3. Calculate Stress & Yield (Detailed Loop)
    # ---------------------------------------------------------
    print("Calculating daily stress factors...")
    
    # Filter for PCODE
    df_pcode = df_daily[df_daily['PCODE'] == target_pcode].copy()
    df_pcode['Year'] = df_pcode['date'].dt.year
    df_pcode = df_pcode.set_index('date', drop=False)
    
    cal_row = df_calendar[df_calendar['PCODE'] == target_pcode]
    if cal_row.empty:
        print("No calendar data for PCODE.")
        return

    f_planting = cal_row.iloc[0]['Maize_1_planting']
    f_harvest = cal_row.iloc[0]['Maize_1_harvest']
    f_end = cal_row.iloc[0]['Maize_1_endofseaso']

    years = df_pcode['Year'].unique()
    
    daily_segments = []
    yearly_stats = []

    # Constants from model_v3
    EPSILON_MAX = 2.8
    HI = 0.35
    RS = 0.18
    MC = 0.125
    C_FRAC = 0.45

    for year in years:
        # Dynamic Search Window
        try:
            s_start, s_end = get_search_window(year, f_planting, f_end)
        except Exception:
            continue
            
        # Find Dynamic Dates
        sos, eos, col_used = calculate_dynamic_dates_v3(df_pcode, s_start, s_end)
        
        # Fallback Logic (simplified from model_v3 check)
        season_mode = "Dynamic"
        if sos is None or eos is None:
            season_mode = "Fixed (Fallback)"
            # Basic Fixed Logic
            start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_planting - 1)
            if f_planting < f_harvest:
                 end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
            else:
                 end_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
        else:
            start_date = sos
            end_date = eos
        
        # Extract Season Data
        mask = (df_pcode['date'] >= start_date) & (df_pcode['date'] <= end_date)
        df_season = df_pcode[mask].copy()
        
        if df_season.empty: continue
        
        # Calculate Stress
        # Ts = exp( - (Temp - T_OPT)^2 / (2 * T_SIGMA^2) )
        df_season['Ts'] = np.exp( - (df_season['Temp_C'] - T_OPT)**2 / (2 * T_SIGMA**2) )
        # Ws = 1.5 / (1.5 + VPD)
        df_season['Ws'] = 1.5 / (1.5 + df_season['VPD_kPa'])
        
        df_season['Limiting_Factor'] = np.where(df_season['Ts'] < df_season['Ws'], 'Ts (Temp)', 'Ws (Water)')
        df_season['Stress_Value'] = np.minimum(df_season['Ts'], df_season['Ws'])
        
        # Calculate Yield
        df_season['NPP_daily'] = (
            df_season['PAR_MJ'] * 
            df_season['FPAR_smooth'] * 
            EPSILON_MAX * 
            df_season['Stress_Value'] * 
            0.5
        )
        
        npp_total = df_season['NPP_daily'].sum()
        biomass = npp_total / C_FRAC
        partitioning = HI / ((1 + RS) * (1 - MC))
        yield_tha = biomass * partitioning * 0.01

        df_season['Season_Year'] = year
        daily_segments.append(df_season)
        
        yearly_stats.append({
            'Year': year,
            'Mean_Ts': df_season['Ts'].mean(),
            'Mean_Ws': df_season['Ws'].mean(),
            'Yield_t_ha': yield_tha,
            'Season_Length': (end_date - start_date).days
        })

    if not daily_segments:
        print("No valid seasons found.")
        return

    df_all_seasons = pd.concat(daily_segments)
    df_yearly_stats = pd.DataFrame(yearly_stats)
    
    # ---------------------------------------------------------
    # 4. Diagnostic Plots
    # ---------------------------------------------------------
    
    # Plot 1: Stress Histograms
    plot_stress_histogram(df_all_seasons, OUTPUT_DIR, target_pcode)
    
    # Plot 2: Limiting Factor Time Series (One representative year or all?)
    # User said "Time series". Plotting all years might be messy. 
    # Let's plot the whole series but zoomed in/interactive, or just a multi-panel.
    # We will plot the last 3 available seasons to keep it readable.
    plot_limiting_factors(daily_segments[-3:], OUTPUT_DIR, target_pcode)
    
    # Plot 3: Spatial (Temporal) Anomaly Check
    plot_anomaly_check(df_yearly_stats, OUTPUT_DIR, target_pcode)
    
    print("\nProcessing Complete.")

def plot_stress_histogram(df, output_dir, pcode):
    """
    X-Axis: Temperature (°C)
    Y-Axis (Left): Frequency (Histogram)
    Y-Axis (Right): Stress Factor Value (0 to 1).
    Curve: Overlay Ts curve.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 1. Histogram of Temperatures
    temps = df['Temp_C']
    ax1.hist(temps, bins=30, color='gray', alpha=0.5, density=False, label='Temp Freq')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Frequency (Days)', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    
    # 2. Stress Curve Overlay
    ax2 = ax1.twinx()
    
    # Range of temps for curve
    t_range = np.linspace(temps.min() - 2, temps.max() + 2, 100)
    ts_curve = np.exp( - (t_range - T_OPT)**2 / (2 * T_SIGMA**2) )
    
    ax2.plot(t_range, ts_curve, color='blue', linewidth=2, label='Ts Curve')
    ax2.set_ylabel('Stress Factor (Ts)', color='blue')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.title(f"Stress Histogram: {pcode}\n(Dist of Crop-Experienced Temps | T_opt={T_OPT}, σ={T_SIGMA})")
    
    # Add vertical line at Opt
    ax2.axvline(T_OPT, color='blue', linestyle='--', alpha=0.3, label=f'Optimum ({T_OPT}°C)')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    save_path = os.path.join(output_dir, "stress_histogram.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def plot_limiting_factors(season_list, output_dir, pcode):
    """
    Plot Ts (Blue) and Ws (Orange) on the same graph.
    Showing last few seasons concatenated or subplots.
    """
    num_seasons = len(season_list)
    fig, axes = plt.subplots(num_seasons, 1, figsize=(12, 4 * num_seasons), sharey=True)
    if num_seasons == 1: axes = [axes]
    
    for i, df in enumerate(season_list):
        ax = axes[i]
        year = df['Season_Year'].iloc[0]
        
        # Plot lines
        ax.plot(df['date'], df['Ts'], color='blue', label='$T_s$ (Temp)', linewidth=1.5)
        ax.plot(df['date'], df['Ws'], color='orange', label='$W_s$ (Water)', linewidth=1.5)
        
        # Fill area roughly to show limiting?
        # Maybe fill between min and 0?
        # Or Just shade the background based on which is lower?
        
        # Highlight regions where Ts < Ws (Temp Limited) vs Ws < Ts (Water Limited)
        # We can fill_between
        ax.fill_between(df['date'], 0, df['Ts'], where=(df['Ts'] < df['Ws']), color='blue', alpha=0.1, interpolate=True)
        ax.fill_between(df['date'], 0, df['Ws'], where=(df['Ws'] <= df['Ts']), color='orange', alpha=0.1, interpolate=True)

        ax.set_title(f"Limiting Factors - Year {year} (T_opt={T_OPT}, σ={T_SIGMA})")
        ax.set_ylabel("Stress Factor (0-1)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='lower right')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "stress_limiting_factors.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def plot_anomaly_check(df_yearly, output_dir, pcode):
    """
    Calculate the mean Ts for every year and plot it against Yield.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = df_yearly['Mean_Ts']
    y = df_yearly['Yield_t_ha']
    years = df_yearly['Year']
    
    ax.scatter(x, y, color='purple', s=100, alpha=0.7)
    
    # Annotate years
    for i, year in enumerate(years):
        ax.text(x.iloc[i], y.iloc[i], str(year), fontsize=9, ha='right', va='bottom')
        
    # Regression line / Trend
    try:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color='gray', linestyle='--', alpha=0.5, label=f'Trend')
    except:
        pass

    ax.set_xlabel("Mean Temperature Stress Factor ($T_s$)\n(Higher = Better Conditions, Closer to Optimal)")
    ax.set_ylabel("Estimated Yield (t/ha)")
    ax.set_title(f"Temporal Anomaly Check: {pcode}\n(T_opt={T_OPT}, σ={T_SIGMA})")
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, "stress_anomaly_check.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_stress_analysis("Zimbabwe")
