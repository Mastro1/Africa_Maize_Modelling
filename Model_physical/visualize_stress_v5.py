"""
Script: visualize_stress_v5.py
Purpose: Detailed visualization of stress factors, critical windows, and phased NPP accumulation
         for Model V5. Focuses on a single PCODE (user-specified or max crop area).
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Model V5 to reuse logic and constants
try:
    from model_v5 import (MaizeYieldModelV5, 
                          T_OPT, T_SIGMA, 
                          VEGETATIVE_WEIGHT, REPRODUCTIVE_WEIGHT,
                          CRITICAL_WINDOW_DAYS, HEAT_THRESHOLD, DROUGHT_WS_THRESHOLD,
                          HEAT_PENALTY, DROUGHT_PENALTY,
                          EPSILON_MAX, HI, RS, MC, C_FRAC)
    from verify_dynamic_calendar_v3_2 import calculate_dates_v3_2
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Improve plot resolution
plt.rcParams['figure.dpi'] = 150

def run_stress_visualization(country, target_pcode=None, year_to_plot=None):
    """
    Runs the stress visualization for a specific country and PCODE.
    
    Parameters:
    - country: str, name of the country.
    - target_pcode: str, optional. If None, selects PCODE with max crop area.
    - year_to_plot: int, optional. If None, plots the last available year with valid data.
    """
    print(f"\n--- Starting V5 Stress Factor Visualization for {country} ---")
    
    # ---------------------------------------------------------
    # 1. Initialize Model V5 (to load data)
    # ---------------------------------------------------------
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
    GADM_DATA_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
    CALENDAR_FILE = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "V5_stress")
    
    # File names (assuming standard naming convention)
    VI_FILE = os.path.join(GADM_DATA_DIR, f"{country.replace(' ', '_')}_admin2_VI_timeseries_GADM.csv")
    
    model = MaizeYieldModelV5(
        data_dir=DATA_DIR,
        gadm_data_dir=GADM_DATA_DIR,
        fpar_file=f"{country.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv",
        era5_new_file=f"{country.replace(' ', '_')}_admin2_new_ERA5_timeseries.csv",
        era5_gadm_file=f"{country.replace(' ', '_')}_admin2_ERA5_timeseries_GADM.csv",
        calendar_file=CALENDAR_FILE,
        output_dir=OUTPUT_DIR,
        country=country,
        vi_file=VI_FILE,
        crop_area_file=CROP_AREA_FILE
    )

    # ---------------------------------------------------------
    # 2. Select Target PCODE
    # ---------------------------------------------------------
    if target_pcode is None:
        target_pcode = model.get_max_area_pcode(country)
        print("No PCODE provided. Auto-selected max area PCODE.")
    
    if not target_pcode:
         # Fallback if crop area file missing/empty
        print("Could not determine max area PCODE. Finding first available from data...")
        # We need to load data first to find a PCODE if crop area fails, 
        # but let's assume we can load data now.
    
    print(f"Target PCODE: {target_pcode}")
    
    # ---------------------------------------------------------
    # 3. Load & Prepare Data
    # ---------------------------------------------------------
    # Load biophysical data
    df_daily = model.load_and_merge_data()
    
    # Check if target PCODE exists in data
    if target_pcode not in df_daily['PCODE'].unique():
        print(f"Error: PCODE {target_pcode} not found in biophysical data.")
        return

    df_daily = model.preprocess_fpar(df_daily)
    df_daily = model.calculate_biophysical_variables(df_daily)
    
    # Load calendar
    df_calendar = model.load_crop_calendar()
    
    # Prepare dynamic calendar data (just for this pcode if possible, but the function does all. 
    # Optimization: The V5 `prepare_calendar_data` prepares ALL PCODES. 
    # For visualization speed, we might want to filter, but `prepare_calendar_data` doesn't take args.
    # We'll run it as is; it shouldn't be too slow for one country.)
    pcode_cal_dict = model.prepare_calendar_data()
    
    if target_pcode not in pcode_cal_dict:
        print(f"Error: PCODE {target_pcode} not found in calendar data.")
        return

    # ---------------------------------------------------------
    # 4. Re-calculate V5 Daily Data for the Target PCODE
    # ---------------------------------------------------------
    print(f"Calculating daily diagnostics for {target_pcode}...")
    
    df_pcode = df_daily[df_daily['PCODE'] == target_pcode].copy()
    df_pcode['Year'] = df_pcode['date'].dt.year
    df_pcode = df_pcode.set_index('date', drop=False)
    
    cal_row = df_calendar[df_calendar['PCODE'] == target_pcode]
    if cal_row.empty:
        print("No static calendar data found.")
        return

    f_planting = cal_row.iloc[0]['Maize_1_planting']
    f_harvest  = cal_row.iloc[0]['Maize_1_harvest']
    f_end      = cal_row.iloc[0]['Maize_1_endofseaso']
    
    df_pcode_cal = pcode_cal_dict[target_pcode]
    years = df_pcode['Year'].unique()
    
    # If year_to_plot is not specified, we'll collect all seasons and then pick the last valid one 
    # or iterate to find a good one.
    
    # Container for the season data we want to plot
    target_season_data = None
    target_season_info = None

    # We iterate backwards to find the latest valid season if year not set
    years_sorted = sorted(years, reverse=True)
    if year_to_plot:
        if year_to_plot in years:
            years_sorted = [year_to_plot]
        else:
            print(f"Requested year {year_to_plot} not in data.")
            return

    for year in years_sorted:
        seasons_config = [{'index': 1, 'planting': int(f_planting), 'endofseason': int(f_end)}]
        
        try:
            season_results = calculate_dates_v3_2(df_pcode_cal, year, seasons_config)
        except Exception:
            continue
            
        if 1 not in season_results:
            continue
            
        sos_date, silking_date, eos_date = season_results[1]['dates']
        
        # Determine dates
        if sos_date is None or eos_date is None:
            # Fallback
            start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_planting - 1)
            if f_planting < f_harvest:
                end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
            else:
                end_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
            silking_date = None # No silking in fallback -> No Phases
        else:
            start_date = sos_date
            end_date = eos_date
            
        # Extract Season
        mask = (df_pcode['date'] >= start_date) & (df_pcode['date'] <= end_date)
        df_season = df_pcode[mask].copy()
        
        if df_season.empty: continue
        if len(df_season) < (end_date - start_date).days * 0.9: continue # Check integrity

        # Calculate Logic
        df_season['Ts'] = np.exp(-(df_season['Temp_C'] - T_OPT)**2 / (2 * T_SIGMA**2))
        df_season['Ws'] = 1.5 / (1.5 + df_season['VPD_kPa'])
        df_season['Stress_Value'] = np.minimum(df_season['Ts'], df_season['Ws'])
        
        # Phased NPP
        # 1. Base NPP
        df_season['NPP_Raw'] = (
            df_season['PAR_MJ'] *
            df_season['FPAR_smooth'] *
            EPSILON_MAX *
            df_season['Stress_Value'] *
            0.5
        )
        
        # 2. Phases
        df_season['Phase'] = 'Unknown'
        df_season['Weight'] = 1.0
        
        if silking_date and start_date <= silking_date <= end_date:
            mask_veg = df_season['date'] <= silking_date
            mask_rep = df_season['date'] > silking_date
            
            df_season.loc[mask_veg, 'Phase'] = 'Vegetative'
            df_season.loc[mask_veg, 'Weight'] = VEGETATIVE_WEIGHT
            
            df_season.loc[mask_rep, 'Phase'] = 'Reproductive'
            df_season.loc[mask_rep, 'Weight'] = REPRODUCTIVE_WEIGHT
        else:
             df_season['Phase'] = 'Unified'
             avg_weight = (VEGETATIVE_WEIGHT + REPRODUCTIVE_WEIGHT) / 2.0
             df_season['Weight'] = avg_weight

        df_season['NPP_Weighted'] = df_season['NPP_Raw'] * df_season['Weight']
        df_season['NPP_Cum'] = df_season['NPP_Weighted'].cumsum()

        # Critical Window Analysis
        heat_penalty = False
        drought_penalty = False
        cw_dates = None
        
        if silking_date and start_date <= silking_date <= end_date:
            cw_start = silking_date - pd.Timedelta(days=CRITICAL_WINDOW_DAYS)
            cw_end   = silking_date + pd.Timedelta(days=CRITICAL_WINDOW_DAYS)
            cw_dates = (cw_start, cw_end)
            
            mask_cw = (df_season['date'] >= cw_start) & (df_season['date'] <= cw_end)
            df_cw = df_season[mask_cw]
            
            if not df_cw.empty:
                if df_cw['Temp_C'].mean() > HEAT_THRESHOLD:
                    heat_penalty = True
                if df_cw['Ws'].mean() < DROUGHT_WS_THRESHOLD:
                    drought_penalty = True

        # Store this season and break (unless we want specific year)
        target_season_data = df_season
        target_season_info = {
            'Year': year, 
            'Start': start_date, 'End': end_date, 'Silking': silking_date,
            'CW': cw_dates,
            'Heat_Penalty': heat_penalty, 'Drought_Penalty': drought_penalty,
            'Yield_Est': (df_season['NPP_Weighted'].sum() / C_FRAC) * (HI / ((1+RS)*(1-MC))) * 0.01
        }
        
        # Apply penalties to info for display
        if heat_penalty: target_season_info['Yield_Est'] *= HEAT_PENALTY
        if drought_penalty: target_season_info['Yield_Est'] *= DROUGHT_PENALTY

        break # Found the latest valid season, stop

    if target_season_data is None:
        print("No valid season found to visualize.")
        return

    # ---------------------------------------------------------
    # 5. Generate Plots
    # ---------------------------------------------------------
    plot_visualization(target_season_data, target_season_info, target_pcode, OUTPUT_DIR)


def plot_visualization(df, info, pcode, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=False)
    
    year = info['Year']
    silking = info['Silking']
    cw = info['CW']
    
    # --- Plot 1: Stress Factors & Critical Window ---
    ax1 = axes[0]
    ax1.plot(df['date'], df['Ts'], color='blue', label='Temp Stress (Ts)', linewidth=1.5, alpha=0.8)
    ax1.plot(df['date'], df['Ws'], color='orange', label='Water Stress (Ws)', linewidth=1.5, alpha=0.8)
    
    # Highlight Critical Window
    if cw:
        ax1.axvspan(cw[0], cw[1], color='red', alpha=0.1, label='Critical Window')
        ax1.axvline(silking, color='red', linestyle='--', alpha=0.5, label='Silking')
        
    ax1.set_ylabel("Stress Factor (0-1)")
    ax1.set_title(f"Stress Factors & Critical Window - {pcode} ({year})")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Annotate Penalties
    penalty_text = []
    if info['Heat_Penalty']: penalty_text.append("HEAT PENALTY TRIGGERED!")
    if info['Drought_Penalty']: penalty_text.append("DROUGHT PENALTY TRIGGERED!")
    
    if penalty_text:
        ax1.text(0.02, 0.05, "\n".join(penalty_text), transform=ax1.transAxes, 
                 color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    # --- Plot 2: Phased NPP Accumulation ---
    ax2 = axes[1]
    
    # Color bars by Phase
    colors = np.where(df['Phase'] == 'Vegetative', 'green', 
                      np.where(df['Phase'] == 'Reproductive', 'gold', 'gray'))
    
    ax2.bar(df['date'], df['NPP_Weighted'], color=colors, alpha=0.6, label='Daily NPP (Weighted)')
    
    # Plot Cumulative on secondary axis
    ax2b = ax2.twinx()
    ax2b.plot(df['date'], df['NPP_Cum'], color='black', linewidth=2, label='Cumulative NPP')
    ax2b.set_ylabel("Cumulative NPP (gC/m²)")
    
    # Legend for phases
    veg_patch = mpatches.Patch(color='green', alpha=0.6, label=f'Veg Phase (w={VEGETATIVE_WEIGHT})')
    rep_patch = mpatches.Patch(color='gold', alpha=0.6, label=f'Rep Phase (w={REPRODUCTIVE_WEIGHT})')
    
    handles, labels = ax2.get_legend_handles_labels() # For bars if labeled? No, custom logic needed
    handles2, labels2 = ax2b.get_legend_handles_labels()
    
    ax2.legend(handles=[veg_patch, rep_patch] + handles2, loc='upper left')
    ax2.set_ylabel("Daily NPP (gC/m²)")
    ax2.set_title(f"Phased NPP Accumulation (Final Yield: {info['Yield_Est']:.2f} t/ha)")
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Temperature Histogram ---
    ax3 = axes[2]
    
    # Separate histogram for Critical Window vs Rest of season?
    # Or just general distribution
    temps = df['Temp_C']
    ax3.hist(temps, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Season Temps')
    
    # If CW exists, overplot
    if cw:
        mask_cw = (df['date'] >= cw[0]) & (df['date'] <= cw[1])
        temps_cw = df.loc[mask_cw, 'Temp_C']
        if not temps_cw.empty:
            ax3.hist(temps_cw, bins=20, color='red', alpha=0.5, label='Crit. Window Temps')
            
    ax3.axvline(T_OPT, color='green', linestyle='--', linewidth=2, label=f'T_opt ({T_OPT}°C)')
    ax3.axvline(HEAT_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Heat Thresh ({HEAT_THRESHOLD}°C)')
    
    ax3.set_xlabel("Temperature (°C)")
    ax3.set_ylabel("Frequency (Days)")
    ax3.set_title("Temperature Distribution & Stress Curve")
    ax3.grid(True, alpha=0.3)

    # Secondary Axis for Stress Curve
    ax3b = ax3.twinx()
    t_min = df['Temp_C'].min() - 2
    t_max = df['Temp_C'].max() + 2
    t_range = np.linspace(t_min, t_max, 200)
    ts_curve = np.exp(-(t_range - T_OPT)**2 / (2 * T_SIGMA**2))
    
    ax3b.plot(t_range, ts_curve, color='blue', linewidth=2, label='Ts Curve')
    ax3b.set_ylabel("Stress Factor (Ts)", color='blue')
    ax3b.set_ylim(0, 1.05)
    ax3b.tick_params(axis='y', labelcolor='blue')
    
    # Combine legends
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{pcode}_{year}_stress.png")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Test run
    # run_stress_visualization("Zimbabwe")
    pass
