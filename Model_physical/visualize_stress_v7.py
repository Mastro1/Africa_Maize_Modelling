"""
Script: visualize_stress_v7.py
Purpose: High-impact scientific visualization of stress factors, critical windows, 
         and phased NPP accumulation for Model V7.
         Follows scientific_graphs_designer SKILL and uses scienceplots.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

# Try to use scienceplots for high-quality scientific figures
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except ImportError:
    print("Warning: 'scienceplots' not installed. Using default style.")
    plt.style.use('ggplot')

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Model V7 to reuse logic
try:
    from model_v7 import MaizeYieldModelV7
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

def run_stress_visualization(country, target_pcode=None, year_to_plot=None, model_instance=None):
    """
    Runs the stress visualization for V7 logic.
    
    Parameters:
    - country: str, name of the country.
    - target_pcode: str, optional.
    - year_to_plot: int, optional.
    - model_instance: MaizeYieldModelV7, optional. If provided, uses its parameters.
    """
    print(f"\n--- [V7 Scientific Visualization] Stress Factors for {country} ---")
    
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
    GADM_DATA_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
    CALENDAR_FILE = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "V7_stress")
    
    VI_FILE = os.path.join(GADM_DATA_DIR, f"{country.replace(' ', '_')}_admin2_VI_timeseries_GADM.csv")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize or Use Model Instance
    if model_instance is None:
        model = MaizeYieldModelV7(
            data_dir=DATA_DIR,
            gadm_data_dir=GADM_DATA_DIR,
            fpar_file=f"{country.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv",
            era5_new_file=f"{country.replace(' ', '_')}_admin2_new_ERA5_timeseries.csv",
            era5_gadm_file=f"{country.replace(' ', '_')}_admin2_ERA5_timeseries_GADM.csv",
            calendar_file=CALENDAR_FILE,
            output_dir=os.path.join(BASE_DIR, "Model_physical", "Results", "V7_model"),
            country=country,
            vi_file=VI_FILE,
            crop_area_file=CROP_AREA_FILE
        )
        model.load_optimized_parameters()
    else:
        model = model_instance

    # Extract parameters for easy access
    P = {
        'T_OPT': model.T_OPT,
        'T_SIGMA': model.T_SIGMA,
        'VEGETATIVE_WEIGHT': model.VEGETATIVE_WEIGHT,
        'REPRODUCTIVE_WEIGHT': model.REPRODUCTIVE_WEIGHT,
        'CRITICAL_WINDOW_DAYS': model.CRITICAL_WINDOW_DAYS,
        'HEAT_THRESHOLD': model.HEAT_THRESHOLD,
        'HEAT_PENALTY': model.HEAT_PENALTY,
        'DROUGHT_WS_THRESHOLD': model.DROUGHT_WS_THRESHOLD,
        'DROUGHT_PENALTY': model.DROUGHT_PENALTY,
        'EPSILON_MAX': model.EPSILON_MAX,
        'HI': model.HI,
        'RS': model.RS,
        'MC': model.MC,
        'C_FRAC': model.C_FRAC
    }

    # 2. Select Target PCODE
    if target_pcode is None:
        target_pcode = model.get_max_area_pcode(country)
        print("  - No PCODE provided. Selecting max area PCODE.")
    
    if not target_pcode:
        print("  - Could not determine PCODE. Exiting.")
        return
    
    print(f"  - Target PCODE: {target_pcode}")
    
    # 3. Load & Prepare Data
    df_daily = model.load_and_merge_data()
    if target_pcode not in df_daily['PCODE'].unique():
        print(f"  - Error: PCODE {target_pcode} not in data.")
        return

    df_daily = model.preprocess_fpar(df_daily)
    df_daily = model.calculate_biophysical_variables(df_daily)
    df_calendar = model.load_crop_calendar()
    pcode_cal_dict = model.prepare_calendar_data()
    
    if target_pcode not in pcode_cal_dict:
        print(f"  - Error: No calendar for {target_pcode}.")
        return

    # 4. Filter for Specific Season
    df_pcode = df_daily[df_daily['PCODE'] == target_pcode].copy()
    df_pcode['Year'] = df_pcode['date'].dt.year
    df_pcode = df_pcode.set_index('date', drop=False)
    
    years_available = sorted(df_pcode['Year'].unique(), reverse=True)
    if year_to_plot and year_to_plot in years_available:
        years_to_scan = [year_to_plot]
    else:
        years_to_scan = years_available

    target_season_data = None
    target_info = None

    for yr in years_to_scan:
        if yr not in pcode_cal_dict[target_pcode]:
            # Fallback to fixed (simplified for viz)
            cal_row = df_calendar[df_calendar['PCODE'] == target_pcode]
            if cal_row.empty: continue
            f_p = int(cal_row.iloc[0]['Maize_1_planting'])
            f_h = int(cal_row.iloc[0]['Maize_1_harvest'])
            sos = pd.Timestamp(year=yr, month=1, day=1) + pd.Timedelta(days=f_p - 1)
            eos = sos + pd.Timedelta(days=(f_h - f_p) % 365)
            silking = None
        else:
            # Take Season 1 for visualization
            s1 = pcode_cal_dict[target_pcode][yr].get(1)
            if not s1: continue
            sos, silking, eos = s1['SOS'], s1['Silking'], s1['EOS']
        
        # Extract season slice
        mask = (df_pcode['date'] >= sos) & (df_pcode['date'] <= eos)
        df_s = df_pcode[mask].copy()
        if df_s.empty or len(df_s) < (eos - sos).days * 0.9: continue

        # --- Calculate Stresses (V7 Logic) ---
        df_s['Ts'] = np.exp(-(df_s['Temp_C'] - P['T_OPT'])**2 / (2 * P['T_SIGMA']**2))
        df_s['f_VPD'] = 1.5 / (1.5 + df_s['VPD_kPa'])
        df_s['Ws'] = df_s['f_VPD']
        min_stress = np.minimum(df_s['Ts'], df_s['Ws'])

        # NPP
        df_s['NPP_daily'] = df_s['PAR_MJ'] * df_s['FPAR_smooth'] * P['EPSILON_MAX'] * min_stress * 0.5
        
        # Phases
        df_s['Weight'] = (P['VEGETATIVE_WEIGHT'] + P['REPRODUCTIVE_WEIGHT']) / 2.0
        df_s['Phase'] = 'Unified'
        if silking and sos <= silking <= eos:
            df_s.loc[df_s['date'] <= silking, 'Weight'] = P['VEGETATIVE_WEIGHT']
            df_s.loc[df_s['date'] <= silking, 'Phase'] = 'Vegetative'
            df_s.loc[df_s['date'] > silking, 'Weight'] = P['REPRODUCTIVE_WEIGHT']
            df_s.loc[df_s['date'] > silking, 'Phase'] = 'Reproductive'
        
        df_s['NPP_W'] = df_s['NPP_daily'] * df_s['Weight']
        df_s['NPP_Cum'] = df_s['NPP_W'].cumsum()

        # Critical Window
        hp, dp = 1.0, 1.0
        cw_dates = None
        if silking and sos <= silking <= eos:
            cw_s = silking - pd.Timedelta(days=P['CRITICAL_WINDOW_DAYS'])
            cw_e = silking + pd.Timedelta(days=P['CRITICAL_WINDOW_DAYS'])
            cw_dates = (cw_s, cw_e)
            df_cw = df_s[(df_s['date'] >= cw_s) & (df_s['date'] <= cw_e)]
            if not df_cw.empty:
                hp = P['HEAT_PENALTY'] if df_cw['Temp_C'].mean() > P['HEAT_THRESHOLD'] else 1.0
                dp = P['DROUGHT_PENALTY'] if df_cw['Ws'].mean() < P['DROUGHT_WS_THRESHOLD'] else 1.0

        target_season_data = df_s
        target_info = {
            'Year': yr, 'SOS': sos, 'EOS': eos, 'Silking': silking, 'CW': cw_dates,
            'HP': hp, 'DP': dp, 'Yield': (df_s['NPP_W'].sum() / P['C_FRAC']) * (P['HI'] / ((1 + P['RS']) * (1 - P['MC']))) * hp * dp * 0.01
        }
        break

    if target_season_data is None:
        print("  - No valid season data found.")
        return

    # 5. Plotting
    plot_scientific_stress(target_season_data, target_info, target_pcode, P, OUTPUT_DIR)

def plot_scientific_stress(df, info, pcode, P, output_dir):
    # Dimensions: IEEE Double Column (7.0 in width)
    width = 7.0
    height = width * 1.2 # Taller for 3-stack
    
    fig, axes = plt.subplots(3, 1, figsize=(width, height), dpi=300)
    
    # --- [1] Stress Factors ---
    ax = axes[0]
    ax.plot(df['date'], df['Ts'], label=r'$T_s$ (Temp Stress)', color='C0', lw=1.2)
    ax.plot(df['date'], df['Ws'], label=r'$W_s$ (Water Stress)', color='C1', lw=1.2)
    
    if info['CW']:
        ax.axvspan(info['CW'][0], info['CW'][1], color='red', alpha=0.1, label='Crit. Window')
        if info['Silking']:
            ax.axvline(info['Silking'], color='red', ls='--', alpha=0.5, label='Silking')
            
    ax.set_ylabel('Stress Factor (0--1)')
    ax.set_title(f'Biophysical Stress Factors: {pcode} ({info["Year"]})')
    ax.legend(loc='lower right', frameon=True, fontsize=8)
    
    # Annotate Penalties
    pen_msgs = []
    if info['HP'] < 1.0: pen_msgs.append(f"Heat Penalty Applied ({info['HP']})")
    if info['DP'] < 1.0: pen_msgs.append(f"Drought Penalty Applied ({info['DP']})")
    if pen_msgs:
        ax.text(0.02, 0.05, "\n".join(pen_msgs), transform=ax.transAxes, color='red', fontsize=7, fontweight='bold')

    # --- [2] NPP Accumulation ---
    ax = axes[1]
    # Use C2 (Green) for Veg, C4 (Purple/Gold-ish) for Rep
    colors = np.where(df['Phase'] == 'Vegetative', 'C2', 'C8')
    ax.bar(df['date'], df['NPP_W'], color=colors, alpha=0.7, label='Daily NPP (Weighted)')
    
    axb = ax.twinx()
    axb.plot(df['date'], df['NPP_Cum'], color='black', lw=1.5, label='Cumulative NPP')
    axb.set_ylabel(r'Cumulative NPP ($\mathrm{gC/m^2}$)')
    
    # Custom patches for phases
    veg_p = mpatches.Patch(color='C2', alpha=0.7, label=f'Veg. Phase ($w={P["VEGETATIVE_WEIGHT"]}$)')
    rep_p = mpatches.Patch(color='C8', alpha=0.7, label=f'Rep. Phase ($w={P["REPRODUCTIVE_WEIGHT"]}$)')
    
    handles_b, labels_b = axb.get_legend_handles_labels()
    ax.legend(handles=[veg_p, rep_p] + handles_b, loc='upper left', frameon=True, fontsize=8)
    ax.set_ylabel(r'Weighted Daily NPP ($\mathrm{gC/m^2/d}$)')
    ax.set_title(f'Phased NPP Accumulation (Yield: {info["Yield"]:.2f} t/ha)')

    # --- [3] Temperature & Stress Curve ---
    ax = axes[2]
    temps = df['Temp_C']
    ax.hist(temps, bins=25, color='C0', alpha=0.3, label='Season $\mathrm{T_{avg}}$')
    
    if info['CW']:
        mask_cw = (df['date'] >= info['CW'][0]) & (df['date'] <= info['CW'][1])
        ax.hist(df.loc[mask_cw, 'Temp_C'], bins=15, color='red', alpha=0.4, label='Crit. Window $\mathrm{T_{avg}}$')
        
    ax.axvline(P['T_OPT'], color='C2', ls='--', lw=1.5, label=f'$T_{{opt}}$ ({P["T_OPT"]}°C)')
    ax.axvline(P['HEAT_THRESHOLD'], color='red', ls=':', lw=1.5, label=f'Heat Thresh ({P["HEAT_THRESHOLD"]}°C)')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Frequency (Days)')
    ax.set_title('Temperature Distribution \\& Gaussian Stress')
    
    # Overlay Stress Curve
    axb = ax.twinx()
    t_axis = np.linspace(temps.min()-2, temps.max()+2, 100)
    ts_vals = np.exp(-(t_axis - P['T_OPT'])**2 / (2 * P['T_SIGMA']**2))
    axb.plot(t_axis, ts_vals, color='C0', lw=1.5, ls='-', label='$T_s$ Curve')
    axb.set_ylabel('Stress Factor ($T_s$)', color='C0')
    axb.set_ylim(0, 1.05)
    
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = axb.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc='upper right', frameon=True, fontsize=8)

    plt.tight_layout()
    
    # Save as PDF for high quality, and PNG for preview
    pdf_path = os.path.join(output_dir, f"{pcode}_{info['Year']}_scientific_stress.pdf")
    png_path = os.path.join(output_dir, f"{pcode}_{info['Year']}_scientific_stress.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    print(f"  - Figures saved: {pdf_path}")
    print(f"  - Preview: {png_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, default="Kenya")
    parser.add_argument("--pcode", type=str, default=None)
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    
    run_stress_visualization(args.country, target_pcode=args.pcode, year_to_plot=args.year)
