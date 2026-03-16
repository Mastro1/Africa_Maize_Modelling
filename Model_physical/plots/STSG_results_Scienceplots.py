import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import scienceplots

plt.style.use(['science','ieee'])

# Project paths
BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
STSG_RESULTS_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "STSG")
CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "plots", "STSG")

# Raw data paths (from STSG_smoothing.py logic)
RAW_NDVI_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
RAW_FPAR_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_location_info(country_name, pcode=None):
    """
    Retrieves administrative names and determines the best PCODE if not provided.
    Cross-references with available STSG data to ensure validity.
    Returns: (pcode, country_display, admin2_name)
    """
    # 1. Load available PCODEs from STSG results to ensure we select one that exists
    country_clean = country_name.replace(' ', '_')
    ndvi_file = os.path.join(STSG_RESULTS_DIR, f"{country_clean}_NDVI_STSG.csv")
    available_pcodes = set()
    if os.path.exists(ndvi_file):
        df_avail = pd.read_csv(ndvi_file, usecols=['PCODE'])
        available_pcodes = set(df_avail['PCODE'].unique())
    
    if not available_pcodes:
        print(f"Warning: No STSG data found for country '{country_name}'.")
        return pcode, country_name, "Unknown Admin2"

    # 2. Load administrative info
    if not os.path.exists(CROP_AREA_FILE):
        print(f"Warning: Crop area file not found at {CROP_AREA_FILE}")
        if pcode is None:
            pcode = sorted(list(available_pcodes))[0]
        return pcode, country_name, "Unknown Admin2"

    df_area = pd.read_csv(CROP_AREA_FILE)
    search_country = country_name.replace("_", " ")
    df_country = df_area[df_area['country'].str.lower() == search_country.lower()]
    
    if df_country.empty:
        print(f"Warning: No administrative data found for country '{country_name}'.")
        if pcode is None:
            pcode = sorted(list(available_pcodes))[0]
        return pcode, country_name, "Unknown Admin2"

    df_country_avail = df_country[df_country['PCODE'].isin(available_pcodes)]
    
    if df_country_avail.empty:
        print(f"Warning: None of the STSG PCodes for {country_name} found in crop area file.")
        if pcode is None:
            pcode = sorted(list(available_pcodes))[0]
        return pcode, country_name, "Unknown Admin2"

    if pcode is None:
        pcode = df_country_avail.groupby('PCODE')['crop_area_ha'].mean().idxmax()
        print(f"Selected PCODE with largest crop area (filtered by availability): {pcode}")

    try:
        row = df_country[df_country['PCODE'] == pcode].iloc[0]
        admin2 = row.get('admin_2', "Unknown Admin2")
        country_display = row.get('country', country_name)
    except IndexError:
        admin2 = "Unknown Admin2"
        country_display = country_name
    
    return pcode, country_display, admin2

def plot_stsg_results(country_name, pcode=None, year=2020):
    """
    Generates a two-panel horizontal plot comparing Original and STSG results.
    """
    # 1. Resolve location information
    pcode, country_display, admin2 = get_location_info(country_name, pcode)
    
    if pcode is None:
        print("Error: Could not determine PCODE.")
        return

    country_clean = country_name.replace(' ', '_')
    
    # 2. Define data paths
    # STSG files
    stsg_ndvi_path = os.path.join(STSG_RESULTS_DIR, f"{country_clean}_NDVI_STSG.csv")
    stsg_fpar_path = os.path.join(STSG_RESULTS_DIR, f"{country_clean}_FPAR_STSG.csv")
    
    # Raw files
    raw_ndvi_path = os.path.join(RAW_NDVI_DIR, f"{country_clean}_admin2_VI_timeseries_GADM.csv")
    raw_fpar_path = os.path.join(RAW_FPAR_DIR, f"{country_clean}_admin2_FPAR_timeseries_GLAD.csv")

    for p in [stsg_ndvi_path, stsg_fpar_path, raw_ndvi_path, raw_fpar_path]:
        if not os.path.exists(p):
            print(f"Error: Required file missing: {p}")
            return

    # 3. Load and merge data
    # NDVI
    df_stsg_ndvi = pd.read_csv(stsg_ndvi_path, parse_dates=['date'])
    df_raw_ndvi = pd.read_csv(raw_ndvi_path, parse_dates=['date'])
    df_ndvi = pd.merge(df_stsg_ndvi, df_raw_ndvi[['date', 'PCODE', 'NDVI_mean']], on=['date', 'PCODE'], how='inner')
    
    # FPAR
    df_stsg_fpar = pd.read_csv(stsg_fpar_path, parse_dates=['date'])
    df_raw_fpar = pd.read_csv(raw_fpar_path, parse_dates=['date'])
    df_fpar = pd.merge(df_stsg_fpar, df_raw_fpar[['date', 'PCODE', 'FPAR_mean']], on=['date', 'PCODE'], how='inner')

    # Filter for PCODE and Year
    df_ndvi_plot = df_ndvi[(df_ndvi['PCODE'] == pcode) & (df_ndvi['date'].dt.year == year)].sort_values('date')
    df_fpar_plot = df_fpar[(df_fpar['PCODE'] == pcode) & (df_fpar['date'].dt.year == year)].sort_values('date')

    if df_ndvi_plot.empty or df_fpar_plot.empty:
        print(f"Error: No data found for PCODE {pcode} in year {year}.")
        return

    # 4. Create the plot using scienceplots style context
    # Width 8.5 cm = 3.346 inches. Height adjusted for side-by-side subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 7*0.35))

    # Plot NDVI
    ax1.plot(df_ndvi_plot['date'].dt.dayofyear, df_ndvi_plot['NDVI_mean'], 'o', markersize=2, alpha=0.5, label='Original')
    ax1.plot(df_ndvi_plot['date'].dt.dayofyear, df_ndvi_plot['NDVI_STSG'], linewidth=1, label='STSG')
    ax1.set_title(f"NDVI ({year})")
    ax1.set_ylabel("NDVI")
    ax1.set_xlabel("Day of Year")
    ax1.legend()

    # Plot fPar
    ax2.plot(df_fpar_plot['date'].dt.dayofyear, df_fpar_plot['FPAR_mean'], 'o', markersize=2, alpha=0.5, label='Original')
    ax2.plot(df_fpar_plot['date'].dt.dayofyear, df_fpar_plot['FPAR_STSG'], linewidth=1, label='STSG')
    ax2.set_title(f"fPar ({year})")
    ax2.set_ylabel("fPar")
    ax2.set_xlabel("Day of Year")
    ax2.legend()

    # Removed super title to follow scientific style (usually handled by captions),
    # but if needed, use a very small one. Keeping it simple now.
    
    # 5. Save the output
    safe_admin2 = str(admin2).replace(" ", "_").replace("/", "_")
    output_filename = f"STSG_Comp_{country_display.replace(' ', '_')}_{safe_admin2}_{pcode}_{year}.pdf" # Changed to PNG for easier viewing if needed
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # plt.savefig(output_path, dpi=300) # Scienceplots handles bbox_inches often via style
    fig.savefig(output_path, dpi=600)
    plt.close()
    
    print(f"Plot saved successfully to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test execution for 2020
    plot_stsg_results("Uganda", year=2020)
