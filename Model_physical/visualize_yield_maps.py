
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import warnings
from statsmodels.nonparametric.kernel_regression import KernelReg

warnings.filterwarnings("ignore")

def smooth_data_kernel_regression(arr: np.array, years: np.array) -> np.ndarray:
    """Smooth the data using kernel regression with a bandwidth of 3 (per verify_hsa.py)."""
    try:
        kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[3])
        return kr.fit()[0]
    except Exception as e:
        print(f"Kernel regression failed: {e}")
        return arr # Fallback

def visualize_yield_maps(country="Angola", version='v5', target_years=[2005, 2010, 2015, 2020]):
    """
    Generates a 4-panel map of detrended yield anomalies.
    """
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    RESULTS_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", f"{version.upper()}_model")
    SHP_PATH = os.path.join(BASE_DIR, "GADM", "gadm41_AFR_shp", "gadm41_AFR_final.shp")
    YIELD_FILE = os.path.join(RESULTS_DIR, f"maize_yield_estimates_{version}.csv")
    GLOBAL_FILE = os.path.join(RESULTS_DIR, f"Global_Maize_Yield_{version.upper()}.csv")
    COUNTRY_FILE = os.path.join(RESULTS_DIR, f"maize_yield_estimates_{version.upper()}_{country.replace(' ', '_')}.csv")

    print(f"\n--- Generating Yield Maps for {country} ({version}) ---")
    
    # Filter Shapefile
    print(f"Loading shapefile for {country}...")
    try:
        gdf = gpd.read_file(SHP_PATH)
        gdf_country = gdf[gdf['ADMIN0'] == country].copy()
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return

    if gdf_country.empty:
        print(f"Error: Country '{country}' not found in shapefile (ADMIN0 column).")
        return

    # Derive ISO prefix from shapefile (GID_0 if available, else first 3 of PCODE in gdf)
    # The shapefile has FNID like 'AGO.1.2_1'. Let's use the GID_0 or just the string before first dot.
    iso_prefix = gdf_country['FNID'].iloc[0].split('.')[0]
    print(f"Identified ISO prefix: {iso_prefix}")

    # 1. Load Yield Data
    if os.path.exists(GLOBAL_FILE):
        print(f"Loading results from global file: {GLOBAL_FILE}")
        df_raw = pd.read_csv(GLOBAL_FILE)
        df_raw = df_raw[df_raw['Country'].str.strip().str.lower() == country.strip().lower()]
    elif os.path.exists(COUNTRY_FILE):
        print(f"Loading results from country file: {COUNTRY_FILE}")
        df_raw = pd.read_csv(COUNTRY_FILE)
    elif os.path.exists(YIELD_FILE):
        print(f"Loading results from: {YIELD_FILE}")
        df_raw = pd.read_csv(YIELD_FILE)
    else:
        print(f"Error: Yield file not found in {RESULTS_DIR}")
        return
    
    # Filter for seasons if V6 (Default to Season 1 for simplicity in anomaly maps)
    if 'Season' in df_raw.columns:
        print("Filtering for Season 1 for Anomaly Maps...")
        df_raw = df_raw[df_raw['Season'] == 1]

    df = df_raw
    
    if df.empty:
        print(f"Warning: No yield data found for ISO prefix {iso_prefix}")
        return

    # Detrend per PCODE
    print("Detrending yield timeseries per PCODE...")
    df['Yield_Anomaly'] = np.nan
    pcodes = df['PCODE'].unique()
    
    for pcode in pcodes:
        sub = df[df['PCODE'] == pcode].sort_values('Year')
        if len(sub) < 5:
            continue
            
        years = sub['Year'].values
        y_vals = sub['Yield_Estimated_t_ha'].values
        
        trend = smooth_data_kernel_regression(y_vals, years)
        trend_safe = np.where(trend == 0, 1, trend)
        anomaly = (y_vals - trend) / trend_safe
        df.loc[sub.index, 'Yield_Anomaly'] = anomaly

    # 4. Prepare Plotting Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    # Common color range for consistency (e.g., -20% to +20%)
    vmin, vmax = -0.2, 0.2
    cmap = 'RdYlGn'

    for i, year in enumerate(target_years):
        ax = axes[i]
        df_year = df[df['Year'] == year]
        
        # Join to GDF
        merged = gdf_country.merge(df_year[['PCODE', 'Yield_Anomaly']], left_on='FNID', right_on='PCODE', how='left')
        
        # Plot background (gray)
        gdf_country.plot(ax=ax, color='#eeeeee', edgecolor='white', linewidth=0.3)
        
        # Plot Data
        if 'Yield_Anomaly' in merged.columns and not merged['Yield_Anomaly'].dropna().empty:
            merged.plot(column='Yield_Anomaly', ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
        
        ax.set_title(f"{year} Detrended Yield Anomaly", fontsize=14)
        ax.set_axis_off()

    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.02]) # [left, bottom, width, height]
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Yield Anomaly (Relative to Trend)', fontsize=12)

    plt.suptitle(f"Maize Yield Anomalies - {country} (Model {version.upper()})", fontsize=18, y=0.95)
    
    output_path = os.path.join(RESULTS_DIR, f"{country}_yield_anomaly_maps.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Maps saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Yield Maps")
    parser.add_argument("--country", type=str, default="Angola")
    parser.add_argument("--version", type=str, default="v5")
    args = parser.parse_args()
    
    visualize_yield_maps(country=args.country, version=args.version)
