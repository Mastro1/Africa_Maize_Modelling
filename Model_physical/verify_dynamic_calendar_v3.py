import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

def calculate_dynamic_dates_v3(df_pcode_year, search_start, search_end):
    """
    Calculates dynamic SOS (20% threshold) and EOS (50% drop threshold) 
    from a smoothed timeseries.
    """
    subset = df_pcode_year[(df_pcode_year["date"] >= search_start) & (df_pcode_year["date"] <= search_end)]
    if subset.empty:
        return None, None, None
    
    # Use smoothed STSG data if available, else fall back to mean
    col = "NDVI_STSG" if "NDVI_STSG" in subset.columns else "NDVI_mean"
    
    v_min = subset[col].min()
    v_max = subset[col].max()
    amplitude = v_max - v_min
    
    # Rule Suggestion: 20% for SOS, 50% Drop for EOS (Maturity)
    threshold_sos = v_min + 0.20 * amplitude
    threshold_eos = v_min + 0.50 * amplitude
    
    peak_idx = subset[col].idxmax()
    
    # SOS Search (Backwards from Peak)
    sos_date = None
    pre_peak = subset.loc[:peak_idx]
    for date, row in pre_peak[::-1].iterrows():
        if row[col] < threshold_sos:
            sos_date = pd.to_datetime(row["date"]) + pd.Timedelta(days=1)
            break
    if sos_date is None:
        sos_date = search_start
            
    # EOS Search (Forwards from Peak)
    eos_date = None
    post_peak = subset.loc[peak_idx:]
    for date, row in post_peak.iterrows():
        if row[col] < threshold_eos:
            eos_date = pd.to_datetime(row["date"]) - pd.Timedelta(days=1)
            break
    if eos_date is None:
        eos_date = search_end
            
    return sos_date, eos_date, col

def get_search_window(year, f_plant, f_end):
    """
    Calculates the search window (start_date, end_date) based on fixed calendar dates.
    Window is +/- 30 days around the fixed planting/end dates.
    """
    start_year = year
    if f_plant > f_end:
        start_year = year - 1
        s_start = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=f_plant - 1 - 30)
        s_end = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_end - 1 + 30)
    else:
        s_start = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_plant - 1 - 30)
        s_end = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_end - 1 + 30)
    return s_start, s_end

def verify_and_plot_v3(country, pcode_selected=None, years_to_plot=[2005, 2010, 2015, 2020]):
    """
    Processes a country, calculates averages, and plots logic.
    """
    # 1. Load Data
    data_path = f"Model_physical/Results/{country}_admin2_STSG_smoothed.csv"
    if not os.path.exists(data_path):
        print(f"Error: Smoothed data not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Load Calendar
    calendar_path = "GADM/crop_calendar/maize_crop_calendar_extraction.csv"
    df_cal = pd.read_csv(calendar_path)
    if 'FNID' in df_cal.columns:
        df_cal = df_cal.rename(columns={'FNID': 'PCODE'})
        
    # 3. Calculate Average Season Length across ALL PCODEs and YEARS
    print(f"Calculating country-wide averages for {country}...")
    all_lengths = []
    
    # Combine data with calendar to iterate efficiently
    unique_pcodes = df["PCODE"].unique()
    
    for pcode in unique_pcodes:
        cal_row = df_cal[df_cal['PCODE'] == pcode]
        if cal_row.empty: continue
        
        f_plant = cal_row.iloc[0]['Maize_1_planting']
        f_end = cal_row.iloc[0]['Maize_1_endofseaso']
        
        df_pcode = df[df["PCODE"] == pcode].set_index("date", drop=False)
        years = df_pcode["date"].dt.year.unique()
        
        for year in years:
            # Search Window
            s_start, s_end = get_search_window(year, f_plant, f_end)
            
            sos, eos, _ = calculate_dynamic_dates_v3(df_pcode, s_start, s_end)
            if sos and eos:
                all_lengths.append((eos - sos).days)
                
    if all_lengths:
        avg_len = np.mean(all_lengths)
        print(f"\n>>> AVERAGE SEASON LENGTH ({country}): {avg_len:.1f} days (N={len(all_lengths)} seasons)")
    else:
        print("No valid seasons detected for average calculation.")

    # 4. Selective Plotting
    if pcode_selected is None:
        pcode_selected = unique_pcodes[0]
        
    cal_row = df_cal[df_cal['PCODE'] == pcode_selected]
    f_plant = cal_row.iloc[0]['Maize_1_planting']
    f_veg = cal_row.iloc[0]['Maize_1_vegetative']
    f_harv = cal_row.iloc[0]['Maize_1_harvest']
    f_end = cal_row.iloc[0]['Maize_1_endofseaso']
    
    dfp = df[df["PCODE"] == pcode_selected].sort_values("date").set_index("date", drop=False)
    
    fig, axes = plt.subplots(len(years_to_plot), 1, figsize=(12, 4 * len(years_to_plot)))
    if len(years_to_plot) == 1: axes = [axes]
    
    for i, year in enumerate(years_to_plot):
        ax = axes[i]
        
        # Search Window Logic
        s_start, s_end = get_search_window(year, f_plant, f_end)
        start_year = s_start.year # Approximate for fixed line plotting logic below
            
        p_start = s_start - pd.Timedelta(days=60)
        p_end = s_end + pd.Timedelta(days=60)
        
        sub = dfp.loc[p_start:p_end]
        if sub.empty: continue
        
        sos, eos, col = calculate_dynamic_dates_v3(dfp, s_start, s_end)
        
        # Plot
        ax.plot(sub["date"], sub["NDVI_mean"], 'o-', color='black', alpha=0.3, label="Raw NDVI", markersize=3)
        ax.plot(sub["date"], sub[col], '-', color='green', label=f"Smoothed ({col})", linewidth=2)
        ax.axvspan(s_start, s_end, color='yellow', alpha=0.05, label="Search Window")
        
        # Fixed Dates
        def doy_to_date(y, doy): return pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=doy - 1)
        ax.axvline(doy_to_date(start_year, f_plant), color='cyan', linestyle='--', alpha=0.5, label="Fixed Plant")
        ax.axvline(doy_to_date(start_year if f_harv > f_plant else start_year+1, f_harv), color='orange', linestyle='--', alpha=0.5, label="Fixed Harv")

        # Dynamic SOS/EOS (Fixed Plotting Bug: Ensuring they appear in every subgraph)
        if sos:
            ax.axvline(sos, color='blue', linewidth=2, label=f"SOS (20%): {sos.date()}")
        if eos:
            ax.axvline(eos, color='red', linewidth=2, label=f"EOS (50% Drop): {eos.date()}")
            
        title = f"Year {year} | PCODE: {pcode_selected}"
        if sos and eos:
            title += f" | Length: {(eos - sos).days} days"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper right', fontsize='x-small', ncol=3)

    plt.tight_layout()
    out_img = f"Model_physical/Results/{country}_dynamic_calendar_V3_{pcode_selected}.png"
    plt.savefig(out_img, dpi=150)
    print(f"Visualization saved to: {out_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, default="Zambia")
    args = parser.parse_args()
    
    #verify_and_plot_v3(args.country)

    # prepare list of all countries
    crop_area = pd.read_csv(r"GADM\crop_areas\africa_crop_areas_glad_filtered.csv")
    countries = crop_area["country"].unique().tolist()

    from STSG_smoothing import run_stsg

    for country in countries:
        print(f"Processing {country}...")
        if os.path.exists(f"Model_physical/Results/{country.replace(' ', '_')}_admin2_STSG_smoothed.csv"):
            print(f"Skipping {country} as STSG smoothed data already exists.")
        else:
            try:
                run_stsg(country.replace(" ", "_"))
            except Exception as e:
                print(f"Failed to process {country}: {str(e)}")
                continue
        try:
            verify_and_plot_v3(country.replace(" ", "_"))
        except Exception as e:
            print(f"Failed to plot {country}: {str(e)}")
            continue
