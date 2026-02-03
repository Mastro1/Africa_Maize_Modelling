
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# STSG HELPER FUNCTIONS
# ============================================================

def compute_reference_curves(df, col="FPAR_mean"):
    """
    For each PCODE: compute mean value per DOY across all years
    """
    if "doy" not in df.columns:
        df["doy"] = df["date"].dt.dayofyear
        
    ref = (
        df.groupby(["PCODE", "doy"])[col]
        .mean()
        .reset_index()
        .pivot(index="doy", columns="PCODE", values=col)
    )
    return ref

def find_similar_pcodes(reference_df, target_pcode, threshold=0.90, max_neighbors=10):
    if target_pcode not in reference_df.columns:
        return []
        
    target_curve = reference_df[target_pcode].values
    similar = []

    for pcode in reference_df.columns:
        if pcode == target_pcode:
            continue
        
        curve = reference_df[pcode].values
        # Strict NaN check from original logic, relaxing slightly to allow calculation if mostly valid?
        # Original: if np.isnan(target_curve).any() or np.isnan(curve).any(): continue
        # We'll use the same strict check to match "apply the same" logic, or fill 0?
        # FPAR might have gaps. Let's keep it strict but maybe fillna(0) on curves first?
        # Let's stick to original logic.
        
        if np.isnan(target_curve).any() or np.isnan(curve).any():
            continue
        
        r, _ = pearsonr(target_curve, curve)
        if r >= threshold:
            similar.append((pcode, r))

    # Sort by correlation desc
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar[:max_neighbors]

def build_initial_estimate(df, reference_df, target_pcode, neighbors, col="FPAR_mean"):
    target_df = df[df["PCODE"] == target_pcode].sort_values("date")
    # Ensure we have dates to merge on
    target_dates = target_df[["date", "doy"]].reset_index(drop=True)
    
    # Container
    initial = np.zeros(len(target_df))
    
    corrs = np.array([r for (_, r) in neighbors])
    if len(neighbors) == 0:
        return target_df[col].values
        
    weights = corrs / corrs.sum()
    
    ref_target = reference_df[target_pcode].reindex(target_df["doy"]).values
    
    for idx, (neighbor_pcode, r) in enumerate(neighbors):
        # Merge neighbor data to ensure alignment with target dates
        nei_df = df[df["PCODE"] == neighbor_pcode]
        merged = pd.merge(target_dates, nei_df[["date", col]], on="date", how="left")
        
        # Fill missing neighbor values (if any) to avoid NaN propagation
        # Using interpolation for short gaps
        val_nei = merged[col].interpolate(method="nearest").fillna(method="bfill").fillna(method="ffill").fillna(0).values
        
        ref_nei = reference_df[neighbor_pcode].reindex(target_df["doy"]).fillna(0.001).values
        # Avoid div by zero
        ref_nei[ref_nei == 0] = 0.001
        
        ratio = val_nei / ref_nei
        pred = ratio * ref_target
        
        # Handle NaNs in pred if they appear
        pred = np.nan_to_num(pred, nan=0.0)
        
        initial += weights[idx] * pred
        
    return initial

def synthesize(raw, initial):
    syn = np.where(np.isnan(raw), initial, raw)
    syn = np.where(raw < initial, initial, raw)
    return syn

def weighted_sg(syn, initial, iters=2, window=5, poly=2):
    current = syn.copy()
    for _ in range(iters):
        # SavGol requires window_length <= x.size
        if len(current) < window:
             # Fallback for very short series
             return current
        try:
            fitted = savgol_filter(current, window_length=window, polyorder=poly)
            mask = current < fitted
            current[mask] = fitted[mask]
        except Exception:
            return current
    return current


def prepare_stsg_data(df, col="FPAR_mean"):
    return compute_reference_curves(df, col=col)

def run_stsg_on_pcode(df, target_pcode, ref_df, col="FPAR_mean"):
    # 2. Find Neighbors
    neighbors = find_similar_pcodes(ref_df, target_pcode)
    print(f"  - Found {len(neighbors)} neighbors for STSG.")
    
    # 3. Initial Estimate
    initial = build_initial_estimate(df, ref_df, target_pcode, neighbors, col=col)
    
    # 4. Synthesize
    target_df = df[df["PCODE"] == target_pcode].sort_values("date")
    raw = target_df[col].values
    syn = synthesize(raw, initial)
    
    # 5. Smooth
    smoothed = weighted_sg(syn, initial, window=30, poly=2)
    
    return smoothed


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

def verify_and_plot_fpar(country, years_to_plot=[2017, 2018, 2019, 2020]):
    """
    Loads FPAR data, finds the max area PCODE, processes it, and plots specific years.
    """
    base_dir = os.getcwd()
    
    col_path = os.path.join("GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    area_path = os.path.join("GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    
    fpar_filename = f"{country.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv"
    fpar_path = os.path.join("Model_physical", "Input", fpar_filename)
    
    if not os.path.exists(fpar_path):
        print(f"Error: FPAR file not found at {fpar_path}")
        return

    # 1. Get Target PCODE
    target_pcode = get_max_area_pcode(country, area_path)
    if not target_pcode: return

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

    # 3. Load Data & Filter to Crop Areas (for valid neighbors)
    print(f"Loading full FPAR data from {fpar_path}...")
    df_fpar = pd.read_csv(fpar_path, parse_dates=['date'])
    df_fpar["doy"] = df_fpar["date"].dt.dayofyear
    df_fpar["year"] = df_fpar["date"].dt.year
    
    # Filter to known crop areas for STSG reference quality
    if os.path.exists(area_path):
        crop_areas = pd.read_csv(area_path)
        df_fpar = df_fpar[df_fpar["PCODE"].isin(crop_areas["PCODE"])].copy()

    # 4. Extract Target & Process Simple Method
    df_pcode = df_fpar[df_fpar['PCODE'] == target_pcode].copy()
    if df_pcode.empty:
        print(f"Error: No FPAR data found for PCODE {target_pcode}")
        return
    print("Processing Simple FPAR (Interpolation + Smooth)...")
    df_pcode = process_fpar_pcode(df_pcode)
    
    # 5. Process STSG Method
    print("Processing STSG FPAR...")
    try:
        ref_df = prepare_stsg_data(df_fpar, col="FPAR_mean")
        stsg_values = run_stsg_on_pcode(df_fpar, target_pcode, ref_df, col="FPAR_mean")
        # Ensure alignment
        # run_stsg_for_pcode sorts by date internally to produce values. 
        # df_pcode comes from process_fpar_pcode which also sorts by date.
        # Lengths should match.
        if len(stsg_values) == len(df_pcode):
            df_pcode['FPAR_STSG'] = stsg_values
        else:
            print(f"Warning: STSG output length {len(stsg_values)} != df length {len(df_pcode)}. Skipping STSG plot.")
            df_pcode['FPAR_STSG'] = np.nan
    except Exception as e:
        print(f"STSG Failed: {e}")
        df_pcode['FPAR_STSG'] = np.nan

    # Set index
    df_pcode = df_pcode.set_index('date', drop=False)
    
    # 6. Plotting
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
        sub = df_pcode.loc[str(plot_start):str(plot_end)]
        
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            continue
            
        # Plot
        ax.plot(sub['date'], sub['FPAR_mean'], 'o-', color='grey', alpha=0.5, label='Raw FPAR')
        ax.plot(sub['date'], sub['FPAR_smooth'], '-', color='green', linewidth=2, label='Simple Smooth (SavGol)')
        
        if 'FPAR_STSG' in sub.columns and not sub['FPAR_STSG'].isna().all():
            ax.plot(sub['date'], sub['FPAR_STSG'], '-', color='blue', linewidth=2, label='STSG Smooth')
        
        cal_plant_date = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=f_plant - 1)
        cal_end_date = pd.Timestamp(year=win_end_year, month=1, day=1) + pd.Timedelta(days=f_end - 1)
        
        ax.axvline(cal_plant_date, color='blue', linestyle='--', label='Calendar Plant')
        ax.axvline(cal_end_date, color='red', linestyle='--', label='Calendar End')
        ax.axvspan(win_start, win_end, color='yellow', alpha=0.1, label='Target Window')
        
        ax.set_title(f"Year {year} | PCODE: {target_pcode}")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper right', ncol=3, fontsize='small')
            
    plt.tight_layout()
    output_filename = f"Model_physical/Results/{country}_FPAR_timeseries_{target_pcode}.png"
    plt.savefig(output_filename, dpi=150)
    print(f"Plot saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, default="Zimbabwe")
    args = parser.parse_args()
    
    verify_and_plot_fpar(args.country)
