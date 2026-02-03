import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import random
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_country_timeseries(country):
    # Handle spaces in country names for filenames
    country_file = country.replace(" ", "_")
    path = f"Model_physical/Input/{country_file}_admin2_VI_timeseries_GADM.csv"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input: {path}")
        
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear

    # Filter locations that are needed
    crop_areas_path = "GADM/crop_areas/africa_crop_areas_glad_filtered.csv"
    if os.path.exists(crop_areas_path):
        crop_areas = pd.read_csv(crop_areas_path)
        df = df[df["PCODE"].isin(crop_areas["PCODE"])].copy()

    return df


# ============================================================
# 2. BUILD REFERENCE CURVES (Climatology)
# ============================================================

def compute_reference_curves(df):
    """
    For each PCODE: compute mean NDVI per DOY across all years.
    Used as the purely temporal 'initial guess' to fill gaps.
    """
    ref = (
        df.groupby(["PCODE", "doy"])["NDVI_mean"]
        .mean()
        .reset_index()
        .pivot(index="doy", columns="PCODE", values="NDVI_mean")
    )
    return ref


# ============================================================
# 3. INITIAL ESTIMATION (Purely Temporal)
# ============================================================

def build_temporal_estimate(sub_df, reference_df, target_pcode):
    """
    Extracts the climatological mean for the specific PCODE as the initial estimate.
    """
    doys = sub_df["doy"].values
    if target_pcode not in reference_df.columns:
        return np.full(len(doys), np.nan)
        
    ref_target = reference_df[target_pcode].reindex(doys).values
    return ref_target


# ============================================================
# 4. SYNTHESIZE RAW + INITIAL
# ============================================================

def synthesize(raw, initial, floor_factor=0.7):
    """
    If raw is nan: use initial.
    If raw < initial * floor_factor: negative noise (clouds) -> use initial * floor_factor.
    Otherwise use raw.
    """
    syn = np.where(np.isnan(raw), initial, raw)
    # Use initial (climatology) with a factor as floor to handle extreme cloud/noise dips
    # Default 0.7 allows 30% drop from mean before lifting.
    syn = np.where(raw < initial * floor_factor, initial * floor_factor, raw)
    return syn


# ============================================================
# 5. WHITTAKER SMOOTHING
# ============================================================

def whittaker_smooth(y, lmbda=10, d=2, weights=None):
    """
    Core Whittaker-Eilers smoother using sparse matrices.
    """
    n = len(y)
    if weights is None:
        weights = np.ones(n)
    
    W = sp.diags(weights, format='csc')
    
    # Second order difference matrix D
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n), format='csc')
    
    A = W + lmbda * (D.T @ D)
    return spsolve(A, weights * y)

def weighted_whittaker(syn, iters=3, lmbda=10):
    """
    Whittaker smoothing with iterative lifting to follow the upper envelope (vegetation).
    """
    if np.all(np.isnan(syn)): return syn
    
    current = syn.copy()
    for _ in range(iters):
        fitted = whittaker_smooth(current, lmbda=lmbda)
        mask = current < fitted
        current[mask] = fitted[mask]
    return current


# ============================================================
# 6. MAIN TEMPORAL WHITTAKER PIPELINE
# ============================================================

def run_whittaker_temporal(country, pcode_selected=None, floor_factor=0.7, lmbda=10, iters=3):
    print(f"\n--- Running Whittaker Temporal Smoothing for {country} ---")
    df = load_country_timeseries(country)
    reference_df = compute_reference_curves(df)

    results = []
    unique_pcodes = df["PCODE"].unique()

    for pcode in unique_pcodes:
        print(f"Processing {pcode} ...")

        # Extract raw NDVI (sorted)
        sub_df = df[df["PCODE"] == pcode].sort_values("date")
        raw = sub_df["NDVI_mean"].values

        # Purely temporal initial estimate (Climatology)
        initial = build_temporal_estimate(sub_df, reference_df, pcode)

        # Synthesis & Smoothing
        syn = synthesize(raw, initial, floor_factor=floor_factor)
        smoothed = weighted_whittaker(syn, lmbda=lmbda, iters=iters)

        out = sub_df.copy()
        out["NDVI_Whittaker"] = smoothed
        results.append(out)

    final = pd.concat(results, ignore_index=True)

    # Save
    os.makedirs("Model_physical/Results", exist_ok=True)
    out_path = f"Model_physical/Results/{country.replace(' ', '_')}_admin2_Whittaker_temporal_smoothed.csv"
    final.to_csv(out_path, index=False)
    print(f"\nSaved output \u2192 {out_path}")

    # Plotting
    if pcode_selected is None:
        pcode_selected = random.choice(unique_pcodes)
        print(f"Random PCODE for visualization: {pcode_selected}")

    plot_results(final, pcode_selected, country)

    return final


# ============================================================
# 7. PLOT FUNCTION (Fixed Calendar Context)
# ============================================================

def plot_results(df, pcode, country="Unknown"):
    """
    Plots results with fixed calendar benchmarks (Plan, Veg, Harv, End).
    No dynamic date detection logic.
    """
    # 0. Ensure dates
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # 1. Load Calendar
    calendar_path = "GADM/crop_calendar/maize_crop_calendar_extraction.csv"
    if not os.path.exists(calendar_path):
        print("Warning: Maize calendar not found. Showing simple plot.")
        _plot_simple(df, pcode, country)
        return

    df_cal = pd.read_csv(calendar_path)
    if 'FNID' in df_cal.columns: df_cal = df_cal.rename(columns={'FNID': 'PCODE'})
    
    cal_row = df_cal[df_cal['PCODE'] == pcode]
    if cal_row.empty:
        print(f"No calendar entry for {pcode}. Showing simple plot.")
        _plot_simple(df, pcode, country)
        return

    f_plant = cal_row.iloc[0]['Maize_1_planting']
    f_veg   = cal_row.iloc[0]['Maize_1_vegetative']
    f_harv  = cal_row.iloc[0]['Maize_1_harvest']
    f_end   = cal_row.iloc[0]['Maize_1_endofseaso']

    # 2. Setup Plot
    YEARS = [2005, 2010, 2015, 2020]
    dfp = df[df["PCODE"] == pcode].sort_values("date")
    
    fig, axes = plt.subplots(len(YEARS), 1, figsize=(12, 4 * len(YEARS)))
    if len(YEARS) == 1: axes = [axes]

    for i, year in enumerate(YEARS):
        ax = axes[i]
        
        # Season Context Window
        start_year = year
        if f_plant > f_end: # Crosses year
            start_year = year - 1
            s_start = pd.Timestamp(year=start_year, month=1, day=1) + pd.Timedelta(days=f_plant - 31)
            s_end   = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_end + 29)
        else:
            s_start = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_plant - 31)
            s_end   = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_end + 29)

        p_start, p_end = s_start - pd.Timedelta(days=60), s_end + pd.Timedelta(days=60)
        sub = dfp[(dfp["date"] >= p_start) & (dfp["date"] <= p_end)]
        if sub.empty: continue

        # Draw Curves
        ax.plot(sub["date"], sub["NDVI_mean"], 'o-', color='black', label='Raw NDVI', alpha=0.6, markersize=4, linewidth=1)
        ax.plot(sub["date"], sub["NDVI_Whittaker"], '-', color='green', label='Whittaker Smoothed (Temporal)', linewidth=2)
        ax.axvspan(s_start, s_end, color='yellow', alpha=0.1, label='Season Context')
        
        # Draw Fixed Benchmarks
        def d2date(y, d): return pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=d - 1)
        
        v_plant = d2date(start_year, f_plant)
        v_veg   = d2date(start_year if f_veg > f_plant else start_year+1, f_veg)
        v_harv  = d2date(start_year if f_harv > f_plant else start_year+1, f_harv)
        v_end   = d2date(start_year if f_end > f_plant else start_year+1, f_end)
        
        ax.axvline(v_plant, color='cyan', linestyle='--', label='Fixed Plant')
        ax.axvline(v_veg,   color='lime', linestyle=':',  label='Fixed Veg')
        ax.axvline(v_harv,  color='orange', linestyle='--', label='Fixed Harv')
        ax.axvline(v_end,   color='magenta', linestyle=':', label='Fixed EndSeaso')

        ax.set_title(f"Year {year} | PCODE: {pcode}")
        ax.set_ylabel("NDVI")
        ax.grid(True)
        if i == 0: ax.legend(loc='upper right', fontsize='small', ncol=2)

    plt.tight_layout()
    plt.savefig(f"Model_physical/Results/{country.replace(' ', '_')}_Whittaker_temporal_visualization_{pcode}.png", dpi=150)
    plt.show()

def _plot_simple(df, pcode, country):
    YEARS = [2005, 2010, 2015, 2020]
    dfp = df[df["PCODE"] == pcode].sort_values("date")
    fig, axs = plt.subplots(len(YEARS), 1, figsize=(12, 16))
    for i, yr in enumerate(YEARS):
        sub = dfp[dfp["year"] == yr]
        if sub.empty: continue
        axs[i].plot(sub["date"], sub["NDVI_mean"], 'o-', color='black', alpha=0.6, markersize=4)
        axs[i].plot(sub["date"], sub["NDVI_Whittaker"], color='green', linewidth=2)
        axs[i].set_title(f"{pcode} — Year {yr}")
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    country = "Malawi"
    run_whittaker_temporal(country)
