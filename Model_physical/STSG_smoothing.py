"""
STSG SMOOTHING MODULE – Unified NDVI & FPAR Smoother
=====================================================
This module performs Spatio-Temporal Savitzky-Golay (STSG) smoothing
on either NDVI or FPAR timeseries data.

Outputs are saved to:  Model_physical/Results/STSG/
  - {Country}_NDVI_STSG.csv
  - {Country}_FPAR_STSG.csv

Usage:
  # From command line:
  python STSG_smoothing.py --country Zambia --variable both

  # As importable module:
  from STSG_smoothing import run_stsg_ndvi, run_stsg_fpar
  run_stsg_ndvi("Zambia")
  run_stsg_fpar("Zambia")
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# TUNABLE PARAMETERS  –  Change these to experiment
# =============================================================

# --- NDVI STSG Parameters ---
NDVI_COL = "NDVI_mean"              # Source column name in input data
NDVI_OUTPUT_COL = "NDVI_STSG"       # Output column name
NDVI_SG_WINDOW = 5                  # Savitzky-Golay window length
NDVI_SG_POLY = 2                    # Polynomial order
NDVI_SG_ITERS = 2                   # Iterative lifting passes
NDVI_NEIGHBOR_THRESHOLD = 0.95      # Min correlation to be a "neighbor"
NDVI_MAX_NEIGHBORS = 10             # Max neighbors to use

# --- FPAR STSG Parameters ---
FPAR_COL = "FPAR_mean"              # Source column name in input data
FPAR_OUTPUT_COL = "FPAR_STSG"       # Output column name
FPAR_SG_WINDOW = 30                 # Savitzky-Golay window length (wider for FPAR)
FPAR_SG_POLY = 2                    # Polynomial order
FPAR_SG_ITERS = 2                   # Iterative lifting passes
FPAR_NEIGHBOR_THRESHOLD = 0.95      # Min correlation (lower for FPAR, sparser data)
FPAR_MAX_NEIGHBORS = 10             # Max neighbors to use

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_STSG_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "STSG")

# =============================================================
# END OF TUNABLE PARAMETERS
# =============================================================


# ============================================================
# CORE STSG FUNCTIONS  (generic – work with any column)
# ============================================================

def compute_reference_curves(df, col):
    """
    For each PCODE: compute mean value per DOY across all years.
    Returns a pivot table: rows = DOY, columns = PCODE.
    """
    if "doy" not in df.columns:
        df = df.copy()
        df["doy"] = df["date"].dt.dayofyear
    ref = (
        df.groupby(["PCODE", "doy"])[col]
        .mean()
        .reset_index()
        .pivot(index="doy", columns="PCODE", values=col)
    )
    return ref


def find_similar_pcodes(reference_df, target_pcode, threshold, max_neighbors):
    """Find spatially similar PCODEs based on Pearson correlation of reference curves."""
    if target_pcode not in reference_df.columns:
        return []

    target_curve = reference_df[target_pcode].values
    similar = []

    for pcode in reference_df.columns:
        if pcode == target_pcode:
            continue
        curve = reference_df[pcode].values
        if np.isnan(target_curve).any() or np.isnan(curve).any():
            continue
        r, _ = pearsonr(target_curve, curve)
        if r >= threshold:
            similar.append((pcode, r))

    similar.sort(key=lambda x: x[1], reverse=True)
    return similar[:max_neighbors]


def build_initial_estimate(df, reference_df, target_pcode, neighbors, col):
    """Build the STSG initial estimate by weighted combination of neighbor ratios."""
    target_df = df[df["PCODE"] == target_pcode].sort_values("date")
    target_dates = target_df[["date", "doy"]].reset_index(drop=True)

    initial = np.zeros(len(target_df))

    corrs = np.array([r for (_, r) in neighbors])
    if len(neighbors) == 0:
        return target_df[col].values

    weights = corrs / corrs.sum()
    ref_target = reference_df[target_pcode].reindex(target_df["doy"]).values

    for idx, (neighbor_pcode, r) in enumerate(neighbors):
        nei_df = df[df["PCODE"] == neighbor_pcode]
        merged = pd.merge(target_dates, nei_df[["date", col]], on="date", how="left")
        val_nei = (merged[col].interpolate(method="nearest")
                   .fillna(method="bfill").fillna(method="ffill").fillna(0).values)
        ref_nei = reference_df[neighbor_pcode].reindex(target_df["doy"]).fillna(0.001).values
        ref_nei[ref_nei == 0] = 0.001

        ratio = val_nei / ref_nei
        pred = ratio * ref_target
        pred = np.nan_to_num(pred, nan=0.0)
        initial += weights[idx] * pred

    return initial


def synthesize(raw, initial):
    """Merge raw + initial: use initial where raw is NaN or below initial."""
    syn = np.where(np.isnan(raw), initial, raw)
    syn = np.where(raw < initial, initial, raw)
    return syn


def weighted_sg(syn, initial, iters, window, poly):
    """Iterative Savitzky-Golay with lifting."""
    current = syn.copy()
    for _ in range(iters):
        if len(current) < window:
            return current
        try:
            fitted = savgol_filter(current, window_length=window, polyorder=poly)
            mask = current < fitted
            current[mask] = fitted[mask]
        except Exception:
            return current
    return current


# ============================================================
# HIGH-LEVEL PIPELINE  (per-PCODE processing)
# ============================================================

def run_stsg_on_pcode(df, target_pcode, ref_df, col, threshold, max_neighbors,
                      sg_window, sg_poly, sg_iters):
    """Run the full STSG pipeline for a single PCODE. Returns smoothed array."""
    neighbors = find_similar_pcodes(ref_df, target_pcode, threshold, max_neighbors)

    # Initial estimate
    initial = build_initial_estimate(df, ref_df, target_pcode, neighbors, col)

    # Synthesize
    target_df = df[df["PCODE"] == target_pcode].sort_values("date")
    raw = target_df[col].values
    syn = synthesize(raw, initial)

    # Smooth
    smoothed = weighted_sg(syn, initial, iters=sg_iters, window=sg_window, poly=sg_poly)
    return smoothed


# ============================================================
# COUNTRY-LEVEL RUNNERS
# ============================================================

def _load_data(country, variable):
    """Load the appropriate timeseries data for a variable."""
    country_clean = country.replace(" ", "_")

    if variable == "ndvi":
        path = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions",
                            f"{country_clean}_admin2_VI_timeseries_GADM.csv")
        col = NDVI_COL
    elif variable == "fpar":
        path = os.path.join(BASE_DIR, "Model_physical", "Input",
                            f"{country_clean}_admin2_FPAR_timeseries_GLAD.csv")
        col = FPAR_COL
    else:
        raise ValueError(f"Unknown variable: {variable}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    # Filter to known crop area PCODEs for quality reference curves
    crop_area_path = os.path.join(BASE_DIR, "GADM", "crop_areas",
                                  "africa_crop_areas_glad_filtered.csv")
    if os.path.exists(crop_area_path):
        crop_areas = pd.read_csv(crop_area_path)
        df = df[df["PCODE"].isin(crop_areas["PCODE"])].copy()

    return df, col


def _run_stsg_country(country, variable, col, output_col,
                      threshold, max_neighbors,
                      sg_window, sg_poly, sg_iters):
    """Generic country-level STSG runner."""
    country_clean = country.replace(" ", "_")

    df, _ = _load_data(country, variable)

    print(f"\n--- STSG Smoothing: {variable.upper()} for {country} ---")
    print(f"  Parameters: window={sg_window}, poly={sg_poly}, iters={sg_iters}, "
          f"threshold={threshold}, max_neighbors={max_neighbors}")

    ref_df = compute_reference_curves(df, col)
    pcodes = df["PCODE"].unique()
    total = len(pcodes)
    print(f"  Processing {total} PCODEs...")

    results = []
    for count, pcode in enumerate(pcodes, 1):
        if count % 20 == 0:
            print(f"    {count}/{total} ...", end="\r")

        try:
            smoothed = run_stsg_on_pcode(df, pcode, ref_df, col,
                                          threshold, max_neighbors,
                                          sg_window, sg_poly, sg_iters)
            sub = df[df["PCODE"] == pcode].sort_values("date").copy()
            if len(smoothed) == len(sub):
                sub[output_col] = smoothed
                sub[output_col] = sub[output_col].clip(0, 1)
            else:
                sub[output_col] = sub[col]
        except Exception:
            sub = df[df["PCODE"] == pcode].copy()
            sub[output_col] = sub[col]

        results.append(sub[["date", "PCODE", output_col]])

    final = pd.concat(results, ignore_index=True)

    # Save
    os.makedirs(OUTPUT_STSG_DIR, exist_ok=True)
    filename = f"{country_clean}_{variable.upper()}_STSG.csv"
    out_path = os.path.join(OUTPUT_STSG_DIR, filename)
    final.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({len(final)} rows)")

    return out_path


def run_stsg_ndvi(country):
    """Run STSG smoothing on NDVI for a country. Returns path to output CSV."""
    return _run_stsg_country(
        country, variable="ndvi", col=NDVI_COL, output_col=NDVI_OUTPUT_COL,
        threshold=NDVI_NEIGHBOR_THRESHOLD, max_neighbors=NDVI_MAX_NEIGHBORS,
        sg_window=NDVI_SG_WINDOW, sg_poly=NDVI_SG_POLY, sg_iters=NDVI_SG_ITERS,
    )


def run_stsg_fpar(country):
    """Run STSG smoothing on FPAR for a country. Returns path to output CSV."""
    return _run_stsg_country(
        country, variable="fpar", col=FPAR_COL, output_col=FPAR_OUTPUT_COL,
        threshold=FPAR_NEIGHBOR_THRESHOLD, max_neighbors=FPAR_MAX_NEIGHBORS,
        sg_window=FPAR_SG_WINDOW, sg_poly=FPAR_SG_POLY, sg_iters=FPAR_SG_ITERS,
    )


# ============================================================
# HELPER: get expected output path (for models to check)
# ============================================================

def get_stsg_path(country, variable):
    """Return the expected output path for a given country + variable."""
    country_clean = country.replace(" ", "_")
    filename = f"{country_clean}_{variable.upper()}_STSG.csv"
    return os.path.join(OUTPUT_STSG_DIR, filename)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run STSG smoothing on NDVI/FPAR")
    parser.add_argument("--country", type=str, default="Zambia",
                        help="Country name (e.g. 'Zambia', 'South Africa')")
    parser.add_argument("--variable", type=str, default="both",
                        choices=["ndvi", "fpar", "both"],
                        help="Which variable to smooth")
    args = parser.parse_args()

    if args.variable in ("ndvi", "both"):
        run_stsg_ndvi(args.country)
    if args.variable in ("fpar", "both"):
        run_stsg_fpar(args.country)

    print("\nDone.")