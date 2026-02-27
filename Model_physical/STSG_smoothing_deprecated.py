import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_country_timeseries(country):
    path = f"RemoteSensing/GADM/extractions/{country}_admin2_VI_timeseries_GADM.csv"
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear

    # Filter locations that are needed
    crop_areas = pd.read_csv("GADM/crop_areas/africa_crop_areas_glad_filtered.csv")
    df = df[df["PCODE"].isin(crop_areas["PCODE"])].copy()

    return df


# ============================================================
# 2. BUILD REFERENCE CURVES
# ============================================================

def compute_reference_curves(df):
    """
    For each PCODE:
    - compute mean NDVI per DOY across all years
    """
    ref = (
        df.groupby(["PCODE", "doy"])["NDVI_mean"]
        .mean()
        .reset_index()
        .pivot(index="doy", columns="PCODE", values="NDVI_mean")
    )
    
    return ref  # rows = DOY, columns = PCODE


# ============================================================
# 3. FIND SIMILAR PCODES
# ============================================================

def find_similar_pcodes(reference_df, target_pcode, threshold=0.95, max_neighbors=10):
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

    return similar[:max_neighbors]


# ============================================================
# 4. INITIAL ESTIMATION (STSG)
# ============================================================

def build_initial_estimate(df, reference_df, target_pcode, neighbors):
    """
    df: full dataframe
    neighbors: list of (pcode, corr)
    """
    if len(neighbors) == 0:  # fallback to raw NDVI
        return df[df["PCODE"] == target_pcode]["NDVI_mean"].values

    # Extract raw curves (for all years)
    target_df = df[df["PCODE"] == target_pcode].sort_values("date")
    ndvi_target = target_df["NDVI_mean"].values
    years = target_df["year"].values
    doys = target_df["doy"].values

    # Pre-compute reference curves
    ref_target = reference_df[target_pcode].reindex(doys).values

    # Container
    initial = np.zeros_like(ndvi_target)
    
    # Convert neighbor correlations into weights
    corrs = np.array([r for (_, r) in neighbors])
    weights = corrs / corrs.sum()

    for idx, (neighbor_pcode, r) in enumerate(neighbors):
        neighbor_df = df[df["PCODE"] == neighbor_pcode].sort_values("date")

        ndvi_nei = neighbor_df["NDVI_mean"].values
        ref_nei = reference_df[neighbor_pcode].reindex(doys).values

        # predicted NDVI (ratio * ref_target) — simplified form of Eq.4
        ratio = ndvi_nei / ref_nei
        pred = ratio * ref_target

        initial += weights[idx] * pred

    return initial


# ============================================================
# 5. SYNTHESIZE RAW + INITIAL
# ============================================================

def synthesize(raw, initial):
    """
    If raw < initial: negative noise -> use initial
    Otherwise use raw.
    """
    syn = np.where(np.isnan(raw), initial, raw)
    syn = np.where(raw < initial, initial, raw)
    return syn


# ============================================================
# 6. WEIGHTED SAVITZKY–GOLAY
# ============================================================

def weighted_sg(syn, initial, iters=2, window=5, poly=2):
    """
    SG smoothing with iterative lifting.
    """
    current = syn.copy()
    for _ in range(iters):
        fitted = savgol_filter(current, window_length=window, polyorder=poly)
        mask = current < fitted
        current[mask] = fitted[mask]
    return current


# ============================================================
# 7. MAIN STSG PIPELINE FOR COUNTRY
# ============================================================

def run_stsg(country):
    df = load_country_timeseries(country)
    reference_df = compute_reference_curves(df)

    results = []

    for pcode in df["PCODE"].unique():
        print(f"Processing {pcode} ...")
        neighbors = find_similar_pcodes(reference_df, pcode)

        # Extract raw NDVI (sorted)
        sub_df = df[df["PCODE"] == pcode].sort_values("date")
        raw = sub_df["NDVI_mean"].values

        # STSG step 1: initial estimate
        initial = build_initial_estimate(df, reference_df, pcode, neighbors)

        # Step 2: merge
        syn = synthesize(raw, initial)

        # Step 3: smooth
        smoothed = weighted_sg(syn, initial)

        out = sub_df.copy()
        out["NDVI_STSG"] = smoothed
        results.append(out)

    final = pd.concat(results, ignore_index=True)

    # Save
    out_path = f"Model_physical/Results/{country}_admin2_STSG_smoothed.csv"
    os.makedirs("Model_physical/Results", exist_ok=True)
    final.to_csv(out_path, index=False)
    print(f"\nSaved output -> {out_path}")

    return final

if __name__ == "__main__":
    country = "Angola"
    run_stsg(country)