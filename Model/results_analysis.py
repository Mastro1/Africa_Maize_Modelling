import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Standardize column names we rely on
    required = [
        "country",
        "PCODE",
        "year",
        "area",
        "production",
        "yield",
        "pred_yield",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in results CSV")
    
    # Optional admin name: prefer 'admin_2' if present; else try 'admin2'; else skip.
    if "admin_2" not in df.columns and "admin2" in df.columns:
        df = df.rename(columns={"admin2": "admin_2"})
    
    # Check if national_trend is already present
    if "national_trend" in df.columns:
        print("national_trend column already present in results CSV")
    
    return df


def add_national_yield(df: pd.DataFrame, admin0_csv_path: Path) -> pd.DataFrame:
    """Attach national yield from the official admin0 file (country-year).

    Expected columns in admin0 CSV: year, area, production, yield, yield_recalc, country.
    We prefer the provided 'yield' if present; else compute production/area; if not,
    fall back to 'yield_recalc'.
    """
    nat = pd.read_csv(admin0_csv_path)
    # Normalize column names
    colmap = {c.lower(): c for c in nat.columns}
    def has(name):
        return name in colmap

    if has("yield"):
        nat["national_yield"] = nat[colmap["yield"]]
    elif has("production") and has("area"):
        nat["national_yield"] = nat[colmap["production"]] / nat[colmap["area"]]
    elif has("yield_recalc"):
        nat["national_yield"] = nat[colmap["yield_recalc"]]
    else:
        raise ValueError("Admin0 CSV missing usable national yield columns")

    # Select and rename join keys safely
    country_col = colmap.get("country", "country")
    year_col = colmap.get("year", "year")
    nat = nat[[country_col, year_col, "national_yield"]].rename(columns={country_col: "country", year_col: "year"})

    # Enforce consistent dtypes for join
    nat["year"] = pd.to_numeric(nat["year"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    nat["country"] = nat["country"].astype(str)
    df["country"] = df["country"].astype(str)

    out = df.merge(nat, on=["country", "year"], how="left")

    # Fallback: compute from results file if merge failed
    if "national_yield" not in out.columns or out["national_yield"].isna().all():
        agg = (
            out.groupby(["country", "year"]).apply(
                lambda g: pd.Series({
                    "ny": g["production"].sum(min_count=1) / g["area"].sum(min_count=1)
                })
            ).reset_index()
        )
        agg = agg.rename(columns={"ny": "national_yield"})
        out = out.drop(columns=[c for c in ["national_yield"] if c in out.columns])
        out = out.merge(agg, on=["country", "year"], how="left")

    return out


def compute_relative_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-location relative z-scores (model-like metric).

    Steps per admin-2 (PCODE):
    - yield_rel = yield / national_yield
    - pred_rel = calibrated_ratio if present else pred_yield / national_yield
    - Use OBS mean and std to standardize both series: z_obs, z_pred.
    """
    df = df.copy()
    df["yield_rel"] = df["yield"] / df["national_yield"]
    if "calibrated_ratio" in df.columns:
        df["pred_yield_rel"] = df["calibrated_ratio"]
    else:
        df["pred_yield_rel"] = df["pred_yield"] / df["national_yield"]

    def _zscore(g: pd.DataFrame):
        mu = g["yield_rel"].mean()
        sd = g["yield_rel"].std(ddof=1)
        sd = sd if sd and np.isfinite(sd) and sd > 0 else np.nan
        g = g.copy()
        g["z_obs"] = (g["yield_rel"] - mu) / sd
        g["z_pred"] = (g["pred_yield_rel"] - mu) / sd
        return g

    df = df.groupby("PCODE", group_keys=False).apply(_zscore)
    return df


def _safe_corr(x: np.ndarray, y: np.ndarray):
    try:
        if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan, np.nan
        r, p = pearsonr(x, y)
        return r, p
    except Exception:
        return np.nan, np.nan


def _safe_spearman(x: np.ndarray, y: np.ndarray):
    try:
        if len(x) < 3:
            return np.nan, np.nan
        r, p = spearmanr(x, y)
        return r, p
    except Exception:
        return np.nan, np.nan


def per_admin_metrics(df: pd.DataFrame, weight_col: str = "crop_area_ha_adjusted") -> pd.DataFrame:
    rows = []
    for (country, pcode), g in df.groupby(["country", "PCODE"], dropna=False):
        g = g.sort_values("year")
        y = g["z_obs"].values
        p = g["z_pred"].values
        n = np.isfinite(y) & np.isfinite(p)
        y = y[n]
        p = p[n]
        if len(y) < 3:
            acc = acc_p = rho = rho_p = var_ratio = nrmse = skill = np.nan
            slope_obs = slope_pred = ac1_obs = ac1_pred = np.nan
            extremes_p = extremes_r = extremes_f1 = np.nan
        else:
            acc, acc_p = _safe_corr(y, p)
            rho, rho_p = _safe_spearman(y, p)
            std_y = np.std(y, ddof=1)
            std_p = np.std(p, ddof=1)
            var_ratio = std_p / std_y if std_y > 0 else np.nan  # amplitude ratio on z-scores
            mse = np.mean((p - y) ** 2)
            nrmse = np.sqrt(mse) / (std_y if std_y > 0 else 1.0)
            denom = np.sum((y - np.mean(y)) ** 2)
            r2_z = 1.0 - (np.sum((y - p) ** 2) / denom) if denom > 0 else np.nan
            skill = 1.0 - (mse / (std_y ** 2 if std_y > 0 else np.nan))
            # Trend consistency (simple linear slope)
            yrs = g.loc[n, "year"].values
            try:
                slope_obs = np.polyfit(yrs, y, 1)[0]
                slope_pred = np.polyfit(yrs, p, 1)[0]
            except Exception:
                slope_obs = slope_pred = np.nan
            # Lag-1 autocorrelation
            def _acf1(a):
                if len(a) < 3:
                    return np.nan
                a = a - a.mean()
                return np.corrcoef(a[:-1], a[1:])[0, 1]

            ac1_obs = _acf1(y)
            ac1_pred = _acf1(p)
            # Extreme detection using observed 20th percentile threshold
            thr = np.nanpercentile(y, 20)
            obs_evt = y <= thr
            pred_evt = p <= thr
            tp = np.sum(pred_evt & obs_evt)
            fp = np.sum(pred_evt & ~obs_evt)
            fn = np.sum(~pred_evt & obs_evt)
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            extremes_f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
                else np.nan
            )
            extremes_p = precision
            extremes_r = recall

        weight = g[weight_col].mean() if weight_col in g.columns else g["area"].mean()
        rows.append(
            {
                "country": country,
                "PCODE": pcode,
                "admin_2": g["admin_2"].iloc[0] if "admin_2" in g.columns else np.nan,
                "n_years": int(len(g)),
                "weight": weight,
                "ACC": acc,
                "ACC_p": acc_p,
                "Spearman": rho,
                "Spearman_p": rho_p,
                "VAR_ratio": var_ratio,
                "nRMSE": nrmse,
                "Skill": skill,
                "R2_z": r2_z,
                "slope_obs": slope_obs,
                "slope_pred": slope_pred,
                "acf1_obs": ac1_obs,
                "acf1_pred": ac1_pred,
                "extreme_precision": extremes_p,
                "extreme_recall": extremes_r,
                "extreme_f1": extremes_f1,
            }
        )
    return pd.DataFrame(rows)


def per_year_spatial_skill(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, year), g in df.groupby(["country", "year"], dropna=False):
        # Spatial correlation across admins for this year (anomalies)
        # Use admins with at least one non-nan anomaly pair
        y = g["z_obs"].values
        p = g["z_pred"].values
        n = np.isfinite(y) & np.isfinite(p)
        if n.sum() < 3:
            r_s, r_p = (np.nan, np.nan)
        else:
            r_s, r_p = spearmanr(y[n], p[n])
        std_y = np.nanstd(y)
        std_p = np.nanstd(p)
        rows.append(
            {
                "country": country,
                "year": year,
                "spatial_spearman": r_s,
                "spatial_var_ratio": (std_p / std_y) if std_y > 0 else np.nan,
                "n_admins": int(n.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_countries(admin_metrics: pd.DataFrame) -> pd.DataFrame:
    def wmean(s, w):
        s = s.astype(float)
        w = w.astype(float)
        if np.all(~np.isfinite(s)):
            return np.nan
        if np.all(~np.isfinite(w)) or w.fillna(0).sum() == 0:
            return s.mean()
        s = s.fillna(0)
        w = w.fillna(0)
        return np.average(s, weights=w)

    summary = (
        admin_metrics.groupby("country").apply(
            lambda g: pd.Series(
                {
                    "admins": len(g),
                    "ACC_wmean": wmean(g["ACC"], g["weight"]),
                    "VAR_wmean": wmean(g["VAR_ratio"], g["weight"]),
                    "nRMSE_wmean": wmean(g["nRMSE"], g["weight"]),
                    "Skill_wmean": wmean(g["Skill"], g["weight"]),
                    "frac_ACC_ge_0_3": (g["ACC"] >= 0.3).mean(),
                    "frac_VAR_in_0_8_1_25": g["VAR_ratio"].between(0.8, 1.25).mean(),
                }
            )
        )
    ).reset_index()
    return summary


def plot_acc_var_quadrants(admin_metrics: pd.DataFrame, out_dir: Path):
    countries = sorted(admin_metrics["country"].dropna().unique())
    n = len(countries)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for idx, country in enumerate(countries):
        ax = axes[idx // cols, idx % cols]
        g = admin_metrics[admin_metrics["country"] == country]
        ax.scatter(g["ACC"], g["VAR_ratio"], s=np.clip(g["weight"].fillna(0) / (g["weight"].max() + 1e-9) * 100, 10, 100), alpha=0.7)
        ax.axvline(0.3, color="grey", ls="--", lw=1)
        ax.axhline(1.0, color="grey", ls="--", lw=1)
        ax.set_title(country)
        ax.set_xlabel("Anomaly Corr (ACC)")
        ax.set_ylabel("Amplitude ratio (Pred/Obs)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, max(2, np.nanpercentile(g["VAR_ratio"], 95)))
    plt.tight_layout()
    fig.savefig(out_dir / "acc_vs_var_quadrants.png", dpi=200)
    plt.close(fig)


def plot_distributions(admin_metrics: pd.DataFrame, out_dir: Path):
    melted = admin_metrics.melt(id_vars=["country"], value_vars=["ACC", "VAR_ratio", "nRMSE", "Skill"], var_name="metric", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="country", y="value", hue="metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "metric_distributions_by_country.png", dpi=200)
    plt.close()


def plot_spatial_skill_lines(spatial_df: pd.DataFrame, out_dir: Path):
    countries = sorted(spatial_df["country"].dropna().unique())
    cols = 3
    rows = int(np.ceil(len(countries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    for idx, country in enumerate(countries):
        ax = axes[idx // cols, idx % cols]
        g = spatial_df[spatial_df["country"] == country].sort_values("year")
        ax.plot(g["year"], g["spatial_spearman"], marker="o")
        ax.axhline(0.0, color="grey", ls="--", lw=1)
        ax.set_title(country)
        ax.set_xlabel("Year")
        ax.set_ylabel("Spatial Spearman (anomalies)")
        ax.set_ylim(-1, 1)
    plt.tight_layout()
    fig.savefig(out_dir / "yearly_spatial_skill.png", dpi=200)
    plt.close(fig)


def plot_anomaly_heatmaps(df: pd.DataFrame, out_dir: Path):
    countries = sorted(df["country"].dropna().unique())
    for country in countries:
        g = df[df["country"] == country].copy()
        # Pivot for observed and predicted standardized anomalies (z-scores)
        obs = g.pivot_table(index="PCODE", columns="year", values="z_obs")
        pred = g.pivot_table(index="PCODE", columns="year", values="z_pred")
        # Align rows/cols
        obs, pred = obs.align(pred, join="inner", axis=0)
        obs, pred = obs.align(pred, join="inner", axis=1)
        if obs.shape[0] < 2 or obs.shape[1] < 2:
            continue
        vmax = np.nanmax(np.abs(obs.values))
        vmax = max(vmax, np.nanmax(np.abs(pred.values)))
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else None
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, obs.shape[0] * 0.15)), sharey=True)
        sns.heatmap(
            obs,
            ax=axes[0],
            cmap="RdBu_r",
            center=0,
            vmax=vmax,
            vmin=-vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Anomaly (z)"},
        )
        axes[0].set_title(f"Observed anomalies — {country}")
        sns.heatmap(
            pred,
            ax=axes[1],
            cmap="RdBu_r",
            center=0,
            vmax=vmax,
            vmin=-vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Anomaly (z)"},
        )
        axes[1].set_title(f"Predicted anomalies — {country}")
        plt.tight_layout()
        fig.savefig(out_dir / f"anomaly" / f"anomaly_heatmaps_{country.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)


def plot_yield_heatmaps(df: pd.DataFrame, out_dir: Path):
    """Render heatmaps of observed vs predicted yields per country.

    Left panel: observed yield heatmap; Right panel: predicted yield heatmap.
    Cells are annotated with the value (one decimal) using a small font.
    """
    countries = sorted(df["country"].dropna().unique())
    for country in countries:
        g = df[df["country"] == country].copy()

        # Pivot tables for observed and predicted yields
        obs = g.pivot_table(index="PCODE", columns="year", values="yield", aggfunc="mean")
        pred = g.pivot_table(index="PCODE", columns="year", values="pred_yield", aggfunc="mean")

        # Align rows and columns
        obs, pred = obs.align(pred, join="inner", axis=0)
        obs, pred = obs.align(pred, join="inner", axis=1)

        if obs.shape[0] < 2 or obs.shape[1] < 2:
            continue

        # Shared color scale across both panels
        vmax = np.nanmax([np.nanmax(obs.values), np.nanmax(pred.values)])
        vmin = np.nanmin([np.nanmin(obs.values), np.nanmin(pred.values)])
        if not np.isfinite(vmax) or not np.isfinite(vmin):
            continue

        fig_height = max(4, obs.shape[0] * 0.15)
        fig, axes = plt.subplots(1, 2, figsize=(14, fig_height), sharey=True)

        sns.heatmap(
            obs,
            ax=axes[0],
            cmap="YlGnBu",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Yield"},
        )
        axes[0].set_title(f"Observed yield — {country}")

        sns.heatmap(
            pred,
            ax=axes[1],
            cmap="YlGnBu",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Yield"},
        )
        axes[1].set_title(f"Predicted yield — {country}")

        plt.tight_layout()
        fig.savefig(out_dir / f"yield" / f"yield_heatmaps_{country.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)


def compute_country_r2_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute multiple country-level skill summaries.

    Outputs per country:
    - R2: row-wise yield vs pred_yield R2 across all admin-years
    - average_location_r2: mean of per-PCODE R2 over years (unweighted)
    - average_location_area_adjusted_r2: weighted mean of per-PCODE R2 using crop_area_ha_adjusted (else area)
    - aggregate_to_ground_r2: R2 between aggregated Model and True time series at the country-year level
    - aggregate_to_ground_corr: Pearson correlation between aggregated Model and True time series
    - aggregate_detrended_to_ground_r2: R2 between detrended (relative deviation from trend) aggregated Model and True time series
    """
    def _r2(y: np.ndarray, p: np.ndarray) -> float:
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() < 3:
            return np.nan
        yv, pv = y[m], p[m]
        ss_res = np.sum((yv - pv) ** 2)
        ss_tot = np.sum((yv - np.mean(yv)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    def _pearson(y: np.ndarray, p: np.ndarray) -> float:
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() < 3:
            return np.nan
        try:
            r, _ = pearsonr(y[m], p[m])
            return float(r)
        except Exception:
            return np.nan

    rows = []
    for country, g in df.groupby("country", dropna=False):
        # 1) Row-wise R2 across all observations
        r2_overall = _r2(g["yield"].values, g["pred_yield"].values)

        # 2) Per-PCODE R2 and averages
        pcode_r2_list = []
        pcode_weight_list = []
        for pcode, gp in g.groupby("PCODE", dropna=False):
            r2_pcode = _r2(gp["yield"].values, gp["pred_yield"].values)
            pcode_r2_list.append(r2_pcode)
            # weight per PCODE: average crop area adjusted if present else area
            if "crop_area_ha_adjusted" in gp.columns:
                w = pd.to_numeric(gp["crop_area_ha_adjusted"], errors="coerce").mean()
            else:
                w = pd.to_numeric(gp.get("area"), errors="coerce").mean()
            pcode_weight_list.append(w)

        pcode_r2 = np.array(pcode_r2_list, dtype=float)
        pcode_w = np.array(pcode_weight_list, dtype=float)
        # Unweighted average
        avg_loc_r2 = float(np.nanmean(pcode_r2)) if pcode_r2.size > 0 else np.nan
        # Weighted average
        if pcode_r2.size > 0 and np.isfinite(pcode_w).any() and np.nansum(np.nan_to_num(pcode_w, nan=0.0)) > 0:
            # Use only finite pairs
            m = np.isfinite(pcode_r2) & np.isfinite(pcode_w)
            if m.any():
                avg_loc_r2_w = float(np.average(pcode_r2[m], weights=pcode_w[m]))
            else:
                avg_loc_r2_w = np.nan
        else:
            avg_loc_r2_w = np.nan

        # 3) Aggregate national time series: True vs Model
        # True: area-weighted observed yield per year
        def _wavg_true(gx: pd.DataFrame) -> float:
            a = pd.to_numeric(gx.get("area"), errors="coerce")
            yv = pd.to_numeric(gx.get("yield"), errors="coerce")
            m = np.isfinite(a) & np.isfinite(yv)
            denom = a[m].sum()
            return float((yv[m] * a[m]).sum() / denom) if denom > 0 else np.nan

        # Model: weighted by crop_area_ha_adjusted if present, else area
        model_weight_col = "crop_area_ha_adjusted" if "crop_area_ha_adjusted" in g.columns else "area"

        def _wavg_model(gx: pd.DataFrame) -> float:
            w = pd.to_numeric(gx.get(model_weight_col), errors="coerce")
            yhat = pd.to_numeric(gx.get("pred_yield"), errors="coerce")
            m = np.isfinite(w) & np.isfinite(yhat)
            denom = w[m].sum()
            return float((yhat[m] * w[m]).sum() / denom) if denom > 0 else np.nan

        true_ts = (
            g.groupby("year", dropna=False).apply(_wavg_true).rename("True").reset_index()
        )
        model_ts = (
            g.groupby("year", dropna=False).apply(_wavg_model).rename("Model").reset_index()
        )
        comp_ts = true_ts.merge(model_ts, on="year", how="outer").sort_values("year")
        agg_r2 = _r2(comp_ts["True"].values, comp_ts["Model"].values)
        agg_corr = _pearson(comp_ts["True"].values, comp_ts["Model"].values)

        # Detrended national time series using relative deviation from each series' own trend
        x = pd.to_numeric(comp_ts["year"], errors="coerce").values

        def _rel_dev(xv: np.ndarray, series: np.ndarray) -> np.ndarray:
            m = np.isfinite(xv) & np.isfinite(series)
            if m.sum() >= 2:
                try:
                    slope, intercept = np.polyfit(xv[m], series[m], 1)
                    trend = slope * xv + intercept
                except Exception:
                    mu = np.nanmean(series[m])
                    trend = np.full_like(series, fill_value=mu, dtype=float)
            else:
                mu = np.nanmean(series[m]) if m.any() else np.nan
                trend = np.full_like(series, fill_value=mu, dtype=float)
            denom = np.where(np.abs(trend) > 1e-9, trend, np.nan)
            return (series - trend) / denom

        d_true = _rel_dev(x, pd.to_numeric(comp_ts["True"], errors="coerce").values)
        d_model = _rel_dev(x, pd.to_numeric(comp_ts["Model"], errors="coerce").values)
        agg_detrended_r2 = _r2(d_true, d_model)

        rows.append(
            {
                "country": country,
                "n_points": int(len(g)),
                "R2": r2_overall,
                "average_location_r2": avg_loc_r2,
                "average_location_area_adjusted_r2": avg_loc_r2_w,
                "aggregate_to_ground_r2": agg_r2,
                "aggregate_to_ground_corr": agg_corr,
                "aggregate_detrended_to_ground_r2": agg_detrended_r2,
            }
        )
    return pd.DataFrame(rows)


def plot_country_r2_barplot(country_r2: pd.DataFrame, out_dir: Path):
    """Render multi-subplot bar charts for multiple country-level metrics.

    Subplots include (if available):
    - R2
    - average_location_r2
    - average_location_area_adjusted_r2
    - aggregate_to_ground_r2
    - aggregate_detrended_to_ground_r2
    - aggregate_to_ground_corr
    """
    metrics = [
        ("R2", "R2 (Yield vs Predicted)", "Country-level R2"),
        ("average_location_r2", "R2", "Avg. per-location R2"),
        ("average_location_area_adjusted_r2", "R2", "Area-adjusted avg. per-location R2"),
        ("aggregate_to_ground_r2", "R2", "Aggregate to ground R2"),
        ("aggregate_detrended_to_ground_r2", "R2", "Aggregate detrended to ground R2"),
        ("aggregate_to_ground_corr", "Pearson r", "Aggregate to ground correlation"),
    ]

    # Keep only metrics present in df
    metrics = [m for m in metrics if m[0] in country_r2.columns]
    if not metrics:
        return

    n_metrics = len(metrics)
    cols = 2
    rows = int(np.ceil(n_metrics / cols))
    n_countries = country_r2["country"].nunique()
    height_per_subplot = max(2.8, 0.35 * n_countries)
    fig_width = 11 * cols
    fig_height = height_per_subplot * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=True)

    for idx, (col, xlabel, title) in enumerate(metrics):
        ax = axes[idx // cols, idx % cols]
        data = country_r2.dropna(subset=[col]).sort_values(col, ascending=True)
        if data.empty:
            ax.axis('off')
            continue
        y_pos = np.arange(len(data))
        ax.barh(y_pos, data[col], color=sns.color_palette("Blues", n_colors=max(3, len(data)))[-1])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["country"])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        avg_val = float(np.nanmean(country_r2[col]))
        ax.axvline(avg_val, color="red", linestyle="--", linewidth=1.2, label=f"Average: {avg_val:.2f}")
        for i, v in enumerate(data[col].values):
            if np.isfinite(v):
                ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
        # X limits
        if col == "aggregate_to_ground_corr":
            left = min(-1.0, float(np.nanmin(data[col])) - 0.05)
            right = 1.0
        else:
            left = min(-0.1, float(np.nanmin(data[col])) - 0.05)
            right = 1.0
        ax.set_xlim(left, right)
        ax.legend(loc="lower right")

    # Hide any unused axes
    for j in range(idx + 1, rows * cols):
        axes[j // cols, j % cols].axis('off')

    fig.savefig(out_dir / "country_r2_barplot.png", dpi=200)
    plt.close(fig)


def compute_national_trend_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-country linear trend of national_yield over years (admin0 proxy).

    Uses country-year means from the provided dataframe to avoid extra I/O.
    Returns a DataFrame with columns ['country','year','national_trend'].
    """
    # Aggregate to country-year to avoid repeated admin-2 rows
    df = df.copy()
    df["country"] = df["country"].astype(str).str.strip()
    agg = df.groupby(["country", "year"], dropna=False)["national_yield"].mean().reset_index()
    rows = []
    for country, g in agg.groupby("country", dropna=False):
        g = g.dropna(subset=["year", "national_yield"]).sort_values("year")
        if len(g) < 2:
            continue
        years = g["year"].astype(float).values
        ny = g["national_yield"].astype(float).values
        try:
            slope, intercept = np.polyfit(years, ny, 1)
            yhat = slope * years + intercept
        except Exception:
            yhat = np.full_like(ny, fill_value=np.nanmean(ny), dtype=float)
        out = pd.DataFrame({
            "country": g["country"].values,
            "year": g["year"].values,
            "national_trend": yhat,
        })
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["country", "year", "national_trend"])
    return pd.concat(rows, ignore_index=True)


def compute_national_trend_from_admin0(admin0_csv_path: Path) -> pd.DataFrame:
    """Compute per-country linear trend using the admin0 CSV directly.

    Robust to schema variations similar to add_national_yield; returns
    ['country','year','national_trend'].
    """
    nat = pd.read_csv(admin0_csv_path)
    colmap = {c.lower(): c for c in nat.columns}
    def has(name: str) -> bool:
        return name in colmap

    if has("yield"):
        nat_y = nat[colmap["yield"]]
    elif has("production") and has("area"):
        nat_y = nat[colmap["production"]] / nat[colmap["area"]]
    elif has("yield_recalc"):
        nat_y = nat[colmap["yield_recalc"]]
    else:
        # No usable columns
        return pd.DataFrame(columns=["country", "year", "national_trend"])

    country_col = colmap.get("country", "country")
    year_col = colmap.get("year", "year")
    tmp = pd.DataFrame({
        "country": nat[country_col].astype(str),
        "year": pd.to_numeric(nat[year_col], errors="coerce"),
        "national_yield": pd.to_numeric(nat_y, errors="coerce"),
    }).dropna(subset=["year", "national_yield"])
    tmp["year"] = tmp["year"].astype(float)

    rows = []
    for country, g in tmp.groupby("country", dropna=False):
        g = g.sort_values("year")
        if len(g) < 2:
            continue
        try:
            slope, intercept = np.polyfit(g["year"].values, g["national_yield"].values, 1)
            yhat = slope * g["year"].values + intercept
        except Exception:
            y = g["national_yield"].values
            yhat = np.full_like(y, fill_value=np.nanmean(y), dtype=float)
        out = pd.DataFrame({
            "country": g["country"].values,
            "year": g["year"].astype(int).values,
            "national_yield": g["national_yield"].values,
            "national_trend": yhat,
        })
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["country", "year", "national_trend"])
    return pd.concat(rows, ignore_index=True)

def plot_residual_heatmaps(df: pd.DataFrame, out_dir: Path):
    """Render heatmaps of observed vs predicted residuals per country.

    Residuals are computed as value - national_trend at the country-year level.
    Left: observed residuals; Right: predicted residuals. Annotated with 1-decimals.
    """
    countries = sorted(df["country"].dropna().unique())
    for country in countries:
        g = df[df["country"] == country].copy()
        obs = g.pivot_table(index="PCODE", columns="year", values="obs_residual", aggfunc="mean")
        pred = g.pivot_table(index="PCODE", columns="year", values="pred_residual", aggfunc="mean")
        obs, pred = obs.align(pred, join="inner", axis=0)
        obs, pred = obs.align(pred, join="inner", axis=1)
        if obs.shape[0] < 2 or obs.shape[1] < 2:
            continue
        vmax = np.nanmax(np.abs([obs.values, pred.values]))
        if not np.isfinite(vmax) or vmax == 0:
            continue
        fig_height = max(4, obs.shape[0] * 0.15)
        fig, axes = plt.subplots(1, 2, figsize=(14, fig_height), sharey=True)
        sns.heatmap(
            obs,
            ax=axes[0],
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Residual (yield - trend)"},
        )
        axes[0].set_title(f"Observed residuals — {country}")
        sns.heatmap(
            pred,
            ax=axes[1],
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Residual (yield - trend)"},
        )
        axes[1].set_title(f"Predicted residuals — {country}")
        plt.tight_layout()
        fig.savefig(out_dir / f"residuals" / f"residual_heatmaps_{country.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)


def plot_country_national_timeseries(df: pd.DataFrame, out_dir: Path):
    """Plot FAO vs area-weighted ground vs area-weighted model yields per country.

    - FAO series: loaded from admin0 CSV file with 'yield' column
    - Ground series (True): area-weighted mean using 'area'
    - Model series: weighted mean using 'crop_area_ha_adjusted' if present, else 'area'
    Adds a linear trend line for FAO.
    """
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country", "year"])  # keep only valid keys

    # Load actual FAO data from admin0 CSV
    admin0_path = Path("all_data/full_features/all_admin0_merged_data_GLAM.csv")
    if admin0_path.exists():
        fao_df = pd.read_csv(admin0_path)
        fao_df["year"] = pd.to_numeric(fao_df["year"], errors="coerce")
        
        # Filter to only include countries that are in the results data
        results_countries = set(df["country"].unique())
        fao_df = fao_df[fao_df["country"].isin(results_countries)]
        
        fao = fao_df[["country", "year", "yield"]].rename(columns={"yield": "FAO"})
        print(f"Loaded FAO data from {admin0_path.name}: {len(fao)} records for {len(fao['country'].unique())} countries")
    else:
        # Fallback to national_yield from results CSV if admin0 not available
        print("Admin0 CSV not found, using national_yield from results CSV")
        fao = (
            df.groupby(["country", "year"], dropna=False)["national_yield"].mean().rename("FAO").reset_index()
        )

    # Ground (True): area-weighted observed yield
    def _wavg_true(g: pd.DataFrame) -> float:
        a = pd.to_numeric(g.get("area"), errors="coerce")
        y = pd.to_numeric(g.get("yield"), errors="coerce")
        m = np.isfinite(a) & np.isfinite(y)
        denom = a[m].sum()
        return float((y[m] * a[m]).sum() / denom) if denom > 0 else np.nan

    true_series = (
        df.groupby(["country", "year"], dropna=False).apply(_wavg_true).rename("True").reset_index()
    )

    # Model: weighted by crop_area_ha_adjusted if available, else area
    model_weight_col = "crop_area_ha_adjusted" if "crop_area_ha_adjusted" in df.columns else "area"

    def _wavg_model(g: pd.DataFrame) -> float:
        w = pd.to_numeric(g.get(model_weight_col), errors="coerce")
        yhat = pd.to_numeric(g.get("pred_yield"), errors="coerce")
        m = np.isfinite(w) & np.isfinite(yhat)
        denom = w[m].sum()
        return float((yhat[m] * w[m]).sum() / denom) if denom > 0 else np.nan

    model_series = (
        df.groupby(["country", "year"], dropna=False).apply(_wavg_model).rename("Model").reset_index()
    )

    # Merge to single table
    comp = fao.merge(true_series, on=["country", "year"], how="outer").merge(
        model_series, on=["country", "year"], how="outer"
    )

    countries = sorted(comp["country"].dropna().unique())
    if not countries:
        return

    cols = 3
    rows = int(np.ceil(len(countries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.2 * rows), squeeze=False)

    for idx, country in enumerate(countries):
        ax = axes[idx // cols, idx % cols]
        g = comp[comp["country"] == country].sort_values("year")
        # Plot series
        ax.plot(g["year"], g["FAO"], color="black", linewidth=2, marker="D", markersize=6, label="FAO")
        ax.plot(g["year"], g["Model"], color="red", linestyle="--", linewidth=2, marker="o", markersize=5, label="Model")
        ax.plot(g["year"], g["True"], color="blue", linewidth=2, marker="s", markersize=5, label="True")

        # Trend line (FAO)
        x = pd.to_numeric(g["year"], errors="coerce").values
        y = pd.to_numeric(g["FAO"], errors="coerce").values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            slope, intercept = np.polyfit(x[m], y[m], 1)
            ax.plot(x[m], slope * x[m] + intercept, color="grey", linestyle=":", linewidth=1.2, label="Trend")

        ax.set_title(country)
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield (t/ha)")
        ax.grid(True, alpha=0.2)

    # Legend once
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "country_national_timeseries.png", dpi=200)
    plt.close(fig)


def plot_country_national_timeseries_detrended(df: pd.DataFrame, out_dir: Path):
    """Plot detrended national series (relative deviation from each series' own trend).

    - Series: FAO, True (area-weighted ground), Model (weighted by crop_area_ha_adjusted if present else area)
    - For each series and country, fit a linear regression over years, compute relative deviation: (value - trend) / trend
    - Same structure/colors/markers as plot_country_national_timeseries
    """
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["country", "year"])  # keep only valid keys

    # Load FAO data as in the original plot
    admin0_path = Path("all_data/full_features/all_admin0_merged_data_GLAM.csv")
    if admin0_path.exists():
        fao_df = pd.read_csv(admin0_path)
        fao_df["year"] = pd.to_numeric(fao_df["year"], errors="coerce")
        results_countries = set(df["country"].unique())
        fao_df = fao_df[fao_df["country"].isin(results_countries)]
        fao = fao_df[["country", "year", "yield"]].rename(columns={"yield": "FAO"})
        print(
            f"Loaded FAO data for detrended plot from {admin0_path.name}: {len(fao)} records for {len(fao['country'].unique())} countries"
        )
    else:
        print("Admin0 CSV not found, using national_yield from results CSV for detrended plot")
        fao = (
            df.groupby(["country", "year"], dropna=False)["national_yield"].mean().rename("FAO").reset_index()
        )

    # Ground (True): area-weighted observed yield
    def _wavg_true(g: pd.DataFrame) -> float:
        a = pd.to_numeric(g.get("area"), errors="coerce")
        y = pd.to_numeric(g.get("yield"), errors="coerce")
        m = np.isfinite(a) & np.isfinite(y)
        denom = a[m].sum()
        return float((y[m] * a[m]).sum() / denom) if denom > 0 else np.nan

    true_series = (
        df.groupby(["country", "year"], dropna=False).apply(_wavg_true).rename("True").reset_index()
    )

    # Model: weighted by crop_area_ha_adjusted if available, else area
    model_weight_col = "crop_area_ha_adjusted" if "crop_area_ha_adjusted" in df.columns else "area"

    def _wavg_model(g: pd.DataFrame) -> float:
        w = pd.to_numeric(g.get(model_weight_col), errors="coerce")
        yhat = pd.to_numeric(g.get("pred_yield"), errors="coerce")
        m = np.isfinite(w) & np.isfinite(yhat)
        denom = w[m].sum()
        return float((yhat[m] * w[m]).sum() / denom) if denom > 0 else np.nan

    model_series = (
        df.groupby(["country", "year"], dropna=False).apply(_wavg_model).rename("Model").reset_index()
    )

    # Merge to single table
    comp = fao.merge(true_series, on=["country", "year"], how="outer").merge(
        model_series, on=["country", "year"], how="outer"
    )

    countries = sorted(comp["country"].dropna().unique())
    if not countries:
        return

    cols = 3
    rows = int(np.ceil(len(countries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.2 * rows), squeeze=False)

    for idx, country in enumerate(countries):
        ax = axes[idx // cols, idx % cols]
        g = comp[comp["country"] == country].sort_values("year")

        x = pd.to_numeric(g["year"], errors="coerce").values

        def _rel_dev(series: pd.Series) -> np.ndarray:
            s = pd.to_numeric(series, errors="coerce").values
            m = np.isfinite(x) & np.isfinite(s)
            if m.sum() >= 2:
                try:
                    slope, intercept = np.polyfit(x[m], s[m], 1)
                    trend = slope * x + intercept
                except Exception:
                    mu = np.nanmean(s[m])
                    trend = np.full_like(s, fill_value=mu, dtype=float)
            else:
                mu = np.nanmean(s[m]) if m.any() else np.nan
                trend = np.full_like(s, fill_value=mu, dtype=float)
            denom = np.where(np.abs(trend) > 1e-9, trend, np.nan)
            return (s - trend) / denom

        rel_fao = _rel_dev(g["FAO"]) if "FAO" in g.columns else np.full(len(g), np.nan)
        rel_model = _rel_dev(g["Model"]) if "Model" in g.columns else np.full(len(g), np.nan)
        rel_true = _rel_dev(g["True"]) if "True" in g.columns else np.full(len(g), np.nan)

        # Plot relative deviations with same style/colors
        ax.plot(g["year"], rel_fao, color="black", linewidth=2, marker="D", markersize=6, label="FAO")
        ax.plot(g["year"], rel_model, color="red", linestyle="--", linewidth=2, marker="o", markersize=5, label="Model")
        ax.plot(g["year"], rel_true, color="blue", linewidth=2, marker="s", markersize=5, label="True")

        # Zero baseline
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.2)

        ax.set_title(country)
        ax.set_xlabel("Year")
        ax.set_ylabel("Relative deviation from trend")
        ax.grid(True, alpha=0.2)

    # Legend once
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "country_national_timeseries_detrended.png", dpi=200)
    plt.close(fig)

def derive_output_dir_from_results(csv_path: Path) -> Path:
    stem_lower = csv_path.stem
    prefix = "results_"
    if stem_lower.lower().startswith(prefix):
        model_name = csv_path.stem[len(prefix):]
    else:
        model_name = csv_path.stem
    # Subfolders for organization
    base = (csv_path.parent.parent / model_name).resolve()
    (base / "plots").mkdir(parents=True, exist_ok=True)
    (base / "metrics").mkdir(parents=True, exist_ok=True)
    # Prepare subfolders for classification
    (base / "plots" / "anomaly").mkdir(parents=True, exist_ok=True)
    (base / "plots" / "yield").mkdir(parents=True, exist_ok=True)
    (base / "plots" / "residuals").mkdir(parents=True, exist_ok=True)
    return base


def run_analysis(results_csv: Path, admin0_csv: Optional[Path] = None, out_dir: Optional[Path] = None) -> None:
    results_csv = Path(results_csv).resolve()
    if admin0_csv is None:
        admin0_csv = Path(__file__).resolve().parent / "all_data" / "full_features" / "all_admin0_merged_data_GLAM.csv"
    else:
        admin0_csv = Path(admin0_csv).resolve()

    if out_dir is None:
        out_dir = derive_output_dir_from_results(results_csv)
    else:
        out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_csv}")
    df = load_results(results_csv)
    # Normalize country formatting early for robust joins
    df["country"] = df["country"].astype(str).str.strip()
    df = add_national_yield(df, admin0_csv)
    df = compute_relative_anomalies(df)

    print("Computing per-admin variability metrics...")
    admin_metrics = per_admin_metrics(df)
    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    admin_metrics.to_csv(metrics_dir / "per_admin_metrics.csv", index=False)

    print("Computing yearly spatial skill...")
    spatial_df = per_year_spatial_skill(df)
    spatial_df.to_csv(metrics_dir / "per_year_spatial_skill.csv", index=False)

    print("Summarizing by country...")
    summary_df = summarize_countries(admin_metrics)
    summary_df.to_csv(metrics_dir / "country_summary.csv", index=False)

    print("Rendering plots...")
    plot_acc_var_quadrants(admin_metrics, plots_dir)
    plot_distributions(admin_metrics, plots_dir)
    plot_spatial_skill_lines(spatial_df, plots_dir)
    plot_anomaly_heatmaps(df, plots_dir)
    plot_yield_heatmaps(df, plots_dir)

    # Country-level R2
    print("Computing and plotting country-level R2...")
    country_r2 = compute_country_r2_scores(df)
    country_r2.to_csv(metrics_dir / "country_r2.csv", index=False)
    plot_country_r2_barplot(country_r2, plots_dir)

    # Residual plots (observed/predicted minus national trend)
    print("Computing residuals and rendering residual heatmaps...")
    
    # Check if national_trend already exists in df
    if "national_trend" in df.columns:
        print("Using existing national_trend column from results CSV")
        df_res = df.copy()
        # Ensure consistent data types
        df_res["country"] = df_res["country"].astype(str).str.strip()
        df_res["year"] = pd.to_numeric(df_res["year"], errors="coerce").astype("Int64")
        missing_ratio = df_res["national_trend"].isna().mean()
        print(f"National trend coverage: {100*(1-missing_ratio):.1f}%")
        
        if missing_ratio >= 1.0:
            print("Warning: national_trend column is empty. Skipping residual heatmaps.")
        else:
            df_res["obs_residual"] = df_res["yield"] - df_res["national_trend"]
            df_res["pred_residual"] = df_res["pred_yield"] - df_res["national_trend"]
            # Debug stats
            n_countries = df_res[~df_res["obs_residual"].isna()]["country"].nunique()
            print(f"Residuals computed for {n_countries} countries")
            plot_residual_heatmaps(df_res, plots_dir)
    else:
        # Compute national trend from admin0-derived national_yield attached to df
        trend_df = compute_national_trend_from_df(df)
        if trend_df.empty or "national_trend" not in trend_df.columns:
            print("Warning: Could not compute national trend (df-derived). Skipping residual heatmaps.")
        else:
            # Ensure dtypes align for merge
            trend_df["country"] = trend_df["country"].astype(str).str.strip()
            trend_df["year"] = pd.to_numeric(trend_df["year"], errors="coerce").astype("Int64")
            df_merge = df.copy()
            df_merge["country"] = df_merge["country"].astype(str).str.strip()
            df_merge["year"] = pd.to_numeric(df_merge["year"], errors="coerce").astype("Int64")
            # Save trend series for transparency
            trend_df.to_csv(metrics_dir / "national_trend.csv", index=False)
            df_res = df_merge.merge(trend_df, on=["country", "year"], how="left")
            missing_ratio = df_res["national_trend"].isna().mean()
            print(f"Trend coverage after merge: {100*(1-missing_ratio):.1f}%")
            if missing_ratio >= 1.0:
                print("Warning: national_trend missing after merge. Skipping residual heatmaps.")
            else:
                df_res["obs_residual"] = df_res["yield"] - df_res["national_trend"]
                df_res["pred_residual"] = df_res["pred_yield"] - df_res["national_trend"]
                # Debug stats
                n_countries = df_res[~df_res["obs_residual"].isna()]["country"].nunique()
                print(f"Residuals computed for {n_countries} countries")
                plot_residual_heatmaps(df_res, plots_dir)

    # Country-level national timeseries comparison (FAO vs True vs Model)
    print("Rendering national timeseries comparison plots...")
    plot_country_national_timeseries(df, plots_dir)
    # Detrended national timeseries (relative deviations from each series' trend)
    plot_country_national_timeseries_detrended(df, plots_dir)

    print("Analysis complete. Outputs saved to:", out_dir)


def main():
    # Simple entrypoint: set RESULTS_CSV here, or select via file dialog if left as None
    file_path = None
    RESULTS_CSV: Optional[Path] = None

    selected_csv = None
    if RESULTS_CSV is not None:
        selected_csv = Path(RESULTS_CSV)
    else:
        # Try an interactive file picker if available
        try:
            import tkinter as tk  # type: ignore
            from tkinter import filedialog  # type: ignore

            root = tk.Tk()
            root.withdraw()
            selected_csv = filedialog.askopenfilename(
                title="Select results CSV",
                filetypes=[("CSV files", "results_*.csv"), ("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=str(Path(__file__).resolve().parent),
            )
            if not selected_csv:
                raise SystemExit("No results CSV selected. Set RESULTS_CSV in the script or select a file when prompted.")
            selected_csv = Path(selected_csv)
        except Exception:
            raise SystemExit(
                "No results CSV configured and file dialog unavailable. Set RESULTS_CSV at the top of main()."
            )

    run_analysis(Path(selected_csv))


if __name__ == "__main__":
    main()


