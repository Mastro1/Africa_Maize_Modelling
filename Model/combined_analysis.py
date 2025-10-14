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
    """Load results CSV with required columns."""
    df = pd.read_csv(csv_path)
    required = [
        "country", "PCODE", "year", "area", "production", "yield", "pred_yield"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in results CSV")

    if "admin_2" not in df.columns and "admin2" in df.columns:
        df = df.rename(columns={"admin2": "admin_2"})

    return df


def add_national_yield(df: pd.DataFrame, admin0_csv_path: Path) -> pd.DataFrame:
    """Attach national yield from admin0 file."""
    nat = pd.read_csv(admin0_csv_path)
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

    country_col = colmap.get("country", "country")
    year_col = colmap.get("year", "year")
    nat = nat[[country_col, year_col, "national_yield"]].rename(columns={country_col: "country", year_col: "year"})

    nat["year"] = pd.to_numeric(nat["year"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    nat["country"] = nat["country"].astype(str)
    df["country"] = df["country"].astype(str)

    out = df.merge(nat, on=["country", "year"], how="left")

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
    """Compute per-location relative z-scores."""
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
    """Safe correlation computation."""
    try:
        if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan, np.nan
        r, p = pearsonr(x, y)
        return r, p
    except Exception:
        return np.nan, np.nan


def _safe_spearman(x: np.ndarray, y: np.ndarray):
    """Safe Spearman correlation computation."""
    try:
        if len(x) < 3:
            return np.nan, np.nan
        r, p = spearmanr(x, y)
        return r, p
    except Exception:
        return np.nan, np.nan


def per_admin_metrics(df: pd.DataFrame, weight_col: str = "area") -> pd.DataFrame:
    """Compute comprehensive per-admin metrics including anomaly correlation and extreme events."""
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
            temporal_r = temporal_p = hit_rate_20pct = false_alarm_rate_20pct = np.nan
        else:
            acc, acc_p = _safe_corr(y, p)
            rho, rho_p = _safe_spearman(y, p)
            std_y = np.std(y, ddof=1)
            std_p = np.std(p, ddof=1)
            var_ratio = std_p / std_y if std_y > 0 else np.nan
            mse = np.mean((p - y) ** 2)
            nrmse = np.sqrt(mse) / (std_y if std_y > 0 else 1.0)
            denom = np.sum((y - np.mean(y)) ** 2)
            r2_z = 1.0 - (np.sum((y - p) ** 2) / denom) if denom > 0 else np.nan
            skill = 1.0 - (mse / (std_y ** 2 if std_y > 0 else np.nan))

            # Trend consistency
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

            # Extreme detection
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

            # Temporal correlation (yield vs pred_yield)
            y_yield = g["yield"].values
            p_yield = g["pred_yield"].values
            n_yield = np.isfinite(y_yield) & np.isfinite(p_yield)
            y_yield_clean = y_yield[n_yield]
            p_yield_clean = p_yield[n_yield]
            if len(y_yield_clean) >= 3:
                temporal_r, temporal_p = _safe_corr(y_yield_clean, p_yield_clean)
            else:
                temporal_r = temporal_p = np.nan

            # Extreme events detection (20th percentile on yields)
            if len(y_yield_clean) >= 5:
                obs_threshold = np.percentile(y_yield_clean, 20)
                pred_threshold = np.percentile(p_yield_clean, 20)
                obs_extreme = y_yield_clean <= obs_threshold
                pred_extreme = p_yield_clean <= pred_threshold
                hits = np.sum(obs_extreme & pred_extreme)
                misses = np.sum(obs_extreme & ~pred_extreme)
                false_alarms = np.sum(~obs_extreme & pred_extreme)
                hit_rate_20pct = hits / (hits + misses) if (hits + misses) > 0 else np.nan
                false_alarm_rate_20pct = false_alarms / (false_alarms + np.sum(~obs_extreme & ~pred_extreme)) if (false_alarms + np.sum(~obs_extreme & ~pred_extreme)) > 0 else np.nan
            else:
                hit_rate_20pct = false_alarm_rate_20pct = np.nan

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
                "temporal_r": temporal_r,
                "temporal_p": temporal_p,
                "hit_rate_20pct": hit_rate_20pct,
                "false_alarm_rate_20pct": false_alarm_rate_20pct,
            }
        )
    return pd.DataFrame(rows)


def per_year_spatial_skill(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly spatial correlation across admins."""
    rows = []
    for (country, year), g in df.groupby(["country", "year"], dropna=False):
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


def compute_temporal_r_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal correlation at national level."""
    rows = []

    for country, g in df.groupby("country", dropna=False):
        years = sorted(g["year"].unique())

        obs_national = []
        pred_national = []

        for year in years:
            year_data = g[g["year"] == year]

            obs_weight = year_data["area"]  # Use area for ground data comparison
            obs_yield = year_data["yield"]
            valid_obs = np.isfinite(obs_weight) & np.isfinite(obs_yield)

            pred_weight = obs_weight
            pred_yield = year_data["pred_yield"]
            valid_pred = np.isfinite(pred_weight) & np.isfinite(pred_yield)

            # Only compute national averages if we have valid data for both observed and predicted
            if valid_obs.sum() > 0 and valid_pred.sum() > 0:
                obs_national_val = np.average(obs_yield[valid_obs], weights=obs_weight[valid_obs])
                pred_national_val = np.average(pred_yield[valid_pred], weights=pred_weight[valid_pred])

                if np.isfinite(obs_national_val) and np.isfinite(pred_national_val):
                    obs_national.append(obs_national_val)
                    pred_national.append(pred_national_val)

        if len(obs_national) >= 3:
            temporal_r, temporal_p = _safe_corr(np.array(obs_national), np.array(pred_national))
        else:
            temporal_r = temporal_p = np.nan

        rows.append({
            "country": country,
            "n_years": len(years),
            "temporal_r_national": temporal_r,
            "temporal_p_national": temporal_p,
        })

    return pd.DataFrame(rows)


def compute_extreme_events_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extreme event detection at national level."""
    rows = []

    for country, g in df.groupby("country", dropna=False):
        years = sorted(g["year"].unique())

        obs_national = []
        pred_national = []

        for year in years:
            year_data = g[g["year"] == year]

            obs_weight = year_data["area"]  # Use area for ground data comparison
            obs_yield = year_data["yield"]
            valid_obs = np.isfinite(obs_weight) & np.isfinite(obs_yield)

            if valid_obs.sum() > 0:
                obs_national_val = np.average(obs_yield[valid_obs], weights=obs_weight[valid_obs])
                obs_national.append(obs_national_val)

            pred_weight = obs_weight
            pred_yield = year_data["pred_yield"]
            valid_pred = np.isfinite(pred_weight) & np.isfinite(pred_yield)

            if valid_pred.sum() > 0:
                pred_national_val = np.average(pred_yield[valid_pred], weights=pred_weight[valid_pred])
                pred_national.append(pred_national_val)

        if len(obs_national) >= 5:
            obs_threshold = np.percentile(obs_national, 20)
            pred_threshold = np.percentile(pred_national, 20)

            obs_extreme = np.array(obs_national) <= obs_threshold
            pred_extreme = np.array(pred_national) <= pred_threshold

            hits = np.sum(obs_extreme & pred_extreme)
            misses = np.sum(obs_extreme & ~pred_extreme)
            false_alarms = np.sum(~obs_extreme & pred_extreme)

            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else np.nan
            false_alarm_rate = false_alarms / (false_alarms + np.sum(~obs_extreme & ~pred_extreme)) if (false_alarms + np.sum(~obs_extreme & ~pred_extreme)) > 0 else np.nan
        else:
            hit_rate = false_alarm_rate = np.nan

        rows.append({
            "country": country,
            "n_years": len(years),
            "hit_rate_20pct_national": hit_rate,
            "false_alarm_rate_20pct_national": false_alarm_rate,
        })

    return pd.DataFrame(rows)


def summarize_countries(admin_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics by country with weighted means."""
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
                    "temporal_r_wmean": wmean(g["temporal_r"], g["weight"]),
                    "hit_rate_20pct_wmean": wmean(g["hit_rate_20pct"], g["weight"]),
                }
            )
        )
    ).reset_index()
    return summary


def compute_country_r2_scores(df: pd.DataFrame, admin_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute multiple country-level skill summaries."""
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
        # Overall R2
        r2_overall = _r2(g["yield"].values, g["pred_yield"].values)

        # Per-PCODE R2 averages
        pcode_r2_list = []
        pcode_weight_list = []
        for pcode, gp in g.groupby("PCODE", dropna=False):
            r2_pcode = _r2(gp["yield"].values, gp["pred_yield"].values)
            pcode_r2_list.append(r2_pcode)
            w = pd.to_numeric(gp.get("area"), errors="coerce").mean()
            pcode_weight_list.append(w)

        pcode_r2 = np.array(pcode_r2_list, dtype=float)
        pcode_w = np.array(pcode_weight_list, dtype=float)

        avg_loc_r2 = float(np.nanmean(pcode_r2)) if pcode_r2.size > 0 else np.nan

        if pcode_r2.size > 0 and np.isfinite(pcode_w).any() and np.nansum(np.nan_to_num(pcode_w, nan=0.0)) > 0:
            m = np.isfinite(pcode_r2) & np.isfinite(pcode_w)
            if m.any():
                avg_loc_r2_w = float(np.average(pcode_r2[m], weights=pcode_w[m]))
            else:
                avg_loc_r2_w = np.nan
        else:
            avg_loc_r2_w = np.nan

        # Aggregate national time series
        def _wavg_true(gx: pd.DataFrame) -> float:
            a = pd.to_numeric(gx.get("area"), errors="coerce")
            yv = pd.to_numeric(gx.get("yield"), errors="coerce")
            m = np.isfinite(a) & np.isfinite(yv)
            denom = a[m].sum()
            return float((yv[m] * a[m]).sum() / denom) if denom > 0 else np.nan

        model_weight_col = "area"  # Use area for ground data comparison

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

        # Detrended national time series
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

        # Median temporal_r from per-admin metrics
        country_admin_metrics = admin_metrics[admin_metrics["country"] == country]
        median_temporal_r = country_admin_metrics["temporal_r"].median() if not country_admin_metrics.empty else np.nan

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
                "median_temporal_r": median_temporal_r,
            }
        )
    return pd.DataFrame(rows)


# Plotting functions
def plot_acc_var_quadrants(admin_metrics: pd.DataFrame, out_dir: Path):
    """Plot ACC vs VAR ratio quadrants."""
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
    """Plot metric distributions by country."""
    melted = admin_metrics.melt(id_vars=["country"], value_vars=["ACC", "VAR_ratio", "nRMSE", "Skill"], var_name="metric", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="country", y="value", hue="metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "metric_distributions_by_country.png", dpi=200)
    plt.close()


def plot_spatial_skill_lines(spatial_df: pd.DataFrame, out_dir: Path):
    """Plot yearly spatial skill lines."""
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


def plot_temporal_r_metrics(admin_metrics: pd.DataFrame, out_dir: Path):
    """Create plots for temporal r metrics."""
    country_summary = admin_metrics.groupby('country').agg({
        'temporal_r': 'median',
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Temporal Correlation by Country (median)
    ax = axes[0]
    data = country_summary.sort_values('temporal_r', ascending=True)
    y_pos = np.arange(len(data))
    ax.barh(y_pos, data['temporal_r'], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['country'])
    ax.set_xlabel('Temporal Correlation (r)')
    ax.set_title('Temporal Correlation by Country (Median)')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Target: r=0.5')
    ax.axvline(data['temporal_r'].median(), color='green', linestyle=':', linewidth=1.5,
               label=f'Overall Median: {data["temporal_r"].median():.3f}')
    for i, v in enumerate(data['temporal_r'].values):
        if np.isfinite(v):
            ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)
    ax.legend()
    ax.set_xlim(-0.2, 1.0)

    # Plot 2: Distribution of Temporal r Across Locations
    ax = axes[1]
    countries = sorted(admin_metrics['country'].unique())
    data_for_violin = [admin_metrics[admin_metrics['country']==c]['temporal_r'].dropna().values
                      for c in countries]
    parts = ax.violinplot(data_for_violin, positions=range(len(countries)),
                         vert=False, showmedians=True, widths=0.7)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries)
    ax.set_xlabel('Temporal Correlation (r)')
    ax.set_title('Distribution of Temporal r Across Locations')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "temporal_r_analysis.png", dpi=200)
    plt.close(fig)


def plot_extreme_events_metrics(admin_metrics: pd.DataFrame, national_extreme: pd.DataFrame, out_dir: Path):
    """Create plots for extreme event detection metrics."""
    country_summary = admin_metrics.groupby('country').agg({
        'hit_rate_20pct': 'mean',
    }).reset_index()

    national_summary = national_extreme[['country', 'hit_rate_20pct_national']].copy()
    national_summary = national_summary.rename(columns={'hit_rate_20pct_national': 'hit_rate_national'})

    combined = country_summary.merge(national_summary, on='country', how='left')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Disaggregated Hit Rate
    ax = axes[0]
    data = combined.sort_values('hit_rate_20pct', ascending=True)
    y_pos = np.arange(len(data))
    ax.barh(y_pos, 100*data['hit_rate_20pct'], color='coral')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['country'])
    ax.set_xlabel('Hit Rate (%)')
    ax.set_title('Extreme Event Detection (20th Percentile) - Disaggregated')
    ax.axvline(70, color='red', linestyle='--', linewidth=1.5, label='Target: 70%')
    overall_median = 100*data['hit_rate_20pct'].median()
    ax.axvline(overall_median, color='green', linestyle=':', linewidth=1.5,
               label=f'Median: {overall_median:.1f}%')
    for i, v in enumerate(data['hit_rate_20pct'].values):
        if np.isfinite(v):
            ax.text(100*v + 1, i, f'{100*v:.0f}%', va='center', fontsize=8)
    ax.legend()
    ax.set_xlim(0, 100)

    # Plot 2: Aggregated Hit Rate
    ax = axes[1]
    data = combined.dropna(subset=['hit_rate_national']).sort_values('hit_rate_national', ascending=True)
    if not data.empty:
        y_pos = np.arange(len(data))
        ax.barh(y_pos, 100*data['hit_rate_national'], color='darkorange')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data['country'])
        ax.set_xlabel('Hit Rate (%)')
        ax.set_title('Extreme Event Detection (20th Percentile) - National Aggregate')
        ax.axvline(70, color='red', linestyle='--', linewidth=1.5, label='Target: 70%')
        national_median = 100*data['hit_rate_national'].median()
        ax.axvline(national_median, color='green', linestyle=':', linewidth=1.5,
                   label=f'Median: {national_median:.1f}%')
        for i, v in enumerate(data['hit_rate_national'].values):
            if np.isfinite(v):
                ax.text(100*v + 1, i, f'{100*v:.0f}%', va='center', fontsize=8)
        ax.legend()
        ax.set_xlim(0, 100)

    plt.tight_layout()
    fig.savefig(out_dir / "extreme_events_analysis.png", dpi=200)
    plt.close(fig)


def plot_country_r2_barplot(country_r2: pd.DataFrame, out_dir: Path):
    """Render multi-subplot bar charts for country-level R2 metrics."""
    metrics = [
        ("R2", "R2 (Yield vs Predicted)", "Country-level R2"),
        ("average_location_r2", "R2", "Avg. per-location R2"),
        ("average_location_area_adjusted_r2", "R2", "Area-adjusted avg. per-location R2"),
        ("aggregate_to_ground_r2", "R2", "Aggregate to ground R2"),
        ("aggregate_detrended_to_ground_r2", "R2", "Aggregate detrended to ground R2"),
        ("aggregate_to_ground_corr", "Pearson r", "Aggregate to ground correlation"),
        ("median_temporal_r", "Pearson r", "Median temporal correlation (per location)"),
    ]

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
        if col == "aggregate_to_ground_corr":
            left = min(-1.0, float(np.nanmin(data[col])) - 0.05)
            right = 1.0
        else:
            left = min(-0.1, float(np.nanmin(data[col])) - 0.05)
            right = 1.0
        ax.set_xlim(left, right)
        ax.legend(loc="lower right")

    for j in range(idx + 1, rows * cols):
        axes[j // cols, j % cols].axis('off')

    fig.savefig(out_dir / "country_r2_barplot.png", dpi=200)
    plt.close(fig)


def plot_yield_heatmaps(df: pd.DataFrame, out_dir: Path):
    """Render heatmaps of observed vs predicted yields."""
    countries = sorted(df["country"].dropna().unique())

    yield_dir = out_dir / "yield_comparison"
    yield_dir.mkdir(exist_ok=True)

    for country in countries:
        g = df[df["country"] == country].copy()

        obs = g.pivot_table(index="PCODE", columns="year", values="yield", aggfunc="mean")
        pred = g.pivot_table(index="PCODE", columns="year", values="pred_yield", aggfunc="mean")

        obs, pred = obs.align(pred, join="inner", axis=0)
        obs, pred = obs.align(pred, join="inner", axis=1)

        if obs.shape[0] < 2 or obs.shape[1] < 2:
            continue

        all_values = np.concatenate([obs.values.flatten(), pred.values.flatten()])
        vmax = np.nanmax(all_values)
        vmin = np.nanmin(all_values)
        if not np.isfinite(vmax) or not np.isfinite(vmin):
            continue

        fig_height = max(4, obs.shape[0] * 0.15)
        fig, axes = plt.subplots(1, 2, figsize=(20, fig_height), sharey=True)

        sns.heatmap(
            obs,
            ax=axes[0],
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Yield (t/ha)"},
        )
        axes[0].set_title(f"Observed Yield — {country}")

        sns.heatmap(
            pred,
            ax=axes[1],
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Yield (t/ha)"},
        )
        axes[1].set_title(f"Predicted Yield — {country}")

        plt.tight_layout()
        fig.savefig(yield_dir / f"yield_comparison_{country.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)


def plot_detrended_yields(df: pd.DataFrame, out_dir: Path):
    """Plot detrended yields showing deviations from linear regression for both observed and predicted."""
    countries = sorted(df["country"].dropna().unique())

    # Use the same yield_comparison subfolder
    yield_dir = out_dir / "yield_comparison"
    yield_dir.mkdir(exist_ok=True)

    for country in countries:
        g = df[df["country"] == country].copy()

        # Calculate detrended yields for each location (both observed and predicted)
        detrended_data_obs = []
        detrended_data_pred = []
        pcodes = g["PCODE"].unique()

        for pcode in pcodes:
            loc_data = g[g["PCODE"] == pcode].sort_values("year")
            if len(loc_data) < 3:
                continue

            years = loc_data["year"].values.astype(float)
            obs_yields = loc_data["yield"].values
            pred_yields = loc_data["pred_yield"].values

            # Fit linear regression for observed yields
            valid_obs = np.isfinite(years) & np.isfinite(obs_yields)
            if valid_obs.sum() >= 3:
                try:
                    slope_obs, intercept_obs = np.polyfit(years[valid_obs], obs_yields[valid_obs], 1)
                    trend_obs = slope_obs * years + intercept_obs
                    detrended_obs = obs_yields - trend_obs

                    for i, year in enumerate(years):
                        if valid_obs[i]:
                            detrended_data_obs.append({
                                "PCODE": pcode,
                                "year": year,
                                "detrended_yield": detrended_obs[i]
                            })
                except:
                    pass

            # Fit linear regression for predicted yields
            valid_pred = np.isfinite(years) & np.isfinite(pred_yields)
            if valid_pred.sum() >= 3:
                try:
                    slope_pred, intercept_pred = np.polyfit(years[valid_pred], pred_yields[valid_pred], 1)
                    trend_pred = slope_pred * years + intercept_pred
                    detrended_pred = pred_yields - trend_pred

                    for i, year in enumerate(years):
                        if valid_pred[i]:
                            detrended_data_pred.append({
                                "PCODE": pcode,
                                "year": year,
                                "detrended_yield": detrended_pred[i]
                            })
                except:
                    pass

        if not detrended_data_obs or not detrended_data_pred:
            continue

        detrended_df_obs = pd.DataFrame(detrended_data_obs)
        detrended_df_pred = pd.DataFrame(detrended_data_pred)

        det_pivot_obs = detrended_df_obs.pivot_table(index="PCODE", columns="year", values="detrended_yield", aggfunc="mean")
        det_pivot_pred = detrended_df_pred.pivot_table(index="PCODE", columns="year", values="detrended_yield", aggfunc="mean")

        # Align rows and columns
        det_pivot_obs, det_pivot_pred = det_pivot_obs.align(det_pivot_pred, join="inner", axis=0)
        det_pivot_obs, det_pivot_pred = det_pivot_obs.align(det_pivot_pred, join="inner", axis=1)

        if det_pivot_obs.shape[0] < 2 or det_pivot_obs.shape[1] < 2:
            continue

        # Shared color scale centered on zero
        all_values = np.concatenate([det_pivot_obs.values.flatten(), det_pivot_pred.values.flatten()])
        vmax = np.nanmax(np.abs(all_values))
        vmin = -vmax

        fig_height = max(4, det_pivot_obs.shape[0] * 0.15)
        fig, axes = plt.subplots(1, 2, figsize=(20, fig_height), sharey=True)

        # Detrended observed yields
        sns.heatmap(
            det_pivot_obs,
            ax=axes[0],
            cmap="RdBu",  # Red (low) to blue (high)
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Detrended Yield Deviation"},
        )
        axes[0].set_title(f"Detrended Observed Yields — {country}")

        # Detrended predicted yields
        sns.heatmap(
            det_pivot_pred,
            ax=axes[1],
            cmap="RdBu",  # Red (low) to blue (high)
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Detrended Yield Deviation"},
        )
        axes[1].set_title(f"Detrended Predicted Yields — {country}")

        plt.tight_layout()
        fig.savefig(yield_dir / f"detrended_yield_comparison_{country.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)


def plot_anomaly_heatmaps(df: pd.DataFrame, out_dir: Path):
    """Render heatmaps of observed vs predicted anomalies."""
    countries = sorted(df["country"].dropna().unique())

    anomaly_dir = out_dir / "anomaly"
    anomaly_dir.mkdir(exist_ok=True)

    for country in countries:
        g = df[df["country"] == country].copy()

        obs = g.pivot_table(index="PCODE", columns="year", values="z_obs")
        pred = g.pivot_table(index="PCODE", columns="year", values="z_pred")

        obs, pred = obs.align(pred, join="inner", axis=0)
        obs, pred = obs.align(pred, join="inner", axis=1)

        if obs.shape[0] < 2 or obs.shape[1] < 2:
            continue

        # Calculate shared color scale
        all_values = np.concatenate([obs.values.flatten(), pred.values.flatten()])
        vmin = np.nanmin(all_values)
        vmax = np.nanmax(all_values)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = -1, 1

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, obs.shape[0] * 0.15)), sharey=True)
        sns.heatmap(
            obs,
            ax=axes[0],
            cmap="RdBu",
            vmax=vmax,
            vmin=vmin,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Anomaly (z)"},
        )
        axes[0].set_title(f"Observed anomalies — {country}")
        sns.heatmap(
            pred,
            ax=axes[1],
            cmap="RdBu",
            vmax=vmax,
            vmin=vmin,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 6},
            cbar_kws={"label": "Anomaly (z)"},
        )
        axes[1].set_title(f"Predicted anomalies — {country}")
        plt.tight_layout()
        fig.savefig(anomaly_dir / f"anomaly_heatmaps_{country.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)


def run_combined_analysis(results_csv: Path, admin0_csv: Optional[Path] = None, out_dir: Optional[Path] = None) -> None:
    """Run the complete combined analysis from both original scripts."""
    results_csv = Path(results_csv).resolve()

    # Default admin0 path
    if admin0_csv is None:
        admin0_csv = Path(__file__).resolve().parent.parent / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat" / "all_admin0_merged_data.csv"
    else:
        admin0_csv = Path(admin0_csv).resolve()

    # Default output directory
    if out_dir is None:
        out_dir = results_csv.parent / "analysis_results"
    else:
        out_dir = Path(out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    metrics_dir = out_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_csv}")
    df = load_results(results_csv)
    df["country"] = df["country"].astype(str).str.strip()

    if admin0_csv.exists():
        df = add_national_yield(df, admin0_csv)
    else:
        print("Warning: Admin0 CSV not found, proceeding without national yield data")

    df = compute_relative_anomalies(df)

    print("Computing per-admin metrics...")
    admin_metrics = per_admin_metrics(df)
    admin_metrics.to_csv(metrics_dir / "per_admin_metrics.csv", index=False)

    print("Computing yearly spatial skill...")
    spatial_df = per_year_spatial_skill(df)
    spatial_df.to_csv(metrics_dir / "per_year_spatial_skill.csv", index=False)

    print("Computing temporal correlations (aggregated)...")
    temporal_agg = compute_temporal_r_aggregated(df)
    temporal_agg.to_csv(metrics_dir / "temporal_r_aggregated.csv", index=False)

    print("Computing extreme events (aggregated)...")
    extreme_agg = compute_extreme_events_aggregated(df)
    extreme_agg.to_csv(metrics_dir / "extreme_events_aggregated.csv", index=False)

    print("Summarizing by country...")
    summary_df = summarize_countries(admin_metrics)
    summary_df.to_csv(metrics_dir / "country_summary.csv", index=False)

    print("Computing country-level R2 scores...")
    country_r2 = compute_country_r2_scores(df, admin_metrics)
    country_r2.to_csv(metrics_dir / "country_r2.csv", index=False)

    print("Creating visualizations...")

    # Core plots from results_analysis.py
    plot_acc_var_quadrants(admin_metrics, plots_dir)
    plot_distributions(admin_metrics, plots_dir)
    plot_spatial_skill_lines(spatial_df, plots_dir)
    plot_anomaly_heatmaps(df, plots_dir)
    plot_country_r2_barplot(country_r2, plots_dir)

    # Additional plots from results_full_vision.py
    plot_temporal_r_metrics(admin_metrics, plots_dir)
    plot_extreme_events_metrics(admin_metrics, extreme_agg, plots_dir)
    plot_yield_heatmaps(df, plots_dir)
    plot_detrended_yields(df, plots_dir)

    print(f"Analysis complete. Results saved to: {out_dir}")


def main():
    """Main entry point with fixed paths."""
    results_csv = Path("EXPERIMENTAL_MODEL\experiment_14_results\predictions_extreme_weighted.csv")
    out_dir = Path("Model/training_results")

    if not results_csv.exists():
        print(f"Error: Results CSV not found at {results_csv}")
        return

    run_combined_analysis(results_csv, out_dir=out_dir)


if __name__ == "__main__":
    main()
