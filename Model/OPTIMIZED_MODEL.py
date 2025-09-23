import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from tqdm.auto import tqdm
try:
    from tqdm_joblib import tqdm_joblib
except Exception:
    from contextlib import contextmanager

    @contextmanager
    def tqdm_joblib(*args, **kwargs):
        yield

import time
import warnings
import json
from datetime import datetime
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
epsilon = 1e-9

# Define allowed feature prefixes
allowed_prefixes = [
    'gdd_', 'precip_',
    'ndvi_', 'evi_', 
    'temp_2m_', 'soil_moisture_', 
    'max_dry_days_', 'heat_stress_days_', 
    'drought',
]

# Best configuration from structure search
BEST_CONFIG = {
    "objective": "normalized_residuals",
    "space_config": "relative_only",
    "crop_area_use": "exclude_crop_area",
    "interaction_type": "soil_weather_interactions",
    "gaez_method": "fao_relatively_adjusted"
}


def add_country_zscores(
    df: pd.DataFrame,
    base_features: list,
    group_col: str = 'country',
    suffix: str = '_zscore'
) -> tuple[pd.DataFrame, list, dict]:
    df = df.copy()
    new_feature_names: list = []
    feature_map: dict = {}
    for feature in base_features:
        if feature not in df.columns:
            continue
        group_mean = df.groupby(group_col)[feature].transform('mean')
        group_std = df.groupby(group_col)[feature].transform('std')
        z_col = f"{feature}{suffix}"
        df[z_col] = (df[feature] - group_mean) / (group_std + epsilon)
        new_feature_names.append(z_col)
        feature_map[feature] = z_col
    return df, new_feature_names, feature_map


def get_golden_cohort():
    return ['Angola', 'Lesotho', 'Zimbabwe', 'Ethiopia', 'Zambia', 'Kenya', 'South Africa', 'Tanzania, United Republic of', 'Nigeria']


def create_target_variable(df, objective_type):
    """Create target variable based on objective type."""
    if objective_type == 'residuals':
        df['target'] = df['yield'] - df['national_trend']
    elif objective_type == 'normalized_residuals':
        df['target'] = (df['yield'] - df['national_trend']) / (df['national_trend'] + epsilon)
    return df


def create_spatial_features(rs_df, space_config):
    """Create spatial features on rs_df based on configuration."""
    rs_df = rs_df.copy()
    spatial_features = []
    
    # Get environmental features from rs_df - limit to key features for performance
    env_features = [col for col in rs_df.columns if any(prefix in col for prefix in allowed_prefixes)][:25]  # Limit features
    print(f"   - Creating spatial features for {len(env_features)} environmental features...")
    
    if space_config in ['relative_only', 'both_relative_and_raw']:
        # Location z-scores within country - vectorized operation
        print(f"     - Computing country z-scores...")
        country_stats = rs_df.groupby('country')[env_features].agg(['mean', 'std'])
        for feature in env_features:
            if feature in rs_df.columns:
                zscore_col = f"{feature}_country_zscore"
                # Vectorized z-score calculation
                rs_df[zscore_col] = (rs_df[feature] - rs_df['country'].map(country_stats[(feature, 'mean')])) / (rs_df['country'].map(country_stats[(feature, 'std')]) + epsilon)
                spatial_features.append(zscore_col)
    
    if space_config in ['raw_stats_only', 'both_relative_and_raw']:
        # Location temporal statistics - vectorized operation
        print(f"     - Computing location temporal statistics...")
        location_stats = rs_df.groupby('PCODE')[env_features].agg(['mean', 'std'])
        for feature in env_features:
            if feature in rs_df.columns:
                mean_col = f"{feature}_location_mean"
                std_col = f"{feature}_location_std"
                # Vectorized calculation
                rs_df[mean_col] = rs_df['PCODE'].map(location_stats[(feature, 'mean')])
                rs_df[std_col] = rs_df['PCODE'].map(location_stats[(feature, 'std')])
                spatial_features.extend([mean_col, std_col])
    
    return rs_df, spatial_features


def apply_gaez_processing(df, admin0_df, gaez_method):
    """Apply GAEZ processing based on method."""
    df = df.copy()
    gaez_features = []
    
    if gaez_method == 'no_gaez':
        # Remove any existing GAEZ features
        df = df.drop(columns=[col for col in df.columns if 'gaez' in col.lower()], errors='ignore')
        return df, gaez_features
    
    if 'gaez_potential_yield' not in df.columns:
        print(f"Warning: gaez_potential_yield not found for method {gaez_method}")
        return df, gaez_features
    
    if gaez_method == 'raw_gaez':
        gaez_features.append('gaez_potential_yield')
    
    elif gaez_method == 'relative_ranking':
        # Rank GAEZ within each country
        df['gaez_country_rank'] = df.groupby('country')['gaez_potential_yield'].rank(method='dense', na_option='bottom')
        gaez_features.append('gaez_country_rank')
    
    elif gaez_method == 'fao_adjusted':
        # gaez_potential_yield / national_trend
        if 'national_trend' in df.columns:
            df['gaez_fao_adjusted'] = df['gaez_potential_yield'] / (df['national_trend'] + epsilon)
            gaez_features.append('gaez_fao_adjusted')
    
    elif gaez_method == 'fao_relatively_adjusted':
        # national_trend × (gaez_potential_yield / fao_long_term_avg_yield) - vectorized
        if 'national_trend' in df.columns:
            fao_long_term = admin0_df.groupby('country')['national_yield'].mean()
            df['country_avg_yield'] = df['country'].map(fao_long_term).fillna(1.0)
            # Vectorized calculation instead of apply
            df['gaez_fao_ratio'] = np.where(
                (pd.notna(df['gaez_potential_yield'])) & (df['country_avg_yield'] > 0),
                df['gaez_potential_yield'] / df['country_avg_yield'],
                np.nan
            )
            df['gaez_fao_relatively_adjusted'] = df['national_trend'] * df['gaez_fao_ratio']
            gaez_features.extend(['gaez_fao_ratio', 'gaez_fao_relatively_adjusted'])
    
    return df, gaez_features


def create_interaction_features(df, interaction_type, spatial_features):
    """Create interaction features based on configuration."""
    df = df.copy()
    interaction_features = []
    
    if interaction_type == 'no_interactions':
        return df, interaction_features
    
    # Get soil features
    soil_features = [col for col in df.columns if col.startswith('T_')]
    
    if interaction_type == 'soil_weather_interactions':
        # Weather z-scores × soil features
        weather_zscores = [col for col in df.columns if col.endswith('_zscore')]
        for weather_zscore in weather_zscores:
            for soil_feat in soil_features:
                if weather_zscore in df.columns and soil_feat in df.columns:
                    # Normalize soil feature
                    soil_norm = (df[soil_feat] - df[soil_feat].mean()) / (df[soil_feat].std() + epsilon)
                    interact_name = f"interact_{weather_zscore}_x_{soil_feat}"
                    df[interact_name] = df[weather_zscore] * soil_norm
                    interaction_features.append(interact_name)
    
    elif interaction_type == 'soil_comprehensive_interactions':
        # All spatial features × soil features
        all_spatial = spatial_features + [col for col in df.columns if col.endswith('_zscore')]
        for spatial_feat in all_spatial:
            for soil_feat in soil_features:
                if spatial_feat in df.columns and soil_feat in df.columns:
                    # Normalize soil feature
                    soil_norm = (df[soil_feat] - df[soil_feat].mean()) / (df[soil_feat].std() + epsilon)
                    interact_name = f"interact_{spatial_feat}_x_{soil_feat}"
                    df[interact_name] = df[spatial_feat] * soil_norm
                    interaction_features.append(interact_name)
    
    return df, interaction_features


def load_data(data_dir, all_countries):
    print("1. Loading all datasets...")
    yield_data_path = data_dir / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat" / "all_admin_gt0_merged_data.csv"
    rs_data_path = data_dir / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat" / "all_admin_gt0_remote_sensing_data.csv"
    admin0_path = data_dir / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat" / "all_admin0_merged_data.csv"
    gaez_yield_path = data_dir / "HarvestStatAfrica" / "yield_potential" / "gaez_maize_yield_potential_HSA.csv"
    soil_data_path = data_dir / "HarvestStatAfrica" / "HWSD" / "HSAfrica_Soil_Stats.csv"

    yield_df = pd.read_csv(yield_data_path)
    rs_df = pd.read_csv(rs_data_path)
    admin0_df = pd.read_csv(admin0_path)

    soil_df = pd.read_csv(soil_data_path)
    soil_df.rename(columns={'FNID': 'PCODE'}, inplace=True)
    soil_features = ['PCODE'] + [col for col in soil_df.columns if col.startswith('T_')]
    yield_df = pd.merge(yield_df, soil_df[soil_features], on='PCODE', how='left')

    yield_df = yield_df[yield_df['country'].isin(all_countries)].copy()
    rs_df = rs_df[rs_df['country'].isin(all_countries)].copy()
    admin0_df = admin0_df[admin0_df['country'].isin(all_countries)].copy()

    admin0_df['national_yield'] = admin0_df['production'] / admin0_df['area']
    admin0_processed = admin0_df[['country', 'year', 'national_yield']].copy()
    yield_df = pd.merge(yield_df, admin0_processed, on=['country', 'year'], how='left')

    gaez_df = pd.read_csv(gaez_yield_path)
    gaez_df.rename(columns={'FNID': 'PCODE', 'yield_mean_t_ha': 'gaez_potential_yield'}, inplace=True)
    yield_df['PCODE'] = yield_df['PCODE'].astype(str)
    gaez_df['PCODE'] = gaez_df['PCODE'].astype(str)
    yield_df = pd.merge(yield_df, gaez_df[['PCODE', 'gaez_potential_yield']], on='PCODE', how='left')

    long_term_nat_yield = admin0_df.groupby('country')['national_yield'].mean().to_dict()
    yield_df['gaez_potential_yield'] = yield_df.apply(
        lambda r: long_term_nat_yield.get(r['country'], 0)
        if pd.isna(r['gaez_potential_yield']) or r['gaez_potential_yield'] < 0.1
        else r['gaez_potential_yield'],
        axis=1,
    )

    yield_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("   - Data loading complete.")
    return yield_df, rs_df, admin0_df


def compute_national_trend(admin0_df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    trend_rows = []
    for country, g in admin0_df.groupby('country'):
        g_clean = g.dropna(subset=['year', 'national_yield'])
        if g_clean.empty:
            continue
        X = g_clean['year'].values.reshape(-1, 1)
        y = g_clean['national_yield'].values
        model = LinearRegression()
        try:
            model.fit(X, y)
            yhat = model.predict(X)
        except Exception:
            yhat = np.full_like(y, fill_value=np.nanmean(y), dtype=float)
        out = pd.DataFrame({
            'country': g_clean['country'].values,
            'year': g_clean['year'].values,
            'national_trend': yhat
        })
        trend_rows.append(out)
    if not trend_rows:
        return pd.DataFrame(columns=['country', 'year', 'national_trend'])
    trend_df = pd.concat(trend_rows, ignore_index=True)
    return trend_df


def _compute_loc10_z_for_series(values: np.ndarray, window: int = 10) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.array([])
    if n < window:
        m = np.nanmean(values)
        s = np.nanstd(values, ddof=1)
        if not np.isfinite(s) or s < epsilon:
            s = np.nanstd(values) + epsilon
        return (values - m) / (s + epsilon)

    half = window // 2
    z = np.empty(n, dtype=float)
    for i in range(n):
        start = i - half
        end = start + window
        if start < 0:
            start = 0
            end = window
        if end > n:
            end = n
            start = n - window
        w = values[start:end]
        m = np.nanmean(w)
        s = np.nanstd(w, ddof=1)
        if not np.isfinite(s) or s < epsilon:
            s = np.nanstd(w) + epsilon
        z[i] = (values[i] - m) / (s + epsilon)
    return z


def add_temporal_loc10_zscores(df: pd.DataFrame, base_features: list, window: int = 10) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    new_feature_names = []
    print(f"     - Computing temporal z-scores for {len(base_features)} features...")
    
    for i, feature in enumerate(base_features):
        if feature not in df.columns:
            continue
        if i % 5 == 0:  # Progress indicator
            print(f"       Processing feature {i+1}/{len(base_features)}: {feature}")
        
        new_col = f"{feature}_loc10_zscore"
        new_feature_names.append(new_col)
        df[new_col] = np.nan
        
        # Group processing with progress for large datasets
        for pcode, g in df.groupby('PCODE'):
            g_sorted = g.sort_values('year')
            vals = g_sorted[feature].values
            z = _compute_loc10_z_for_series(vals, window=window)
            df.loc[g_sorted.index, new_col] = z
    
    return df, new_feature_names


def get_pca_profiles(profiling_df, admin0_df, profile_features, n_components=5, fit_countries=None):
    print("Creating Country Fingerprints using PCA...")
    env_profiles = profiling_df.groupby('country')[profile_features].mean()
    prod_profiles = admin0_df.groupby('country')['national_yield'].mean().to_frame('long_term_avg_yield')
    full_profiles = pd.concat([env_profiles, prod_profiles], axis=1).dropna()

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    if fit_countries is not None:
        fit_idx = [c for c in fit_countries if c in full_profiles.index]
        if len(fit_idx) < 2:
            scaled_all = scaler.fit_transform(full_profiles)
            principal_components = pca.fit_transform(scaled_all)
        else:
            scaled_fit = scaler.fit_transform(full_profiles.loc[fit_idx])
            pca.fit(scaled_fit)
            principal_components = pca.transform(scaler.transform(full_profiles))
    else:
        scaled_all = scaler.fit_transform(full_profiles)
        principal_components = pca.fit_transform(scaled_all)

    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_profiles = pd.DataFrame(data=principal_components, columns=pc_columns, index=full_profiles.index)
    print(f"   - PCA complete. {sum(pca.explained_variance_ratio_):.1%} of variance.")
    return pca_profiles


def get_analogue_strategy(target_profile, training_profiles, distance_threshold, min_analogues=3):
    distances = training_profiles.apply(lambda row: euclidean(row, target_profile), axis=1)
    close_analogues = distances[distances < distance_threshold]
    if len(close_analogues) >= min_analogues:
        similarity_weights = 1 / (close_analogues + epsilon)
        normalized_weights = (similarity_weights / similarity_weights.sum()).to_dict()
        return 'specialist', normalized_weights
    else:
        return 'global', training_profiles.index.tolist()


def run_model_fold_adaptive_rfecv(train_df, eval_df, yield_features, country_profiles, distance_threshold):
    """Enhanced model fold with analogue-based training + RFECV+PCA."""
    eval_country = eval_df['country'].unique()[0]
    
    # Determine training strategy based on PCA profiles
    if eval_country not in country_profiles.index:
        strategy, train_info = 'global', train_df['country'].unique().tolist()
    else:
        target_profile = country_profiles.loc[eval_country]
        training_profiles = country_profiles.drop(eval_country)
        strategy, train_info = get_analogue_strategy(target_profile, training_profiles, distance_threshold)

    if strategy == 'specialist':
        analogue_weights = train_info
        analogue_countries = list(analogue_weights.keys())
        specialist_train_df = train_df[train_df['country'].isin(analogue_countries)].copy()
        if specialist_train_df.empty:
            return pd.DataFrame()
        print(f"    [{eval_country}] Strategy: Specialist. Training on {len(analogue_countries)} close analogues")
        final_train_df = specialist_train_df
        sample_weights = specialist_train_df['country'].map(analogue_weights)
    else:
        fallback_countries = train_info
        global_train_df = train_df[train_df['country'].isin(fallback_countries)]
        print(f"    [{eval_country}] Strategy: Global Fallback. Training on {len(fallback_countries)} countries.")
        final_train_df = global_train_df
        sample_weights = None
    
    # Step 1: Feature selection with RFECV on analogue/global training data
    X_train_full = final_train_df[yield_features].fillna(0)
    y_train = final_train_df['target']
    groups_train = final_train_df['country']
    
    # RFECV with GroupKFold
    estimator = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    n_groups = groups_train.nunique()
    
    if n_groups > 1:
        cv = GroupKFold(n_splits=min(3, n_groups))
        rfecv = RFECV(
            estimator=estimator,
            step=0.3,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            min_features_to_select=max(10, min(15, len(yield_features) // 5)),
        )
        
        try:
            rfecv.fit(X_train_full, y_train, groups=groups_train)
            selected_features = [f for f, selected in zip(yield_features, rfecv.support_) if selected]
        except Exception as e:
            print(f"    Warning: RFECV failed ({str(e)}), using top 15 features by importance")
            estimator.fit(X_train_full, y_train, sample_weight=sample_weights)
            importances = pd.Series(estimator.feature_importances_, index=yield_features).sort_values(ascending=False)
            selected_features = importances.head(15).index.tolist()
    else:
        # Single country - use feature importance
        estimator.fit(X_train_full, y_train, sample_weight=sample_weights)
        importances = pd.Series(estimator.feature_importances_, index=yield_features).sort_values(ascending=False)
        selected_features = importances.head(15).index.tolist()
    
    print(f"    Selected {len(selected_features)} features via RFECV/importance")
    
    # Step 2: PCA on selected features
    X_train_selected = final_train_df[selected_features].fillna(0)
    X_eval_selected = eval_df[selected_features].fillna(0)
    
    # Apply PCA
    scaler = StandardScaler()
    pca = PCA(n_components=min(8, len(selected_features), X_train_selected.shape[0] - 1))
    
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    X_eval_scaled = scaler.transform(X_eval_selected)
    X_eval_pca = pca.transform(X_eval_scaled)
    
    print(f"    PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Step 3: Train final model on PCA components
    final_model = ExtraTreesRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=5)
    final_model.fit(X_train_pca, y_train, sample_weight=sample_weights)
    
    # Predict
    eval_df = eval_df.copy()
    eval_df['pred_target'] = final_model.predict(X_eval_pca)
    
    # Convert back to yield predictions
    if BEST_CONFIG['objective'] == 'residuals':
        eval_df['pred_yield'] = eval_df['pred_target'] + eval_df['national_trend']
    elif BEST_CONFIG['objective'] == 'normalized_residuals':
        eval_df['pred_yield'] = eval_df['pred_target'] * eval_df['national_trend'] + eval_df['national_trend']
    
    eval_df['pred_yield'] = eval_df['pred_yield'].clip(lower=0)
    
    return eval_df


def compute_country_r2_scores(df: pd.DataFrame) -> pd.DataFrame:
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
        # Row-wise R2 across all observations
        r2_overall = _r2(g["yield"].values, g["pred_yield"].values)

        rows.append({
            "country": country,
            "n_points": int(len(g)),
            "R2": r2_overall,
        })
    return pd.DataFrame(rows)


def evaluate_and_report(model_name, results_df):
    print("\n" + "="*70 + f"\n      RESULTS FOR MODEL: {model_name}\n" + "="*70)
    country_metrics = []
    for country in sorted(results_df['country'].unique()):
        country_df = results_df[results_df['country'] == country]
        valid_yield_country_df = country_df.dropna(subset=['yield', 'pred_yield'])
        if not valid_yield_country_df.empty and len(valid_yield_country_df) > 1:
            r2 = r2_score(valid_yield_country_df['yield'], valid_yield_country_df['pred_yield'])
            mae = mean_absolute_error(valid_yield_country_df['yield'], valid_yield_country_df['pred_yield'])
            bias = valid_yield_country_df['pred_yield'].mean() - valid_yield_country_df['yield'].mean()
        else:
            r2, mae, bias = np.nan, np.nan, np.nan
        country_metrics.append({'country': country, 'Yield R2': r2, 'Yield MAE': mae, 'Bias': bias})

    summary_df = pd.DataFrame(country_metrics).set_index('country')
    avg_r2 = summary_df['Yield R2'].mean()

    print("\n--- Per-Country Performance ---\\n" + summary_df.to_string(
        formatters={'Yield R2': '{:,.3f}'.format, 'Yield MAE': '{:,.3f}'.format, 'Bias': '{:,.3f}'.format}
    ))
    print("\n" + "-"*35 + f"\n  Average R2:  {avg_r2:.3f} (Std Dev: {summary_df['Yield R2'].std():.3f})")
    print(f"  Countries with R2 > 0.3: {(summary_df['Yield R2'] > 0.3).sum()}/{len(summary_df.dropna())}\\n" + "-" * 35)
    
    # Create performance plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Country R2 scores
    plt.subplot(2, 2, 1)
    summary_df['Yield R2'].plot(kind='bar')
    plt.axhline(y=0.3, color='red', linestyle='--', label='R2 = 0.3 threshold')
    plt.title('R2 Score by Country')
    plt.ylabel('R2 Score')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plot 2: Predictions vs Observations
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['yield'], results_df['pred_yield'], alpha=0.6)
    min_val = min(results_df['yield'].min(), results_df['pred_yield'].min())
    max_val = max(results_df['yield'].max(), results_df['pred_yield'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'Predictions vs Observations (R2={avg_r2:.3f})')
    plt.legend()
    
    # Plot 3: Residuals
    plt.subplot(2, 2, 3)
    residuals = results_df['pred_yield'] - results_df['yield']
    plt.scatter(results_df['pred_yield'], residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Yield')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Plot 4: MAE by Country
    plt.subplot(2, 2, 4)
    summary_df['Yield MAE'].plot(kind='bar', color='orange')
    plt.title('MAE by Country')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

    return avg_r2


def main():
    start_time = time.time()
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    print("="*80)
    print("         OPTIMIZED MODEL WITH BEST CONFIGURATION + RFECV PCA")
    print("="*80)
    print(f"Configuration: {BEST_CONFIG}")

    # Load base data
    countries_to_use = get_golden_cohort()
    yield_df_orig, rs_df_orig, admin0_df = load_data(PROJECT_ROOT, countries_to_use)

    # Create the "universe" DataFrame by merging RS data with available yield data
    print("\n2. Creating a unified DataFrame for all locations and years...")
    # Ensure PCODE is string for merging
    rs_df_orig['PCODE'] = rs_df_orig['PCODE'].astype(str)
    yield_df_orig['PCODE'] = yield_df_orig['PCODE'].astype(str)
    
    # Use RS data as the base (left dataframe) to keep all location-years
    # The suffixes=('', '_yield_gt') handles overlapping columns except for the key columns
    key_cols = ['PCODE', 'year', 'country']
    yield_cols = [col for col in yield_df_orig.columns if col not in rs_df_orig.columns or col in key_cols]
    universe_df = pd.merge(rs_df_orig, yield_df_orig[yield_cols], on=key_cols, how='left')
    
    print(f"   - Unified data shape: {universe_df.shape}")
    print(f"   - Year range: {universe_df['year'].min()}-{universe_df['year'].max()}")
    print(f"   - Observations with yield data: {universe_df['yield'].notna().sum()}")
    print(f"   - Observations without yield data: {universe_df['yield'].isna().sum()}")

    # Fill soil data across the entire universe_df
    soil_cols = [col for col in universe_df.columns if col.startswith('T_')]
    for col in soil_cols:
        # First, fill within-country, then globally
        universe_df[col] = universe_df.groupby('country')[col].transform(lambda x: x.fillna(x.median()))
        universe_df[col].fillna(universe_df[col].median(), inplace=True)

    # Add trend data to the entire universe_df
    trend_df = compute_national_trend(admin0_df)
    universe_df = universe_df.merge(trend_df, on=['country', 'year'], how='left')

    print("\n3. Applying best configuration preprocessing to the unified DataFrame...")
    
    # Apply spatial features configuration on the unified_df
    universe_df, spatial_features = create_spatial_features(universe_df, BEST_CONFIG['space_config'])
    
    # Apply temporal z-scores to the unified_df
    essential_env_features = [col for col in universe_df.columns if any(prefix in col for prefix in allowed_prefixes)][:20]
    print(f"   - Applying temporal z-scores to {len(essential_env_features)} environmental features...")
    universe_df, temporal_z_features = add_temporal_loc10_zscores(universe_df, essential_env_features, window=10)
    
    # Apply GAEZ processing
    universe_df, gaez_features = apply_gaez_processing(universe_df, admin0_df, BEST_CONFIG['gaez_method'])
    
    # Create target variable (will be NaN for rows without yield)
    universe_df = create_target_variable(universe_df, BEST_CONFIG['objective'])
    
    # Handle crop area
    if BEST_CONFIG['crop_area_use'] == 'exclude_crop_area':
        if 'crop_area_ha_adjusted' in universe_df.columns:
            universe_df = universe_df.drop(columns=['crop_area_ha_adjusted'])
    
    # Create interaction features
    universe_df, interaction_features = create_interaction_features(universe_df, BEST_CONFIG['interaction_type'], spatial_features)
    
    # Collect all features for modeling
    all_features = []
    all_features.extend([col for col in universe_df.columns if col.startswith('T_')])  # Soil
    all_features.extend(spatial_features)
    all_features.extend(temporal_z_features)
    all_features.extend(gaez_features)
    all_features.extend(interaction_features)
    if BEST_CONFIG['crop_area_use'] == 'include_crop_area' and 'crop_area_ha_adjusted' in universe_df.columns:
        all_features.append('crop_area_ha_adjusted')
    
    # Filter valid features and clean data
    valid_features = [f for f in all_features if f in universe_df.columns and universe_df[f].dtype != 'object']
    universe_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # The main dataframe for CV now contains all data
    yield_df = universe_df.copy()
    
    trainable_points = yield_df['target'].notna().sum()
    print(f"   Total features: {len(valid_features)}, Total data points: {len(yield_df)}, Trainable points: {trainable_points}")
    print(f"   Feature categories: Soil={len([f for f in valid_features if f.startswith('T_')])}, Spatial={len(spatial_features)}, Temporal={len(temporal_z_features)}, GAEZ={len(gaez_features)}, Interactions={len(interaction_features)}")
    
    print("\n4. Creating PCA profiles for analogue-based training...")
    
    # Create PCA profiles for country similarity (using the full rs_df is fine here)
    env_features_for_pca = [col for col in rs_df_orig.columns if any(prefix in col for prefix in allowed_prefixes)][:30]
    stress_features_for_pca = ['max_dry_days_plant_harv', 'heat_stress_days_plant_harv']
    profile_features = [f for f in env_features_for_pca + stress_features_for_pca if f in rs_df_orig.columns]
    
    pca_profiles = get_pca_profiles(rs_df_orig, admin0_df, profile_features, n_components=5)
    
    print("\n5. Running cross-validation with Analogue-based RFECV+PCA...")
    
    # Run cross-validation on the full dataset
    groups = yield_df['country']
    logo = LeaveOneGroupOut()
    all_predictions = []
    
    # We split the entire dataset, then filter for training
    for i, (train_idx, eval_idx) in enumerate(logo.split(yield_df, groups=groups)):
        full_train_df = yield_df.iloc[train_idx].copy()
        eval_df = yield_df.iloc[eval_idx].copy()
        eval_country = eval_df['country'].unique()[0]
        
        # Actual training data must have a valid target
        train_df = full_train_df.dropna(subset=['target']).copy()
        
        print(f"\n  Fold {i+1}: Evaluating {eval_country}. Training on {train_df['country'].nunique()} countries ({len(train_df)} points).")
        print(f"  Predicting on {len(eval_df)} total points for {eval_country}.")

        # Calculate distance threshold dynamically
        train_countries = train_df['country'].unique().tolist()
        if len(train_countries) > 1 and all(c in pca_profiles.index for c in train_countries):
            dists = [
                euclidean(pca_profiles.loc[c1], pca_profiles.loc[c2])
                for c1 in train_countries for c2 in train_countries if c1 != c2
            ]
            distance_threshold = np.percentile(dists, 50) if dists else 0.0
        else:
            distance_threshold = 0.0
        
        # Run enhanced model fold with analogue-based RFECV+PCA
        result_df = run_model_fold_adaptive_rfecv(train_df, eval_df, valid_features, pca_profiles, distance_threshold)
        all_predictions.append(result_df)
    
    # Combine results and evaluate
    print("\n6. Evaluating results...")
    results_df = pd.concat(all_predictions, ignore_index=True)
    
    # Compute scores (evaluation functions already handle NaNs in 'yield' column)
    scores_df = compute_country_r2_scores(results_df)
    avg_r2 = evaluate_and_report("OPTIMIZED_MODEL_RFECV_PCA", results_df)
    
    # Save results
    results_dir = PROJECT_ROOT / "Model" / "training_results"
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / "model_predictions.csv", index=False)
    scores_df.to_csv(results_dir / "model_scores.csv", index=False)
    
    total_time = (time.time() - start_time) / 60.0
    print(f"\n" + "="*80)
    print(f"OPTIMIZED MODEL COMPLETE - Total time: {total_time:.1f} minutes")
    print(f"Average R2: {avg_r2:.3f}")
    print(f"Full predictions for {len(results_df)} location-years saved to: {results_dir}")
    print("="*80)



if __name__ == "__main__":
    main()
