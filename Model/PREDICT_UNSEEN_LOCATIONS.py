import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean
from tqdm.auto import tqdm
try:
    from tqdm_joblib import tqdm_joblib
except Exception:
    from contextlib import contextmanager

    @contextmanager
    def tqdm_joblib(*args, **kwargs):
        yield

import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")
epsilon = 1e-9

# Define allowed feature prefixes
allowed_prefixes = [
    'gdd_', 'precip_',
    'ndvi_', 'evi_', 
    'temp_2m_', 'soil_moisture_', 
    'max_dry_days_', 'heat_stress_days_', 
    'drought_',
]

# Best configuration from structure search
BEST_CONFIG = {
    "objective": "normalized_residuals", # possible: normalized_residuals, residuals
    "space_config": "relative_only", # possible: None, relative_only, raw_stats_only, both_relative_and_raw
    "crop_area_use": "exclude_crop_area", # possible: exclude_crop_area, include_crop_area
    "interaction_type": "no_interactions", # possible: no_interactions, soil_weather_interactions, soil_comprehensive_interactions
    "gaez_method": "raw_gaez" # possible: no_gaez, raw_gaez, relative_ranking, fao_adjusted, fao_relatively_adjusted
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
    
    # Use dynamic environmental features filtering based on allowed_prefixes
    env_features = [col for col in rs_df.columns if any(prefix in col for prefix in allowed_prefixes)]
    print(f"   - Creating spatial features for {len(env_features)} environmental features...")
    print(f"   - Available features: {len(env_features)} from allowed_prefixes filtering")
    
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


def apply_gaez_processing_predict(df, national_yields_dict, gaez_method):
    """Apply GAEZ processing for prediction - using known national yields."""
    df = df.copy()
    gaez_features = []
    
    if gaez_method == 'no_gaez':
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
        # Use average national yield for each country
        df['national_trend_avg'] = df['country'].map(national_yields_dict)
        df['gaez_fao_adjusted'] = df['gaez_potential_yield'] / (df['national_trend_avg'] + epsilon)
        gaez_features.append('gaez_fao_adjusted')
    
    elif gaez_method == 'fao_relatively_adjusted':
        # Use average national yield for trend - vectorized
        df['national_trend_avg'] = df['country'].map(national_yields_dict).fillna(1.0)
        # Vectorized calculation instead of apply
        df['gaez_fao_ratio'] = np.where(
            (pd.notna(df['gaez_potential_yield'])) & (df['national_trend_avg'] > 0),
            df['gaez_potential_yield'] / df['national_trend_avg'],
            np.nan
        )
        df['gaez_fao_relatively_adjusted'] = df['national_trend_avg'] * df['gaez_fao_ratio']
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


def compute_national_trend(admin0_df: pd.DataFrame) -> pd.DataFrame:
    """Compute national yield trends using linear regression for each country."""
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


def load_training_data(data_dir, countries):
    """Load training data with yield information."""
    print("Loading training datasets...")
    yield_data_path = data_dir / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat" / "all_admin_gt0_merged_data.csv"
    rs_data_path = data_dir / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat" / "all_admin_gt0_remote_sensing_data.csv"
    admin0_path = data_dir / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat" / "all_admin0_merged_data.csv"
    gaez_yield_path = data_dir / "HarvestStatAfrica" / "yield_potential" / "gaez_maize_yield_potential_HSA.csv"
    soil_data_path = data_dir / "HarvestStatAfrica" / "HWSD" / "HSAfrica_Soil_Stats.csv"
    
    # IMPORTANT: Load COMPLETE Africa admin0 data for trend calculation
    complete_admin0_path = data_dir / "RemoteSensing" / "GADM" / "preprocessed_concat" / "admin0_merged_data_GLAM_concat.csv"
    print(f"   - Loading complete Africa admin0 data from: {complete_admin0_path}")

    yield_df = pd.read_csv(yield_data_path)
    rs_df = pd.read_csv(rs_data_path)
    admin0_df = pd.read_csv(admin0_path)  # Training admin0 (for training data processing)
    complete_admin0_df = pd.read_csv(complete_admin0_path)  # Complete admin0 (for trend calculation)

    # IMPORTANT: Calculate national yield trends for ALL African countries using complete dataset
    print("   - Computing national yield trends for ALL African countries using complete dataset...")
    
    # The complete dataset has 'yield' column directly, but let's also calculate from production/area for consistency
    if 'yield' in complete_admin0_df.columns:
        complete_admin0_df['national_yield'] = complete_admin0_df['yield']
        print(f"   - Using 'yield' column from complete dataset")
    else:
        complete_admin0_df['national_yield'] = complete_admin0_df['production'] / complete_admin0_df['area']
        print(f"   - Calculated yield from production/area in complete dataset")
    
    # Compute proper linear trends for all countries using complete dataset
    all_trends_df = compute_national_trend(complete_admin0_df)
    print(f"   - Computed trend data for {all_trends_df['country'].nunique()} countries across {len(all_trends_df)} country-year combinations")
    
    # Also keep the long-term averages for fallback calculations
    all_national_yields = complete_admin0_df.groupby('country')['national_yield'].mean().to_dict()
    print(f"   - Found complete FAO yield data for {len(all_national_yields)} countries")
    
    # Add soil data
    soil_df = pd.read_csv(soil_data_path)
    soil_df.rename(columns={'FNID': 'PCODE'}, inplace=True)
    soil_features = ['PCODE'] + [col for col in soil_df.columns if col.startswith('T_')]
    yield_df = pd.merge(yield_df, soil_df[soil_features], on='PCODE', how='left')

    # Calculate national yield for training admin0 data BEFORE filtering
    admin0_df['national_yield'] = admin0_df['production'] / admin0_df['area']

    # Filter to training countries AFTER calculating national yields
    yield_df = yield_df[yield_df['country'].isin(countries)].copy()
    rs_df = rs_df[rs_df['country'].isin(countries)].copy()
    admin0_df_filtered = admin0_df[admin0_df['country'].isin(countries)].copy()

    # Add national yield for training data
    admin0_processed = admin0_df_filtered[['country', 'year', 'national_yield']].copy()
    yield_df = pd.merge(yield_df, admin0_processed, on=['country', 'year'], how='left')

    # Add GAEZ
    gaez_df = pd.read_csv(gaez_yield_path)
    gaez_df.rename(columns={'FNID': 'PCODE', 'yield_mean_t_ha': 'gaez_potential_yield'}, inplace=True)
    yield_df['PCODE'] = yield_df['PCODE'].astype(str)
    gaez_df['PCODE'] = gaez_df['PCODE'].astype(str)
    yield_df = pd.merge(yield_df, gaez_df[['PCODE', 'gaez_potential_yield']], on='PCODE', how='left')

    # Fill missing GAEZ with national average (using training countries only for this)
    training_national_yields = admin0_df_filtered.groupby('country')['national_yield'].mean().to_dict()
    yield_df['gaez_potential_yield'] = yield_df.apply(
        lambda r: training_national_yields.get(r['country'], 0)
        if pd.isna(r['gaez_potential_yield']) or r['gaez_potential_yield'] < 0.1
        else r['gaez_potential_yield'],
        axis=1,
    )

    yield_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return yield_df, rs_df, admin0_df_filtered, all_national_yields, all_trends_df


def load_prediction_data(prediction_rs_path_admin2, prediction_rs_path_admin1, soil_data_path, gaez_yield_path):
    """Load prediction data from the all_Africa_predictions dataset."""
    print("Loading all-Africa prediction dataset...")
    
    # Load the new prediction remote sensing data
    rs_df_admin2 = pd.read_csv(prediction_rs_path_admin2)
    rs_df_admin1 = pd.read_csv(prediction_rs_path_admin1)
    rs_df = pd.concat([rs_df_admin2, rs_df_admin1], ignore_index=True)
    print(f"   - Loaded prediction data: {len(rs_df):,} records")
    print(f"   - Countries in prediction data: {rs_df['country'].nunique()}")
    
    # Clean country names - remove empty strings
    rs_df = rs_df[rs_df['country'].notna() & (rs_df['country'] != '')].copy()
    print(f"   - After cleaning: {len(rs_df):,} records from {rs_df['country'].nunique()} countries")
    
    # Add soil data
    soil_df = pd.read_csv(soil_data_path)
    soil_df.rename(columns={'FNID': 'PCODE'}, inplace=True)
    soil_features = ['PCODE'] + [col for col in soil_df.columns if col.startswith('T_')]
    rs_df = pd.merge(rs_df, soil_df[soil_features], on='PCODE', how='left')
    print(f"   - Added soil data for {rs_df[soil_features[1:]].notna().any(axis=1).sum():,} locations")
    
    # Add GAEZ data
    gaez_df = pd.read_csv(gaez_yield_path)
    gaez_df.rename(columns={'FNID': 'PCODE', 'yield_mean_t_ha': 'gaez_potential_yield'}, inplace=True)
    rs_df['PCODE'] = rs_df['PCODE'].astype(str)
    gaez_df['PCODE'] = gaez_df['PCODE'].astype(str)
    rs_df = pd.merge(rs_df, gaez_df[['PCODE', 'gaez_potential_yield']], on='PCODE', how='left')
    print(f"   - Added GAEZ data for {rs_df['gaez_potential_yield'].notna().sum():,} locations")
    
    rs_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return rs_df


def compute_national_trend(admin0_df: pd.DataFrame) -> pd.DataFrame:
    """Compute national trends from historical data."""
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

    # Sort once at the beginning
    df = df.sort_values(['PCODE', 'year'])

    for i, feature in enumerate(base_features):
        if feature not in df.columns:
            continue
        if i % 5 == 0:  # Progress indicator
            print(f"       Processing feature {i+1}/{len(base_features)}: {feature}")

        new_col = f"{feature}_loc10_zscore"
        new_feature_names.append(new_col)

        # Vectorized approach: use groupby + rolling window
        df[new_col] = (
            df.groupby('PCODE')[feature]
            .transform(lambda x: (x - x.rolling(window=window, center=True, min_periods=1).mean())
                      / (x.rolling(window=window, center=True, min_periods=1).std(ddof=1) + epsilon))
        )

        # Handle edge cases where rolling std is NaN or 0
        df[new_col] = df[new_col].fillna(
            df.groupby('PCODE')[feature]
            .transform(lambda x: (x - x.mean()) / (x.std(ddof=1) + epsilon))
        )

    return df, new_feature_names


def get_pca_profiles(profiling_df, admin0_df, profile_features, n_components=5, fit_countries=None):
    print("Creating Country Fingerprints using PCA...")
    env_profiles = profiling_df.groupby('country')[profile_features].mean()
    print(f"   Environmental profiles created for: {sorted(env_profiles.index.tolist())}")
    
    # Only add production profiles where available (golden cohort countries)
    prod_profiles = admin0_df.groupby('country')['national_yield'].mean().to_frame('long_term_avg_yield')
    print(f"   Production profiles available for: {sorted(prod_profiles.index.tolist())}")
    
    # For countries without production data, use 0 as placeholder
    all_countries = env_profiles.index.tolist()
    full_profiles = env_profiles.copy()
    full_profiles['long_term_avg_yield'] = 0.0  # Default value
    
    # Update with actual production values where available
    for country in prod_profiles.index:
        if country in full_profiles.index:
            full_profiles.loc[country, 'long_term_avg_yield'] = prod_profiles.loc[country, 'long_term_avg_yield']
    
    print(f"   Full profiles created for: {sorted(full_profiles.index.tolist())}")

    # Fill NaN values with global means instead of dropping countries
    if full_profiles.isnull().any().any():
        print(f"   Filling {full_profiles.isnull().sum().sum()} NaN values with global means")
        for col in full_profiles.columns:
            if full_profiles[col].isnull().any():
                global_mean = full_profiles[col].mean()
                if not np.isnan(global_mean):
                    full_profiles[col].fillna(global_mean, inplace=True)

    # Only drop countries that still have all NaN after filling
    full_profiles = full_profiles.dropna(how='all')

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


def predict_for_country_with_analogues(country_data, training_df, feature_columns, country_profiles, distance_threshold, scaler, selected_features):
    """Predict for a specific country using analogue-based approach."""
    target_country = country_data['country'].iloc[0]

    # Determine training strategy based on PCA profiles
    if target_country not in country_profiles.index:
        # Last resort: global strategy for countries without profiles
        strategy, train_info = 'global', training_df['country'].unique().tolist()
        print(f"   [{target_country}] No PCA profile available - using global strategy")
    else:
        target_profile = country_profiles.loc[target_country]
        # Only consider golden cohort countries as potential trainers
        training_countries = training_df['country'].unique()
        training_profiles = country_profiles.loc[training_countries]
        strategy, train_info = get_analogue_strategy(target_profile, training_profiles, distance_threshold)

    if strategy == 'specialist':
        analogue_weights = train_info
        analogue_countries = list(analogue_weights.keys())
        specialist_train_df = training_df[training_df['country'].isin(analogue_countries)].copy()
        if specialist_train_df.empty:
            print(f"   [{target_country}] No analogue data available - skipping")
            return country_data.copy()

        weights_str = {k: f'{v:.3f}' for k, v in analogue_weights.items()}
        print(f"   [{target_country}] SPECIALIST training on {len(analogue_countries)} analogues: {weights_str}")
        final_train_df = specialist_train_df
        sample_weights = specialist_train_df['country'].map(analogue_weights)
    else:
        fallback_countries = train_info
        global_train_df = training_df[training_df['country'].isin(fallback_countries)]
        print(f"   [{target_country}] GLOBAL training on {len(fallback_countries)} countries: {fallback_countries}")
        final_train_df = global_train_df
        sample_weights = None

    # Train model on analogues/global data (no PCA)
    X_train_selected = final_train_df[selected_features].fillna(0)
    X_train_scaled = scaler.transform(X_train_selected)
    y_train = final_train_df['target']

    # Train final model directly on scaled features
    model = ExtraTreesRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=3)
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # Predict for target country
    X_target = country_data[selected_features].fillna(0)
    X_target_scaled = scaler.transform(X_target)

    country_data = country_data.copy()
    country_data['pred_target'] = model.predict(X_target_scaled)

    return country_data


def train_prediction_model_with_analogues(training_df, feature_columns):
    """Train components needed for analogue-based prediction."""
    print("Setting up analogue-based prediction model...")

    # Feature selection with RFECV on all training data
    X_train = training_df[feature_columns].fillna(0)
    y_train = training_df['target']
    groups_train = training_df['country']

    # RFECV with GroupKFold
    estimator = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    n_groups = groups_train.nunique()
    cv = GroupKFold(n_splits=min(5, n_groups))

    rfecv = RFECV(
        estimator=estimator,
        step=0.3,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        min_features_to_select=max(10, min(20, len(feature_columns) // 5)),
    )

    with tqdm_joblib(tqdm(desc="Feature selection for analogue model", total=None)):
        try:
            rfecv.fit(X_train, y_train, groups=groups_train)
            selected_features = [f for f, selected in zip(feature_columns, rfecv.support_) if selected]
        except Exception as e:
            print(f"Warning: RFECV failed ({str(e)}), using top 20 features by importance")
            estimator.fit(X_train, y_train)
            importances = pd.Series(estimator.feature_importances_, index=feature_columns).sort_values(ascending=False)
            selected_features = importances.head(20).index.tolist()

    print(f"Selected {len(selected_features)} features via RFECV/importance")

    # Prepare data for correlation analysis
    X_train_selected = training_df[selected_features].fillna(0)

    # Train importance model to get feature importances for correlation pruning
    importance_model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    importance_model.fit(X_train_selected, y_train)
    feature_importances = pd.Series(importance_model.feature_importances_, index=selected_features).sort_values(ascending=False)

    # Step 1.5: Correlation pruning - remove highly correlated features (>0.7), keeping most important ones
    if len(selected_features) > 1:
        # Calculate correlation matrix
        corr_matrix = X_train_selected.corr().abs()

        # Find features to remove due to high correlation
        features_to_remove = set()
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                feat1, feat2 = selected_features[i], selected_features[j]
                if feat1 not in features_to_remove and feat2 not in features_to_remove:
                    if corr_matrix.loc[feat1, feat2] > 0.7:
                        # Keep the more important feature
                        imp1 = feature_importances[feat1] if feat1 in feature_importances.index else 0
                        imp2 = feature_importances[feat2] if feat2 in feature_importances.index else 0
                        if imp1 >= imp2:
                            features_to_remove.add(feat2)
                        else:
                            features_to_remove.add(feat1)

        # Apply pruning
        selected_features = [f for f in selected_features if f not in features_to_remove]
        print(f"    After correlation pruning (|corr| > 0.7): {len(selected_features)} features remain")

        # Update training data with pruned features
        X_train_selected = training_df[selected_features].fillna(0)
        # Re-train importance model on pruned features
        importance_model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        importance_model.fit(X_train_selected, y_train)
        feature_importances = pd.Series(importance_model.feature_importances_, index=selected_features).sort_values(ascending=False)

    print(f"Final selected {len(selected_features)} features for analogue-based prediction")

    # Setup scaler (no PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)

    return scaler, None, selected_features


def merge_with_existing_yields(prediction_df, existing_yield_data):
    """Merge predictions with existing yield data where available."""
    print("Merging predictions with existing yield data...")
    
    # Create a comprehensive dataset
    result_df = prediction_df.copy()
    
    # Add existing yields where available
    if existing_yield_data is not None and not existing_yield_data.empty:
        existing_keys = ['PCODE', 'year', 'country']
        existing_data = existing_yield_data[existing_keys + ['yield']].copy()
        existing_data['has_ground_truth'] = True
        
        result_df = pd.merge(result_df, existing_data, on=existing_keys, how='left', suffixes=('', '_existing'))
        result_df['has_ground_truth'] = result_df['has_ground_truth'].fillna(False)
        
        # Create final yield column: use existing where available, prediction otherwise
        result_df['final_yield'] = np.where(
            result_df['has_ground_truth'], 
            result_df['yield'], 
            result_df['pred_yield']
        )
        
        print(f"   Found existing yields for {result_df['has_ground_truth'].sum():,} locations")
        print(f"   Using predictions for {(~result_df['has_ground_truth']).sum():,} locations")
    else:
        result_df['has_ground_truth'] = False
        result_df['final_yield'] = result_df['pred_yield']
        print(f"   No existing yield data found, using predictions for all {len(result_df):,} locations")
    
    return result_df


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    print("="*80)
    print("         MAIZE YIELD PREDICTION FOR UNSEEN LOCATIONS")
    print("="*80)
    print(f"Configuration: {BEST_CONFIG}")
    
    # Step 1: Load training data from golden cohort countries ONLY
    print("\n1. Loading training data...")
    training_countries = get_golden_cohort()
    print(f"   - IMPORTANT: Training ONLY on golden cohort countries: {training_countries}")
    print(f"   - All other countries will be prediction targets only")
    
    training_yield_df, training_rs_df, admin0_df, national_yields, all_trends_df = load_training_data(
        PROJECT_ROOT, training_countries
    )

    # Keep a copy of original RS data for PCA profile features (before any processing)
    raw_rs_df = training_rs_df.copy()
    
    # Step 2: Load prediction data from all_Africa_predictions dataset
    print("\n2. Loading prediction data...")
    prediction_rs_df = load_prediction_data(
        PROJECT_ROOT / "RemoteSensing" / "GADM" / "preprocessed_concat" / "admin2_remote_sensing_data_GLAM_concat.csv",
        PROJECT_ROOT / "RemoteSensing" / "GADM" / "preprocessed_concat" / "admin1_remote_sensing_data_GLAM_concat.csv",
        PROJECT_ROOT / "GADM" / "HWSD" / "Africa_GADM_Soil_Stats.csv",
        PROJECT_ROOT / "GADM" / "yield_potential" / "gaez_maize_yield_potential.csv"
    )
    
    # Fill missing GAEZ with continental average
    continental_gaez_avg = prediction_rs_df['gaez_potential_yield'].median()
    prediction_rs_df['gaez_potential_yield'].fillna(continental_gaez_avg, inplace=True)
    
    print(f"   Training data: {len(training_yield_df):,} observations from {len(training_countries)} GOLDEN COHORT countries")
    print(f"   Prediction data: {len(prediction_rs_df):,} location-years from {prediction_rs_df['country'].nunique()} countries")
    
    # Check for overlapping countries
    training_countries_set = set(training_yield_df['country'].unique())
    prediction_countries_set = set(prediction_rs_df['country'].unique())
    overlapping_countries = training_countries_set.intersection(prediction_countries_set)
    prediction_only_countries = prediction_countries_set - training_countries_set
    
    print(f"   Countries in both training and prediction: {len(overlapping_countries)} - {sorted(overlapping_countries)}")
    print(f"   Countries only in prediction data: {len(prediction_only_countries)} - {sorted(prediction_only_countries)}")
    
    # Verify golden cohort limitation
    actual_training_countries = set(training_yield_df['country'].unique())
    expected_training_countries = set(training_countries)
    if actual_training_countries != expected_training_countries:
        print(f"   WARNING: Training countries mismatch!")
        print(f"   Expected: {expected_training_countries}")
        print(f"   Actual: {actual_training_countries}")
    else:
        print(f"   Confirmed: Training limited to golden cohort countries only")
    
    # Step 3: Prepare training data with best configuration
    print("\n3. Preparing training data...")
    
    # Fill soil data
    soil_cols = [col for col in training_yield_df.columns if col.startswith('T_')]
    for col in soil_cols:
        training_yield_df[col].fillna(training_yield_df[col].median(), inplace=True)
        prediction_rs_df[col].fillna(prediction_rs_df[col].median(), inplace=True)
    
    # Add trends
    trend_df = compute_national_trend(admin0_df)
    training_yield_df = training_yield_df.merge(trend_df, on=['country', 'year'], how='left')
    
    # Apply preprocessing to training data using dynamic allowed_prefixes filtering
    training_rs_df, spatial_features = create_spatial_features(training_rs_df, BEST_CONFIG['space_config'])
    essential_env_features = [col for col in training_rs_df.columns if any(prefix in col for prefix in allowed_prefixes) and not col in spatial_features]
    print(f"   - Applying temporal z-scores to {len(essential_env_features)} environmental features...")
    training_rs_df, temporal_z_features = add_temporal_loc10_zscores(training_rs_df, essential_env_features, window=10)

    # Determine PCA profile features from RAW environmental features (before spatial processing)
    # This ensures consistency between training and prediction datasets
    raw_env_features = [col for col in raw_rs_df.columns if any(prefix in col for prefix in allowed_prefixes)]

    # Merge features to training yield data
    merge_cols = [col for col in ['PCODE', 'year', 'country'] if col in training_rs_df.columns and col in training_yield_df.columns]
    feature_cols = spatial_features + temporal_z_features
    if merge_cols and feature_cols:
        training_yield_df = training_yield_df.merge(training_rs_df[merge_cols + feature_cols], on=merge_cols, how='left')
    
    # Apply GAEZ and interaction processing to training data
    training_yield_df, gaez_features = apply_gaez_processing_predict(training_yield_df, national_yields, BEST_CONFIG['gaez_method'])
    training_yield_df = create_target_variable(training_yield_df, BEST_CONFIG['objective'])
    
    if BEST_CONFIG['crop_area_use'] == 'exclude_crop_area':
        training_yield_df = training_yield_df.drop(columns=['crop_area_ha_adjusted'], errors='ignore')
    
    training_yield_df, interaction_features = create_interaction_features(training_yield_df, BEST_CONFIG['interaction_type'], spatial_features)
    
    # Collect features
    all_features = []
    all_features.extend([col for col in training_yield_df.columns if col.startswith('T_')])  # Soil
    all_features.extend(spatial_features)
    all_features.extend(temporal_z_features)
    all_features.extend(gaez_features)
    all_features.extend(interaction_features)
    if BEST_CONFIG['crop_area_use'] == 'include_crop_area' and 'crop_area_ha_adjusted' in training_yield_df.columns:
        all_features.append('crop_area_ha_adjusted')
    
    valid_features = [f for f in all_features if f in training_yield_df.columns]
    training_yield_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    training_yield_df.dropna(subset=['target'] + valid_features, inplace=True)
    
    print(f"   Training features: {len(valid_features)}")
    print(f"   Training data points after cleaning: {len(training_yield_df):,}")
    
    # Step 4: Create PCA profiles for analogue-based prediction
    print("\n4. Creating PCA profiles for country similarity...")
    
    # Get list of training countries
    training_countries_list = training_yield_df['country'].unique().tolist()
    
    # Create PCA profiles using RAW environmental features (consistent between training and prediction)
    profile_features = raw_env_features
    
    # Combine training and prediction data for PCA profiles - ensure all countries get profiles
    print(f"   Creating profiles using training countries: {training_countries_list}")
    pred_countries = [c for c in prediction_rs_df['country'].unique() if pd.notna(c)]
    print(f"   Creating profiles for prediction countries: {sorted(pred_countries)}")
    
    all_rs_data = pd.concat([training_rs_df, prediction_rs_df], ignore_index=True)
    all_countries = [c for c in all_rs_data['country'].unique() if pd.notna(c)]
    print(f"   Combined dataset countries: {sorted(all_countries)}")
    
    country_profiles = get_pca_profiles(all_rs_data, admin0_df, profile_features, n_components=5)
    print(f"   Countries with PCA profiles: {sorted(country_profiles.index.tolist())}")
    
    # Calculate distance threshold
    if len(training_countries_list) > 1:
        dists = [
            euclidean(country_profiles.loc[c1], country_profiles.loc[c2])
            for c1 in training_countries_list for c2 in training_countries_list if c1 != c2
        ]
        distance_threshold = np.percentile(dists, 50)
        print(f"   Distance threshold: {distance_threshold:.3f}")
    else:
        distance_threshold = 0.0
    
    # Step 6: Apply same preprocessing to prediction data
    print("\n6. Preprocessing prediction data...")
    
    # Apply exactly the same feature engineering pipeline as training data
    # 1. Spatial features 
    prediction_rs_df, spatial_features_pred = create_spatial_features(prediction_rs_df, BEST_CONFIG['space_config'])
    print(f"   - Created spatial features: {len(spatial_features_pred)}")
    
    # 2. Temporal z-scores (using same dynamic filtering as training)
    essential_env_features_pred = [col for col in prediction_rs_df.columns if any(prefix in col for prefix in allowed_prefixes) and not col in spatial_features_pred]
    print(f"   - Applying temporal z-scores to {len(essential_env_features_pred)} environmental features...")
    prediction_rs_df, temporal_z_features_pred = add_temporal_loc10_zscores(prediction_rs_df, essential_env_features_pred, window=10)
    print(f"   - Created temporal z-score features: {len(temporal_z_features_pred)}")
    
    # 3. GAEZ processing using average national yields
    prediction_rs_df, gaez_features_pred = apply_gaez_processing_predict(prediction_rs_df, national_yields, BEST_CONFIG['gaez_method'])
    print(f"   - Created GAEZ features: {len(gaez_features_pred)}")
    
    # 4. Handle crop area according to configuration
    if BEST_CONFIG['crop_area_use'] == 'exclude_crop_area':
        prediction_rs_df = prediction_rs_df.drop(columns=['crop_area_ha_adjusted'], errors='ignore')
    
    # 5. Interaction features (using same spatial features as training)
    prediction_rs_df, interaction_features_pred = create_interaction_features(prediction_rs_df, BEST_CONFIG['interaction_type'], spatial_features_pred)
    print(f"   - Created interaction features: {len(interaction_features_pred)}")
    
    # Clean up infinite values
    prediction_rs_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Verify that all training features are present in prediction data
    missing_features = [f for f in valid_features if f not in prediction_rs_df.columns]
    if missing_features:
        print(f"   WARNING: {len(missing_features)} training features missing in prediction data:")
        for feat in missing_features[:10]:  # Show first 10
            print(f"     - {feat}")
        if len(missing_features) > 10:
            print(f"     ... and {len(missing_features) - 10} more")
    
    available_features = [f for f in valid_features if f in prediction_rs_df.columns]
    print(f"   - Prediction features available: {len(available_features)}/{len(valid_features)}")
    
    # Update valid_features to only include features present in prediction data
    if missing_features:
        print(f"   - Updating feature list to exclude {len(missing_features)} missing features")
        valid_features = available_features
    
    # Set up analogue-based model components with available features
    print("\n7. Setting up analogue-based prediction model with available features...")
    scaler, _, selected_features = train_prediction_model_with_analogues(training_yield_df, valid_features)
    print(f"   - Model trained with {len(selected_features)} selected features")
    
    # Step 8: Make predictions using analogue-based approach
    print("\n8. Making predictions using analogue-based approach...")
    print("   Detailed training strategy for each target country:")
    
    # Predict country by country using analogue approach
    all_predictions = []
    
    # Group prediction data by country
    for target_country, country_data in prediction_rs_df.groupby('country'):
        if len(country_data) == 0:
            continue
            
        # Predict for this country using analogue-based approach
        country_predictions = predict_for_country_with_analogues(
            country_data,
            training_yield_df,
            valid_features,
            country_profiles,
            distance_threshold,
            scaler,
            selected_features
        )
        
        all_predictions.append(country_predictions)
    
    # Combine all predictions
    prediction_rs_df = pd.concat(all_predictions, ignore_index=True)
    
    # Convert to yield predictions using year-specific national trends
    print("\n   Diagnostic: National trend assignment (year-specific):")
    
    # Merge with year-specific trend data
    prediction_countries = prediction_rs_df['country'].unique()
    prediction_years = prediction_rs_df['year'].unique()
    print(f"   Processing {len(prediction_countries)} countries, {len(prediction_years)} years ({prediction_years.min()}-{prediction_years.max()})")
    
    # First, try to match with year-specific trends
    prediction_rs_df = prediction_rs_df.merge(
        all_trends_df[['country', 'year', 'national_trend']], 
        on=['country', 'year'], 
        how='left'
    )
    
    # Count matches and missing data
    matched_trend_data = prediction_rs_df['national_trend'].notna().sum()
    total_records = len(prediction_rs_df)
    print(f"   Year-specific trend matches: {matched_trend_data:,}/{total_records:,} ({100*matched_trend_data/total_records:.1f}%)")
    
    # For missing trend data, use fallback strategy
    missing_trend_mask = prediction_rs_df['national_trend'].isna()
    if missing_trend_mask.sum() > 0:
        print(f"   Applying fallback for {missing_trend_mask.sum():,} records without year-specific trends...")
        
        # Check which countries have some trend data vs none at all
        countries_with_trends = set(all_trends_df['country'].unique())
        countries_without_trends = set(prediction_countries) - countries_with_trends
        
        if countries_without_trends:
            print(f"   Countries without ANY trend data: {sorted(countries_without_trends)}")
            # Calculate regional average trend for countries without any data
            fao_values = [y for y in national_yields.values() if y > 0.5]
            regional_avg = np.mean(fao_values)
            print(f"   Using regional average fallback: {regional_avg:.2f} t/ha for missing countries")
            
            # Fill missing trends with regional average
            prediction_rs_df.loc[missing_trend_mask, 'national_trend'] = regional_avg
        else:
            # For countries with some trend data but missing years, interpolate or use country average
            print("   Filling missing years with country-specific averages...")
            country_avg_trends = all_trends_df.groupby('country')['national_trend'].mean()
            for country in prediction_countries:
                if country in country_avg_trends:
                    country_mask = (prediction_rs_df['country'] == country) & missing_trend_mask
                    if country_mask.sum() > 0:
                        prediction_rs_df.loc[country_mask, 'national_trend'] = country_avg_trends[country]
    
    # Rename for consistency with rest of code
    prediction_rs_df['national_trend_avg'] = prediction_rs_df['national_trend']
    
    print(f"   Final national trends: {prediction_rs_df['national_trend_avg'].min():.2f} - {prediction_rs_df['national_trend_avg'].max():.2f} t/ha")
    print(f"   Trend variation by country: {prediction_rs_df.groupby('country')['national_trend_avg'].std().mean():.3f} avg std dev")
    
    if BEST_CONFIG['objective'] == 'residuals':
        prediction_rs_df['pred_yield'] = prediction_rs_df['pred_target'] + prediction_rs_df['national_trend_avg']
    elif BEST_CONFIG['objective'] == 'normalized_residuals':
        prediction_rs_df['pred_yield'] = prediction_rs_df['pred_target'] * prediction_rs_df['national_trend_avg'] + prediction_rs_df['national_trend_avg']
    
    prediction_rs_df['pred_yield'] = prediction_rs_df['pred_yield'].clip(lower=0)
    
    print(f"\n   Predictions completed for {len(prediction_rs_df):,} locations")
    print(f"   Predicted yield range: {prediction_rs_df['pred_yield'].min():.2f} - {prediction_rs_df['pred_yield'].max():.2f} t/ha")
    print(f"   Mean predicted yield: {prediction_rs_df['pred_yield'].mean():.2f} t/ha")
    
    # Step 8: Try to merge with existing yield data (only for overlapping countries)
    print("\n8. Merging with existing yield data where available...")
    try:
        existing_yield_path = PROJECT_ROOT / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat" / "all_admin_gt0_merged_data.csv"
        existing_yield_df = pd.read_csv(existing_yield_path)
        
        # Only use existing yields for golden cohort countries (where we have ground truth)
        existing_yield_df_filtered = existing_yield_df[
            existing_yield_df['country'].isin(training_countries_set)
        ].copy()
        
        print(f"   - Found existing yields for {len(existing_yield_df_filtered):,} golden cohort observations")
        final_df = merge_with_existing_yields(prediction_rs_df, existing_yield_df_filtered)
    except Exception as e:
        print(f"   Warning: Could not load existing yield data ({str(e)}), using predictions only")
        final_df = prediction_rs_df.copy()
        final_df['has_ground_truth'] = False
        final_df['final_yield'] = final_df['pred_yield']
    
    # Step 9: Save results
    print("\n9. Saving results...")
    results_dir = PROJECT_ROOT / "Model" / "africa_results"
    results_dir.mkdir(exist_ok=True)
    
    # Select key columns for output - check what's available in the dataset
    base_columns = ['PCODE', 'country', 'year']
    if 'season_index' in final_df.columns:
        base_columns.append('season_index')
    
    output_columns = base_columns + ['pred_yield', 'has_ground_truth', 'final_yield']
    
    # Add existing yield if available
    if 'yield' in final_df.columns:
        output_columns.insert(-1, 'yield')
    
    # Add some key features for analysis (using prediction dataset column names)
    if 'crop_area_ha_adjusted' in final_df.columns:
        output_columns.append('crop_area_ha_adjusted')
    elif 'crop_area_ha' in final_df.columns:
        output_columns.append('crop_area_ha')
    if 'national_trend_avg' in final_df.columns:
        output_columns.append('national_trend_avg')
    
    available_output_cols = [col for col in output_columns if col in final_df.columns]
    output_df = final_df[available_output_cols].copy()
    
    # Save main results
    output_path = results_dir / "all_africa_maize_yield_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"   Main predictions saved to: {output_path}")
    
    # Save summary statistics
    summary_stats = {
        'total_locations': int(len(output_df)),
        'countries_covered': int(output_df['country'].nunique()),
        'years_covered': [int(y) for y in sorted(output_df['year'].unique())],
        'locations_with_ground_truth': int(output_df.get('has_ground_truth', pd.Series([False]*len(output_df))).sum()),
        'mean_predicted_yield': float(output_df['pred_yield'].mean()),
        'median_predicted_yield': float(output_df['pred_yield'].median()),
        'yield_range': [float(output_df['pred_yield'].min()), float(output_df['pred_yield'].max())],
        'countries': list(sorted([str(c) for c in output_df['country'].unique() if pd.notna(c)])),
        'model_config': BEST_CONFIG,
        'selected_features': selected_features,
        'training_countries': training_countries,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / "all_africa_predictions_summary.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"   Summary statistics saved to: {results_dir / 'all_africa_predictions_summary.json'}")
    
    # Country-level summary
    country_summary = output_df.groupby('country').agg({
        'pred_yield': ['count', 'mean', 'median', 'std'],
        'final_yield': ['mean', 'median']
    }).round(3)
    country_summary.columns = ['locations', 'pred_mean', 'pred_median', 'pred_std', 'final_mean', 'final_median']
    country_summary.to_csv(results_dir / "all_africa_predictions_by_country.csv")
    print(f"   Country-level summary saved to: {results_dir / 'all_africa_predictions_by_country.csv'}")
    
    print("="*80)
    print("PREDICTION COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"Total locations: {len(output_df):,}")
    print(f"Countries covered: {output_df['country'].nunique()}")
    print(f"Years: {min(output_df['year'])} - {max(output_df['year'])}")
    print(f"Mean predicted yield: {output_df['pred_yield'].mean():.2f} t/ha")
    if 'has_ground_truth' in output_df.columns:
        gt_count = output_df['has_ground_truth'].sum()
        pred_count = (~output_df['has_ground_truth']).sum()
        print(f"Locations with ground truth: {gt_count:,} ({gt_count/len(output_df)*100:.1f}%)")
        print(f"Locations with predictions only: {pred_count:,} ({pred_count/len(output_df)*100:.1f}%)")
    print("="*80)


if __name__ == "__main__":
    main()