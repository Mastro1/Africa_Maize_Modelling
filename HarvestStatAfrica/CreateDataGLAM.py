import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import time
from scipy import stats
import json
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Import visualization functions
from DataPlots import create_period_visualization, create_yield_correlation_plots
from FilterCropAreas import filter_crop_areas
from CropAreas import process_optimized_30m_smart, format_and_export_30m_complete


# Set plotly renderer for better display
pio.renderers.default = "browser"

def detect_admin_level(ground_data):
    """
    Detects the administrative level of the ground truth data.
    """
    ground_data_admin2_array = ground_data["admin_2"].copy()
    if ground_data_admin2_array.eq('none').all():
        print("   🔎 Admin level 1 detected.")
        return 1
    else:
        print("   🔎 Admin level 2 detected.")
        return 2

def filter_data_for_season(df, start_doy, end_doy, harvest_year):
    """
    Filters a DataFrame for a specific season, handling year boundaries.
    The season is defined by start and end days of the year (DOY) and a harvest year.
    """
    if pd.isna(start_doy) or pd.isna(end_doy):
        return pd.DataFrame()  # Return empty if period is undefined

    df_filtered = df.copy()
    if 'doy' not in df_filtered.columns:
        df_filtered['doy'] = df_filtered['date'].dt.dayofyear

    if start_doy <= end_doy:
        # Season is within a single calendar year, which is the harvest year
        season_year = harvest_year
        mask = (df_filtered['year'] == season_year) & (df_filtered['doy'] >= start_doy) & (df_filtered['doy'] <= end_doy)
    else:
        # Season crosses a year boundary
        planting_year = harvest_year - 1
        mask = (
            ((df_filtered['year'] == planting_year) & (df_filtered['doy'] >= start_doy)) |
            ((df_filtered['year'] == harvest_year) & (df_filtered['doy'] <= end_doy))
        )
    return df_filtered[mask]


def calculate_vi_metrics(vi_data, start_doy, end_doy, harvest_year):
    """
    Calculate integral, max, min, and range of NDVI/EVI for a given period.
    """
    vi_season = filter_data_for_season(vi_data, start_doy, end_doy, harvest_year)
    
    if vi_season.empty:
        return pd.Series({
            'ndvi_integral': np.nan, 'max_ndvi': np.nan, 'min_ndvi': np.nan, 'range_ndvi': np.nan,
            'evi_integral': np.nan, 'max_evi': np.nan, 'min_evi': np.nan, 'range_evi': np.nan
        })

    vi_season = vi_season.sort_values('date')
    
    # Create a continuous time delta for accurate trapezoidal integration across year boundaries
    time_delta = (vi_season['date'] - vi_season['date'].iloc[0]).dt.total_seconds() / (24 * 3600)
    
    ndvi_values = vi_season['NDVI_mean'].values
    evi_values = vi_season['EVI_mean'].values

    return pd.Series({
        'ndvi_integral': np.trapezoid(ndvi_values, time_delta) if len(ndvi_values) > 1 else 0,
        'max_ndvi': np.max(ndvi_values),
        'min_ndvi': np.min(ndvi_values),
        'range_ndvi': np.max(ndvi_values) - np.min(ndvi_values),
        'evi_integral': np.trapezoid(evi_values, time_delta) if len(evi_values) > 1 else 0,
        'max_evi': np.max(evi_values),
        'min_evi': np.min(evi_values),
        'range_evi': np.max(evi_values) - np.min(evi_values)
    })


def calculate_vi_max_doy(vi_data, start_doy, end_doy, harvest_year):
    """
    Calculate the day-of-year when maximum NDVI and EVI occur for a given period.
    """
    vi_season = filter_data_for_season(vi_data, start_doy, end_doy, harvest_year)
    
    if vi_season.empty:
        return pd.Series({
            'max_ndvi_doy': np.nan,
            'max_evi_doy': np.nan
        })

    vi_season = vi_season.sort_values('date')
    
    # Find indices of maximum values
    max_ndvi_idx = vi_season['NDVI_mean'].idxmax()
    max_evi_idx = vi_season['EVI_mean'].idxmax()
    
    # Get the day-of-year for maximum values
    max_ndvi_doy = vi_season.loc[max_ndvi_idx, 'doy']
    max_evi_doy = vi_season.loc[max_evi_idx, 'doy']

    return pd.Series({
        'max_ndvi_doy': max_ndvi_doy,
        'max_evi_doy': max_evi_doy
    })


def calculate_ndwi_metrics(ndwi_data, start_doy, end_doy, harvest_year):
    """
    Calculate integral, max, min, and range of NDWI for a given period.
    """
    ndwi_season = filter_data_for_season(ndwi_data, start_doy, end_doy, harvest_year)
    
    if ndwi_season.empty:
        return pd.Series({
            'ndwi_integral': np.nan, 'max_ndwi': np.nan, 'min_ndwi': np.nan, 'range_ndwi': np.nan
        })

    ndwi_season = ndwi_season.sort_values('date')
    time_delta = (ndwi_season['date'] - ndwi_season['date'].iloc[0]).dt.total_seconds() / (24 * 3600)
    ndwi_values = ndwi_season['NDWI_mean'].values

    return pd.Series({
        'ndwi_integral': np.trapezoid(ndwi_values, time_delta) if len(ndwi_values) > 1 else 0,
        'max_ndwi': np.max(ndwi_values),
        'min_ndwi': np.min(ndwi_values),
        'range_ndwi': np.max(ndwi_values) - np.min(ndwi_values),
    })


def calculate_gdd(era5_data, start_doy, end_doy, harvest_year, base_temperature=283.15):
    """
    Calculate Growing Degree Days (GDD) for a given period.
    """
    gdd_season = filter_data_for_season(era5_data, start_doy, end_doy, harvest_year)

    if gdd_season.empty:
        return np.nan

    gdd_season['GDD'] = (gdd_season['temperature_2m'] - base_temperature).clip(lower=0)
    return gdd_season['GDD'].sum()


def calculate_precipitation_metrics(era5_data, start_doy, end_doy, harvest_year):
    """
    Calculate total precipitation for a given period.
    """
    precip_season = filter_data_for_season(era5_data, start_doy, end_doy, harvest_year)
    
    if precip_season.empty:
        return np.nan
        
    return precip_season['total_precipitation_sum'].sum()


def calculate_max_consecutive_dry_days(era5_data, start_doy, end_doy, harvest_year, dry_threshold=1.0):
    """
    Calculate maximum consecutive dry days during the growing season.
    
    Parameters:
    - era5_data: DataFrame with ERA5 data including 'total_precipitation_sum'
    - start_doy, end_doy: Day of year for season boundaries  
    - harvest_year: Year of harvest
    - dry_threshold: Precipitation threshold in mm below which day is considered dry (default: 1.0mm)
    
    Returns:
    - max_consecutive_dry_days: Maximum number of consecutive days with precipitation < threshold
    """
    season_data = filter_data_for_season(era5_data, start_doy, end_doy, harvest_year)
    
    if season_data.empty:
        return np.nan
    
    # Sort by date to ensure proper sequence
    season_data = season_data.sort_values('date').reset_index(drop=True)
    
    # Convert precipitation to mm (assuming it's already in mm from ERA5)
    precip_mm = season_data['total_precipitation_sum'].values
    
    # Identify dry days (precipitation < threshold)
    dry_days = precip_mm < dry_threshold
    
    # Find maximum consecutive dry spell
    max_consecutive = 0
    current_consecutive = 0
    
    for is_dry in dry_days:
        if is_dry:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def calculate_heat_stress_days(era5_data, start_doy, end_doy, harvest_year, heat_threshold=308.15):
    """
    Calculate number of heat stress days during the growing season.
    
    Parameters:
    - era5_data: DataFrame with ERA5 data including 'temperature_2m_max'
    - start_doy, end_doy: Day of year for season boundaries
    - harvest_year: Year of harvest  
    - heat_threshold: Maximum temperature threshold in Kelvin above which maize experiences heat stress
                     (default: 308.15K = 35°C, critical for maize during flowering)
    
    Returns:
    - heat_stress_days: Number of days with maximum temperature > threshold
    """
    season_data = filter_data_for_season(era5_data, start_doy, end_doy, harvest_year)
    
    if season_data.empty:
        return np.nan
    
    # Count days where maximum temperature exceeds threshold
    heat_stress_days = (season_data['temperature_2m_max'] > heat_threshold).sum()
    
    return heat_stress_days


def calculate_temperature_metrics(era5_data, start_doy, end_doy, harvest_year):
    """
    Calculate comprehensive temperature metrics for a given period.
    
    Parameters:
    - era5_data: DataFrame with ERA5 data including temperature columns
    - start_doy, end_doy: Day of year for season boundaries
    - harvest_year: Year of harvest
    
    Returns:
    - Series with temperature metrics: mean, min, max, range for 2m, 2m_min, 2m_max
    """
    season_data = filter_data_for_season(era5_data, start_doy, end_doy, harvest_year)
    
    if season_data.empty:
        return pd.Series({
            'temp_2m_mean': np.nan, 'temp_2m_min': np.nan, 'temp_2m_max': np.nan, 'temp_2m_range': np.nan,
            'temp_2m_min_mean': np.nan, 'temp_2m_min_min': np.nan, 'temp_2m_min_max': np.nan, 'temp_2m_min_range': np.nan,
            'temp_2m_max_mean': np.nan, 'temp_2m_max_min': np.nan, 'temp_2m_max_max': np.nan, 'temp_2m_max_range': np.nan
        })
    
    # Temperature 2m (mean daily temperature)
    temp_2m = season_data['temperature_2m'].values
    # Temperature 2m min (daily minimum temperature)
    temp_2m_min = season_data['temperature_2m_min'].values  
    # Temperature 2m max (daily maximum temperature)
    temp_2m_max = season_data['temperature_2m_max'].values
    
    return pd.Series({
        # Mean daily temperature metrics
        'temp_2m_mean': np.mean(temp_2m),
        'temp_2m_min': np.min(temp_2m), 
        'temp_2m_max': np.max(temp_2m),
        'temp_2m_range': np.max(temp_2m) - np.min(temp_2m),
        # Daily minimum temperature metrics
        'temp_2m_min_mean': np.mean(temp_2m_min),
        'temp_2m_min_min': np.min(temp_2m_min),
        'temp_2m_min_max': np.max(temp_2m_min), 
        'temp_2m_min_range': np.max(temp_2m_min) - np.min(temp_2m_min),
        # Daily maximum temperature metrics
        'temp_2m_max_mean': np.mean(temp_2m_max),
        'temp_2m_max_min': np.min(temp_2m_max),
        'temp_2m_max_max': np.max(temp_2m_max),
        'temp_2m_max_range': np.max(temp_2m_max) - np.min(temp_2m_max)
    })


def calculate_soil_moisture_metrics(era5_data, start_doy, end_doy, harvest_year):
    """
    Calculate comprehensive soil moisture metrics for a given period.
    
    Parameters:
    - era5_data: DataFrame with ERA5 data including 'volumetric_soil_water_layer_1'
    - start_doy, end_doy: Day of year for season boundaries
    - harvest_year: Year of harvest
    
    Returns:
    - Series with soil moisture metrics: mean, min, max, range
    """
    season_data = filter_data_for_season(era5_data, start_doy, end_doy, harvest_year)
    
    if season_data.empty:
        return pd.Series({
            'soil_moisture_mean': np.nan, 'soil_moisture_min': np.nan, 
            'soil_moisture_max': np.nan, 'soil_moisture_range': np.nan
        })
    
    soil_values = season_data['volumetric_soil_water_layer_1'].values
    
    return pd.Series({
        'soil_moisture_mean': np.mean(soil_values),
        'soil_moisture_min': np.min(soil_values),
        'soil_moisture_max': np.max(soil_values),
        'soil_moisture_range': np.max(soil_values) - np.min(soil_values)
    })


def calculate_soil_moisture_anomalies(pcode_era5_data, harvest_year, p_doy, v_doy, h_doy, climatology):
    """
    Calculate soil moisture drought indicators for vegetation and grain-filling periods.
    """
    # Determine planting year
    planting_year = harvest_year - 1 if p_doy > h_doy else harvest_year
    season_planting_date = pd.to_datetime(f"{planting_year}-01-01") + pd.to_timedelta(p_doy - 1, unit='d')

    # Define analysis window for the current season
    season_end_date = season_planting_date + pd.Timedelta(days=300) # Generous window
    current_season_data = pcode_era5_data[
        (pcode_era5_data['date'] >= season_planting_date) & (pcode_era5_data['date'] <= season_end_date)
    ].copy()

    if current_season_data.empty:
        return pd.Series({'veg_max_drought': 0, 'veg_sum_drought': 0, 'flow_max_drought': 0, 'flow_sum_drought': 0})

    current_season_data['days_after_planting'] = (current_season_data['date'] - season_planting_date).dt.days
    
    # Exponential smoothing
    alpha = 0.3
    current_season_data['soil_moisture_smoothed'] = current_season_data['volumetric_soil_water_layer_1'].ewm(alpha=alpha, adjust=False).mean()

    # Merge with climatology and calculate anomaly
    current_season_data = current_season_data.merge(climatology, on='days_after_planting', how='left')
    current_season_data['soil_moisture_anomaly'] = current_season_data['soil_moisture_smoothed'] - current_season_data['climatology_soil_moisture']
    
    # Define periods in Days After Planting (DAP)
    dap_v = (pd.to_datetime(f"{harvest_year if v_doy > p_doy else harvest_year + 1}-01-01") + pd.to_timedelta(v_doy - 1, 'd') - season_planting_date).days
    dap_h = (pd.to_datetime(f"{harvest_year if h_doy > p_doy else harvest_year + 1}-01-01") + pd.to_timedelta(h_doy - 1, 'd') - season_planting_date).days

    # Veg period: planting to vegetative
    veg_data = current_season_data[(current_season_data['days_after_planting'] >= 0) & (current_season_data['days_after_planting'] <= dap_v)]
    veg_negative = veg_data[veg_data['soil_moisture_anomaly'] < 0]['soil_moisture_anomaly']
    veg_max_drought = veg_negative.min() if not veg_negative.empty else 0
    veg_sum_drought = veg_negative.sum() if not veg_negative.empty else 0

    # Flow period: vegetative to harvest
    flow_data = current_season_data[(current_season_data['days_after_planting'] > dap_v) & (current_season_data['days_after_planting'] <= dap_h)]
    flow_negative = flow_data[flow_data['soil_moisture_anomaly'] < 0]['soil_moisture_anomaly']
    flow_max_drought = flow_negative.min() if not flow_negative.empty else 0
    flow_sum_drought = flow_negative.sum() if not flow_negative.empty else 0

    return pd.Series({
        'veg_max_drought': veg_max_drought, 'veg_sum_drought': veg_sum_drought,
        'flow_max_drought': flow_max_drought, 'flow_sum_drought': flow_sum_drought
    })


def precompute_soil_climatology(pcode_era5_data, pcode_calendar):
    """Pre-computes the long-term average soil moisture for each day after planting."""
    climatology_data = []
    
    min_year = pcode_era5_data['year'].min()
    max_year = pcode_era5_data['year'].max()

    for year in range(min_year, max_year + 1):
        for season_idx in [1, 2]:
            p_doy = pcode_calendar[f'Maize_{season_idx}_planting']
            h_doy = pcode_calendar[f'Maize_{season_idx}_harvest']

            if pd.isna(p_doy) or pd.isna(h_doy):
                continue

            planting_year = year - 1 if p_doy > h_doy else year
            season_planting_date = pd.to_datetime(f"{planting_year}-01-01") + pd.to_timedelta(p_doy - 1, unit='d')
            
            season_end_date = season_planting_date + pd.Timedelta(days=300)
            
            season_data = pcode_era5_data[
                (pcode_era5_data['date'] >= season_planting_date) & (pcode_era5_data['date'] <= season_end_date)
            ].copy()

            if not season_data.empty:
                season_data['days_after_planting'] = (season_data['date'] - season_planting_date).dt.days
                season_data['harvest_year'] = year
                climatology_data.append(season_data[['days_after_planting', 'harvest_year', 'volumetric_soil_water_layer_1']])

    if not climatology_data:
        return None
        
    all_years_df = pd.concat(climatology_data)
    climatology = all_years_df.groupby('days_after_planting')['volumetric_soil_water_layer_1'].mean().reset_index()
    climatology.rename(columns={'volumetric_soil_water_layer_1': 'climatology_soil_moisture'}, inplace=True)
    
    return climatology


def adjust_crop_areas_to_fao(admin_level_df, admin0_df, country_calendar):
    """
    Proportionally adjust district-level crop areas to match FAO national totals,
    accounting for multiple seasons per PCODE without using ground truth data.
    """
    print("   📏 Adjusting sub-national crop areas to match FAO totals...")
    
    # Get FAO national area by year
    fao_areas = admin0_df[['year', 'area', 'production']].copy()
    fao_areas.rename(columns={'area': 'fao_national_area', 'production': 'fao_national_production'}, inplace=True)
    
    # Calculate sub-national area sums by year from our data
    district_sums = admin_level_df.groupby('year')[['area', 'production']].sum().reset_index()
    district_sums.rename(columns={'area': 'district_total_area', 'production': 'district_total_production'}, inplace=True)
    
    # Merge to create scaling factors for area and production
    area_comparison = pd.merge(fao_areas, district_sums, on='year', how='inner')
    area_comparison['scaling_factor_area'] = (area_comparison['fao_national_area'] / area_comparison['district_total_area']).replace([np.inf, -np.inf], 1)
    area_comparison['scaling_factor_production'] = (area_comparison['fao_national_production'] / area_comparison['district_total_production']).replace([np.inf, -np.inf], 1)

    # Apply scaling factors to the original admin-level data
    admin_adjusted = pd.merge(
        admin_level_df,
        area_comparison[['year', 'scaling_factor_area', 'scaling_factor_production']], 
        on='year', 
        how='left'
    )
    
    admin_adjusted['scaling_factor_area'] = admin_adjusted['scaling_factor_area'].fillna(1)
    admin_adjusted['scaling_factor_production'] = admin_adjusted['scaling_factor_production'].fillna(1)
    
    admin_adjusted['area_adjusted'] = admin_adjusted['area'] * admin_adjusted['scaling_factor_area']
    admin_adjusted['production_adjusted'] = admin_adjusted['production'] * admin_adjusted['scaling_factor_production']
    admin_adjusted['yield_adjusted'] = admin_adjusted['production_adjusted'] / admin_adjusted['area_adjusted']
    
    # NEW APPROACH: Adjust crop_area_ha with season-specific scaling factors
    # Calculate total crop area per season and year
    season_totals = admin_level_df.groupby(['year', 'season_index'])['crop_area_ha'].sum().reset_index()
    season_totals.rename(columns={'crop_area_ha': 'season_total_area'}, inplace=True)
    
    # Calculate overall total per year (sum across all seasons)
    yearly_totals = season_totals.groupby('year')['season_total_area'].sum().reset_index()
    yearly_totals.rename(columns={'season_total_area': 'total_all_seasons'}, inplace=True)
    
    # Merge to get proportions
    season_proportions = pd.merge(season_totals, yearly_totals, on='year')
    season_proportions['season_proportion'] = season_proportions['season_total_area'] / season_proportions['total_all_seasons']
    
    # Merge with FAO data to get target areas per season
    season_targets = pd.merge(season_proportions, fao_areas[['year', 'fao_national_area']], on='year')
    season_targets['fao_season_target'] = season_targets['fao_national_area'] * season_targets['season_proportion']
    
    # Calculate season-specific scaling factors
    season_targets['scaling_factor_crop_season'] = (
        season_targets['fao_season_target'] / season_targets['season_total_area']
    ).replace([np.inf, -np.inf], 1)
    
    # Merge season-specific scaling factors back to main dataframe
    admin_adjusted = pd.merge(
        admin_adjusted,
        season_targets[['year', 'season_index', 'scaling_factor_crop_season', 'season_proportion', 'fao_season_target']], 
        on=['year', 'season_index'], 
        how='left'
    )
    admin_adjusted['scaling_factor_crop_season'] = admin_adjusted['scaling_factor_crop_season'].fillna(1)
    
    # Apply season-specific scaling factor to each crop area
    admin_adjusted['crop_area_ha_adjusted'] = (
        admin_adjusted['crop_area_ha'] * admin_adjusted['scaling_factor_crop_season']
    )

    # Verification: Check that sum of adjusted areas matches FAO total
    if not admin_adjusted.empty:
        sample_year = admin_adjusted['year'].min()
        verif_df = admin_adjusted[admin_adjusted['year'] == sample_year]
        total_adjusted = verif_df['crop_area_ha_adjusted'].sum()
        fao_for_year = fao_areas[fao_areas['year'] == sample_year]['fao_national_area']
        fao_value = fao_for_year.values[0] if not fao_for_year.empty else "N/A"
        
        print(f"Verification for year {sample_year}:")
        print(f"  Total adjusted crop area = {total_adjusted:.2f}")
        print(f"  FAO national area = {fao_value}")
        print(f"  Difference = {abs(total_adjusted - fao_value) if fao_value != 'N/A' else 'N/A'}")
        
        # Show totals per season with different scaling factors
        season_verification = verif_df.groupby('season_index').agg({
            'crop_area_ha_adjusted': 'sum',
            'scaling_factor_crop_season': 'first',
            'season_proportion': 'first',
            'fao_season_target': 'first'
        }).reset_index()
        
        print(f"  Season-specific scaling results:")
        for _, row in season_verification.iterrows():
            season_count = len(verif_df[verif_df['season_index'] == row['season_index']])
            print(f"    Season {row['season_index']}: {row['crop_area_ha_adjusted']:.2f} ha "
                  f"(target: {row['fao_season_target']:.2f}, "
                  f"proportion: {row['season_proportion']:.1%}, "
                  f"scaling: {row['scaling_factor_crop_season']:.3f}, "
                  f"{season_count} PCODEs)")
        
        # Show sample of different scaling by season
        sample_seasons = verif_df.groupby('season_index').head(1)
        print(f"  Sample seasonal adjustments with different factors:")
        for _, row in sample_seasons.iterrows():
            print(f"    PCODE {row['PCODE']}, Season {row['season_index']}: "
                  f"{row['crop_area_ha']:.2f} -> {row['crop_area_ha_adjusted']:.2f} "
                  f"(factor: {row['scaling_factor_crop_season']:.3f})")

    print("   ✅ Area adjustment complete.")
    return admin_adjusted


def detect_yield_outliers(df):
    """
    Detects outliers in the 'yield' column and sets qc_flag = 1.
    Outliers are defined as values significantly above the 75th percentile.
    """
    if 'yield' not in df.columns or df['yield'].isnull().all():
        print("   ⚠️ Yield column not found or is all NaN. Skipping outlier detection.")
        return df

    print("   🔎 Detecting yield outliers...")
    
    # Calculate IQR for the 'yield' column
    Q1 = df['yield'].quantile(0.25)
    Q3 = df['yield'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Avoid issues with zero IQR
    if IQR == 0:
        print("   ⚠️ IQR is zero, cannot detect outliers. Skipping.")
        return df
        
    # Define a strict upper bound for outlier detection
    upper_bound = Q3 + 3.0 * IQR  # Using 3.0 multiplier for strictness
    
    # Identify outliers
    outlier_condition = df['yield'] > upper_bound
    num_outliers = outlier_condition.sum()
    
    if num_outliers > 0:
        print(f"   Found {num_outliers} yield outlier(s). Setting qc_flag to 1.")
        # Update qc_flag for outliers
        df.loc[outlier_condition, 'qc_flag'] = 1
    else:
        print("   No significant yield outliers found.")
        
    return df


def run_glam_data_preparation(country_name="Benin", crop_name="Maize", plot=True):
    """
    Main function to run the GLAM-based data preparation process with yield data integration.
    
    Parameters:
    - country_name: Name of the country
    - crop_name: Name of the crop (should be "Maize")
    - admin_level: Administrative level (1 or 2)
    - plot: Whether to create visualizations
    """
    
    print(f"--- Starting GLAM Data Preparation with Yield Data for {country_name} ---")
    
    base_path = Path.cwd()
    country_name_fs = country_name.replace(' ', '_').replace(',', '_').replace("'", "_")
    # Create output directories
    output_dir = base_path / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Data Loading ---
    try:
        print("   Loading input data...")
        
        # GLAM Crop Calendar
        glam_calendar_path = base_path / "HarvestStatAfrica" / "crop_calendar" / "maize_crop_calendar_extraction.csv"
        glam_calendar = pd.read_csv(glam_calendar_path)
        country_calendar = glam_calendar[glam_calendar['ADMIN0'] == country_name].copy()
        if country_calendar.empty: 
            raise ValueError(f"No GLAM calendar data for {country_name}")

        # Season Mapping
        season_mapping_path = base_path / "HarvestStatAfrica" / "crop_calendar" / "cropCalendarMatching.json"
        with open(season_mapping_path, 'r') as f:
            season_mapping = json.load(f)
        if country_name not in season_mapping:
            raise ValueError(f"No season mapping for {country_name}")

        # FAO Data
        fao_path = base_path / "FAOSTAT" / "faostat_maize.csv"
        fao_df = pd.read_csv(fao_path)
        fao = fao_df[fao_df['Area'] == country_name].copy()
        if fao.empty: 
            raise ValueError(f"No FAO data for {country_name}")

        # Ground Truth Yield Data
        ground_data_path = base_path / "HarvestStatAfrica" / "data" / "hvstat_africa_data_v1.0.csv"
        ground_data = pd.read_csv(ground_data_path)
        ground_data = ground_data[
            (ground_data['country'] == country_name) & 
            (ground_data['product'] == crop_name) &
            (ground_data['harvest_year'] >= 2001)
        ].copy()

        ground_data = ground_data.dropna(subset=['yield', 'area', 'production'])

        admin_level = detect_admin_level(ground_data)

        if ground_data.empty:
            raise ValueError(f"No ground truth data for {country_name}")

        # Crop Map Data from HarvestStatAfrica/crop_areas
        crop_map_filtered_path = base_path / "HarvestStatAfrica" / "crop_areas" / f"{country_name_fs}_crop_areas_glad_filtered.csv"
        crop_map_path = base_path / "HarvestStatAfrica" / "crop_areas" / f"{country_name_fs}_crop_areas_glad.csv"
        
        if not crop_map_path.exists():
            results, admin_level_map = process_optimized_30m_smart(country_name=country_name, batch_size=10)

            include_admin1_agg = (admin_level_map == 'admin2')  # Only aggregate if we have admin2 data
            output_file = format_and_export_30m_complete(results, admin_level_map, country_name, include_admin1_agg)

        if not crop_map_filtered_path.exists():
            crop_map_filtered_path = filter_crop_areas(country_name=country_name)
            
        crop_map = pd.read_csv(crop_map_filtered_path)
        if crop_map.empty: 
            raise ValueError(f"No crop area data for {country_name}")

        # Remote Sensing Data from RemoteSensing/HarvestStatAfrica/extractions
        rs_path = base_path / "RemoteSensing" / "HarvestStatAfrica" / "extractions"
        
        # Check for NDVI data to determine admin level
        ndvi_file = rs_path / f"{country_name_fs}_admin{admin_level}_VI_timeseries_GLAD.csv"
        if ndvi_file.exists():
            detected_admin_level = admin_level
        elif admin_level == 2:
            ndvi_file_admin1 = rs_path / f"{country_name_fs}_admin1_VI_timeseries_GLAD.csv"
            if ndvi_file_admin1.exists():
                detected_admin_level = 1
                ndvi_file = ndvi_file_admin1
                print(f"   Admin 2 not available, using Admin 1 data")
            else:
                raise FileNotFoundError(f"No Admin 1 or 2 remote sensing data found for {country_name}")
        else:
            raise FileNotFoundError(f"No Admin {admin_level} remote sensing data found for {country_name}")
        
        print(f"   Detected Admin Level: {detected_admin_level}")

        ndvi_path = rs_path / f"{country_name_fs}_admin{detected_admin_level}_VI_timeseries_GLAD.csv"
        ndwi_path = rs_path / f"{country_name_fs}_admin{detected_admin_level}_NDWI_timeseries_GLAD_monthly.csv"
        era5_path = rs_path / f"{country_name_fs}_admin{detected_admin_level}_ERA5_timeseries.csv"
        
        ndvi = pd.read_csv(ndvi_path)
        ndwi = pd.read_csv(ndwi_path)
        era5 = pd.read_csv(era5_path)

        print("   ✓ All input data loaded successfully.")

    except (FileNotFoundError, ValueError) as e:
        print(f"   ❌ ERROR: {e}")
        return

    # --- 2. Data Processing ---
    print("   Processing yield data and season mapping...")

    # Detect and flag outliers before handling QC flags
    ground_data = detect_yield_outliers(ground_data)

    # Clean and process ground truth data
    # Handle quality control flags by interpolation
    qc_flags = ground_data[ground_data['qc_flag'] == 1]
    print(f"   Found {len(qc_flags)} data points with quality flags (including outliers)")

    for idx, row in qc_flags.iterrows():
        fnid = row['fnid']
        year = row['harvest_year']
        
        # Get values from previous and next year for the same FNID
        prev_year = ground_data[(ground_data['fnid'] == fnid) & (ground_data['harvest_year'] == year - 1)]
        next_year = ground_data[(ground_data['fnid'] == fnid) & (ground_data['harvest_year'] == year + 1)]
        
        # Calculate average for area and production, then compute yield as production/area
        if not prev_year.empty and not next_year.empty:
            avg_area = (prev_year['area'].iloc[0] + next_year['area'].iloc[0]) / 2
            avg_production = (prev_year['production'].iloc[0] + next_year['production'].iloc[0]) / 2
            avg_yield = avg_production / avg_area if avg_area != 0 else np.nan
            ground_data.loc[idx, 'area'] = avg_area
            ground_data.loc[idx, 'production'] = avg_production
            ground_data.loc[idx, 'yield'] = avg_yield
        else:
            # If no previous or next year, use the mean of the fnid
            avg_area = ground_data[(ground_data['fnid'] == fnid)]['area'].mean()
            avg_production = ground_data[(ground_data['fnid'] == fnid)]['production'].mean()
            avg_yield = avg_production / avg_area if avg_area != 0 else np.nan
            ground_data.loc[idx, 'area'] = avg_area
            ground_data.loc[idx, 'production'] = avg_production
            ground_data.loc[idx, 'yield'] = avg_yield

    # Map seasons to Maize 1/2 using the JSON mapping
    ground_data['maize_season'] = ground_data['season_name'].map(season_mapping[country_name])
    ground_data = ground_data.dropna(subset=['maize_season'])

    # Aggregate by PCODE, year, and maize season (sum area/production, recalculate yield)
    print("   Aggregating data by PCODE, year, and maize season...")
    
    ground_data_grouped = ground_data.groupby(['fnid', 'harvest_year', 'maize_season']).agg({
        'area': 'sum',
        'production': 'sum',
        'yield': 'mean',
        'qc_flag': 'max'  # Keep original mean for comparison
    }).reset_index()
    
    # Recalculate yield from aggregated area and production
    ground_data_grouped['yield_recalc'] = ground_data_grouped['production'] / ground_data_grouped['area']
    ground_data_grouped.rename(columns={
        'fnid': 'PCODE', 
        'harvest_year': 'year',
        'yield': 'yield_original_mean',
        'yield_recalc': 'yield'
    }, inplace=True)

    # Create season_index column (1 for Maize 1, 2 for Maize 2)
    ground_data_grouped['season_index'] = ground_data_grouped['maize_season'].map({'Maize 1': 1, 'Maize 2': 2})

    # Pre-filter PCODEs for efficiency
    print("   Pre-filtering administrative regions...")
    calendar_pcodes = country_calendar[['FNID']].drop_duplicates().rename(columns={'FNID': 'PCODE'})
    map_pcodes = crop_map[['PCODE']].drop_duplicates()
    yield_pcodes = ground_data_grouped[['PCODE']].drop_duplicates()
    
    # Get intersection of all three datasets
    active_pcodes_df = pd.merge(calendar_pcodes, map_pcodes, on='PCODE', how='inner')
    active_pcodes_df = pd.merge(active_pcodes_df, yield_pcodes, on='PCODE', how='inner')
    active_pcodes = active_pcodes_df['PCODE'].unique()
    
    # Further filter by PCODEs present in remote sensing data
    original_rs_pcodes = ndvi['PCODE'].unique()
    valid_pcodes = [p for p in active_pcodes if p in original_rs_pcodes]
    
    print(f"   Found {len(original_rs_pcodes)} total regions in RS data.")
    print(f"   Processing {len(valid_pcodes)} active regions with calendar, crop area, yield, and RS data.")

    # --- 3. Remote Sensing Data Pre-processing ---
    print("   Pre-processing remote sensing data...")
    
    # FAO processing
    fao = fao[fao['Year'] >= 2001]
    fao = fao.pivot(index='Year', columns='Element', values='Value').reset_index()
    fao.rename(columns={'Production': 'production', 'Yield': 'yield', 'Area harvested': 'area'}, inplace=True)
    fao["yield"] = fao["yield"] / 1000  # convert to t/ha
    fao = fao.rename(columns={'Year': 'year'})
    fao['year'] = fao['year'].astype(int)

    # RS Data preprocessing
    for df in [ndvi, ndwi, era5]:
        if 'month' in df.columns and 'date' not in df.columns:
            df.rename(columns={'month': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['doy'] = df['date'].dt.dayofyear

    # --- 4. Main Processing Loop ---
    print("   Processing remote sensing data for each region and season...")
    
    # Separate datasets for remote sensing only and merged data
    remote_sensing_results = []
    merged_results = []
    
    # Determine year range from remote sensing data only (not limited by yield data)
    min_year = int(ndvi['year'].min())
    max_year = int(ndvi['year'].max())

    for pcode in valid_pcodes:
        pcode_calendar = country_calendar[country_calendar['FNID'] == pcode]
        if pcode_calendar.empty:
            continue
        pcode_calendar = pcode_calendar.iloc[0]

        # Filter RS data for the current PCODE
        pcode_ndvi = ndvi[ndvi['PCODE'] == pcode]
        pcode_ndwi = ndwi[ndwi['PCODE'] == pcode]
        pcode_era5 = era5[era5['PCODE'] == pcode]

        # Pre-compute soil moisture climatology
        soil_climatology = precompute_soil_climatology(pcode_era5, pcode_calendar)

        for harvest_year in range(min_year, max_year + 1):
            for season_idx in [1, 2]:
                # Get DOYs for the current season
                p_doy = pcode_calendar[f'Maize_{season_idx}_planting']
                v_doy = pcode_calendar[f'Maize_{season_idx}_vegetative']
                h_doy = pcode_calendar[f'Maize_{season_idx}_harvest']
                e_doy = pcode_calendar[f'Maize_{season_idx}_endofseaso']

                # Skip if season is not defined
                if pd.isna(p_doy):
                    continue

                # Check if we have yield data for this PCODE, year, and season
                yield_data = ground_data_grouped[
                    (ground_data_grouped['PCODE'] == pcode) & 
                    (ground_data_grouped['year'] == harvest_year) & 
                    (ground_data_grouped['season_index'] == season_idx)
                ]
                
                # Calculate metrics for different periods (always calculate for remote sensing data)
                vi_plant_harv = calculate_vi_metrics(pcode_ndvi, p_doy, h_doy, harvest_year)
                vi_plant_veg = calculate_vi_metrics(pcode_ndvi, p_doy, v_doy, harvest_year)
                vi_veg_harv = calculate_vi_metrics(pcode_ndvi, v_doy, h_doy, harvest_year)
                vi_plant_end = calculate_vi_metrics(pcode_ndvi, p_doy, e_doy, harvest_year)
                
                # Calculate day-of-year for maximum VI values during plant_end period
                vi_max_doy_plant_end = calculate_vi_max_doy(pcode_ndvi, p_doy, e_doy, harvest_year)

                # NDWI metrics for all periods (expand to match NDVI granularity)
                ndwi_plant_harv = calculate_ndwi_metrics(pcode_ndwi, p_doy, h_doy, harvest_year)
                ndwi_plant_veg = calculate_ndwi_metrics(pcode_ndwi, p_doy, v_doy, harvest_year)
                ndwi_veg_harv = calculate_ndwi_metrics(pcode_ndwi, v_doy, h_doy, harvest_year)
                ndwi_plant_end = calculate_ndwi_metrics(pcode_ndwi, p_doy, e_doy, harvest_year)

                # GDD metrics for all periods
                gdd_plant_harv = calculate_gdd(pcode_era5, p_doy, h_doy, harvest_year)
                gdd_plant_veg = calculate_gdd(pcode_era5, p_doy, v_doy, harvest_year)
                gdd_veg_harv = calculate_gdd(pcode_era5, v_doy, h_doy, harvest_year)
                gdd_plant_end = calculate_gdd(pcode_era5, p_doy, e_doy, harvest_year)
                
                # Precipitation metrics for all periods
                precip_plant_harv = calculate_precipitation_metrics(pcode_era5, p_doy, h_doy, harvest_year)
                precip_plant_veg = calculate_precipitation_metrics(pcode_era5, p_doy, v_doy, harvest_year)
                precip_veg_harv = calculate_precipitation_metrics(pcode_era5, v_doy, h_doy, harvest_year)
                precip_plant_end = calculate_precipitation_metrics(pcode_era5, p_doy, e_doy, harvest_year)
                
                # Consecutive dry days for all periods
                max_dry_days_plant_harv = calculate_max_consecutive_dry_days(pcode_era5, p_doy, h_doy, harvest_year)
                max_dry_days_plant_veg = calculate_max_consecutive_dry_days(pcode_era5, p_doy, v_doy, harvest_year)
                max_dry_days_veg_harv = calculate_max_consecutive_dry_days(pcode_era5, v_doy, h_doy, harvest_year)
                max_dry_days_plant_end = calculate_max_consecutive_dry_days(pcode_era5, p_doy, e_doy, harvest_year)
                
                # Heat stress days for all periods
                heat_stress_days_plant_harv = calculate_heat_stress_days(pcode_era5, p_doy, h_doy, harvest_year)
                heat_stress_days_plant_veg = calculate_heat_stress_days(pcode_era5, p_doy, v_doy, harvest_year)
                heat_stress_days_veg_harv = calculate_heat_stress_days(pcode_era5, v_doy, h_doy, harvest_year)
                heat_stress_days_plant_end = calculate_heat_stress_days(pcode_era5, p_doy, e_doy, harvest_year)
                
                # Temperature metrics for all periods
                temp_plant_harv = calculate_temperature_metrics(pcode_era5, p_doy, h_doy, harvest_year)
                temp_plant_veg = calculate_temperature_metrics(pcode_era5, p_doy, v_doy, harvest_year)
                temp_veg_harv = calculate_temperature_metrics(pcode_era5, v_doy, h_doy, harvest_year)
                temp_plant_end = calculate_temperature_metrics(pcode_era5, p_doy, e_doy, harvest_year)
                
                # Soil moisture metrics for all periods
                soil_plant_harv = calculate_soil_moisture_metrics(pcode_era5, p_doy, h_doy, harvest_year)
                soil_plant_veg = calculate_soil_moisture_metrics(pcode_era5, p_doy, v_doy, harvest_year)
                soil_veg_harv = calculate_soil_moisture_metrics(pcode_era5, v_doy, h_doy, harvest_year)
                soil_plant_end = calculate_soil_moisture_metrics(pcode_era5, p_doy, e_doy, harvest_year)
                
                # Soil moisture anomaly metrics
                if soil_climatology is not None:
                    soil_metrics = calculate_soil_moisture_anomalies(pcode_era5, harvest_year, p_doy, v_doy, h_doy, soil_climatology)
                else:
                    soil_metrics = pd.Series({'veg_max_drought': np.nan, 'veg_sum_drought': np.nan, 'flow_max_drought': np.nan, 'flow_sum_drought': np.nan})

                # Assemble remote sensing data (always add to remote sensing results)
                remote_sensing_result = {
                    'PCODE': pcode,
                    'year': harvest_year,
                    'season_index': season_idx,
                    # GDD for all periods
                    'gdd_plant_harv': gdd_plant_harv,
                    'gdd_plant_veg': gdd_plant_veg,
                    'gdd_veg_harv': gdd_veg_harv,
                    'gdd_plant_end': gdd_plant_end,
                    # Precipitation for all periods
                    'precip_plant_harv': precip_plant_harv,
                    'precip_plant_veg': precip_plant_veg,
                    'precip_veg_harv': precip_veg_harv,
                    'precip_plant_end': precip_plant_end,
                    # Consecutive dry days for all periods
                    'max_dry_days_plant_harv': max_dry_days_plant_harv,
                    'max_dry_days_plant_veg': max_dry_days_plant_veg,
                    'max_dry_days_veg_harv': max_dry_days_veg_harv,
                    'max_dry_days_plant_end': max_dry_days_plant_end,
                    # Heat stress days for all periods
                    'heat_stress_days_plant_harv': heat_stress_days_plant_harv,
                    'heat_stress_days_plant_veg': heat_stress_days_plant_veg,
                    'heat_stress_days_veg_harv': heat_stress_days_veg_harv,
                    'heat_stress_days_plant_end': heat_stress_days_plant_end,
                }
                
                # Add prefixed metrics for vegetation indices (all periods)
                remote_sensing_result.update({f'{k}_plant_harv': v for k, v in vi_plant_harv.items()})
                remote_sensing_result.update({f'{k}_plant_veg': v for k, v in vi_plant_veg.items()})
                remote_sensing_result.update({f'{k}_veg_harv': v for k, v in vi_veg_harv.items()})
                remote_sensing_result.update({f'{k}_plant_end': v for k, v in vi_plant_end.items()})
                
                # Add day-of-year for maximum VI values during plant_end period
                remote_sensing_result.update({f'{k}_plant_end': v for k, v in vi_max_doy_plant_end.items()})
                
                # Add NDWI metrics for all periods
                remote_sensing_result.update({f'{k}_plant_harv': v for k, v in ndwi_plant_harv.items()})
                remote_sensing_result.update({f'{k}_plant_veg': v for k, v in ndwi_plant_veg.items()})
                remote_sensing_result.update({f'{k}_veg_harv': v for k, v in ndwi_veg_harv.items()})
                remote_sensing_result.update({f'{k}_plant_end': v for k, v in ndwi_plant_end.items()})
                
                # Add temperature metrics for all periods
                remote_sensing_result.update({f'{k}_plant_harv': v for k, v in temp_plant_harv.items()})
                remote_sensing_result.update({f'{k}_plant_veg': v for k, v in temp_plant_veg.items()})
                remote_sensing_result.update({f'{k}_veg_harv': v for k, v in temp_veg_harv.items()})
                remote_sensing_result.update({f'{k}_plant_end': v for k, v in temp_plant_end.items()})
                
                # Add soil moisture metrics for all periods
                remote_sensing_result.update({f'{k}_plant_harv': v for k, v in soil_plant_harv.items()})
                remote_sensing_result.update({f'{k}_plant_veg': v for k, v in soil_plant_veg.items()})
                remote_sensing_result.update({f'{k}_veg_harv': v for k, v in soil_veg_harv.items()})
                remote_sensing_result.update({f'{k}_plant_end': v for k, v in soil_plant_end.items()})
                
                # Add soil moisture anomaly metrics (legacy format)
                remote_sensing_result.update(soil_metrics)
                
                remote_sensing_results.append(remote_sensing_result)

                # Only add to merged results if we have yield data
                if not yield_data.empty:
                    # Assemble results with yield data
                    yield_row = yield_data.iloc[0]
                    merged_result = remote_sensing_result.copy()  # Start with remote sensing data
                    merged_result.update({
                        'maize_season': yield_row['maize_season'],
                        'area': yield_row['area'],
                        'production': yield_row['production'],
                        'yield': yield_row['yield'],
                        'yield_original_mean': yield_row['yield_original_mean'],
                        'qc_flag': yield_row['qc_flag'],
                    })
                    merged_results.append(merged_result)

    if not remote_sensing_results:
        print("   ❌ No data processed. Exiting.")
        return

    # --- 5. Final Processing and Export ---
    print("   Finalizing data and exporting...")
    
    # Process remote sensing data
    remote_sensing_df = pd.DataFrame(remote_sensing_results)
    
    # Merge with crop area data for remote sensing data
    crop_map_admin = crop_map[['PCODE', 'year', 'crop_area_ha', 'total_area_ha', 'crop_percentage']]
    remote_sensing_df = pd.merge(remote_sensing_df, crop_map_admin, on=['PCODE', 'year'], how='left')

     # Add country name as the first column in merged_results_df
    remote_sensing_df.insert(0, 'country', country_name)

    
    # Process merged data (only combinations with yield data)
    merged_results_df = pd.DataFrame(merged_results)
    
    if not merged_results_df.empty:
        # Merge with crop area data for merged data
        merged_results_df = pd.merge(merged_results_df, crop_map_admin, on=['PCODE', 'year'], how='inner')

        # Adjust areas to match FAO totals for merged data
        #merged_results_df = adjust_crop_areas_to_fao(merged_results_df, fao, country_calendar)

        # Add country name as the first column in merged_results_df
        merged_results_df.insert(0, 'country', country_name)

        # Map PCODE to admin_1 and admin_2
        ground_data_match = ground_data[["fnid", "admin_1", "admin_2"]].rename(columns={"fnid": "PCODE"}).drop_duplicates()
        merged_results_df = pd.merge(merged_results_df, ground_data_match, on=['PCODE'], how='left')

        # Move admin_1 and admin_2 to the beginning of the dataframe, after country
        merged_results_df = merged_results_df[['country', 'admin_1', 'admin_2'] + [col for col in merged_results_df.columns if col not in ['country', 'admin_1', 'admin_2']]]

        # Add admin_level to the dataframe
        merged_results_df.insert(1, 'admin_level', admin_level)

        # Create Admin 0 (National) Data by aggregating from results and merging with FAO
        national_aggregated = merged_results_df.groupby(['year']).agg({
            'area': 'sum',
            'production': 'sum',
            'yield': 'mean'
        }).reset_index()
        national_aggregated['yield_recalc'] = national_aggregated['production'] / national_aggregated['area']
        
        # Merge with FAO data for complete national dataset
        national_df = pd.merge(fao, national_aggregated[['year', 'yield_recalc']], on='year', how='left')
        national_df['country'] = country_name
        national_df['PCODE'] = f"{country_name_fs}_COUNTRY"
    else:
        # If no merged data, create empty dataframes with proper structure
        merged_results_df = pd.DataFrame()
        national_df = pd.DataFrame()

    # Export files
    merged_output_path = output_dir / f"{country_name_fs}_admin{admin_level}_merged_data_GLAM.csv"
    rs_output_path = output_dir / f"{country_name_fs}_admin{admin_level}_remote_sensing_data_GLAM.csv"
    national_output_path = output_dir / f"{country_name_fs}_admin0_merged_data_GLAM.csv"

    # Export merged data (only combinations with yield data)
    if not merged_results_df.empty:
        merged_results_df.to_csv(merged_output_path, index=False)
        print(f"   ✓ Merged data exported to: {merged_output_path}")
    else:
        print("   ⚠️ No merged data to export (no yield data available)")
    
    # Export remote sensing data (complete time series for all locations and years)
    remote_sensing_df.to_csv(rs_output_path, index=False)
    print(f"   ✓ Remote sensing data exported to: {rs_output_path}")
    
    # Export national data
    if not national_df.empty:
        national_df.to_csv(national_output_path, index=False)
        print(f"   ✓ National data exported to: {national_output_path}")

    # --- 6. Visualizations ---
    if plot and not merged_results_df.empty:
        print("   Creating visualizations...")
        
        # Summary statistics
        print(f"\n=== SUMMARY ===")
        print(f"Final merged dataset shape: {merged_results_df.shape}")
        print(f"Years covered: {merged_results_df['year'].min()} - {merged_results_df['year'].max()}")
        print(f"Number of unique PCODEs: {merged_results_df['PCODE'].nunique()}")
        print(f"Number of seasons: {merged_results_df['season_index'].nunique()}")
        print(f"Maize seasons: {merged_results_df['maize_season'].unique()}")

        # Create yield correlation analysis
        correlations = create_yield_correlation_plots(merged_results_df, country_name)

        
        # Create period visualization for a sample PCODE
        if not ndvi.empty:
            sample_pcode = merged_results_df['PCODE'].iloc[0]
            sample_season = merged_results_df[merged_results_df['PCODE'] == sample_pcode].iloc[0]
            season_idx = int(sample_season['season_index'])
            harvest_year_viz = int(sample_season['year'])
            
            ndvi_sample = ndvi[ndvi['PCODE'] == sample_pcode].copy()
            
            # Get the calendar for this PCODE
            pcode_calendar_viz = country_calendar[country_calendar['FNID'] == sample_pcode].iloc[0]
            p_doy = pcode_calendar_viz[f'Maize_{season_idx}_planting']
            h_doy = pcode_calendar_viz[f'Maize_{season_idx}_harvest']
            
            print(f"Creating visualization for PCODE: {sample_pcode}, Season: {season_idx}, Year: {harvest_year_viz}")
            
            # Use the original visualization function adapted for DOY
            planting_year_viz = harvest_year_viz - 1 if p_doy > h_doy else harvest_year_viz
            harvest_year_actual = harvest_year_viz
            
            # Convert DOY to approximate months for the visualization
            planting_month_approx = int(p_doy / 30) + 1 if p_doy <= 365 else 12
            harvest_month_approx = int(h_doy / 30) + 1 if h_doy <= 365 else 12
            
            create_period_visualization(
                data=ndvi_sample,
                date_col='date',
                value_col='NDVI_mean',
                title=f"GLAM Vegetation Indices - {country_name} (PCODE: {sample_pcode}, Season: {season_idx})",
                planting_month=planting_month_approx,
                harvest_month=harvest_month_approx,
                planting_year_offset=-1 if p_doy > h_doy else 0,
                harvest_year_offset=0,
                periods={f"Maize {season_idx} Growing Season": (
                    pd.Timestamp(year=planting_year_viz, month=planting_month_approx, day=15),
                    pd.Timestamp(year=harvest_year_actual, month=harvest_month_approx, day=15)
                )},
                harvest_year_to_show=harvest_year_viz,
                additional_value_cols=['EVI_mean']
            )

    print(f"--- GLAM Data Preparation for {country_name} Completed ---")
    return merged_results_df if not merged_results_df.empty else remote_sensing_df


def worker(args):
    """
    Worker function for multiprocessing. Calls run_glam_data_preparation.
    """
    country_name, crop_name, plot = args
    try:
        print(f"🚀 Starting processing for {country_name}...")
        run_glam_data_preparation(
            country_name=country_name,
            crop_name=crop_name,
            plot=plot
        )
        print(f"✅ Successfully processed {country_name}.")
        return country_name, "Success", None
    except Exception as e:
        error_message = f"Error processing {country_name}: {e}"
        logging.error(error_message, exc_info=True)
        print(f"❌ {error_message}")
        return country_name, "Failure", str(e)


def run_glam_for_all_countries(crop_name="Maize", force_redo=False, plot=True):
    """
    Runs the GLAM data preparation for all maize-producing countries in the dataset.
    
    Parameters:
    - crop_name: The crop to process (default: "Maize").
    - force_redo: If True, re-processes all countries even if output files exist.
    - plot: If True, generates plots for each country (not recommended for batch processing).
    """
    # Setup logging
    output_dir = Path.cwd() / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / "processing_errors.log"
    
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',  # Overwrite log file each time
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("--- Starting GLAM Data Preparation for All Countries ---")

    # 1. Get list of countries and their admin levels
    try:
        data_path = Path.cwd() / "HarvestStatAfrica" / "data" / "hvstat_africa_data_v1.0.csv"
        df = pd.read_csv(data_path)
        maize_df = df[df['product'].str.contains(crop_name, na=False)]
        unique_countries = sorted(maize_df['country'].unique())

        countries_to_process = []
        for country in unique_countries:
            country_df = maize_df[maize_df['country'] == country]
            
            # Determine admin level based on 'admin_2' column
            admin_2_values = country_df['admin_2'].dropna().astype(str).str.lower()
            if admin_2_values.eq('none').all() or len(admin_2_values) == 0:
                admin_level = 1
            else:
                admin_level = 2

            # 2. Check for existing files if not force_redo
            country_name_fs = country.replace(' ', '_').replace(',', '_').replace("'", "_")
            
            # Since the script can auto-detect and fall back on admin level, we check both possible output files.
            output_file_admin1 = output_dir / f"{country_name_fs}_admin1_merged_data_GLAM.csv"
            output_file_admin2 = output_dir / f"{country_name_fs}_admin2_merged_data_GLAM.csv"
            
            if not force_redo and (output_file_admin1.exists() or output_file_admin2.exists()):
                print(f"⏭️ Skipping {country}, output already exists.")
                continue
            
            countries_to_process.append((country, crop_name, plot))

    except FileNotFoundError:
        error_msg = f"❌ ERROR: Could not find ground truth data at {data_path}"
        logging.error(error_msg)
        print(error_msg)
        return

    if not countries_to_process:
        print("✅ All countries are already processed.")
        return

    print(f"\nFound {len(countries_to_process)} countries to process.")
    
    # 3. Use multiprocessing to run the analysis
    num_processes = min(cpu_count(), len(countries_to_process))
    print(f"Running processing on {num_processes} cores...")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(worker, countries_to_process)

    # 4. Log results
    print("\n--- Processing Complete ---")
    successful = sorted([r[0] for r in results if r[1] == "Success"])
    failed = sorted([r[0] for r in results if r[1] == "Failure"])

    print(f"\n✅ Successful countries ({len(successful)}): {successful}")
    if failed:
        print(f"\n❌ Failed countries ({len(failed)}): {failed}")
        print(f"Check '{log_file_path}' for detailed error messages.")
    
    print("----------------------------")


if __name__ == "__main__":
    # Run the analysis for all countries
    run_glam_for_all_countries(force_redo=False, plot=False) 
    #run_glam_data_preparation(country_name="Kenya", crop_name="Maize", plot=False)
