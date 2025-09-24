import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import time
from scipy import stats
import json
from pathlib import Path
import multiprocessing
from itertools import product
import geopandas as gpd
import json

# Set plotly renderer for better display
pio.renderers.default = "browser"


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

def adjust_crop_areas_to_fao(admin_level_df, admin0_df, country_name):
    """
    Proportionally adjust district-level (Admin1/2) crop areas to match FAO national totals.
    """
    print("   📏 Adjusting sub-national crop areas to match FAO totals...")
    
    # Ensure a unique set of PCODEs for the adjustment
    if 'season_index' in admin_level_df.columns:
        admin_level_df_unique = admin_level_df.drop_duplicates(subset=['PCODE', 'year'])
    else:
        admin_level_df_unique = admin_level_df
    
    # Get FAO national area by year
    fao_areas = admin0_df[['year', 'area']].copy()
    fao_areas.rename(columns={'area': 'fao_national_area'}, inplace=True)
    
    # Calculate sub-national area sums by year from our data
    district_sums = admin_level_df_unique.groupby('year')['crop_area_ha'].sum().reset_index()
    district_sums.rename(columns={'crop_area_ha': 'district_total_area'}, inplace=True)
    
    # Merge to create scaling factors
    area_comparison = pd.merge(fao_areas, district_sums, on='year', how='inner')
    
    # Avoid division by zero if district total is 0
    area_comparison['scaling_factor'] = (area_comparison['fao_national_area'] / area_comparison['district_total_area']).replace([np.inf, -np.inf], 1)
    
    # Apply scaling factors to the original admin-level data
    admin_adjusted = pd.merge(
        admin_level_df,
        area_comparison[['year', 'scaling_factor']], 
        on='year', 
        how='left'
    )
    
    # Fill NaN scaling factors (for years without FAO data) with 1 to keep original area
    admin_adjusted['scaling_factor'] = admin_adjusted['scaling_factor'].fillna(1)
    
    # Create adjusted crop area column
    admin_adjusted['crop_area_ha_adjusted'] = admin_adjusted['crop_area_ha'] * admin_adjusted['scaling_factor']
    
    # Add country column
    admin_adjusted['country'] = country_name
    
    print("   ✅ Area adjustment complete.")
    return admin_adjusted

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


def worker_glam(task):
    """
    Worker function for multiprocessing GLAM data preparation.
    Takes a single task and calls the main data preparation function.
    """
    country_name = task
    try:
        run_glam_data_preparation(country_name=country_name)
        return (f"SUCCESS: Country={country_name} (GLAM)", None)
    except Exception as e:
        error_msg = f"FAILED: Country={country_name} (GLAM). Error: {type(e).__name__} - {e}"
        return (None, error_msg)


def run_glam_data_preparation(country_name="Benin", crop_name="Maize"):
    """
    Main function to run the data preparation process for a single country
    using the granular GLAM crop calendar.
    """
    print(f"--- Starting GLAM Data Preparation for {country_name} ---")
    
    base_path = Path.cwd()
    gadm_data_path = base_path / "GADM"
    country_name_fs = country_name.replace(' ', '_').replace(',', '_').replace("'", "_")

    # --- 1. Data Loading ---
    try:
        print("   Loading input data...")
        # GLAM Crop Calendar
        glam_calendar_path = gadm_data_path / "crop_calendar" / "maize_crop_calendar_extraction.csv"
        glam_calendar = pd.read_csv(glam_calendar_path)
        country_calendar = glam_calendar[glam_calendar['ADMIN0'] == country_name].copy()
        if country_calendar.empty: raise ValueError(f"No GLAM calendar data for {country_name}")

        # FAO Data
        fao_path = base_path / "FAOSTAT" / "faostat_maize.csv"
        fao_df = pd.read_csv(fao_path)
        fao = fao_df[fao_df['Area'] == country_name].copy()
        if fao.empty: raise ValueError(f"No FAO data for {country_name}")

        # Crop Map Data (for area adjustment)
        crop_map_path = gadm_data_path / "crop_areas" / "africa_crop_areas_glad_filtered.csv"
        crop_map_full = pd.read_csv(crop_map_path)
        crop_map = crop_map_full[crop_map_full['country'] == country_name].copy()
        if crop_map.empty: raise ValueError(f"No crop area data for {country_name}")

        # Remote Sensing Data
        # We need to determine which admin levels are available for the country
        rs_path = base_path / "RemoteSensing" / "GADM" / "extractions"
        available_admin_levels = []
        
        if (rs_path / f"{country_name_fs}_admin2_VI_timeseries_GADM.csv").exists():
            available_admin_levels.append(2)
        if (rs_path / f"{country_name_fs}_admin1_VI_timeseries_GADM.csv").exists():
            available_admin_levels.append(1)
            
        if not available_admin_levels:
            raise FileNotFoundError(f"No Admin 1 or 2 remote sensing data found for {country_name}")
        
        # FAO processing (needed for all admin levels)
        fao = fao[fao['Year'] >= 2001]
        fao = fao.pivot(index='Year', columns='Element', values='Value').reset_index()
        fao.rename(columns={'Production': 'production', 'Yield': 'yield', 'Area harvested': 'area'}, inplace=True)
        fao["yield"] = fao["yield"] / 1000  # convert to t/ha
        fao = fao.rename(columns={'Year': 'year'})
        fao['year'] = fao['year'].astype(int)
        
        print("   All input data loaded successfully.")
        
        # Process all available admin levels
        for admin_level in available_admin_levels:
            print(f"   Processing Admin Level: {admin_level}")
            process_admin_level_glam(country_name, admin_level, country_calendar, fao, crop_map, gadm_data_path)
    
    except (FileNotFoundError, ValueError) as e:
        print(f"   ERROR: {e}")
        return
    
    print(f"--- GLAM Data Preparation for {country_name} Finished ---")


def process_admin_level_glam(country_name, admin_level, country_calendar, fao, crop_map, gadm_data_path):
    """
    Process a specific admin level for GLAM data preparation.
    """

    base_path = Path.cwd()
    gadm_data_path = base_path / "GADM"

    country_name_fs = country_name.replace(' ', '_').replace(',', '_').replace("'", "_")
    
    try:
        # Remote Sensing Data for this admin level
        rs_path = base_path / "RemoteSensing" / "GADM" / "extractions"
        
        ndvi_path = rs_path / f"{country_name_fs}_admin{admin_level}_VI_timeseries_GADM.csv"
        ndwi_path = rs_path / f"{country_name_fs}_admin{admin_level}_NDWI_timeseries_GADM_monthly.csv"
        era5_path = rs_path / f"{country_name_fs}_admin{admin_level}_ERA5_timeseries_GADM.csv"
        
        ndvi = pd.read_csv(ndvi_path)
        ndwi = pd.read_csv(ndwi_path)
        era5 = pd.read_csv(era5_path)

        print(f"   ✓ Admin {admin_level} remote sensing data loaded successfully.")

    except (FileNotFoundError, ValueError) as e:
        print(f"   ERROR: {e}")
        return

    # --- 2. Pre-filter PCODEs for efficiency ---
    print("   Pre-filtering administrative regions...")
    calendar_pcodes = country_calendar[['FNID']].drop_duplicates().rename(columns={'FNID': 'PCODE'})
    map_pcodes = crop_map[['PCODE']].drop_duplicates()
    active_pcodes_df = pd.merge(calendar_pcodes, map_pcodes, on='PCODE', how='inner')
    active_pcodes = active_pcodes_df['PCODE'].unique()
    
    # Further filter by PCODEs that are actually present in the remote sensing data to avoid errors
    original_rs_pcodes = ndvi['PCODE'].unique()
    valid_pcodes = [p for p in active_pcodes if p in original_rs_pcodes]
    print(f"   Found {len(original_rs_pcodes)} total regions in RS data.")
    print(f"   Processing {len(valid_pcodes)} active regions with calendar, crop area, and RS data.")


    # --- 3. Data Pre-processing ---
    print("   Pre-processing remote sensing data...")
    # RS Data
    for df in [ndvi, ndwi, era5]:
        if 'month' in df.columns and 'date' not in df.columns:
            df.rename(columns={'month': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['doy'] = df['date'].dt.dayofyear
    
    # --- 4. Main Processing Loop ---
    all_results = []
    unique_pcodes = valid_pcodes # Use the pre-filtered list of PCODEs
    min_year = ndvi['year'].min()
    max_year = ndvi['year'].max()

    print(f"   Processing {len(unique_pcodes)} PCODEs from {min_year} to {max_year}...")

    for pcode in unique_pcodes:
        pcode_calendar = country_calendar[country_calendar['FNID'] == pcode]
        # This check should not be necessary now, but as a safeguard:
        if pcode_calendar.empty:
            continue
        pcode_calendar = pcode_calendar.iloc[0]

        # Filter RS data for the current PCODE once
        pcode_ndvi = ndvi[ndvi['PCODE'] == pcode]
        pcode_ndwi = ndwi[ndwi['PCODE'] == pcode]
        pcode_era5 = era5[era5['PCODE'] == pcode]

        # Pre-compute soil moisture climatology for this PCODE
        soil_climatology = precompute_soil_climatology(pcode_era5, pcode_calendar)
        if soil_climatology is None:
            # print(f"   - WARNING: Could not compute soil climatology for PCODE {pcode}. Skipping soil metrics.")
            pass

        for harvest_year in range(min_year, max_year + 1):
            for season_idx in [1, 2]:
                # Get DOYs for the current season
                p_doy = pcode_calendar[f'Maize_{season_idx}_planting']
                v_doy = pcode_calendar[f'Maize_{season_idx}_vegetative']
                h_doy = pcode_calendar[f'Maize_{season_idx}_harvest']
                e_doy = pcode_calendar[f'Maize_{season_idx}_endofseaso']

                # Skip if season is not defined (no planting date)
                if pd.isna(p_doy):
                    continue
        
                # --- Calculate metrics for different periods ---
                # VI Metrics (NDVI & EVI) for all periods
                vi_plant_harv = calculate_vi_metrics(pcode_ndvi, p_doy, h_doy, harvest_year)
                vi_plant_veg = calculate_vi_metrics(pcode_ndvi, p_doy, v_doy, harvest_year)
                vi_veg_harv = calculate_vi_metrics(pcode_ndvi, v_doy, h_doy, harvest_year)
                vi_plant_end = calculate_vi_metrics(pcode_ndvi, p_doy, e_doy, harvest_year)
                
                # Check if we have valid NDVI data for this location-year-season
                vi_season_test = filter_data_for_season(pcode_ndvi, p_doy, e_doy, harvest_year)
                if vi_season_test.empty or vi_season_test['NDVI_mean'].isna().all():
                    continue  # Skip this location-year-season combination entirely
                
                # Calculate day-of-year for maximum VI values during plant_end period
                vi_max_doy_plant_end = calculate_vi_max_doy(pcode_ndvi, p_doy, e_doy, harvest_year)

                # NDWI Metrics for all periods
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
                
                # Soil Moisture Anomaly Metrics
                if soil_climatology is not None:
                    soil_metrics = calculate_soil_moisture_anomalies(pcode_era5, harvest_year, p_doy, v_doy, h_doy, soil_climatology)
                else:
                    soil_metrics = pd.Series({'veg_max_drought': np.nan, 'veg_sum_drought': np.nan, 'flow_max_drought': np.nan, 'flow_sum_drought': np.nan})

                # --- Assemble results ---
                result = {
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
                
                # Add prefixed vegetation indices metrics for all periods
                result.update({f'{k}_plant_harv': v for k, v in vi_plant_harv.items()})
                result.update({f'{k}_plant_veg': v for k, v in vi_plant_veg.items()})
                result.update({f'{k}_veg_harv': v for k, v in vi_veg_harv.items()})
                result.update({f'{k}_plant_end': v for k, v in vi_plant_end.items()})
                
                # Add day-of-year for maximum VI values during plant_end period
                result.update({f'{k}_plant_end': v for k, v in vi_max_doy_plant_end.items()})
                
                # Add NDWI metrics for all periods
                result.update({f'{k}_plant_harv': v for k, v in ndwi_plant_harv.items()})
                result.update({f'{k}_plant_veg': v for k, v in ndwi_plant_veg.items()})
                result.update({f'{k}_veg_harv': v for k, v in ndwi_veg_harv.items()})
                result.update({f'{k}_plant_end': v for k, v in ndwi_plant_end.items()})
                
                # Add temperature metrics for all periods
                result.update({f'{k}_plant_harv': v for k, v in temp_plant_harv.items()})
                result.update({f'{k}_plant_veg': v for k, v in temp_plant_veg.items()})
                result.update({f'{k}_veg_harv': v for k, v in temp_veg_harv.items()})
                result.update({f'{k}_plant_end': v for k, v in temp_plant_end.items()})
                
                # Add soil moisture metrics for all periods
                result.update({f'{k}_plant_harv': v for k, v in soil_plant_harv.items()})
                result.update({f'{k}_plant_veg': v for k, v in soil_plant_veg.items()})
                result.update({f'{k}_veg_harv': v for k, v in soil_veg_harv.items()})
                result.update({f'{k}_plant_end': v for k, v in soil_plant_end.items()})
                
                # Add soil moisture anomaly metrics (legacy format)
                result.update(soil_metrics)
                
                all_results.append(result)

    if not all_results:
        print(f"   ❌ No data processed for admin {admin_level}. Skipping.")
        return

    # --- 5. Final Aggregation and Export ---
    print(f"   Aggregating results for admin {admin_level} and exporting files...")
    results_df = pd.DataFrame(all_results)
    
    # Merge with crop area data
    crop_map_admin = crop_map.rename(columns={'PCODE':'FNID'})[['FNID', 'year', 'crop_area_ha', 'total_area_ha', 'crop_percentage']]
    results_df = pd.merge(results_df, crop_map_admin.rename(columns={'FNID':'PCODE'}), on=['PCODE', 'year'], how='left')

    # Adjust sub-national areas to match FAO totals
    results_df = adjust_crop_areas_to_fao(results_df, fao, country_name)
    
    # Move country column to first position
    cols = results_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('country')))
    results_df = results_df[cols]

    # Define output paths
    output_dir = base_path / "RemoteSensing" / "GADM" / "preprocessed"
    output_dir.mkdir(exist_ok=True)
    
    rs_output_path = output_dir / f"{country_name_fs}_admin{admin_level}_remote_sensing_data_GLAM.csv"
    results_df.to_csv(rs_output_path, index=False)
    print(f"   ✓ Admin {admin_level} remote sensing data exported to: {rs_output_path}")
    
    # Create national data only once (when processing admin level 2, or admin level 1 if admin 2 doesn't exist)
    admin2_path = rs_path / f"{country_name_fs}_admin2_VI_timeseries_GADM.csv"
    should_create_national = (admin_level == 2) or (admin_level == 1 and not admin2_path.exists())
    
    if should_create_national:
        # Create Admin 0 (National) Data
        has_admin2 = crop_map['admin_2'].notna().any()
        crop_map_for_agg = crop_map[crop_map['admin_2'].notna()] if has_admin2 else crop_map
        
        crop_map_agg = crop_map_for_agg.groupby('year').agg(
            total_area_ha=('total_area_ha', 'sum'),
            crop_area_ha=('crop_area_ha', 'sum')
        ).reset_index()
        crop_map_agg['crop_percentage'] = (crop_map_agg['crop_area_ha'] / crop_map_agg['total_area_ha']).fillna(0)

        # Merge with FAO data  
        national_df = pd.merge(fao, crop_map_agg[['year', 'crop_area_ha', 'crop_percentage']], on='year', how='inner')
        national_df['country'] = country_name
        national_df['PCODE'] = f"{country_name_fs}_COUNTRY"
        
        # Move country column to first position
        cols = national_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('country')))
        national_df = national_df[cols]
        
        national_output_path = output_dir / f"{country_name_fs}_admin0_merged_data_GLAM.csv"
        national_df.to_csv(national_output_path, index=False)
        print(f"   ✓ National data exported to: {national_output_path}")


def process_all_africa_glam(force_redo=False):
    """
    Process all African countries using GLAM crop calendar data with parallel processing.
    Automatically detects available admin levels and processes both admin1 and admin2 where available.
    """
    print("=== Starting Africa-Wide GLAM Data Preparation (Parallel) ===")
    
    log_file_path = "RemoteSensing/GADM/extractions/glam_data_creation_errors.log"
    errors = []

    # Load GLAM calendar to get list of countries
    base_path = Path.cwd()
    gadm_data_path = base_path / "GADM"
    
    try:
        glam_calendar_path = gadm_data_path / "crop_calendar" / "maize_crop_calendar_extraction.csv"
        glam_calendar = pd.read_csv(glam_calendar_path)
        available_countries = sorted(glam_calendar['ADMIN0'].unique())
        print(f"✅ Found {len(available_countries)} countries in GLAM calendar")
    except FileNotFoundError:
        print("❌ CRITICAL: GLAM calendar file not found. Aborting.")
        return
    
    # Check which countries have remote sensing data available
    rs_path = base_path / "RemoteSensing" / "GADM" / "extractions"
    final_countries = []
    
    for country in available_countries:
        country_name_fs = country.replace(' ', '_').replace(',', '_').replace("'", "_")
        
        # Check if output already exists and force_redo is False
        if not force_redo:
            output_dir = base_path / "RemoteSensing" / "GADM" / "preprocessed"
            admin1_exists = (output_dir / f"{country_name_fs}_admin1_remote_sensing_data_GLAM.csv").exists()
            admin2_exists = (output_dir / f"{country_name_fs}_admin2_remote_sensing_data_GLAM.csv").exists()
            national_exists = (output_dir / f"{country_name_fs}_admin0_merged_data_GLAM.csv").exists()
            
            # Skip if files already exist
            if (admin1_exists or admin2_exists) and national_exists:
                continue
        
        # Check if remote sensing data exists
        has_admin1 = (rs_path / f"{country_name_fs}_admin1_VI_timeseries_GADM.csv").exists()
        has_admin2 = (rs_path / f"{country_name_fs}_admin2_VI_timeseries_GADM.csv").exists()
        
        if has_admin1 or has_admin2:
            final_countries.append(country)
        else:
            error_msg = f"No remote sensing data found for {country}"
            errors.append(error_msg)
    
    if not final_countries:
        print("\n✅ All countries have already been processed or no data available.")
        return
    
    print(f"\nProcessing {len(final_countries)} countries with available data...")
    
    # Process countries in parallel
    num_processes = min(multiprocessing.cpu_count(), len(final_countries))
    print(f"Starting multiprocessing with {num_processes} worker processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for success_msg, error_msg in pool.imap_unordered(worker_glam, final_countries):
            if error_msg:
                print(f"--- {error_msg} ---")
                errors.append(error_msg)
            else:
                print(f"--- {success_msg} ---")
    
    # Write error log
    if errors:
        print(f"\nCompleted. Encountered {len(errors)} errors.")
        with open(log_file_path, 'w') as f:
            f.write("=== GLAM Data Creation Error Log ===\n\n")
            for error in sorted(errors):
                f.write(error + "\n")
        print(f"See error log for details: {log_file_path}")
    else:
        print("\n✅ All countries processed successfully with no errors!")


if __name__ == "__main__":
    # Process all Africa with GLAM calendar
    process_all_africa_glam(force_redo=False)
    
    # Or process single country
    #run_glam_data_preparation(country_name="Tanzania")