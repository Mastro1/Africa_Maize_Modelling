import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

def load_data(pcode, base_path):
    """
    Loads all necessary data for a given PCODE.
    """
    # 1. Determine admin level and country from crop areas file
    crop_areas_path = base_path / "GADM" / "crop_areas" / "africa_crop_areas_glad_filtered.csv"
    crop_areas_df = pd.read_csv(crop_areas_path)
    
    pcode_info = crop_areas_df[crop_areas_df['PCODE'] == pcode]
    if pcode_info.empty:
        raise ValueError(f"PCODE {pcode} not found in crop areas file.")
    
    # Use the first row for info, it should be consistent for a pcode
    pcode_info_row = pcode_info.iloc[0]
    country = pcode_info_row['country']
    
    if pd.notna(pcode_info_row['admin_2']):
        admin_level = 2
        admin1_name = pcode_info_row['admin_1']
        admin2_name = pcode_info_row['admin_2']
        location_name = f"{country}, {admin1_name}, {admin2_name}"
    else:
        admin_level = 1
        admin1_name = pcode_info_row['admin_1']
        location_name = f"{country}, {admin1_name}"

    print(f"Determined Admin Level: {admin_level} for {location_name}")

    # 2. Construct file paths and load remote sensing data
    rs_path = base_path / "RemoteSensing" / "GADM" / "extractions"
    country_fs = country.replace(' ', '_').replace(',', '_').replace("'", "_")
    
    vi_path = rs_path / f"{country_fs}_admin{admin_level}_VI_timeseries_GADM.csv"
    ndwi_path = rs_path / f"{country_fs}_admin{admin_level}_NDWI_timeseries_GADM_monthly.csv"
    era5_path = rs_path / f"{country_fs}_admin{admin_level}_ERA5_timeseries_GADM.csv"
    
    try:
        vi_df = pd.read_csv(vi_path)
        ndwi_df = pd.read_csv(ndwi_path)
        era5_df = pd.read_csv(era5_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find remote sensing data for {country} at admin level {admin_level}. Searched for: {e.filename}")

    # Filter data for the specific pcode
    vi_df = vi_df[vi_df['PCODE'] == pcode].copy()
    ndwi_df = ndwi_df[ndwi_df['PCODE'] == pcode].copy()
    era5_df = era5_df[era5_df['PCODE'] == pcode].copy()
    
    # 3. Load crop calendar data
    calendar_path = base_path / "GADM" / "crop_calendar" / "maize_crop_calendar_extraction.csv"
    calendar_df = pd.read_csv(calendar_path)
    pcode_calendar = calendar_df[calendar_df['FNID'] == pcode]
    
    if pcode_calendar.empty:
        raise ValueError(f"Crop calendar data not found for PCODE {pcode} (FNID).")
    
    pcode_calendar = pcode_calendar.iloc[0]

    return {
        "vi": vi_df,
        "ndwi": ndwi_df,
        "era5": era5_df,
        "calendar": pcode_calendar,
        "location_name": location_name
    }

def process_data_for_ag_year(data, year):
    """
    Processes the loaded data for a single agricultural year to calculate climatologies and derived variables.
    """
    # Unpack data
    vi_df = data['vi']
    ndwi_df = data['ndwi']
    era5_df = data['era5']
    calendar = data['calendar']

    # --- 1. Pre-processing and Date Handling ---
    for df in [vi_df, ndwi_df, era5_df]:
        if 'month' in df.columns and 'date' not in df.columns:
            df.rename(columns={'month': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['doy'] = df['date'].dt.dayofyear
        df['year'] = df['date'].dt.year
    
    # Calculate Mean Temp for all years on the main dataframe
    era5_df['temp_mean'] = (era5_df['temperature_2m_max'] + era5_df['temperature_2m_min']) / 2
    
    # --- 2. Define Agricultural Year Period ---
    p_doy = calendar['Maize_1_planting']
    v_doy = calendar['Maize_1_vegetative']
    h_doy = calendar['Maize_1_harvest']
    eos_doy = calendar['Maize_1_endofseaso']

    if pd.isna(p_doy) or pd.isna(eos_doy):
        raise ValueError("Planting and End of Season DOY must be available.")
        
    season_crosses_year = p_doy > eos_doy
    planting_year = year - 1 if season_crosses_year else year
    
    planting_date = pd.Timestamp(planting_year, 1, 1) + pd.Timedelta(days=p_doy - 1)
    end_of_season_date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=eos_doy - 1)
    
    start_date = planting_date - pd.DateOffset(months=1)
    end_date = end_of_season_date + pd.DateOffset(months=1)

    # --- 3. Isolate Current Agricultural Year Data ---
    vi_year = vi_df[(vi_df['date'] >= start_date) & (vi_df['date'] <= end_date)].copy()
    ndwi_year = ndwi_df[(ndwi_df['date'] >= start_date) & (ndwi_df['date'] <= end_date)].copy()
    era5_year = era5_df[(era5_df['date'] >= start_date) & (era5_df['date'] <= end_date)].copy()
    
    # --- 4. Calculate Climatologies (10-year mean, min, max) ---
    climatology_dfs = {}
    vi_agg = {'NDVI_mean': ['mean', 'min', 'max'], 'EVI_mean': ['mean', 'min', 'max']}
    ndwi_agg = {'NDWI_mean': ['mean', 'min', 'max']}
    era5_agg = {
        'temperature_2m_min': ['mean', 'min', 'max'],
        'temperature_2m_max': ['mean', 'min', 'max'],
        'total_precipitation_sum': ['mean', 'min', 'max'],
        'volumetric_soil_water_layer_1': ['mean', 'min', 'max'],
        'temp_mean': ['mean']
    }
    aggregations = {'vi': vi_agg, 'ndwi': ndwi_agg, 'era5': era5_agg}

    for df, name in zip([vi_df, ndwi_df, era5_df], ['vi', 'ndwi', 'era5']):
        climo_df_period = df[(df['year'] >= 2002) & (df['year'] < year)].copy()
        climatology = climo_df_period.groupby('doy').agg(aggregations[name]).reset_index()
        climatology.columns = ['_'.join(col).strip('_') for col in climatology.columns.values]
        
        # Ensure climatology covers all days of the year by interpolating
        full_doy_range = pd.DataFrame({'doy': range(1, 367)})
        climatology = pd.merge(full_doy_range, climatology, on='doy', how='left').set_index('doy')
        climatology = climatology.interpolate(method='linear', limit_direction='both').reset_index()
        
        # Create a master date range for the full agricultural season
        master_dates = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date)})
        master_dates['doy'] = master_dates['date'].dt.dayofyear
        
        # Merge the master dates with the doy-based climatology
        final_climatology = pd.merge(master_dates, climatology, on='doy', how='left')
        climatology_dfs[name] = final_climatology
        
    # --- 5. Cumulative Precipitation Recalculation (from planting date) ---
    era5_year_season = era5_year[era5_year['date'] >= planting_date].copy()
    if not era5_year_season.empty:
        era5_year_season['precip_cumulative'] = era5_year_season['total_precipitation_sum'].cumsum()
        era5_year = pd.merge(era5_year, era5_year_season[['date', 'precip_cumulative']], on='date', how='left')

    climo_era5_season = climatology_dfs['era5'][climatology_dfs['era5']['date'] >= planting_date].copy()
    if not climo_era5_season.empty:
        climo_era5_season['precip_cumulative_mean'] = climo_era5_season['total_precipitation_sum_mean'].cumsum()
        climatology_dfs['era5'] = pd.merge(climatology_dfs['era5'], climo_era5_season[['date', 'precip_cumulative_mean']], on='date', how='left')

    # Merge all necessary climatology mean columns to era5_year for comparison plots
    era5_year = pd.merge(
        era5_year, 
        climatology_dfs['era5'][['date', 'volumetric_soil_water_layer_1_mean', 'precip_cumulative_mean']], 
        on='date', 
        how='left'
    )

    # --- 6. Get Calendar Dates as datetime objects ---
    harvest_date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=h_doy - 1)
    vegetative_date = None
    if pd.notna(v_doy):
        days_to_veg = (v_doy - p_doy) if v_doy >= p_doy else (365 - p_doy + v_doy)
        vegetative_date = planting_date + pd.Timedelta(days=int(days_to_veg))

    calendar_dates = {
        'planting': planting_date, 
        'vegetative': vegetative_date, 
        'harvest': harvest_date,
        'endofseaso': end_of_season_date
    }

    return {
        "vi_year": vi_year,
        "ndwi_year": ndwi_year,
        "era5_year": era5_year,
        "climatologies": climatology_dfs,
        "calendar_dates": calendar_dates,
        "start_date": start_date,
        "end_date": end_date
    }

def add_plot(fig, row, col, year_data, climo_data, year_col, climo_mean_col, climo_min_col, climo_max_col, calendar_dates, year, color='blue'):
    """Helper function to add a single plot with climatology range to the dashboard."""
    if year_data.empty or climo_data.empty: return

    climo_dates = climo_data['date']
    year_dates = year_data['date']

    # Climatology range
    fig.add_trace(go.Scatter(
        x=climo_dates, y=climo_data[climo_max_col], mode='lines', line=dict(width=0), showlegend=False
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=climo_dates, y=climo_data[climo_min_col], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(211, 211, 211, 0.5)', name='Climatology Range', showlegend=False
    ), row=row, col=col)
    
    # Climatology mean
    fig.add_trace(go.Scatter(
        x=climo_dates, y=climo_data[climo_mean_col], mode='lines',
        line=dict(color='grey', dash='dash'), name='10-Year Mean', showlegend=False
    ), row=row, col=col)
    
    # Current year data
    fig.add_trace(go.Scatter(
        x=year_dates, y=year_data[year_col], mode='lines',
        line=dict(color=color, width=2), name=f'{year} Data', showlegend=False
    ), row=row, col=col)

    # Calendar lines
    color_map = {'planting': 'green', 'vegetative': 'orange', 'harvest': 'red', 'endofseaso': 'saddlebrown'}
    for stage, stage_date in calendar_dates.items():
        if pd.notna(stage_date) and stage in color_map:
            fig.add_vline(x=stage_date, line_width=1, line_dash="dash", line_color=color_map.get(stage), row=row, col=col)

def add_comparison_plot(fig, row, col, year_data, year_col, mean_col, calendar_dates, year, year_color='black', mean_color='blue'):
    """Helper function for plots comparing year data to a mean with shaded difference."""
    if year_data.empty: return

    year_dates = year_data['date']
    year_values = year_data[year_col]
    mean_values = year_data[mean_col]
    
    if mean_values.isnull().all(): return # Skip if no mean data

    # Green fill for above average
    fig.add_trace(go.Scatter(
        x=year_dates, y=np.maximum(year_values, mean_values), mode='lines', line=dict(width=0), showlegend=False
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=year_dates, y=mean_values, fill='tonexty', fillcolor='rgba(144, 238, 144, 0.5)',
        mode='lines', line=dict(width=0), showlegend=False
    ), row=row, col=col)

    # Pink fill for below average
    fig.add_trace(go.Scatter(
        x=year_dates, y=mean_values, mode='lines', line=dict(width=0), showlegend=False
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=year_dates, y=np.minimum(year_values, mean_values), fill='tonexty', fillcolor='rgba(255, 182, 193, 0.5)',
        mode='lines', line=dict(width=0), showlegend=False
    ), row=row, col=col)
    
    # Plot main lines on top of shading
    fig.add_trace(go.Scatter(
        x=year_dates, y=year_values, mode='lines', line=dict(color=year_color, width=2), name=f'{year} Data', showlegend=False
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=year_dates, y=mean_values, mode='lines', line=dict(color=mean_color, width=1.5), name='10-Year Mean', showlegend=False
    ), row=row, col=col)

    # Calendar lines
    color_map = {'planting': 'green', 'vegetative': 'orange', 'harvest': 'red', 'endofseaso': 'saddlebrown'}
    for stage, stage_date in calendar_dates.items():
        if pd.notna(stage_date) and stage in color_map:
            fig.add_vline(x=stage_date, line_width=1, line_dash="dash", line_color=color_map.get(stage), row=row, col=col)


def generate_vegetation_dashboard(pcode="ZMB.2.3_2", year=2019):
    """
    Generates a vegetation dashboard with NDVI and EVI for a specific location and year.
    """
    print(f"Generating vegetation dashboard for PCODE: {pcode}, Year: {year}")

    # Define paths
    base_path = Path.cwd()
    dashboard_dir = base_path / "Plots" / "output"
    dashboard_dir.mkdir(exist_ok=True)

    # --- 1. Data Loading ---
    print("Step 1: Loading data...")
    try:
        data = load_data(pcode, base_path)
        location_name = data["location_name"]
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Data Processing ---
    print("Step 2: Processing data...")
    processed_data = process_data_for_ag_year(data, year)

    # --- 3. Vegetation Visualization ---
    print("Step 3: Creating vegetation visualization...")

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("NDVI and EVI",)
    )

    # Unpack processed data for plotting
    vi_year = processed_data["vi_year"]
    climatologies = processed_data["climatologies"]
    calendar_dates = processed_data["calendar_dates"]
    start_date = processed_data["start_date"]
    end_date = processed_data["end_date"]

    # Add NDVI plot
    add_plot(fig, 1, 1, vi_year, climatologies['vi'], 'NDVI_mean', 'NDVI_mean_mean', 'NDVI_mean_min', 'NDVI_mean_max', calendar_dates, year, 'green')

    # Add EVI plot (if EVI_mean column exists and climatology data is available)
    if 'EVI_mean' in vi_year.columns and 'EVI_mean_mean' in climatologies['vi'].columns:
        add_plot(fig, 1, 1, vi_year, climatologies['vi'], 'EVI_mean', 'EVI_mean_mean', 'EVI_mean_min', 'EVI_mean_max', calendar_dates, year, '#fccd32')

    # --- 4. Final Touches and Save ---
    fig.update_layout(
        title_text=f"Vegetation Dashboard for {location_name} - {year}",
        title_x=0.5,
        height=600,
        width=900,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Times New Roman", size=12),
        bargap=0
    )
    fig.update_xaxes(tickformat='%b %Y', range=[start_date, end_date])

    output_filename = f"{pcode.replace('.', '_')}_{year}_vegetation_dashboard.html"
    os.makedirs(dashboard_dir, exist_ok=True)
    output_path = dashboard_dir / output_filename
    fig.write_html(output_path)

    print(f"Vegetation dashboard saved to: {output_path}")
    return output_path


def generate_climate_dashboard(pcode="ZMB.2.3_2", year=2019):
    """
    Generates a climate dashboard for a specific location and year.
    """
    print(f"Generating climate dashboard for PCODE: {pcode}, Year: {year}")

    base_path = Path.cwd()
    dashboard_dir = base_path / "Plots" / "output"
    dashboard_dir.mkdir(exist_ok=True)

    # --- 1. Data Loading ---
    print("Step 1: Loading data...")
    try:
        data = load_data(pcode, base_path)
        location_name = data["location_name"]
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Data Processing ---
    print("Step 2: Processing data...")
    processed_data = process_data_for_ag_year(data, year)

    # --- 3. Climate Visualization ---
    print("Step 3: Creating climate visualization...")

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Precipitation", "Cumulative Precipitation", "Soil Moisture",
            "Min Temperature", "Max Temperature", "Mean Temperature"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.06
    )

    # Unpack processed data for plotting
    era5_year = processed_data["era5_year"]
    climatologies = processed_data["climatologies"]
    calendar_dates = processed_data["calendar_dates"]
    start_date = processed_data["start_date"]
    end_date = processed_data["end_date"]

    # --- Add plots to the grid ---
    # Row 1: Precipitation
    if not era5_year.empty:
        fig.add_trace(go.Bar(
            x=era5_year['date'],
            y=era5_year['total_precipitation_sum'],
            marker_color='darkblue',
            marker_line_color='darkblue', # This is the fix
            showlegend=False
        ), row=1, col=1)
        color_map = {'planting': 'green', 'vegetative': 'orange', 'harvest': 'red', 'endofseaso': 'saddlebrown'}
        for stage, stage_date in calendar_dates.items():
            if pd.notna(stage_date) and stage in color_map:
                fig.add_vline(x=stage_date, line_width=1, line_dash="dash", line_color=color_map.get(stage), row=1, col=1)

    # Row 1: Cumulative Precipitation & Soil Moisture
    add_comparison_plot(fig, 1, 2, era5_year, 'precip_cumulative', 'precip_cumulative_mean', calendar_dates, year, year_color='darkblue', mean_color='black')
    add_comparison_plot(fig, 1, 3, era5_year, 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_1_mean', calendar_dates, year, year_color='brown', mean_color='black')

    # Row 2: Temperatures
    add_plot(fig, 2, 1, era5_year, climatologies['era5'], 'temperature_2m_min', 'temperature_2m_min_mean', 'temperature_2m_min_min', 'temperature_2m_min_max', calendar_dates, year, 'lightblue')
    add_plot(fig, 2, 2, era5_year, climatologies['era5'], 'temperature_2m_max', 'temperature_2m_max_mean', 'temperature_2m_max_min', 'temperature_2m_max_max', calendar_dates, year, 'orange')
    
    # Row 2: Mean Temperature Anomaly Plot
    if not era5_year.empty:
        temp_mean = era5_year['temp_mean']
        threshold = 283.15
        # Add threshold line as a shape for better subplot support
        fig.add_shape(
            type="line",
            x0=era5_year['date'].min(),
            x1=era5_year['date'].max(),
            y0=threshold,
            y1=threshold,
            line=dict(color="black", width=3, dash="solid"),
            row=2, col=3
        )

        # Add annotation above the threshold line
        fig.add_annotation(
            x=era5_year['date'].mean(),  # Center horizontally
            y=threshold + 1,  # Just above the line
            text="Base Temperature Maize: 283.15 K",
            showarrow=False,
            font=dict(size=10, color="black"),
            row=2, col=3
        )
        fig.add_trace(go.Scatter(
            x=era5_year['date'], y=np.maximum(temp_mean, threshold),
            mode='lines', line=dict(width=0), showlegend=False
        ), row=2, col=3)
        fig.add_trace(go.Scatter(
            x=era5_year['date'], y=[threshold] * len(era5_year['date']),
            fill='tonexty', fillcolor='rgba(255, 0, 0, 0.3)',
            mode='lines', line=dict(width=0), showlegend=False
        ), row=2, col=3)
        fig.add_trace(go.Scatter(
            x=era5_year['date'], y=temp_mean, mode='lines', 
            line=dict(color='red'), showlegend=False
        ), row=2, col=3)
        for stage, stage_date in calendar_dates.items():
            if pd.notna(stage_date) and stage in color_map:
                fig.add_vline(x=stage_date, line_width=1, line_dash="dash", line_color=color_map.get(stage), row=2, col=3)

    # --- 4. Final Touches and Save ---
    fig.update_layout(
        title_text=f"Climate Dashboard for {location_name} - {year}",
        title_x=0.5,
        height=600,
        width=1200, # Increased width for better viewing
        showlegend=True,
        legend=dict(title="Legend"),
        template="plotly_white",
        font=dict(family="Times New Roman", size=12),
        bargap=0
    )

    # Set y-axis range for mean temperature plot to make threshold visible
    threshold = 283.15
    fig.update_yaxes(range=[threshold - 5, era5_year['temp_mean'].max() + 5], row=2, col=3)

    # Add y-axis labels for all subplots
    fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Precipitation (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Soil Moisture (m³/m³)", row=1, col=3)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=2)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=3)

    # Hide x-axis labels for top row subplots (only show on bottom row)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    
    # Add invisible traces for calendar legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Planting', line=dict(color='green', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Vegetative', line=dict(color='orange', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Harvest', line=dict(color='red', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='End of Season', line=dict(color='saddlebrown', dash='dash', width=2)))

    fig.update_xaxes(tickformat='%b %Y', range=[start_date, end_date])

    output_filename = f"{pcode.replace('.', '_')}_{year}_climate_dashboard.html"
    os.makedirs(dashboard_dir, exist_ok=True)
    output_path = dashboard_dir / output_filename
    fig.write_html(output_path)

    print(f"Climate dashboard saved to: {output_path}")
    return output_path


def generate_dashboard(pcode="ZMB.2.3_2", year=2019):
    """
    Generates separate vegetation and climate dashboards for a specific location and year.
    """
    print(f"Generating separate dashboards for PCODE: {pcode}, Year: {year}")

    # Generate vegetation dashboard
    #veg_path = generate_vegetation_dashboard(pcode, year)

    # Generate climate dashboard
    climate_path = generate_climate_dashboard(pcode, year)

    print(f"\nDashboards generated:")
    #print(f"Vegetation: {veg_path}")
    print(f"Climate: {climate_path}")

    return climate_path


if __name__ == "__main__":
    # Example usage - generates both vegetation and climate dashboards
    climate_path = generate_dashboard(pcode="ZMB.2.3_2", year=2018)
    print(f"Generated dashboards at: {climate_path}")
