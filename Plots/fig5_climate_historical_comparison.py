import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
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
    
    # Calculate Mean Temp for all years on the main dataframe (convert to Celsius)
    era5_df['temp_mean'] = ((era5_df['temperature_2m_max'] + era5_df['temperature_2m_min']) / 2) - 273.15
    
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

    # Convert temperature columns from Kelvin to Celsius in current year data
    # Note: temp_mean is already converted when calculated, so exclude it
    temp_cols_year = [col for col in era5_year.columns if 'temperature' in col and 'temp_mean' not in col]
    for col in temp_cols_year:
        era5_year[col] = era5_year[col] - 273.15
    
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

    # Temperature conversion factor from Kelvin to Celsius
    temp_conversion = 273.15
    aggregations = {'vi': vi_agg, 'ndwi': ndwi_agg, 'era5': era5_agg}

    for df, name in zip([vi_df, ndwi_df, era5_df], ['vi', 'ndwi', 'era5']):
        climo_df_period = df[(df['year'] >= 2002) & (df['year'] < year)].copy()
        climatology = climo_df_period.groupby('doy').agg(aggregations[name]).reset_index()
        climatology.columns = ['_'.join(col).strip('_') for col in climatology.columns.values]

        # Convert temperature columns from Kelvin to Celsius
        if name == 'era5':
            temp_cols = [col for col in climatology.columns if 'temperature' in col or 'temp_mean' in col]
            for col in temp_cols:
                climatology[col] = climatology[col] - temp_conversion

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


def add_plot_matplotlib(ax, year_data, climo_data, year_col, climo_mean_col, climo_min_col, climo_max_col, calendar_dates, year, color='blue', title=None, ylabel=None):
    """Helper function to add a single plot with climatology range to the matplotlib dashboard."""
    if year_data.empty or climo_data.empty:
        return

    climo_dates = climo_data['date']
    year_dates = year_data['date']

    # Climatology range
    ax.fill_between(climo_dates, climo_data[climo_min_col], climo_data[climo_max_col], 
                    color='lightgray', alpha=0.5, label='_nolegend_')

    # Climatology mean
    ax.plot(climo_dates, climo_data[climo_mean_col], color='gray', linestyle='--', label='_nolegend_')

    # Current year data
    ax.plot(year_dates, year_data[year_col], color=color, linewidth=2, label='_nolegend_')

    # Calendar lines
    color_map = {'planting': 'green', 'vegetative': 'orange', 'harvest': 'red', 'endofseaso': 'saddlebrown'}
    for stage, stage_date in calendar_dates.items():
        if pd.notna(stage_date) and stage in color_map:
            ax.axvline(x=stage_date, color=color_map.get(stage), linestyle='--', linewidth=1)
            
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)  # Add horizontal grid lines
    if title:
        ax.set_title(title, fontfamily='Times New Roman')
    if ylabel:
        ax.set_ylabel(ylabel, fontfamily='Times New Roman')


def add_comparison_plot_matplotlib(ax, year_data, year_col, mean_col, calendar_dates, year, title=None, ylabel=None, year_color='black', mean_color='blue'):
    """Helper function for matplotlib plots comparing year data to a mean with shaded difference."""
    if year_data.empty:
        return

    year_dates = year_data['date']
    year_values = year_data[year_col]
    mean_values = year_data[mean_col]
    
    if mean_values.isnull().all():
        return  # Skip if no mean data

    # Green fill for above average
    ax.fill_between(year_dates, np.maximum(year_values, mean_values), mean_values, 
                   color='lightgreen', alpha=0.5)
    
    # Pink fill for below average
    ax.fill_between(year_dates, np.minimum(year_values, mean_values), mean_values, 
                   color='lightcoral', alpha=0.5)

    # Plot main lines on top of shading (without legend labels)
    ax.plot(year_dates, year_values, color=year_color, linewidth=2, label='_nolegend_')
    ax.plot(year_dates, mean_values, color=mean_color, linewidth=1.5, label='_nolegend_')

    # Calendar lines
    color_map = {'planting': 'green', 'vegetative': 'orange', 'harvest': 'red', 'endofseaso': 'saddlebrown'}
    for stage, stage_date in calendar_dates.items():
        if pd.notna(stage_date) and stage in color_map:
            ax.axvline(x=stage_date, color=color_map.get(stage), linestyle='--', linewidth=1)
            
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)  # Add horizontal grid lines
    if title:
        ax.set_title(title, fontfamily='Times New Roman')
    if ylabel:
        ax.set_ylabel(ylabel, fontfamily='Times New Roman')


def generate_climate_dashboard_matplotlib(pcode="ZMB.2.3_2", year=2019):
    """
    Generates a matplotlib climate dashboard for a specific location and year.
    """
    print(f"Generating matplotlib climate dashboard for PCODE: {pcode}, Year: {year}")

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

    # --- 3. Climate Visualization with Matplotlib ---
    print("Step 3: Creating matplotlib climate visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    # Set the font to Times New Roman globally
    plt.rcParams['font.family'] = 'Times New Roman'
    
    fig.suptitle(f"Climate Dashboard for {location_name} - {year}", fontsize=16, fontfamily='Times New Roman')

    # Unpack processed data for plotting
    era5_year = processed_data["era5_year"]
    climatologies = processed_data["climatologies"]
    calendar_dates = processed_data["calendar_dates"]
    start_date = processed_data["start_date"]
    end_date = processed_data["end_date"]

    # --- Add plots to the grid ---

    # Row 1: Precipitation (bar plot)
    if not era5_year.empty:
        # Calculate bar width based on the time span to ensure proper spacing
        if len(era5_year['date']) > 1:
            # Calculate the average time delta between dates
            time_delta = (era5_year['date'].iloc[-1] - era5_year['date'].iloc[0]) / len(era5_year)
            bar_width = max(0.8, time_delta.days * 0.8)  # Use 80% of the average day spacing
        else:
            bar_width = 1
            
        axes[0, 0].bar(era5_year['date'], era5_year['total_precipitation_sum'], 
                       color='darkblue', width=0.8, label=f'{year} Data', 
                       edgecolor='darkblue')  # Match the original styling with edge color
        color_map = {'planting': 'green', 'vegetative': 'orange', 'harvest': 'red', 'endofseaso': 'saddlebrown'}
        for stage, stage_date in calendar_dates.items():
            if pd.notna(stage_date) and stage in color_map:
                axes[0, 0].axvline(x=stage_date, color=color_map.get(stage), linestyle='--', linewidth=1)
        
        axes[0, 0].set_title("Precipitation", fontfamily='Times New Roman')
        axes[0, 0].set_ylabel("Precipitation (mm)", fontfamily='Times New Roman')
        # Set x-axis formatting for this subplot (but make labels invisible)
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[0, 0].tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
        axes[0, 0].grid(True, axis='y', linestyle='-', alpha=0.3)  # Add horizontal grid lines

    # Row 1: Cumulative Precipitation (comparison plot)
    add_comparison_plot_matplotlib(axes[0, 1], era5_year, 'precip_cumulative', 'precip_cumulative_mean', 
                                   calendar_dates, year, 
                                   title="Cumulative Precipitation", 
                                   ylabel="Cumulative Precipitation (mm)",
                                   year_color='darkblue', mean_color='black')
    # Set x-axis formatting for this subplot (but make labels invisible)
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[0, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0, 1].tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
    axes[0, 1].grid(True, axis='y', linestyle='-', alpha=0.3)  # Add horizontal grid lines

    # Row 1: Soil Moisture (comparison plot)
    add_comparison_plot_matplotlib(axes[0, 2], era5_year, 'volumetric_soil_water_layer_1', 
                                   'volumetric_soil_water_layer_1_mean', calendar_dates, year, 
                                   title="Soil Moisture",
                                   ylabel="Soil Moisture (m³/m³)",
                                   year_color='brown', mean_color='black')
    # Set x-axis formatting for this subplot (but make labels invisible)
    axes[0, 2].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[0, 2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0, 2].tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
    axes[0, 2].grid(True, axis='y', linestyle='-', alpha=0.3)  # Add horizontal grid lines

    # Row 2: Min Temperature (with climatology range)
    add_plot_matplotlib(axes[1, 0], era5_year, climatologies['era5'], 'temperature_2m_min', 
                        'temperature_2m_min_mean', 'temperature_2m_min_min', 'temperature_2m_min_max', 
                        calendar_dates, year, 'lightblue',
                        title="Min Temperature",
                        ylabel="Temperature (°C)")
    # Set x-axis formatting for this subplot (labels should be visible)
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[1, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha="right", fontfamily='Times New Roman')
    axes[1, 0].grid(True, axis='y', linestyle='-', alpha=0.3)  # Add horizontal grid lines

    # Row 2: Max Temperature (with climatology range)
    add_plot_matplotlib(axes[1, 1], era5_year, climatologies['era5'], 'temperature_2m_max', 
                        'temperature_2m_max_mean', 'temperature_2m_max_min', 'temperature_2m_max_max', 
                        calendar_dates, year, 'orange',
                        title="Max Temperature",
                        ylabel="Temperature (°C)")
    # Set x-axis formatting for this subplot (labels should be visible)
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[1, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha="right", fontfamily='Times New Roman')
    axes[1, 1].grid(True, axis='y', linestyle='-', alpha=0.3)  # Add horizontal grid lines

    # Row 2: Mean Temperature Anomaly Plot
    if not era5_year.empty:
        temp_mean = era5_year['temp_mean']
        threshold = 10.0  # Base temperature for maize in Celsius (283.15 K - 273.15)
        
        # Add threshold line (without legend label)
        axes[1, 2].axhline(y=threshold, color='black', linewidth=3, linestyle='-', label='_nolegend_')
        
        # Plot temperature data (without legend label)
        axes[1, 2].plot(era5_year['date'], temp_mean, color='red', linewidth=2, label='_nolegend_')
        
        # Fill area above threshold
        axes[1, 2].fill_between(era5_year['date'], np.maximum(temp_mean, threshold), threshold, 
                               where=(temp_mean >= threshold), color='red', alpha=0.3)
        
        # Calendar lines
        for stage, stage_date in calendar_dates.items():
            if pd.notna(stage_date) and stage in color_map:
                axes[1, 2].axvline(x=stage_date, color=color_map.get(stage), linestyle='--', linewidth=1)
        
        axes[1, 2].set_title("Mean Temperature", fontfamily='Times New Roman')
        axes[1, 2].set_ylabel("Temperature (°C)", fontfamily='Times New Roman')
        axes[1, 2].text(era5_year['date'].iloc[len(era5_year['date'])//2], threshold + 1, 
                        'Base Temperature Maize: 10°C', fontsize=10, ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))
        # Set x-axis formatting for this subplot (labels should be visible)
        axes[1, 2].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axes[1, 2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(axes[1, 2].get_xticklabels(), rotation=45, ha="right", fontfamily='Times New Roman')
        axes[1, 2].grid(True, axis='y', linestyle='-', alpha=0.3)  # Add horizontal grid lines

    # Adjust layout and add legend
    plt.tight_layout()
    
    # Add overall legend at the bottom (only for calendar markers, not data series)
    from matplotlib.lines import Line2D
    calendar_handles = [
        Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='Planting'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=1, label='Vegetative'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Harvest'),
        Line2D([0], [0], color='saddlebrown', linestyle='--', linewidth=1, label='End of Season')
    ]
    
    fig.legend(calendar_handles, ['Planting', 'Vegetative', 'Harvest', 'End of Season'], 
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    
    # Save the dashboard
    output_filename = f"{pcode.replace('.', '_')}_{year}_climate_dashboard.png"
    output_path = dashboard_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Matplotlib climate dashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage - generates matplotlib climate dashboard
    climate_path = generate_climate_dashboard_matplotlib(pcode="ZMB.2.3_2", year=2018)
    print(f"Generated matplotlib climate dashboard at: {climate_path}")