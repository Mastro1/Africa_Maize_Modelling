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

    try:
        vi_df = pd.read_csv(vi_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find remote sensing data for {country} at admin level {admin_level}. Searched for: {e.filename}")

    # Filter data for the specific pcode
    vi_df = vi_df[vi_df['PCODE'] == pcode].copy()

    # 3. Load crop calendar data
    calendar_path = base_path / "GADM" / "crop_calendar" / "maize_crop_calendar_extraction.csv"
    calendar_df = pd.read_csv(calendar_path)
    pcode_calendar = calendar_df[calendar_df['FNID'] == pcode]

    if pcode_calendar.empty:
        raise ValueError(f"Crop calendar data not found for PCODE {pcode} (FNID).")

    pcode_calendar = pcode_calendar.iloc[0]

    return {
        "vi": vi_df,
        "calendar": pcode_calendar,
        "location_name": location_name
    }


def process_data_for_ag_year(data, current_year, years_back=5):
    """
    Processes VI data for agricultural years, creating a continuous DOY for plotting.
    """
    vi_df = data['vi']
    calendar = data['calendar']

    vi_df['date'] = pd.to_datetime(vi_df['date'])
    vi_df['doy'] = vi_df['date'].dt.dayofyear
    vi_df['year'] = vi_df['date'].dt.year

    p_doy = calendar['Maize_1_planting']
    h_doy = calendar['Maize_1_harvest']
    v_doy = calendar['Maize_1_vegetative']
    eos_doy = calendar['Maize_1_endofseaso']

    if pd.isna(p_doy) or pd.isna(eos_doy):
        raise ValueError("Planting and End of Season DOY must be available.")

    season_crosses_year = p_doy > eos_doy

    # --- Create a continuous 'plot_doy' for x-axis based on planting date ---
    def calculate_plot_doy(doy, year, planting_year):
        if season_crosses_year:
            if year == planting_year:
                return doy - p_doy
            else: # harvest year
                return (365 - p_doy) + doy
        else:  # Season within one calendar year
            return doy - p_doy

    # --- Filter data for historical agricultural years ---
    vi_historical = pd.DataFrame()
    years_to_process = range(current_year - years_back, current_year + 1)
    
    for harvest_year in years_to_process:
        planting_year = harvest_year - 1 if season_crosses_year else harvest_year
        
        planting_date = pd.Timestamp(planting_year, 1, 1) + pd.Timedelta(days=p_doy - 1)
        end_of_season_date = pd.Timestamp(harvest_year, 1, 1) + pd.Timedelta(days=eos_doy - 1)
        
        start_date = planting_date - pd.DateOffset(months=1)
        end_date = end_of_season_date + pd.DateOffset(months=1)

        season_df = vi_df[(vi_df['date'] >= start_date) & (vi_df['date'] <= end_date)].copy()
            
        if not season_df.empty:
            season_df['agricultural_year'] = harvest_year
            # The lambda function captures planting_year from the loop's scope
            season_df['plot_doy'] = season_df.apply(
                lambda row: calculate_plot_doy(row['doy'], row['year'], planting_year), 
                axis=1
            )
            vi_historical = pd.concat([vi_historical, season_df])
            
    # --- Process Climatology ---
    climo_df = vi_df[(vi_df['year'] >= 2002) & (vi_df['year'] < current_year)].copy()
    vi_agg = {'NDVI_mean': ['mean', 'min', 'max'], 'EVI_mean': ['mean', 'min', 'max']}
    climatology = climo_df.groupby('doy').agg(vi_agg).reset_index()
    climatology.columns = ['_'.join(col).strip('_') for col in climatology.columns.values]
    
    # Create a synthetic year for climatology plot_doy calculation
    climatology['year'] = np.where(climatology['doy'] >= p_doy, current_year -1, current_year) if season_crosses_year else current_year
    climatology['plot_doy'] = climatology.apply(lambda row: calculate_plot_doy(row['doy'], row['year'], current_year-1 if season_crosses_year else current_year), axis=1)
    climatology = climatology.sort_values('plot_doy')


    # --- Generate X-axis ticks and labels ---
    planting_year_current = current_year - 1 if season_crosses_year else current_year
    planting_date_current = pd.Timestamp(planting_year_current, 1, 1) + pd.Timedelta(days=p_doy - 1)
    end_of_season_date_current = pd.Timestamp(current_year, 1, 1) + pd.Timedelta(days=eos_doy - 1)
    
    start_range = planting_date_current - pd.DateOffset(months=1)
    end_range = end_of_season_date_current + pd.DateOffset(months=1)
    
    tick_dates = pd.date_range(start=start_range.replace(day=1), end=end_range, freq='MS')
    tickvals = []
    ticktext = []
    
    for date in tick_dates:
        plot_doy_val = (date - planting_date_current).days
        tickvals.append(plot_doy_val)
        ticktext.append(date.strftime('%b %Y'))

    # --- Calendar line positions ---
    harvest_date_current = pd.Timestamp(current_year, 1, 1) + pd.Timedelta(days=h_doy - 1)
    vegetative_plot_doy = None
    if pd.notna(v_doy):
        days_to_veg = (v_doy - p_doy) if v_doy >= p_doy else (365 - p_doy + v_doy)
        vegetative_plot_doy = days_to_veg

    harvest_plot_doy = (harvest_date_current - planting_date_current).days
    end_of_season_plot_doy = (end_of_season_date_current - planting_date_current).days
    
    calendar_plot_doy = {
        'planting': 0,
        'vegetative': vegetative_plot_doy,
        'harvest': harvest_plot_doy,
        'endofseaso': end_of_season_plot_doy
    }

    return {
        "vi_historical": vi_historical,
        "climatology": climatology,
        "years_to_include": list(years_to_process),
        "tickvals": tickvals,
        "ticktext": ticktext,
        "calendar_plot_doy": calendar_plot_doy
    }


def generate_ndvi_evi_historical_dashboard_matplotlib(pcode="ZMB.2.3_2", current_year=2019):
    """
    Generates a matplotlib historical NDVI and EVI dashboard showing past 5 years with transparency.
    """
    print(f"Generating matplotlib NDVI/EVI historical dashboard for PCODE: {pcode}, Year: {current_year}")

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
    print("Step 2: Processing historical data...")
    processed_data = process_data_for_ag_year(data, current_year)

    # --- 3. Visualization with Matplotlib ---
    print("Step 3: Creating matplotlib NDVI/EVI historical visualization...")

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    fig.suptitle(f"NDVI & EVI Historical Comparison for {location_name}", fontsize=16, fontfamily='Times New Roman')

    # Unpack processed data
    vi_historical = processed_data["vi_historical"]
    years_to_include = processed_data["years_to_include"]
    climatology = processed_data["climatology"]
    tickvals = processed_data["tickvals"]
    ticktext = processed_data["ticktext"]
    calendar_plot_doy = processed_data["calendar_plot_doy"]
    
    # Extract the end of season plot DOY for limiting x-axis
    end_of_season_plot_doy = calendar_plot_doy.get('endofseaso', max(tickvals) if tickvals else 0)
    
    # Add climatology plots first (background)
    # NDVI climatology range
    if 'NDVI_mean_max' in climatology.columns and 'NDVI_mean_min' in climatology.columns:
        ax.fill_between(climatology['plot_doy'], climatology['NDVI_mean_min'], climatology['NDVI_mean_max'], 
                       color='lightgray', alpha=0.5, label='10-Year Range')
        ax.plot(climatology['plot_doy'], climatology['NDVI_mean_mean'], color='gray', linestyle='--', label='10-Year Mean')
    
    # EVI climatology range if available
    if 'EVI_mean_max' in climatology.columns and 'EVI_mean_min' in climatology.columns:
        ax.fill_between(climatology['plot_doy'], climatology['EVI_mean_min'], climatology['EVI_mean_max'], 
                       color='lightgray', alpha=0.3)
        ax.plot(climatology['plot_doy'], climatology['EVI_mean_mean'], color='darkgray', linestyle='--')

    # Define colors for different years
    color_map = {
        current_year: {'ndvi': 'green', 'evi': '#fccd32'},
    }
    historical_colors = ['#B0C4DE', '#FFB6C1', '#98FB98', '#E6E6FA', '#F5DEB3'] # Light colors
    for i, year in enumerate(years_to_include[:-1]):
        color_map[year] = {
            'ndvi': historical_colors[i % len(historical_colors)],
            'evi': historical_colors[i % len(historical_colors)]
        }

    # Plot data for each agricultural year
    for year in years_to_include:
        year_data = vi_historical[vi_historical['agricultural_year'] == year]
        if year_data.empty:
            continue

        opacity = 1.0 if year == current_year else 0.5
        line_width = 3 if year == current_year else 1.5

        # Show both in legend for current year
        if year == current_year:
            if 'NDVI_mean' in year_data.columns:
                ax.plot(year_data['plot_doy'], year_data['NDVI_mean'], 
                       color=color_map[year]['ndvi'], linewidth=line_width, 
                       alpha=opacity, label=f'NDVI {year} (Current)')
            if 'EVI_mean' in year_data.columns:
                ax.plot(year_data['plot_doy'], year_data['EVI_mean'], 
                       color=color_map[year]['evi'], linewidth=line_width, 
                       alpha=opacity, label=f'EVI {year} (Current)')
        else:
            # Show one entry for historical years
            legend_shown = False
            if 'NDVI_mean' in year_data.columns:
                ax.plot(year_data['plot_doy'], year_data['NDVI_mean'], 
                       color=color_map[year]['ndvi'], linewidth=line_width, 
                       alpha=opacity, label=f'{year}' if not legend_shown else "")
                legend_shown = True
            if 'EVI_mean' in year_data.columns:
                ax.plot(year_data['plot_doy'], year_data['EVI_mean'], 
                       color=color_map[year]['evi'], linewidth=line_width, 
                       alpha=opacity, label=f'{year}' if not legend_shown else "")

    # Add calendar lines and annotations
    stage_names = {'planting': 'Planting', 'vegetative': 'Vegetative', 'harvest': 'Harvest', 'endofseaso': 'End of Season'}
    stage_colors = {'planting': 'green', 'vegetative': 'orange', 'harvest': 'red', 'endofseaso': 'saddlebrown'}
    
    for stage, plot_doy in calendar_plot_doy.items():
        if pd.notna(plot_doy):
            ax.axvline(x=plot_doy, color=stage_colors[stage], linestyle='--', linewidth=2, label='_nolegend_')
            
            # Add vertical text annotation
            y_pos = ax.get_ylim()[0]  # Position at bottom of y-axis
            ax.text(plot_doy + 1, y_pos + 0.01, stage_names[stage], 
                   rotation=90, color=stage_colors[stage], fontsize=12, 
                   fontfamily='Times New Roman', verticalalignment='bottom')

    # Set x-axis ticks and labels
    ax.set_xticks(tickvals)
    ax.set_xticklabels(ticktext, rotation=45, ha="right", fontfamily='Times New Roman')
    
    # Add grid
    ax.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # Add labels
    ax.set_xlabel("Date", fontfamily='Times New Roman')
    ax.set_ylabel("Index Value", fontfamily='Times New Roman')

    # Add legend
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, 
                       fontsize=10, title="Legend")
    legend.get_title().set_fontfamily('Times New Roman')

    # Set x-axis limits to ensure the plot doesn't extend too far
    ax.set_xlim(left=min(tickvals)+15, right=end_of_season_plot_doy + 15)  # +30 days (~1 month) after end of season

    # Adjust layout to accommodate legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save the dashboard
    output_filename = f"{pcode.replace('.', '_')}_{current_year}_ndvi_evi_historical.png"
    output_path = dashboard_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Matplotlib NDVI/EVI historical dashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    output_path = generate_ndvi_evi_historical_dashboard_matplotlib(pcode="ZMB.2.3_2", current_year=2018)
    print(f"Dashboard generated at: {output_path}")