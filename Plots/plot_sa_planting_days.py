import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_south_africa_planting_days(country_name, doy=True):
    """
    Generates and saves a side-by-side map plot of maize planting days.

    Parameters:
    - country_name (str): Name of the country to plot
    - doy (bool): If True, display day of year. If False, display months. Default is True.
    """
    # Set Times New Roman font globally
    plt.rcParams['font.family'] = 'Times New Roman'

    print("Loading data...")
    try:
        # Load the extracted calendar data and the admin boundaries
        calendar_data_path = 'GADM/crop_calendar/maize_crop_calendar_extraction.csv'
        admin_boundaries_path = 'GADM/gadm41_AFR_shp/gadm41_AFR_2_processed.shp'

        calendar_df = pd.read_csv(calendar_data_path)
        admin_gdf = gpd.read_file(admin_boundaries_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter for country
    sa_admin_gdf = admin_gdf[admin_gdf['ADMIN0'] == country_name]
    sa_calendar_df = calendar_df[calendar_df['ADMIN0'] == country_name]

    # Merge the datasets
    # The 'FNID' column should be the common key
    merged_gdf = sa_admin_gdf.merge(sa_calendar_df, on=['FNID', 'ADMIN0', 'ADMIN1', 'ADMIN2'], how='left')

    # Convert day of year to months if doy=False
    if not doy:
        def doy_to_month(doy_value):
            if pd.isna(doy_value) or doy_value is None:
                return pd.NA
            try:
                # Convert to float first, then int to handle any decimal values
                doy_int = int(float(doy_value))
                # Use datetime to convert day of year to month
                from datetime import datetime
                date = datetime.strptime(f'2020-{doy_int:03d}', '%Y-%j')
                return date.month
            except (ValueError, TypeError):
                return pd.NA

        # Apply conversion to both maize planting columns
        merged_gdf['Maize_1_planting'] = merged_gdf['Maize_1_planting'].apply(doy_to_month)
        merged_gdf['Maize_2_planting'] = merged_gdf['Maize_2_planting'].apply(doy_to_month)

        # Ensure the converted columns are numeric
        merged_gdf['Maize_1_planting'] = pd.to_numeric(merged_gdf['Maize_1_planting'], errors='coerce')
        merged_gdf['Maize_2_planting'] = pd.to_numeric(merged_gdf['Maize_2_planting'], errors='coerce')

    # Create the plot with reduced spacing between subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    # Adjust spacing between subplots - tweak the 0.05 value manually if needed
    plt.subplots_adjust(wspace=0.01)  # Reduce horizontal space between subplots

    # Set title based on doy parameter
    if doy:
        fig.suptitle(f'Maize Planting Days in {country_name} (Day of Year)', fontsize=16)
    else:
        fig.suptitle(f'Maize Planting Months in {country_name}', fontsize=16)

    # Calculate shared normalization for both plots
    if doy:
        # For day-of-year, use the actual data range
        min_value = min(merged_gdf['Maize_1_planting'].min(), merged_gdf['Maize_2_planting'].min())
        max_value = max(merged_gdf['Maize_1_planting'].max(), merged_gdf['Maize_2_planting'].max())
    else:
        # For months, always use the full range from 1 to 12 (January to December)
        min_value = 1
        max_value = 12
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

    # Define colormap for the plots
    colormap = 'RdYlBu'  # Red-Yellow-Blue colormap

    # Plot for Maize 1 with shared normalization
    merged_gdf.plot(column='Maize_1_planting', ax=ax1, legend=False,
                    norm=norm, cmap=colormap, missing_kwds={'color': 'lightgrey', 'label': 'No Data'})
    merged_gdf.boundary.plot(ax=ax1, linewidth=0.5, color='black')
    # Set title based on doy parameter
    if doy:
        ax1.set_title('Maize 1 Planting Days')
    else:
        ax1.set_title('Maize 1 Planting Months')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_aspect('equal')

    # Plot for Maize 2 with shared normalization
    merged_gdf.plot(column='Maize_2_planting', ax=ax2, legend=False,
                    norm=norm, cmap=colormap, missing_kwds={'color': 'lightgrey', 'label': 'No Data'})
    merged_gdf.boundary.plot(ax=ax2, linewidth=0.5, color='black')
    # Set title based on doy parameter
    if doy:
        ax2.set_title('Maize 2 Planting Days')
    else:
        ax2.set_title('Maize 2 Planting Months')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('')  # Hide y-axis label for the second plot
    ax2.set_aspect('equal')

    # Create a single shared colorbar using the same colormap
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', shrink=0.8, aspect=30)

    # Set colorbar label and ticks based on doy parameter
    if doy:
        cbar.set_label('Planting Day of Year')
    else:
        cbar.set_label('Planting Month')
        # Set ticks to show all months from January to December
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Show all months from 1 to 12 (January to December)
        tick_positions = list(range(1, 13))  # Months 1-12
        tick_labels = month_names
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)


    # Save the plot
    scale_type = 'days' if doy else 'months'
    output_path = f'Plots/output/{country_name}_maize_planting_{scale_type}.png'
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Plot saved successfully.")

if __name__ == '__main__':
    # Generate both versions
    print("Generating day-of-year version...")
    plot_south_africa_planting_days('Kenya', doy=True)

    print("Generating month version (with full Jan-Dec colorbar)...")
    plot_south_africa_planting_days('Kenya', doy=False)   