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
from scipy import stats

# Set plotly renderer for better display
pio.renderers.default = "browser"

def create_period_visualization(data, date_col, value_col, title, planting_month, harvest_month, 
                              planting_year_offset, harvest_year_offset, periods=None, harvest_year_to_show=2020, 
                              additional_value_cols=None):
    """
    Create a visualization showing data over two consecutive years with highlighted periods
    Uses the same date logic as the main calculation functions for accurate period representation
    
    Parameters:
    - data: DataFrame with time series data
    - date_col: name of date column
    - value_col: name of primary value column to plot
    - title: plot title
    - planting_month, harvest_month: crop calendar months
    - planting_year_offset, harvest_year_offset: year offsets from crop calendar
    - periods: dict of period names and their datetime ranges
    - harvest_year_to_show: the harvest year of the season to display
    - additional_value_cols: list of additional value columns to plot (optional)
    """
    
    # Calculate the actual calendar years for planting and harvest
    actual_planting_year = harvest_year_to_show + planting_year_offset
    actual_harvest_year = harvest_year_to_show + harvest_year_offset
    
    # Create season date range - from planting to harvest
    season_start = pd.Timestamp(year=actual_planting_year, month=planting_month, day=1)
    season_end = pd.Timestamp(year=actual_harvest_year, month=harvest_month, day=28)  # Use day 28 to avoid month-end issues
    
    # Extend the window to show full context (add 2 months before and after)
    plot_start = season_start - pd.DateOffset(months=2)
    plot_end = season_end + pd.DateOffset(months=2)
    
    # Filter data for the extended period
    data_filtered = data.copy()
    data_filtered['datetime'] = pd.to_datetime(data_filtered[date_col])
    
    # Get data for the plotting window
    mask = (data_filtered['datetime'] >= plot_start) & (data_filtered['datetime'] <= plot_end)
    data_plot = data_filtered[mask].copy()
    
    if data_plot.empty:
        print(f"No data available for season {actual_planting_year}-{actual_harvest_year}")
        return
    
    # Sort by date
    data_plot = data_plot.sort_values('datetime')
    
    # Create figure
    fig = go.Figure()
    
    # Add original data
    fig.add_trace(go.Scatter(
        x=data_plot['datetime'],
        y=data_plot[value_col],
        mode='lines',
        name=f'{value_col} (Primary)',
        line=dict(color='red', width=2),
        opacity=1
    ))
    
    # Add additional value columns if provided
    if additional_value_cols:
        colors = ['blue', 'green', 'purple', 'orange', 'brown']  # Additional colors for extra lines
        for i, col in enumerate(additional_value_cols):
            if col in data_plot.columns:
                fig.add_trace(go.Scatter(
                    x=data_plot['datetime'],
                    y=data_plot[col],
                    mode='lines',
                    name=f'{col}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.8
                ))
    
    # Add smoothed data (rolling mean)
    if len(data_plot) > 10:
        window_size = min(10, len(data_plot) // 3)
        smoothed = data_plot[value_col].rolling(window=window_size, center=True).mean()
        fig.add_trace(go.Scatter(
            x=data_plot['datetime'],
            y=smoothed,
            mode='lines',
            name=f'Smoothed (window={window_size})',
            line=dict(color='lightgreen', width=1), opacity=0.7
        ))
    
    # Calculate and add climatology (multi-year average for same period)
    # Create a "day of season" for climatology calculation
    all_years_data = data_filtered.copy()
    
    # For climatology, we need to align all years to the same reference period
    # We'll use day-of-year but handle cross-year seasons properly
    climatology_data = []
    
    for year in all_years_data['year'].unique():
        year_planting_year = year + planting_year_offset
        year_harvest_year = year + harvest_year_offset
        
        try:
            year_season_start = pd.Timestamp(year=year_planting_year, month=planting_month, day=1)
            year_season_end = pd.Timestamp(year=year_harvest_year, month=harvest_month, day=28)
            
            year_mask = (all_years_data['datetime'] >= year_season_start) & (all_years_data['datetime'] <= year_season_end)
            year_data = all_years_data[year_mask].copy()
            
            if not year_data.empty:
                # Calculate days from season start for alignment
                year_data['days_from_season_start'] = (year_data['datetime'] - year_season_start).dt.days
                climatology_data.append(year_data[['days_from_season_start', value_col]])
        except:
            continue
    
    if climatology_data:
        clim_df = pd.concat(climatology_data, ignore_index=True)
        climatology = clim_df.groupby('days_from_season_start')[value_col].mean().reset_index()
        
        # Convert back to datetime for plotting (using the current season's dates)
        climatology['datetime'] = season_start + pd.to_timedelta(climatology['days_from_season_start'], unit='D')
        
        # Filter climatology to plot window
        clim_mask = (climatology['datetime'] >= plot_start) & (climatology['datetime'] <= plot_end)
        climatology_plot = climatology[clim_mask]
        
        if not climatology_plot.empty:
            fig.add_trace(go.Scatter(
                x=climatology_plot['datetime'],
                y=climatology_plot[value_col],
                mode='lines',
                name='Climatology (multi-year mean)',
                line=dict(color='lightgray', width=1), 
                opacity=0.7
            ))
    
    # Add period highlighting if provided
    if periods:
        for period_name, (period_start, period_end) in periods.items():
            # Add shaded region
            fig.add_vrect(
                x0=period_start,
                x1=period_end,
                fillcolor="yellow" if "vegetation" in period_name.lower() else "cyan",
                opacity=0.3,
                layer="below",
                line_width=0,
            )
            
            # Add period label
            fig.add_annotation(
                x=period_start + (period_end - period_start) / 2,
                y=data_plot[value_col].max() * 0.9,
                text=period_name,
                showarrow=False,
                font=dict(size=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
    
    # Add vertical lines for key dates using shapes instead of add_vline
    fig.add_shape(
        type="line",
        x0=season_start, x1=season_start,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="blue", width=2, dash="dash")
    )
    fig.add_annotation(
        x=season_start, y=0.95, yref="paper",
        text="Planting", showarrow=False,
        bgcolor="lightblue", bordercolor="blue"
    )
    
    fig.add_shape(
        type="line",
        x0=season_end, x1=season_end,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="orange", width=2, dash="dash")
    )
    fig.add_annotation(
        x=season_end, y=0.95, yref="paper",
        text="Harvest", showarrow=False,
        bgcolor="lightyellow", bordercolor="orange"
    )
    
    # Update layout
    fig.update_layout(
        title=f"{title} - Season {actual_planting_year}/{actual_harvest_year}",
        xaxis_title="Date",
        yaxis_title=value_col,
        legend=dict(x=0.02, y=0.98),
        width=1200,
        height=400,
        showlegend=True
    )
    
    # Format x-axis to show months clearly
    fig.update_xaxes(
        tickformat="%b %Y",
        dtick="M1",  # Monthly ticks
        range=[plot_start, plot_end]
    )
    
    fig.show()
    
    return fig

# Create a summary visualization showing the relationship between variables

def scatter_plots(merged, country_name):
    COUNTRY_NAME = country_name
    if not merged.empty:
        print(f"\nCreating summary visualization for all {merged['PCODE'].nunique()} admin units")
        print(f"Total data points: {len(merged)}")
        
        # Calculate interaction feature
        merged['ndvi_per_gdd'] = merged['ndvi_integral'] / merged['total_GDD']
        
        # Create subplots (now 7 rows to add max_ndvi)
        fig = make_subplots(
            rows=7, cols=2,
            subplot_titles=(
                'NDVI Integral vs Yield', 'EVI Integral vs Yield',
                'Total GDD vs Yield', 'Vegetation Precipitation (Normalized) vs Yield', 
                'Flowering Precipitation (Normalized) vs Yield', 'Vegetation Drought (Max) vs Yield', 
                'Flowering Drought (Max) vs Yield', 'Vegetation Drought (Sum) vs Yield',
                'Flowering Drought (Sum) vs Yield', 'NDVI per GDD vs Yield',
                'NDWI Integral vs Yield', 'Max NDWI vs Yield',
                'Max NDVI vs Yield', ''
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Helper function to add trend line and calculate R²
        def add_trend_line(fig, x_data, y_data, row, col, color='red'):
            # Remove any NaN values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            if len(x_clean) > 1:
                # Calculate linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                
                # Create trend line
                x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_trend = slope * x_trend + intercept
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(x=x_trend, y=y_trend,
                              mode='lines', name=f'Trend (R²={r_value**2:.3f})',
                              line=dict(color=color, width=2, dash='dash'),
                              showlegend=False),
                    row=row, col=col
                )
                
                # Add R² annotation
                fig.add_annotation(
                    x=x_clean.max() * 0.95, y=y_clean.max() * 0.95,
                    text=f"R² = {r_value**2:.3f}<br>p = {p_value:.3f}",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    row=row, col=col
                )
        
        # NDVI vs Yield
        fig.add_trace(
            go.Scatter(x=merged['ndvi_integral'], y=merged['yield'],
                      mode='markers', name='NDVI Integral',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="NDVI Integral: %{x:.2f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='blue', size=4, opacity=0.6)),
            row=1, col=1
        )
        add_trend_line(fig, merged['ndvi_integral'].values, merged['yield'].values, 1, 1, 'darkblue')
        
        # EVI vs Yield
        fig.add_trace(
            go.Scatter(x=merged['evi_integral'], y=merged['yield'],
                      mode='markers', name='EVI Integral',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="EVI Integral: %{x:.2f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='green', size=4, opacity=0.6)),
            row=1, col=2
        )
        add_trend_line(fig, merged['evi_integral'].values, merged['yield'].values, 1, 2, 'darkgreen')
        
        # GDD vs Yield
        fig.add_trace(
            go.Scatter(x=merged['total_GDD'], y=merged['yield'],
                      mode='markers', name='Total GDD',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="Total GDD: %{x:.1f} K·days<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='red', size=4, opacity=0.6)),
            row=2, col=1
        )
        add_trend_line(fig, merged['total_GDD'].values, merged['yield'].values, 2, 1, 'darkred')
        
        # Vegetation Precipitation vs Yield (normalized)
        fig.add_trace(
            go.Scatter(x=merged['veg_precip_norm'], y=merged['yield'],
                      mode='markers', name='Veg Precip (Normalized)',
                      text=[f"PCODE: {pcode}<br>Year: {year}<br>Area: {area:.1f} ha" for pcode, year, area in zip(merged['PCODE'], merged['year'], merged['total_area_ha'])],
                      hovertemplate="Veg Precip: %{x:.4f} mm/km²<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='purple', size=4, opacity=0.6)),
            row=2, col=2
        )
        add_trend_line(fig, merged['veg_precip_norm'].values, merged['yield'].values, 2, 2, 'darkmagenta')
        
        # Flowering Precipitation vs Yield (normalized)
        fig.add_trace(
            go.Scatter(x=merged['flow_precip_norm'], y=merged['yield'],
                      mode='markers', name='Flow Precip (Normalized)',
                      text=[f"PCODE: {pcode}<br>Year: {year}<br>Area: {area:.1f} km²" for pcode, year, area in zip(merged['PCODE'], merged['year'], merged['total_area_ha'])],
                      hovertemplate="Flow Precip: %{x:.4f} mm/km²<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='orange', size=4, opacity=0.6)),
            row=3, col=1
        )
        add_trend_line(fig, merged['flow_precip_norm'].values, merged['yield'].values, 3, 1, 'darkorange')
        
        # Vegetation Drought (Max) vs Yield
        fig.add_trace(
            go.Scatter(x=merged['veg_max_drought'], y=merged['yield'],
                      mode='markers', name='Veg Max Drought',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="Veg Max Drought: %{x:.4f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='brown', size=4, opacity=0.6)),
            row=3, col=2
        )
        add_trend_line(fig, merged['veg_max_drought'].values, merged['yield'].values, 3, 2, 'darkred')
        
        # Flowering Drought (Max) vs Yield
        fig.add_trace(
            go.Scatter(x=merged['flow_max_drought'], y=merged['yield'],
                      mode='markers', name='Flow Max Drought',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="Flow Max Drought: %{x:.4f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='cyan', size=4, opacity=0.6)),
            row=4, col=1
        )
        add_trend_line(fig, merged['flow_max_drought'].values, merged['yield'].values, 4, 1, 'darkcyan')
        
        # Vegetation Drought (Sum) vs Yield
        fig.add_trace(
            go.Scatter(x=merged['veg_sum_drought'], y=merged['yield'],
                      mode='markers', name='Veg Sum Drought',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="Veg Sum Drought: %{x:.4f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='darkred', size=4, opacity=0.6)),
            row=4, col=2
        )
        add_trend_line(fig, merged['veg_sum_drought'].values, merged['yield'].values, 4, 2, 'maroon')
        
        # Flowering Drought (Sum) vs Yield
        fig.add_trace(
            go.Scatter(x=merged['flow_sum_drought'], y=merged['yield'],
                      mode='markers', name='Flow Sum Drought',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="Flow Sum Drought: %{x:.4f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='darkgoldenrod', size=4, opacity=0.6)),
            row=5, col=1
        )
        add_trend_line(fig, merged['flow_sum_drought'].values, merged['yield'].values, 5, 1, 'goldenrod')
        
        # NDVI per GDD vs Yield (interaction feature)
        fig.add_trace(
            go.Scatter(x=merged['ndvi_per_gdd'], y=merged['yield'],
                      mode='markers', name='NDVI per GDD',
                      text=[f"PCODE: {pcode}<br>Year: {year}<br>NDVI: {ndvi:.2f}<br>GDD: {gdd:.1f}" for pcode, year, ndvi, gdd in zip(merged['PCODE'], merged['year'], merged['ndvi_integral'], merged['total_GDD'])],
                      hovertemplate="NDVI per GDD: %{x:.6f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='mediumorchid', size=4, opacity=0.6)),
            row=5, col=2
        )
        add_trend_line(fig, merged['ndvi_per_gdd'].values, merged['yield'].values, 5, 2, 'darkorchid')
        
        # NDWI Integral vs Yield
        fig.add_trace(
            go.Scatter(x=merged['ndwi_integral'], y=merged['yield'],
                      mode='markers', name='NDWI Integral',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="NDWI Integral: %{x:.2f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='navy', size=4, opacity=0.6)),
            row=6, col=1
        )
        add_trend_line(fig, merged['ndwi_integral'].values, merged['yield'].values, 6, 1, 'darkblue')
        
        # Max NDWI vs Yield
        fig.add_trace(
            go.Scatter(x=merged['max_ndwi'], y=merged['yield'],
                      mode='markers', name='Max NDWI',
                      text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                      hovertemplate="Max NDWI: %{x:.3f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                      marker=dict(color='teal', size=4, opacity=0.6)),
            row=6, col=2
        )
        add_trend_line(fig, merged['max_ndwi'].values, merged['yield'].values, 6, 2, 'darkcyan')

        # Max NDVI vs Yield (NEW)
        if 'max_ndvi' in merged.columns:
            fig.add_trace(
                go.Scatter(x=merged['max_ndvi'], y=merged['yield'],
                          mode='markers', name='Max NDVI',
                          text=[f"PCODE: {pcode}<br>Year: {year}" for pcode, year in zip(merged['PCODE'], merged['year'])],
                          hovertemplate="Max NDVI: %{x:.3f}<br>Yield: %{y:.2f} mt/ha<br>%{text}<extra></extra>",
                          marker=dict(color='forestgreen', size=4, opacity=0.6)),
                row=7, col=1
            )
            add_trend_line(fig, merged['max_ndvi'].values, merged['yield'].values, 7, 1, 'darkgreen')

        # Update layout
        fig.update_layout(
            title=f"Yield Relationships - {COUNTRY_NAME} (All Admin Units, n={len(merged)})",
            height=1850,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="NDVI Integral", row=1, col=1)
        fig.update_xaxes(title_text="EVI Integral", row=1, col=2)
        fig.update_xaxes(title_text="Total GDD (K·days)", row=2, col=1)
        fig.update_xaxes(title_text="Vegetation Precip (mm/km²)", row=2, col=2)
        fig.update_xaxes(title_text="Flowering Precip (mm/km²)", row=3, col=1)
        fig.update_xaxes(title_text="Veg Max Drought", row=3, col=2)
        fig.update_xaxes(title_text="Flow Max Drought", row=4, col=1)
        fig.update_xaxes(title_text="Veg Sum Drought", row=4, col=2)
        fig.update_xaxes(title_text="Flow Sum Drought", row=5, col=1)
        fig.update_xaxes(title_text="NDVI per GDD", row=5, col=2)
        fig.update_xaxes(title_text="NDWI Integral", row=6, col=1)
        fig.update_xaxes(title_text="Max NDWI", row=6, col=2)
        fig.update_xaxes(title_text="Max NDVI", row=7, col=1)
        
        for r in range(1,8):
            for c in range(1,3):
                fig.update_yaxes(title_text="Yield (mt/ha)", row=r, col=c)
        
        fig.show()
        
        # Print correlation statistics
        print("\nCorrelation Analysis:")
        print("="*40)
        correlations = {
            'NDVI Integral': merged['ndvi_integral'].corr(merged['yield']),
            'EVI Integral': merged['evi_integral'].corr(merged['yield']),
            'Total GDD': merged['total_GDD'].corr(merged['yield']),
            'Vegetation Precip (Norm)': merged['veg_precip_norm'].corr(merged['yield']),
            'Flowering Precip (Norm)': merged['flow_precip_norm'].corr(merged['yield']),
            'Veg Max Drought': merged['veg_max_drought'].corr(merged['yield']),
            'Flow Max Drought': merged['flow_max_drought'].corr(merged['yield']),
            'Veg Sum Drought': merged['veg_sum_drought'].corr(merged['yield']),
            'Flow Sum Drought': merged['flow_sum_drought'].corr(merged['yield']),
            'NDVI per GDD': merged['ndvi_per_gdd'].corr(merged['yield']),
            'NDWI Integral': merged['ndwi_integral'].corr(merged['yield']),
            'Max NDWI': merged['max_ndwi'].corr(merged['yield']),
            'Min NDWI': merged['min_ndwi'].corr(merged['yield']),
            'Range NDWI': merged['range_ndwi'].corr(merged['yield'])
        }
        
        # Also show comparison with original (non-normalized) precipitation values
        print("Comparison - Original vs Normalized Precipitation Correlations:")
        print(f"{'Variable':<25} {'Original':<10} {'Normalized':<12} {'Difference':<10}")
        print("-" * 60)
        orig_veg_corr = merged['veg_precip'].corr(merged['yield'])
        norm_veg_corr = merged['veg_precip_norm'].corr(merged['yield'])
        orig_flow_corr = merged['flow_precip'].corr(merged['yield'])
        norm_flow_corr = merged['flow_precip_norm'].corr(merged['yield'])
        
        print(f"{'Vegetation Precip':<25} {orig_veg_corr:<10.3f} {norm_veg_corr:<12.3f} {norm_veg_corr-orig_veg_corr:<10.3f}")
        print(f"{'Flowering Precip':<25} {orig_flow_corr:<10.3f} {norm_flow_corr:<12.3f} {norm_flow_corr-orig_flow_corr:<10.3f}")
        print()
        
        print("All Correlations (using normalized precipitation):")
        
        for var, corr in correlations.items():
            print(f"{var:20s}: r = {corr:6.3f}")
        
        # Find strongest correlations
        strongest_pos = max(correlations.items(), key=lambda x: x[1] if x[1] > 0 else -1)
        strongest_neg = min(correlations.items(), key=lambda x: x[1] if x[1] < 0 else 1)
        
        print(f"\nStrongest positive correlation: {strongest_pos[0]} (r = {strongest_pos[1]:.3f})")
        if strongest_neg[1] < 0:
            print(f"Strongest negative correlation: {strongest_neg[0]} (r = {strongest_neg[1]:.3f})")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - All visualizations generated!")
    print("="*60)


def create_yield_correlation_plots(df, country_name):
    """
    For new GLAM calendar system.

    Create comprehensive scatter plots showing yield vs all remote sensing variables
    with R² scores and p-values using Plotly.
    Now, marker shape encodes season: circles for Maize 1, triangles for Maize 2.
    Each season uses a similar but distinct color for each variable group.
    """
    print("   Creating yield correlation analysis...")
    
    # Define remote sensing variables to analyze
    rs_variables = [
        'ndvi_integral_plant_harv', 
        'ndvi_integral_plant_veg',
        'ndvi_integral_veg_harv',
        'max_ndvi_plant_harv',
        'max_ndvi_plant_veg',
        'max_ndvi_veg_harv',
        'max_ndvi_doy_plant_end',  # New variable: day-of-year for max NDVI
        'max_evi_doy_plant_end',   # New variable: day-of-year for max EVI
        'gdd_plant_harv',
        'precip_plant_veg', 'precip_veg_harv',
        'ndwi_integral_plant_harv', 'max_ndwi_plant_harv', 'range_ndwi_plant_harv',
        'veg_max_drought', 'veg_sum_drought', 'flow_max_drought', 'flow_sum_drought'
    ]
    
    # Color mapping for variables and seasons
    color_map_season = {
        'ndvi': {'Maize 1': 'green', 'Maize 2': 'orange'},
        'max_ndvi': {'Maize 1': 'green', 'Maize 2': 'orange'},
        'max_ndvi_doy': {'Maize 1': 'darkgreen', 'Maize 2': 'darkorange'},  # New: DOY colors
        'max_evi_doy': {'Maize 1': 'forestgreen', 'Maize 2': 'chocolate'},  # New: DOY colors
        'gdd': {'Maize 1': 'red', 'Maize 2': 'orange'},
        'precip': {'Maize 1': 'blue', 'Maize 2': 'orange'},
        'ndwi': {'Maize 1': 'teal', 'Maize 2': 'orange'},
        'max_ndwi': {'Maize 1': 'teal', 'Maize 2': 'orange'},
        'range_ndwi': {'Maize 1': 'teal', 'Maize 2': 'orange'},
        'drought': {'Maize 1': 'darkgoldenrod', 'Maize 2': 'orange'}
    }

    # Marker symbol mapping for seasons
    season_symbol_map = {
        'Maize 1': 'circle',
        'Maize 2': 'triangle-up',
    }

    # Filter variables that actually exist in the dataframe
    available_vars = [var for var in rs_variables if var in df.columns]
    
    if not available_vars:
        print("   ❌ No remote sensing variables found for correlation analysis")
        return
    
    # Remove rows with missing yield data
    df_clean = df.dropna(subset=['yield']).copy()
    
    # Replace infinite values with NaN to avoid calculation errors
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if df_clean.empty:
        print("   ❌ No yield data available for correlation analysis")
        return
    
    print(f"   Analyzing {len(available_vars)} variables against yield for {len(df_clean)} data points")
    
    # Calculate correlations and statistics
    correlations = []
    for var in available_vars:
        # Remove rows where this variable is NaN
        var_data = df_clean[[var, 'yield', 'maize_season']].dropna()
        
        if len(var_data) < 10:  # Need at least 10 points for meaningful correlation
            continue
            
        x = var_data[var].values
        y = var_data['yield'].values
        seasons = var_data['maize_season'].values
        
        # Check for constant input to avoid errors in regression
        is_constant = len(np.unique(x)) == 1
        r, r2, p_value, slope, intercept = np.nan, np.nan, np.nan, np.nan, np.nan

        if not is_constant:
            try:
                # Calculate correlation coefficient and p-value
                r, p_value = stats.pearsonr(x, y)
                r2 = r**2
                
                # Calculate linear regression for trend line
                slope, intercept, r_value, p_val_reg, std_err = stats.linregress(x, y)
            except ValueError:
                # This might happen in rare edge cases, treat as constant
                is_constant = True
        
        correlations.append({
            'variable': var,
            'r': r,
            'r2': r2,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'n_points': len(var_data),
            'data': var_data,
            'is_constant': is_constant
        })
    
    # Manually add the new plot for area vs. crop_area_ha_adjusted
    if 'crop_area_ha_adjusted' in df_clean.columns and 'area' in df_clean.columns:
        area_plot_data = df_clean[['crop_area_ha_adjusted', 'area', 'maize_season']].dropna()
        if not area_plot_data.empty and len(area_plot_data) >= 10:
            x_area = area_plot_data['crop_area_ha_adjusted'].values
            y_area = area_plot_data['area'].values

            is_constant = len(np.unique(x_area)) == 1 or len(np.unique(y_area)) == 1
            r, r2, p_value, slope, intercept = np.nan, np.nan, np.nan, np.nan, np.nan

            if not is_constant:
                try:
                    r, p_value = stats.pearsonr(x_area, y_area)
                    r2 = r**2
                    slope, intercept, _, _, _ = stats.linregress(x_area, y_area)
                except ValueError:
                    is_constant = True
            
            correlations.append({
                'variable': 'crop_area_ha_adjusted',
                'y_variable': 'area',
                'r': r, 'r2': r2, 'p_value': p_value, 'slope': slope, 'intercept': intercept,
                'n_points': len(area_plot_data),
                'data': area_plot_data,
                'is_constant': is_constant
            })

    if not correlations:
        print("   ❌ No valid correlations could be calculated")
        return
    
    # Create subplot layout (2 columns, as many rows as needed)
    n_vars = len(correlations)
    n_cols = 2
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    # Create subplot titles with just the variable names
    subplot_titles = [
        'Area vs Adj. Crop Area' if corr.get('y_variable') == 'area' else corr['variable'].replace('_', ' ').title() 
        for corr in correlations
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles
    )
    
    # Add scatter plots and trend lines
    for i, corr in enumerate(correlations):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        is_area_plot = corr.get('y_variable') == 'area'
        
        # Determine y-variable and labels
        if is_area_plot:
            y_var = 'area'
            y_label = 'Area (ha)'
            x_label = 'Adj. Crop Area (ha)'
            y_unit = 'ha'
        else:
            y_var = 'yield'
            y_label = 'Yield (t/ha)'
            x_label = corr['variable'].replace('_', ' ').title()
            y_unit = 't/ha'
            
        var_data = corr['data']
        x = var_data[corr['variable']].values
        y = var_data[y_var].values
        seasons = var_data['maize_season'].values
        
        # Determine variable group for color
        variable_name = corr['variable']
        color_group = None
        for prefix in color_map_season.keys():
            if variable_name.startswith(prefix):
                color_group = prefix
                break
        if color_group is None:
            color_group = 'ndvi'  # fallback
        
        # Plot each season separately for shape and color
        for season in ['Maize 1', 'Maize 2']:
            mask = (seasons == season)
            if not np.any(mask):
                continue
                
            marker_color = 'purple' if y_var == 'area' else color_map_season[color_group][season]

            fig.add_trace(
                go.Scatter(
                    x=x[mask], y=y[mask],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=marker_color,
                        symbol=season_symbol_map.get(season, 'circle'),
                        opacity=0.7
                    ),
                    name=f"{corr['variable']} - {season}",
                    legendgroup=season,
                    showlegend=(i == 0),  # Only show legend for first subplot
                    hovertemplate=f"<b>{corr['variable']}</b><br>Season: {season}<br>Value: %{{x:.2f}}<br>{y_var.title()}: %{{y:.2f}} {y_unit}<br><extra></extra>"
                ),
                row=row, col=col
            )
        
        # Add trend line for correlation plots, or 1:1 line for area comparison
        if is_area_plot:
            # For area plot, make axes equal and add a 1:1 diagonal line
            min_val = min(np.nanmin(x), np.nanmin(y))
            max_val = max(np.nanmax(x), np.nanmax(y))
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='grey', width=2, dash='dash'),
                    name='1:1 Line',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            # Update axes for this subplot to be square
            fig.update_xaxes(title_text=x_label, range=[min_val, max_val], row=row, col=col)
            fig.update_yaxes(title_text=y_label, range=[min_val, max_val], row=row, col=col)
            
        elif not corr['is_constant']:
            # Add trend line for other plots
            color_line = 'red' if corr['p_value'] < 0.05 else 'grey'
            x_trend = np.linspace(x.min(), x.max(), 100)
            y_trend = corr['slope'] * x_trend + corr['intercept']
            
            fig.add_trace(
                go.Scatter(
                    x=x_trend, y=y_trend,
                    mode='lines',
                    line=dict(color=color_line, width=2),
                    name='Trend',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            # Standard axis update for non-area plots
            fig.update_yaxes(title_text=y_label, row=row, col=col)
            fig.update_xaxes(title_text=x_label, row=row, col=col)
        else:
            # Axis update for constant value plots
            fig.update_yaxes(title_text=y_label, row=row, col=col)
            fig.update_xaxes(title_text=x_label, row=row, col=col)


        # Add annotation for R² and p-value
        if corr['is_constant']:
            annotation_text = "Constant value"
        else:
            r2_text = f"{corr['r2']:.3f}" if not np.isnan(corr['r2']) else "N/A"
            p_text = f"{corr['p_value']:.3f}" if not np.isnan(corr['p_value']) else "N/A"
            annotation_text = f"R² = {r2_text}<br>p = {p_text}"

        fig.add_annotation(
            x=np.percentile(x, 95), y=np.percentile(y, 98),
            text=annotation_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1,
            row=row, col=col
        )
        
        # This part is now handled within the conditional blocks above
        # fig.update_yaxes(title_text=y_label, row=row, col=col)
        # fig.update_xaxes(title_text=x_label, row=row, col=col)

    # Update layout
    fig.update_layout(
        title=f"Yield Correlation Analysis - {country_name}<br>" +
              f"<sub>Remote Sensing Variables vs Maize Yield</sub>",
        height=350 * n_rows,
        showlegend=True,
        font=dict(size=10),
        margin=dict(t=100, b=50, l=50, r=50),
        legend=dict(
            title="Maize Season",
            itemsizing='constant',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Show the plot
    fig.show()
    
    # Print summary statistics
    print(f"\n   📊 CORRELATION SUMMARY:")
    print(f"   {'Variable':<35} {'R²':<8} {'p-value':<10} {'N points':<10} {'Significance'}")
    print(f"   {'-'*80}")
    
    # Sort by R² value for reporting
    correlations_sorted = sorted([c for c in correlations if not c['is_constant']], key=lambda x: abs(x['r2']), reverse=True)

    for corr in correlations_sorted[:15]:  # Show top 15 correlations
        significance = "***" if corr['p_value'] < 0.001 else "**" if corr['p_value'] < 0.01 else "*" if corr['p_value'] < 0.05 else ""
        y_var_name = corr.get('y_variable', 'yield')
        print(f"   {corr['variable']:<35} vs {y_var_name:<10} {corr['r2']:<8.3f} {corr['p_value']:<10.3f} {corr['n_points']:<10} {significance}")
    
    if len(correlations_sorted) > 15:
        print(f"   ... and {len(correlations_sorted) - 15} more variables")
    
    print(f"\n   Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    
    return correlations


def plot_comparison(ethiopia_df, burkina_faso_df):
    """
    Create a comparison plot between Ethiopia and Burkina Faso showing NDVI integral, 
    plant harvest, and GDD for both countries in a 2x2 subplot layout using Plotly.
    
    Parameters:
    - ethiopia_df: DataFrame with Ethiopia data
    - burkina_faso_df: DataFrame with Burkina Faso data
    """
    # Filter data to remove rows with missing values
    eth_df = ethiopia_df.dropna(subset=['ndvi_integral_plant_harv', 'gdd_plant_harv', 'yield']).copy()
    bfa_df = burkina_faso_df.dropna(subset=['ndvi_integral_plant_harv', 'gdd_plant_harv', 'yield']).copy()
    
    # Create subplots (2 rows, 2 columns) - restructured so same country is in same column
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Ethiopia - NDVI Integral vs Yield',
            'Burkina Faso - NDVI Integral vs Yield',
            'Ethiopia - GDD vs Yield',
            'Burkina Faso - GDD vs Yield'
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.06
    )
    
    # Helper function to add trend line and calculate R²
    def add_trend_line(fig, x_data, y_data, row, col, color='red'):
        # Remove any NaN values
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) > 1:
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            # Create trend line
            x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_trend = slope * x_trend + intercept
            
            # Add trend line
            fig.add_trace(
                go.Scatter(x=x_trend, y=y_trend,
                          mode='lines', name='Trend',
                          line=dict(color=color, width=2, dash='dash'),
                          showlegend=False),
                row=row, col=col
            )
            
            # Add R² annotation
            fig.add_annotation(
                x=x_clean.max() * 0.95, y=y_clean.max() * 0.95,
                text=f"R² = {r_value**2:.3f}<br>p = {p_value:.3f}",
                showarrow=False,
                font=dict(size=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                row=row, col=col
            )
    
    # Plot 1: Ethiopia NDVI integral vs yield (top left)
    fig.add_trace(
        go.Scatter(
            x=eth_df['ndvi_integral_plant_harv'],
            y=eth_df['yield'],
            mode='markers',
            marker=dict(color='green', size=6, opacity=0.6),
            name='Ethiopia - NDVI',
            text=[f"Year: {year}" for year in eth_df['year']],
            hovertemplate="NDVI Integral: %{x:.2f}<br>Yield: %{y:.2f} t/ha<br>%{text}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add trend line for Ethiopia NDVI
    add_trend_line(fig, eth_df['ndvi_integral_plant_harv'].values, eth_df['yield'].values, 1, 1, 'darkgreen')
    
    # Plot 2: Burkina Faso NDVI integral vs yield (top right)
    fig.add_trace(
        go.Scatter(
            x=bfa_df['ndvi_integral_plant_harv'],
            y=bfa_df['yield'],
            mode='markers',
            marker=dict(color='green', size=6, opacity=0.6),
            name='Burkina Faso - NDVI',
            text=[f"Year: {year}" for year in bfa_df['year']],
            hovertemplate="NDVI Integral: %{x:.2f}<br>Yield: %{y:.2f} t/ha<br>%{text}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add trend line for Burkina Faso NDVI
    add_trend_line(fig, bfa_df['ndvi_integral_plant_harv'].values, bfa_df['yield'].values, 1, 2, 'darkgreen')
    
    # Plot 3: Ethiopia GDD vs yield (bottom left)
    fig.add_trace(
        go.Scatter(
            x=eth_df['gdd_plant_harv'],
            y=eth_df['yield'],
            mode='markers',
            marker=dict(color='red', size=6, opacity=0.6),
            name='Ethiopia - GDD',
            text=[f"Year: {year}" for year in eth_df['year']],
            hovertemplate="GDD: %{x:.1f} K·days<br>Yield: %{y:.2f} t/ha<br>%{text}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Add trend line for Ethiopia GDD
    add_trend_line(fig, eth_df['gdd_plant_harv'].values, eth_df['yield'].values, 2, 1, 'darkred')
    
    # Plot 4: Burkina Faso GDD vs yield (bottom right)
    fig.add_trace(
        go.Scatter(
            x=bfa_df['gdd_plant_harv'],
            y=bfa_df['yield'],
            mode='markers',
            marker=dict(color='red', size=6, opacity=0.6),
            name='Burkina Faso - GDD',
            text=[f"Year: {year}" for year in bfa_df['year']],
            hovertemplate="GDD: %{x:.1f} K·days<br>Yield: %{y:.2f} t/ha<br>%{text}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # Add trend line for Burkina Faso GDD
    add_trend_line(fig, bfa_df['gdd_plant_harv'].values, bfa_df['yield'].values, 2, 2, 'darkred')
    
    # Update layout to match the style of other plots
    fig.update_layout(
        title='Comparison of Ethiopia and Burkina Faso',
        title_x=0.5,
        height=600,
        width=800,
        showlegend=False,
        font=dict(family="Times New Roman", size=12)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="NDVI Integral (Plant-Harvest)", row=1, col=1)
    fig.update_xaxes(title_text="NDVI Integral (Plant-Harvest)", row=1, col=2)
    fig.update_xaxes(title_text="GDD (Plant-Harvest) (K·days)", row=2, col=1)
    fig.update_xaxes(title_text="GDD (Plant-Harvest) (K·days)", row=2, col=2)
    
    fig.update_yaxes(title_text="Yield (t/ha)", row=1, col=1)
    fig.update_yaxes(title_text="Yield (t/ha)", row=1, col=2)
    fig.update_yaxes(title_text="Yield (t/ha)", row=2, col=1)
    fig.update_yaxes(title_text="Yield (t/ha)", row=2, col=2)
    
    fig.show()


if __name__ == "__main__":
    # Example usage of plot_comparison function:
    # Load data for Ethiopia and Burkina Faso
    ethiopia_df = pd.read_csv("ASR/tools/glam_results/Ethiopia_admin2_merged_data_GLAM.csv")
    burkina_faso_df = pd.read_csv("ASR/tools/glam_results/Burkina_Faso_admin2_merged_data_GLAM.csv")
    plot_comparison(ethiopia_df, burkina_faso_df)
    
    countries = ['Ethiopia', 'Angola', 'Burkina_Faso']
    for country in countries:
        country = country.replace(' ', '_').replace(',', '_')
        try:
            df = pd.read_csv(f"ASR/tools/glam_results/{country}_admin2_merged_data_GLAM.csv")
            print(f"   Loaded {country} admin2 data")
        except FileNotFoundError:
            df = pd.read_csv(f"ASR/tools/glam_results/{country}_admin1_merged_data_GLAM.csv")
            print(f"   Loaded {country} admin1 data")

        # Assuming 'df' is your loaded DataFrame

    
        #create_yield_correlation_plots(df, country)
