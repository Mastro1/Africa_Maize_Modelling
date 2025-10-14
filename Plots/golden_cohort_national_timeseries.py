import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

PROJECT_ROOT = os.getcwd()

# Define the golden cohort
golden_cohort = ['Angola', 'Lesotho', 'Zimbabwe', 'Ethiopia', 'Zambia', 'Kenya', 'South Africa', 'Tanzania, United Republic of', 'Nigeria']

def load_data():
    """Load the optimized model predictions and FAO data."""
    # Load optimized model predictions
    print("Loading optimized model predictions...")
    predictions_df = pd.read_csv(
        os.path.join(PROJECT_ROOT, 'Model', "training_results",'model_predictions.csv'),
        usecols=['country', 'year', 'yield', 'pred_yield', 'national_yield', 'national_trend', 'crop_area_ha']
    )
    
    # Load FAO data (this is already at national level)
    print("Loading FAO data...")
    fao_df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "RemoteSensing", "HarvestStatAfrica", "preprocessed_concat", "all_admin0_merged_data.csv"),
        usecols=['country', 'year', 'yield']
    )
    fao_df.rename(columns={'yield': 'fao_yield'}, inplace=True)
    
    return predictions_df, fao_df

def filter_golden_cohort_data(predictions_df, fao_df):
    """Filter data for golden cohort countries."""
    # Filter predictions for golden cohort
    predictions_filtered = predictions_df[predictions_df['country'].isin(golden_cohort)].copy()
    
    # Filter FAO data for golden cohort
    fao_filtered = fao_df[fao_df['country'].isin(golden_cohort)].copy()
    
    return predictions_filtered, fao_filtered

def compute_weighted_series(df, value_col, weight_col='crop_area_ha'):
    """Compute weighted average series by year and country."""
    def wavg(group):
        d = group[value_col]
        w = group[weight_col]

        # First try weighted average with valid weights
        weight_mask = pd.notnull(d) & pd.notnull(w)
        if weight_mask.sum() > 0:
            return np.average(d[weight_mask], weights=w[weight_mask])

        # Fall back to simple average if weights are not available
        value_mask = pd.notnull(d)
        if value_mask.sum() > 0:
            return np.mean(d[value_mask])

        # Return NaN if no valid data
        return np.nan

    weighted_series = df.groupby(['country', 'year'], group_keys=False).apply(wavg).reset_index()
    weighted_series.columns = ['country', 'year', value_col]
    return weighted_series

def prepare_plotting_data(predictions_filtered, fao_filtered):
    """Prepare data for plotting."""
    # Compute area-weighted ground truth series from admin-level data
    print("Computing area-weighted ground truth series...")
    ground_series = compute_weighted_series(predictions_filtered, 'yield')
    
    # Compute area-weighted model series from admin-level data
    print("Computing area-weighted model predictions series...")
    model_series = compute_weighted_series(predictions_filtered, 'pred_yield')
    
    # FAO series is already at country level
    print("Preparing FAO series...")
    fao_series = fao_filtered.dropna(subset=['fao_yield'])
    
    # Merge series
    print("Merging series...")
    merged = fao_series.merge(ground_series, on=['country', 'year'], how='outer', suffixes=('_fao', '_ground'))
    merged = merged.merge(model_series, on=['country', 'year'], how='outer')
    merged.rename(columns={'pred_yield': 'yield_model'}, inplace=True)
    
    return merged

def compute_trend_line(fao_series_data):
    """Compute linear trend line for FAO data."""
    # Remove NaN values
    clean_data = fao_series_data.dropna()
    if len(clean_data) < 2:
        return None, None
    
    X = clean_data['year'].values.reshape(-1, 1)
    y = clean_data['fao_yield'].values
    
    try:
        model = LinearRegression()
        model.fit(X, y)
        y_trend = model.predict(X)
        return X.flatten(), y_trend
    except:
        return None, None


def compute_relative_deviation(series_data, years):
    """Compute relative deviation from trend for a series.
    
    For each series, fit a linear regression over years, 
    compute relative deviation: (value - trend) / trend
    """
    x = pd.to_numeric(years, errors="coerce").values
    s = pd.to_numeric(series_data, errors="coerce").values
    m = np.isfinite(x) & np.isfinite(s)
    
    if m.sum() >= 2:
        try:
            slope, intercept = np.polyfit(x[m], s[m], 1)
            trend = slope * x + intercept
        except Exception:
            mu = np.nanmean(s[m])
            trend = np.full_like(s, fill_value=mu, dtype=float)
    else:
        mu = np.nanmean(s[m]) if m.any() else np.nan
        trend = np.full_like(s, fill_value=mu, dtype=float)
    
    denom = np.where(np.abs(trend) > 1e-9, trend, np.nan)
    return (s - trend) / denom

def create_plot_without_model(data):
    """Create the 3x3 grid plot without model predictions."""
    # Set up the plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Comparison FAO Timeseries to Aggregated HarvestStat Data', 
                 fontsize=16, fontname='Times New Roman')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Define colors
    fao_color = '#24245c'      # Blue from graph designer
    ground_color = '#fccd32'   # Yellow from graph designer
    trend_color = 'gray'
    
    # Create legend handles manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=fao_color, marker='D', linestyle='-', markersize=6, linewidth=2, label='FAO'),
        Line2D([0], [0], color=ground_color, marker='s', linestyle='-', markersize=5, linewidth=2, label='Ground Data'),
        Line2D([0], [0], color=trend_color, linestyle=':', linewidth=1.2, label='Trend')
    ]
    
    # Plot each country
    for i, country in enumerate(golden_cohort):
        if i >= 9:  # Only plot first 9 countries
            break
            
        ax = axes_flat[i]
        
        # Filter data for this country
        country_data = data[data['country'] == country].sort_values('year')
        
        if country_data.empty:
            ax.set_title(country, fontname='Times New Roman', fontsize=12)
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontname='Times New Roman')
            continue
        
        # Plot FAO data (blue diamonds) - only connect points with consecutive years
        fao_clean = country_data.dropna(subset=['fao_yield'])
        if not fao_clean.empty:
            # Sort by year and group consecutive years together
            fao_clean = fao_clean.sort_values('year').copy()
            fao_clean['year_diff'] = fao_clean['year'].diff()
            fao_clean['group'] = (fao_clean['year_diff'] != 1).cumsum()
            
            # Plot each group separately to avoid connecting non-consecutive years
            for group_id, group_data in fao_clean.groupby('group'):
                ax.plot(group_data['year'], group_data['fao_yield'], 
                       color=fao_color, linewidth=2, marker='D', markersize=6, label='FAO' if group_id == 0 else "")
        
        # Plot ground data (yellow squares) - only connect points with consecutive years
        ground_clean = country_data.dropna(subset=['yield'])
        if not ground_clean.empty:
            # Sort by year and group consecutive years together
            ground_clean = ground_clean.sort_values('year').copy()
            ground_clean['year_diff'] = ground_clean['year'].diff()
            ground_clean['group'] = (ground_clean['year_diff'] != 1).cumsum()
            
            # Plot each group separately to avoid connecting non-consecutive years
            for group_id, group_data in ground_clean.groupby('group'):
                ax.plot(group_data['year'], group_data['yield'], 
                       color=ground_color, linewidth=2, marker='s', markersize=5, label='Ground' if group_id == 0 else "")
        
        # Compute and plot trend line based on FAO data for full year range
        if not fao_clean.empty and len(fao_clean) >= 2:
            # Get the full year range from FAO data
            min_year = fao_clean['year'].min()
            max_year = fao_clean['year'].max()
            year_range = np.arange(min_year, max_year + 1)
            
            # Fit trend on available FAO data
            X = fao_clean['year'].values.reshape(-1, 1)
            y = fao_clean['fao_yield'].values
            
            try:
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict for full year range
                X_full = year_range.reshape(-1, 1)
                y_trend = model.predict(X_full)
                
                ax.plot(year_range, y_trend, color=trend_color, linestyle=':', 
                       linewidth=1.2, label='Trend')
            except:
                pass
        
        # Formatting
        ax.set_title(country, fontname='Times New Roman', fontsize=12)
        ax.set_xlabel('Year', fontname='Times New Roman', fontsize=12)
        ax.set_ylabel('Yield (t/ha)', fontname='Times New Roman', fontsize=12)
        ax.grid(True, alpha=0.2)
        
        # Set font for tick labels
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontsize(10)
        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontsize(10)
    
    # Add legend at the bottom center
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, prop={'family': 'Times New Roman', 'size': 10}, bbox_to_anchor=(0.5, -0.02))
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(PROJECT_ROOT, 'Plots', 'output', 'golden_cohort_national_timeseries.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot without model saved to: {output_path}")
    return output_path

def create_plot_with_model(data):
    """Create the 3x3 grid plot with model predictions."""
    # Set up the plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Comparison FAO Timeseries to Aggregated HarvestStat Data with Model Predictions', 
                 fontsize=16, fontname='Times New Roman')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Define colors
    fao_color = '#24245c'      # Blue from graph designer
    ground_color = '#fccd32'   # Yellow from graph designer
    model_color = 'red'        # Red for model predictions
    trend_color = 'gray'
    
    # Create legend handles manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=fao_color, marker='D', linestyle='-', markersize=6, linewidth=2, label='FAO'),
        Line2D([0], [0], color=ground_color, marker='s', linestyle='-', markersize=5, linewidth=2, label='Ground Data'),
        Line2D([0], [0], color=model_color, marker='o', linestyle='--', markersize=5, linewidth=2, label='Model'),
        Line2D([0], [0], color=trend_color, linestyle=':', linewidth=1.2, label='Trend')
    ]
    
    # Plot each country
    for i, country in enumerate(golden_cohort):
        if i >= 9:  # Only plot first 9 countries
            break
            
        ax = axes_flat[i]
        
        # Filter data for this country
        country_data = data[data['country'] == country].sort_values('year')
        
        if country_data.empty:
            ax.set_title(country, fontname='Times New Roman', fontsize=12)
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontname='Times New Roman')
            continue
        
        # Plot FAO data (blue diamonds) - only connect points with consecutive years
        fao_clean = country_data.dropna(subset=['fao_yield'])
        if not fao_clean.empty:
            # Sort by year and group consecutive years together
            fao_clean = fao_clean.sort_values('year').copy()
            fao_clean['year_diff'] = fao_clean['year'].diff()
            fao_clean['group'] = (fao_clean['year_diff'] != 1).cumsum()
            
            # Plot each group separately to avoid connecting non-consecutive years
            for group_id, group_data in fao_clean.groupby('group'):
                ax.plot(group_data['year'], group_data['fao_yield'], 
                       color=fao_color, linewidth=2, marker='D', markersize=6, label='FAO' if group_id == 0 else "")
        
        # Plot ground data (yellow squares) - only connect points with consecutive years
        ground_clean = country_data.dropna(subset=['yield'])
        if not ground_clean.empty:
            # Sort by year and group consecutive years together
            ground_clean = ground_clean.sort_values('year').copy()
            ground_clean['year_diff'] = ground_clean['year'].diff()
            ground_clean['group'] = (ground_clean['year_diff'] != 1).cumsum()
            
            # Plot each group separately to avoid connecting non-consecutive years
            for group_id, group_data in ground_clean.groupby('group'):
                ax.plot(group_data['year'], group_data['yield'], 
                       color=ground_color, linewidth=2, marker='s', markersize=5, label='Ground' if group_id == 0 else "")
        
        # Plot model predictions (red line) - only connect points with consecutive years
        model_clean = country_data.dropna(subset=['yield_model'])
        if not model_clean.empty:
            # Sort by year and group consecutive years together
            model_clean = model_clean.sort_values('year').copy()
            model_clean['year_diff'] = model_clean['year'].diff()
            model_clean['group'] = (model_clean['year_diff'] != 1).cumsum()
            
            # Plot each group separately to avoid connecting non-consecutive years
            for group_id, group_data in model_clean.groupby('group'):
                ax.plot(group_data['year'], group_data['yield_model'], 
                       color=model_color, linewidth=2, marker='o', markersize=5, label='Model' if group_id == 0 else "")
        
        # Compute and plot trend line based on FAO data for full year range
        if not fao_clean.empty and len(fao_clean) >= 2:
            # Get the full year range from FAO data
            min_year = fao_clean['year'].min()
            max_year = fao_clean['year'].max()
            year_range = np.arange(min_year, max_year + 1)
            
            # Fit trend on available FAO data
            X = fao_clean['year'].values.reshape(-1, 1)
            y = fao_clean['fao_yield'].values
            
            try:
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict for full year range
                X_full = year_range.reshape(-1, 1)
                y_trend = model.predict(X_full)
                
                ax.plot(year_range, y_trend, color=trend_color, linestyle=':', 
                       linewidth=1.2, label='Trend')
            except:
                pass
        
        # Formatting
        ax.set_title(country, fontname='Times New Roman', fontsize=12)
        ax.set_xlabel('Year', fontname='Times New Roman', fontsize=12)
        ax.set_ylabel('Yield (t/ha)', fontname='Times New Roman', fontsize=12)
        ax.grid(True, alpha=0.2)
        
        # Set font for tick labels
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontsize(10)
        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontsize(10)
    
    # Add legend at the bottom center
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, prop={'family': 'Times New Roman', 'size': 10}, bbox_to_anchor=(0.5, -0.02))
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(PROJECT_ROOT, 'Plots', 'output', 'golden_cohort_national_timeseries_with_model.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot with model saved to: {output_path}")
    return output_path


def create_plot_detrended(data):
    """Create the 3x3 grid plot with detrended values."""
    # Set up the plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Detrended Comparison FAO Timeseries to Aggregated HarvestStat Data with Model Predictions', 
                 fontsize=16, fontname='Times New Roman')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Define colors
    fao_color = '#24245c'      # Blue from graph designer
    ground_color = '#fccd32'   # Yellow from graph designer
    model_color = 'red'        # Red for model predictions
    
    # Create legend handles manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=fao_color, marker='D', linestyle='-', markersize=6, linewidth=2, label='FAO'),
        Line2D([0], [0], color=ground_color, marker='s', linestyle='-', markersize=5, linewidth=2, label='Ground Data'),
        Line2D([0], [0], color=model_color, marker='o', linestyle='--', markersize=5, linewidth=2, label='Model')
    ]
    
    # Plot each country
    for i, country in enumerate(golden_cohort):
        if i >= 9:  # Only plot first 9 countries
            break
            
        ax = axes_flat[i]
        
        # Filter data for this country
        country_data = data[data['country'] == country].sort_values('year')
        
        if country_data.empty:
            ax.set_title(country, fontname='Times New Roman', fontsize=12)
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontname='Times New Roman')
            continue
        
        # Compute detrended values for each series
        fao_clean = country_data.dropna(subset=['fao_yield'])
        ground_clean = country_data.dropna(subset=['yield'])
        model_clean = country_data.dropna(subset=['yield_model'])
        
        # Compute relative deviations
        if not fao_clean.empty:
            rel_fao = compute_relative_deviation(fao_clean['fao_yield'], fao_clean['year'])
            fao_clean = fao_clean.copy()
            fao_clean['fao_yield_detrended'] = rel_fao
            
        if not ground_clean.empty:
            rel_ground = compute_relative_deviation(ground_clean['yield'], ground_clean['year'])
            ground_clean = ground_clean.copy()
            ground_clean['yield_detrended'] = rel_ground
            
        if not model_clean.empty:
            rel_model = compute_relative_deviation(model_clean['yield_model'], model_clean['year'])
            model_clean = model_clean.copy()
            model_clean['yield_model_detrended'] = rel_model
        
        # Plot detrended FAO data (blue diamonds) - only connect points with consecutive years
        if not fao_clean.empty and 'fao_yield_detrended' in fao_clean.columns:
            # Sort by year and group consecutive years together
            fao_clean = fao_clean.sort_values('year').copy()
            fao_clean['year_diff'] = fao_clean['year'].diff()
            fao_clean['group'] = (fao_clean['year_diff'] != 1).cumsum()
            
            # Plot each group separately to avoid connecting non-consecutive years
            for group_id, group_data in fao_clean.groupby('group'):
                ax.plot(group_data['year'], group_data['fao_yield_detrended'], 
                       color=fao_color, linewidth=2, marker='D', markersize=6, label='FAO' if group_id == 0 else "")
        
        # Plot detrended ground data (yellow squares) - only connect points with consecutive years
        if not ground_clean.empty and 'yield_detrended' in ground_clean.columns:
            # Sort by year and group consecutive years together
            ground_clean = ground_clean.sort_values('year').copy()
            ground_clean['year_diff'] = ground_clean['year'].diff()
            ground_clean['group'] = (ground_clean['year_diff'] != 1).cumsum()
            
            # Plot each group separately to avoid connecting non-consecutive years
            for group_id, group_data in ground_clean.groupby('group'):
                ax.plot(group_data['year'], group_data['yield_detrended'], 
                       color=ground_color, linewidth=2, marker='s', markersize=5, label='Ground' if group_id == 0 else "")
        
        # Plot detrended model predictions (red line) - only connect points with consecutive years
        if not model_clean.empty and 'yield_model_detrended' in model_clean.columns:
            # Sort by year and group consecutive years together
            model_clean = model_clean.sort_values('year').copy()
            model_clean['year_diff'] = model_clean['year'].diff()
            model_clean['group'] = (model_clean['year_diff'] != 1).cumsum()
            
            # Plot each group separately to avoid connecting non-consecutive years
            for group_id, group_data in model_clean.groupby('group'):
                ax.plot(group_data['year'], group_data['yield_model_detrended'], 
                       color=model_color, linewidth=2, marker='o', markersize=5, label='Model' if group_id == 0 else "")
        
        # Zero baseline
        ax.axhline(0.0, color='grey', linestyle=':', linewidth=1.2)
        
        # Formatting
        ax.set_title(country, fontname='Times New Roman', fontsize=12)
        ax.set_xlabel('Year', fontname='Times New Roman', fontsize=12)
        ax.set_ylabel('Relative deviation from trend', fontname='Times New Roman', fontsize=12)
        ax.grid(True, alpha=0.2)
        
        # Set font for tick labels
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontsize(10)
        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
            tick.set_fontsize(10)
    
    # Add legend at the bottom center
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, prop={'family': 'Times New Roman', 'size': 10}, bbox_to_anchor=(0.5, -0.02))
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(PROJECT_ROOT, 'Plots', 'output', 'golden_cohort_national_timeseries_detrended.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detrended plot saved to: {output_path}")
    return output_path

def main():
    """Main function to generate the plots."""
    print("Starting generation of golden cohort national timeseries plots...")
    
    # Load data
    predictions_df, fao_df = load_data()
    
    # Filter for golden cohort
    predictions_filtered, fao_filtered = filter_golden_cohort_data(predictions_df, fao_df)
    
    # Prepare data for plotting
    plotting_data = prepare_plotting_data(predictions_filtered, fao_filtered)
    
    # Create and save plot without model
    output_path_without_model = create_plot_without_model(plotting_data)
    
    # Create and save plot with model
    output_path_with_model = create_plot_with_model(plotting_data)
    
    # Create and save detrended plot
    output_path_detrended = create_plot_detrended(plotting_data)
    
    print("Plot generation complete!")

if __name__ == "__main__":
    main()