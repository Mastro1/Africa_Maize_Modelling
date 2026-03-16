import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots
from scipy.stats import linregress
from matplotlib.lines import Line2D
from statsmodels.nonparametric.kernel_regression import KernelReg

# Set global scientific style
plt.style.use(['science', 'ieee'])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_V7 = os.path.join(PROJECT_ROOT, 'Model_physical', 'Results', 'V7_model', 'Global_Maize_Yield_V7.csv')
CROP5MIN = os.path.join(PROJECT_ROOT, 'ResultsComparison', 'cropyield5min', 'aggregate_admin2', 'maize_yield_admin2_1982_2015.csv')
GDHY = os.path.join(PROJECT_ROOT, 'ResultsComparison', 'GDHY', 'aggregate_admin2', 'maize_yield_admin2_1981_2016.csv')
SPAM = os.path.join(PROJECT_ROOT, 'ResultsComparison', 'SPAM', 'aggregate_admin2', 'spam_maize_yield_admin2_2000_2020.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Model_physical', 'plots', 'output')

def smooth_data_kernel_regression(arr: np.array, years: np.array, bw: int = 3) -> np.ndarray:
    """
    Smooth the data using kernel regression.
    """
    if len(arr) < 3:
        return arr
    try:
        kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[bw])
        return kr.fit()[0]
    except:
        return arr

def calculate_anomaly(df, val_col, method='kernel', **kwargs):
    res = []
    for pcode in df['PCODE'].unique():
        loc_df = df[df['PCODE'] == pcode].sort_values('Year').copy()
        if len(loc_df) < 3:
            continue
        ys = loc_df[val_col].values
        yrs = loc_df['Year'].values
        
        if method == 'kernel':
            trend = smooth_data_kernel_regression(ys, yrs, bw=kwargs.get('bw', 3))
        elif method == 'linear':
            slope, intercept, r_value, p_value, std_err = linregress(yrs, ys)
            trend = slope * yrs + intercept
        else:
            raise ValueError(f"Unknown method {method}")
            
        with np.errstate(divide='ignore', invalid='ignore'):
            anom = (ys - trend) / trend
            
        loc_df[val_col] = anom  # Replace raw yield with anomaly
        res.append(loc_df)
    
    if not res:
        return pd.DataFrame()
    
    out = pd.concat(res, ignore_index=True)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out.dropna(subset=[val_col])

def main():
    print("Loading data...")
    # 1. Load V7 Data
    df_v7 = pd.read_csv(RESULTS_V7, usecols=['PCODE', 'Year', 'Season', 'Yield_Estimated_t_ha'])
    df_v7 = df_v7[df_v7['Season'] == 1].copy()
    
    # 2. Load Benchmark Data
    df_crop = pd.read_csv(CROP5MIN)
    df_gdhy = pd.read_csv(GDHY)
    df_spam = pd.read_csv(SPAM)
    
    # Standardize column names for merging
    df_crop = df_crop.rename(columns={'FNID': 'PCODE', 'year': 'Year', 'yield': 'Yield_Crop'})
    df_gdhy = df_gdhy.rename(columns={'FNID': 'PCODE', 'year': 'Year', 'yield': 'Yield_GDHY'})
    df_spam = df_spam.rename(columns={'FNID': 'PCODE', 'year': 'Year', 'yield': 'Yield_SPAM'})
    
    print("Calculating detrended anomalies (kernel regression with bw=3 for most, bw=4 for SPAM)...")
    df_v7 = calculate_anomaly(df_v7, 'Yield_Estimated_t_ha', method='kernel', bw=3)
    df_crop = calculate_anomaly(df_crop, 'Yield_Crop', method='kernel', bw=3)
    df_gdhy = calculate_anomaly(df_gdhy, 'Yield_GDHY', method='kernel', bw=3)
    df_spam = calculate_anomaly(df_spam, 'Yield_SPAM', method='kernel', bw=4)
    
    years = [2000, 2005, 2010]
    
    # Extract standard scienceplot colors (sober, accessible)
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {2000: prop_cycle[0], 2005: prop_cycle[1], 2010: prop_cycle[2]}
    
    # 3. Figure Setup (IEEE Double Column, 3x3 Grid)
    width = 7.0 
    height = width  # Square figure for 3x3 layout
    fig, axes = plt.subplots(3, 3, figsize=(width, height), sharey=True, sharex=True)
    
    benchmarks = [
        ('GlobalCropYield5min', df_crop, 'Yield_Crop'),
        ('SPAM', df_spam, 'Yield_SPAM'),
        ('GDHY', df_gdhy, 'Yield_GDHY')
    ]
    
    # 4. Process and Plot
    for col_idx, (title, df_benchmark, col) in enumerate(benchmarks):
        # Merge individual benchmark with V7
        df_merged = df_v7.merge(df_benchmark, on=['PCODE', 'Year'], how='inner')
        
        for row_idx, year in enumerate(years):
            ax = axes[row_idx, col_idx]
            df_y = df_merged[df_merged['Year'] == year].dropna(subset=['Yield_Estimated_t_ha', col])
            
            if df_y.empty:
                print(f"Warning: No data for {title} in {year}")
                # Clean up empty axes
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_xticks(np.arange(-1.5, 1.6, 0.5))
                ax.set_yticks(np.arange(-1.5, 1.6, 0.5))
                ax.grid(True, linestyle='--', linewidth=0.2, alpha=0.4)
                if row_idx == 0:
                    ax.set_title(title, fontsize=9)
                if row_idx == 2:
                    ax.set_xlabel(f"Reported Anomaly", fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(f"Simulated LUE-Net Africa Anomaly ({year})", fontsize=8)
                continue
                
            x = df_y[col].values
            y = df_y['Yield_Estimated_t_ha'].values
            
            # Scatter Plot (all points use the standard color now)
            ax.scatter(x, y, s=1.5, c=prop_cycle[0], alpha=0.4, edgecolors='none', zorder=2)
            
            # Linear Regression
            if len(x) > 1:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                r2 = r_value**2
                
                # Plot regression line across the data range
                line_x = np.array([-1.5, 1.5])
                line_y = slope * line_x + intercept
                
                ax.plot(line_x, line_y, c=prop_cycle[1], linewidth=1.5, alpha=0.8, zorder=3)
                
                # R2 Annotation
                ax.text(0.05, 0.95, f"$R^2 = {r2:.2f}$", 
                        transform=ax.transAxes, color=prop_cycle[1], 
                        fontsize=7, verticalalignment='top')
                        
        # Reference Lines (0 anomaly crosshairs and 1:1 fit line)
            ax.axline((0, 0), slope=1, color='k', linestyle='--', alpha=0.3, linewidth=0.8, zorder=1)
            ax.axhline(0, color='k', linestyle='-', alpha=0.1, linewidth=0.8, zorder=1)
            ax.axvline(0, color='k', linestyle='-', alpha=0.1, linewidth=0.8, zorder=1)

            # Axes Formatting
            
            # Symmetrical limits for anomalies (typically between -1.0 and 1.5)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            
            ax.set_xticks(np.arange(-1.5, 1.6, 0.5))
            ax.set_yticks(np.arange(-1.5, 1.6, 0.5))
            
            ax.grid(True, linestyle='--', linewidth=0.2, alpha=0.4)
            
            # Only set titles on the top row
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
                
            # Only set x-labels on the bottom row
            if row_idx == 2:
                ax.set_xlabel(f"Reported Anomaly", fontsize=8)
            
            # Only set y-labels on the left column, including the specific year
            if col_idx == 0:
                ax.set_ylabel(f"LUE-Net Africa Anomaly \n({year})", fontsize=8)
            

    # Adjust layout
    plt.tight_layout()
    
    # 5. Export
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_pdf = os.path.join(OUTPUT_DIR, 'fig_yield_comparison_anomaly.pdf')
    out_png = os.path.join(OUTPUT_DIR, 'fig_yield_comparison_anomaly.png')
    
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(out_png, format='png', bbox_inches='tight', dpi=300)
    
    print()
    print(f"Graph successfully generated and exported to:")
    print(f" - {out_pdf}")
    print(f" - {out_png}")
    print("Done!")

if __name__ == "__main__":
    main()
