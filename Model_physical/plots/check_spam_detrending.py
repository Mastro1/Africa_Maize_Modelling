import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots
from statsmodels.nonparametric.kernel_regression import KernelReg

# Set global scientific style
plt.style.use(['science', 'ieee'])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SPAM_PATH = os.path.join(PROJECT_ROOT, 'ResultsComparison', 'SPAM', 'aggregate_admin2', 'spam_maize_yield_admin2_2000_2020.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Model_physical', 'plots', 'output')

def smooth_data_kernel_regression(arr: np.array, years: np.array) -> np.ndarray:
    """
    Smooth the data using kernel regression with a bandwidth of 3.
    Matches the logic in fig_yield_comparison_V7.py
    """
    if len(arr) < 3:
        return arr
    try:
        kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[3])
        return kr.fit()[0]
    except Exception as e:
        print(f"Error in smoothing: {e}")
        return arr

def main():
    print(f"Loading SPAM data from: {SPAM_PATH}")
    if not os.path.exists(SPAM_PATH):
        print(f"Error: SPAM file not found at {SPAM_PATH}")
        return

    df = pd.read_csv(SPAM_PATH)
    df = df.rename(columns={'FNID': 'PCODE', 'year': 'Year', 'yield': 'Yield_SPAM'})
    
    # Extract Country from PCODE (e.g., 'DZA.2_1' -> 'DZA')
    df['Country'] = df['PCODE'].str.split('.').str[0]
    
    countries = df['Country'].unique()
    print(f"Found {len(countries)} countries in SPAM data.")
    
    # Select locations with some zeros to investigate
    pcode_zeros = df.groupby('PCODE')['Yield_SPAM'].apply(lambda x: (x == 0).sum()).reset_index()
    zero_pcode_list = pcode_zeros[pcode_zeros['Yield_SPAM'] > 0]['PCODE'].unique()
    
    print(f"Found {len(zero_pcode_list)} locations with at least one zero yield.")
    
    target_pcodes = zero_pcode_list[:9] # Take first 9 for visualization

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, pcode in enumerate(target_pcodes):
        ax = axes[i]
        loc_df = df[df['PCODE'] == pcode].sort_values('Year')
        country = df[df['PCODE'] == pcode]['Country'].iloc[0]

        yrs = loc_df['Year'].values
        ys = loc_df['Yield_SPAM'].values
        
        # Calculate linear trend
        if len(yrs) > 1:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(yrs, ys)
            trend = slope * yrs + intercept
        else:
            trend = ys
        
        # Calculate anomaly
        with np.errstate(divide='ignore', invalid='ignore'):
            anom = (ys - trend) / trend
        
        # Plotting
        ax.scatter(yrs, ys, s=10, label='Raw Yield', color='black', alpha=0.6)
        ax.plot(yrs, trend, label='Linear Trend', color='red', linewidth=1.5)
        
        # Second axis for anomaly
        ax_anom = ax.twinx()
        ax_anom.bar(yrs, anom, alpha=0.3, color='blue', label='Anomaly', width=0.5)
        ax_anom.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax_anom.set_ylim(-1, 1) # Typical anomaly range
        
        ax.set_title(f"{country} ({pcode})", fontsize=10)
        if i >= 6:
            ax.set_xlabel("Year")
        if i % 3 == 0:
            ax.set_ylabel("Yield (t/ha)")
        if i % 3 == 2:
            ax_anom.set_ylabel("Anomaly (%)")
        else:
            ax_anom.set_yticklabels([])

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_png = os.path.join(OUTPUT_DIR, 'spam_detrending_check.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {out_png}")

if __name__ == "__main__":
    main()
