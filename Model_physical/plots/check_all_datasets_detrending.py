import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots
from statsmodels.nonparametric.kernel_regression import KernelReg

# Set global scientific style
plt.style.use(['science', 'ieee'])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Model_physical', 'plots', 'output')

DATASETS = {
    'SPAM': {
        'path': os.path.join(PROJECT_ROOT, 'ResultsComparison', 'SPAM', 'aggregate_admin2', 'spam_maize_yield_admin2_2000_2020.csv'),
        'col': 'yield'
    },
    'GDHY': {
        'path': os.path.join(PROJECT_ROOT, 'ResultsComparison', 'GDHY', 'aggregate_admin2', 'maize_yield_admin2_1981_2016.csv'),
        'col': 'yield'
    },
    'Crop5min': {
        'path': os.path.join(PROJECT_ROOT, 'ResultsComparison', 'cropyield5min', 'aggregate_admin2', 'maize_yield_admin2_1982_2015.csv'),
        'col': 'yield'
    }
}

def smooth_data_kernel_regression(arr: np.array, years: np.array) -> np.ndarray:
    if len(arr) < 3:
        return arr
    try:
        kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[4])
        return kr.fit()[0]
    except:
        return arr

def visualize_dataset(name, info, target_countries):
    print(f"Processing dataset: {name}")
    if not os.path.exists(info['path']):
        print(f"Warning: File not found {info['path']}")
        return

    df = pd.read_csv(info['path'])
    df = df.rename(columns={'FNID': 'PCODE', 'year': 'Year', info['col']: 'Yield'})
    df['Country'] = df['PCODE'].str.split('.').str[0]
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, country in enumerate(target_countries):
        ax = axes[i]
        country_df = df[df['Country'] == country]
        if country_df.empty:
            ax.text(0.5, 0.5, f"No Data for {country}", ha='center')
            continue
            
        pcodes = country_df['PCODE'].unique()
        selected_pcode = None
        for pcode in pcodes:
            loc_df = country_df[country_df['PCODE'] == pcode].sort_values('Year')
            # Look for 10+ years of data if possible, else 5
            if len(loc_df) >= 10 and loc_df['Yield'].sum() > 0:
                selected_pcode = pcode
                break
        
        if not selected_pcode:
            for pcode in pcodes:
                loc_df = country_df[country_df['PCODE'] == pcode].sort_values('Year')
                if len(loc_df) >= 5 and loc_df['Yield'].sum() > 0:
                    selected_pcode = pcode
                    break
        
        if not selected_pcode:
            selected_pcode = pcodes[0]
            loc_df = country_df[country_df['PCODE'] == selected_pcode].sort_values('Year')

        yrs = loc_df['Year'].values
        ys = loc_df['Yield'].values
        trend = smooth_data_kernel_regression(ys, yrs)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            anom = (ys - trend) / trend
        
        ax.scatter(yrs, ys, s=8, label='Raw', color='black', alpha=0.5)
        ax.plot(yrs, trend, color='red', linewidth=1.2)
        
        ax_anom = ax.twinx()
        ax_anom.bar(yrs, anom, alpha=0.2, color='blue', width=0.6)
        ax_anom.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax_anom.set_ylim(-1, 1)
        
        ax.set_title(f"{country} ({selected_pcode})", fontsize=9)
        if i >= 6: ax.set_xlabel("Year")
        if i % 3 == 0: ax.set_ylabel("Yield (t/ha)")
        if i % 3 == 2: ax_anom.set_ylabel("Anomaly")
        else: ax_anom.set_yticklabels([])

    plt.suptitle(f"Yield Detrending (KR bw=3) - {name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(OUTPUT_DIR, f'detrending_check_{name}.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_png}")

def main():
    target_countries = ['ETH', 'KEN', 'NGA', 'ZAF', 'TZA', 'MWI', 'ZMB', 'GHA', 'MLI']
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name, info in DATASETS.items():
        visualize_dataset(name, info, target_countries)

if __name__ == "__main__":
    main()
