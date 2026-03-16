import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.metrics import r2_score

def smooth_data_kernel_regression(arr: np.array, years: np.array) -> np.ndarray:
    """
    Smooth the data using kernel regression with a bandwith of 3.

    Args:
        arr: The input data array to be smoothed.

    Returns:
        A numpy array containing the smoothed data.
    
    Notes:
        documentation: https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html

    """
    # Perform kernel regression on the input array
    kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[3])

    # Obtain the smoothed data
    smoothed_data = kr.fit()[0]

    return smoothed_data

def verify_fao(country_name, input_results_dir=None, version='V6', save_output=True, df_model=None):
    """
    Verifies model results against FAO data, aggregating dual seasons with 0.7/0.3 weighting.
    If save_output is False, it will not create plots or tables.
    Returns: comparison DataFrame, corr_ker (kernel detrended correlation), r2_ker (kernel detrended R2).
    """
    # Defaults
    if input_results_dir is None:
        if 'V7_Optuna' in version or 'V7_Final' in version:
            input_results_dir = r'Model_physical/Results/V7_model_optimized'
        else:
            input_results_dir = r'Model_physical/Results/V6_model'
    
    # 1. Load Model Results if not provided
    if df_model is None:
        # Try global file first, then country-specific
        global_path = os.path.join(input_results_dir, f'Global_Maize_Yield_{version}.csv')
        country_path = os.path.join(input_results_dir, f'maize_yield_estimates_{version}_{country_name.replace(" ", "_")}.csv')
        
        if os.path.exists(global_path):
            print(f"Loading model results from {global_path}...")
            df_model_all = pd.read_csv(global_path)
            # Robust filtering for country
            df_model = df_model_all[df_model_all['Country'].str.strip().str.lower() == country_name.strip().lower()].copy()
            
            if df_model.empty:
                print(f"Warning: No data for '{country_name}' in global file.")
                print(f"Available countries in global file: {df_model_all['Country'].unique()}")
        elif os.path.exists(country_path):
            print(f"Loading model results from {country_path}...")
            df_model = pd.read_csv(country_path)
        else:
            print(f"Error: Model results not found in {input_results_dir}")
            return None, None
    else:
        # Use provided df_model directly
        print("Using provided model results dataframe...")

    if df_model.empty:
        print(f"Error: No data for {country_name} in model results.")
        return None, None, None

    # --- Sanity Checks & Outlier Removal ---
    print("\nApplying Sanity Checks:")
    initial_count = len(df_model)
    
    # 1. Yield Outliers (e.g., > 12 t/ha)
    yield_threshold = 12.0
    df_model = df_model[df_model['Yield_Estimated_t_ha'] <= yield_threshold].copy()
    yield_filtered = initial_count - len(df_model)
    if yield_filtered > 0:
        print(f"  - Removed {yield_filtered} records exceeding {yield_threshold} t/ha.")

    # 2. Season Length Outliers (e.g., < 50 days)
    # Convert dates to datetime if they aren't already
    df_model['Start_Date'] = pd.to_datetime(df_model['Start_Date'])
    df_model['End_Date'] = pd.to_datetime(df_model['End_Date'])
    df_model['Season_Length_Days'] = (df_model['End_Date'] - df_model['Start_Date']).dt.days
    
    length_threshold = 50
    final_count_before_len = len(df_model)
    df_model = df_model[df_model['Season_Length_Days'] >= length_threshold].copy()
    length_filtered = final_count_before_len - len(df_model)
    if length_filtered > 0:
        print(f"  - Removed {length_filtered} records with season length < {length_threshold} days.")

    print(f"  - Final clean record count: {len(df_model)} (Filtered {initial_count - len(df_model)} outliers).\n")

    # 2. Load Crop Area Data (for weighting)
    crop_area_path = r'GADM/crop_areas/africa_crop_areas_glad_filtered.csv'
    df_area = pd.read_csv(crop_area_path)
    # Filter for the specific country in crop area data
    print(f"Filtering crop area data for: {country_name}")
    df_area_country = df_area[df_area['country'].str.strip().str.lower() == country_name.strip().lower()].copy()
    
    if df_area_country.empty:
        print(f"Error: No crop area data found for country '{country_name}'. Check spelling.")
        print(f"Available countries in area file (sample): {df_area['country'].unique()[:10]}...")
        return None, None, None

    # 3. Dual Season Aggregation at PCODE level
    # Weighting: 0.7 for Season 1, 0.3 for Season 2
    print("Aggregating dual seasons (Weights: S1=0.7, S2=0.3)...")
    
    def aggregate_seasons(group):
        y1 = group[group['Season'] == 1]['Yield_Estimated_t_ha'].mean()
        y2 = group[group['Season'] == 2]['Yield_Estimated_t_ha'].mean()
        
        # Fixed blend logic requested by user
        w1, w2 = 0.7, 0.3
        total_w = 0
        blend_y = 0
        
        if not np.isnan(y1):
            blend_y += y1 * w1
            total_w += w1
        if not np.isnan(y2):
            blend_y += y2 * w2
            total_w += w2
            
        return blend_y / total_w if total_w > 0 else np.nan

    # Group by PCODE and Year to get an annual 'Modelled Yield' per district
    df_annual_pcode = df_model.groupby(['PCODE', 'Year']).apply(aggregate_seasons).reset_index(name='Yield_Weighted')
    df_annual_pcode = df_annual_pcode.rename(columns={'Year': 'year'})

    # 4. Merge with Crop Area
    cols_area = ['PCODE', 'crop_area_ha'] # Crop area is static in this file
    df_area_subset = df_area_country[['PCODE', 'crop_area_ha']].drop_duplicates()
    
    df_merged = pd.merge(df_annual_pcode, df_area_subset, on='PCODE', how='inner')
    
    if df_merged.empty:
        print(f"Warning: Merged dataframe is empty for {country_name}.")
        print("Possible reasons: PCODE mismatch between model results and crop area data.")
        print(f"Model PCODEs sample: {df_annual_pcode['PCODE'].unique()[:5]}")
        print(f"Area PCODEs sample: {df_area_subset['PCODE'].unique()[:5]}")
        return None, None, None

    # 5. National Roll-up (Area-Weighted)
    print("Calculating national yield estimate...")
    df_merged['production_proxy'] = df_merged['Yield_Weighted'] * df_merged['crop_area_ha']
    
    national_stats = df_merged.groupby('year').apply(
        lambda x: pd.Series({
            'Model_Yield_t_ha': x['production_proxy'].sum() / x['crop_area_ha'].sum(),
            'Total_Area_ha': x['crop_area_ha'].sum(),
            'Sample_Size': len(x)
        })
    ).reset_index()

    # 6. Load FAO Data
    fao_path = r'FAOSTAT/faostat_maize.csv'
    df_fao = pd.read_csv(fao_path)
    df_fao_country = df_fao[
        (df_fao['Area'] == country_name) & 
        (df_fao['Element'] == 'Yield') & 
        (df_fao['Item'] == 'Maize (corn)')
    ].copy()
    
    if df_fao_country.empty:
        print(f"Error: No FAO yield data found for {country_name}.")
        # Use simple fuzzy match check/suggestion
        possible_matches = df_fao[df_fao['Area'].str.contains(country_name, case=False, na=False)]['Area'].unique()
        if len(possible_matches) > 0:
            print(f"Did you mean: {possible_matches}?")
        return None, None, None

    unit = df_fao_country['Unit'].iloc[0]
    scale = 1000.0 if unit == 'kg/ha' else (10000.0 if unit == 'hg/ha' else 1.0)
    df_fao_country['FAO_Yield_t_ha'] = df_fao_country['Value'] / scale
    df_fao_country = df_fao_country[['Year', 'FAO_Yield_t_ha']].rename(columns={'Year': 'year'})

    # 7. Comparison
    comparison = pd.merge(national_stats, df_fao_country, on='year', how='inner')
    comparison = comparison[comparison['year'] != 2000] # Standard exclusion
    
    if comparison.empty:
        print("No overlapping years found.")
        return None, None, None

    # --- Detrending Analysis ---
    print("Performing detrending analysis...")
    
    # 1. Linear Trend
    model_coeffs = np.polyfit(comparison['year'], comparison['Model_Yield_t_ha'], 1)
    model_linear_trend = np.polyval(model_coeffs, comparison['year'])
    comparison['Model_Yield_Linear_Trend'] = model_linear_trend
    comparison['Model_Dev_Linear'] = (comparison['Model_Yield_t_ha'] - model_linear_trend) / model_linear_trend
    
    fao_coeffs = np.polyfit(comparison['year'], comparison['FAO_Yield_t_ha'], 1)
    fao_linear_trend = np.polyval(fao_coeffs, comparison['year'])
    comparison['FAO_Yield_Linear_Trend'] = fao_linear_trend
    comparison['FAO_Dev_Linear'] = (comparison['FAO_Yield_t_ha'] - fao_linear_trend) / fao_linear_trend
    
    # 2. Kernel Trend
    comparison['Model_Yield_Kernel_Trend'] = smooth_data_kernel_regression(comparison['Model_Yield_t_ha'].values, comparison['year'].values)
    comparison['Model_Dev_Kernel'] = (comparison['Model_Yield_t_ha'] - comparison['Model_Yield_Kernel_Trend']) / comparison['Model_Yield_Kernel_Trend']
    
    comparison['FAO_Yield_Kernel_Trend'] = smooth_data_kernel_regression(comparison['FAO_Yield_t_ha'].values, comparison['year'].values)
    comparison['FAO_Dev_Kernel'] = (comparison['FAO_Yield_t_ha'] - comparison['FAO_Yield_Kernel_Trend']) / comparison['FAO_Yield_Kernel_Trend']

    # Metrics
    corr_orig = comparison['Model_Yield_t_ha'].corr(comparison['FAO_Yield_t_ha'])
    corr_lin = comparison['Model_Dev_Linear'].corr(comparison['FAO_Dev_Linear'])
    corr_ker = comparison['Model_Dev_Kernel'].corr(comparison['FAO_Dev_Kernel'])
    
    try:
        r2_ker = r2_score(comparison['FAO_Dev_Kernel'], comparison['Model_Dev_Kernel'])
    except Exception:
        r2_ker = np.nan
    
    print(f"\nVerification Results for {country_name} (Dual Season V6):")
    print(f"Raw Correlation: {corr_orig:.4f}")
    print(f"Linear Detrended Correlation: {corr_lin:.4f}")
    print(f"Kernel Detrended Correlation: {corr_ker:.4f}")
    print(f"Kernel Detrended R2: {r2_ker:.4f}")
    
    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Subplot 1: Original Data with Trends
    ax1 = axes[0]
    ax1.plot(comparison['year'], comparison['Model_Yield_t_ha'], label='Model V6 (0.7S1+0.3S2)', marker='o', color='blue', alpha=0.7)
    ax1.plot(comparison['year'], comparison['Model_Yield_Linear_Trend'], linestyle='--', color='blue', alpha=0.5, label='Model Linear Trend')
    ax1.plot(comparison['year'], comparison['Model_Yield_Kernel_Trend'], linestyle=':', color='blue', alpha=0.8, label='Model Kernel Trend')
    
    ax1.plot(comparison['year'], comparison['FAO_Yield_t_ha'], label='FAO National Yield', marker='x', linestyle='--', color='orange', alpha=0.7)
    ax1.plot(comparison['year'], comparison['FAO_Yield_Linear_Trend'], linestyle='--', color='orange', alpha=0.5, label='FAO Linear Trend')
    ax1.plot(comparison['year'], comparison['FAO_Yield_Kernel_Trend'], linestyle=':', color='orange', alpha=0.8, label='FAO Kernel Trend')
    
    ax1.set_title(f'{country_name}: FAO vs Model V6 (Original Yields)\nRaw Corr: {corr_orig:.2f}')
    ax1.set_ylabel('Yield (t/ha)')
    ax1.legend(ncol=2, fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Linear Detrended
    ax2 = axes[1]
    ax2.plot(comparison['year'], comparison['Model_Dev_Linear'], label='Model Dev (Linear)', marker='o', color='blue')
    ax2.plot(comparison['year'], comparison['FAO_Dev_Linear'], label='FAO Dev (Linear)', marker='x', color='orange')
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax2.set_title(f'Linear Detrended (Relative Deviation)\nCorrelation: {corr_lin:.2f}')
    ax2.set_ylabel('Relative Deviation')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Kernel Detrended
    ax3 = axes[2]
    ax3.plot(comparison['year'], comparison['Model_Dev_Kernel'], label='Model Dev (Kernel)', marker='o', color='blue')
    ax3.plot(comparison['year'], comparison['FAO_Dev_Kernel'], label='FAO Dev (Kernel)', marker='x', color='orange')
    ax3.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax3.set_title(f'Kernel Detrended (Relative Deviation)\nCorrelation: {corr_ker:.2f}')
    ax3.set_ylabel('Relative Deviation')
    ax3.set_xlabel('Year')
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_output:
        # Save results
        # Create dir if not exists
        os.makedirs(input_results_dir, exist_ok=True)
        
        output_plot = os.path.join(input_results_dir, f'{country_name}_fao_v7_optimized_plot.png')
        plt.savefig(output_plot)
        print(f"Verification plot saved to: {output_plot}")

        output_csv = os.path.join(input_results_dir, f'{country_name}_fao_v7_optimized_table.csv')
        comparison.to_csv(output_csv, index=False)
        print(f"Verification table saved to: {output_csv}")
    
    plt.close(fig)

    return comparison, corr_ker, r2_ker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Dual-Season Model against FAO")
    parser.add_argument("--country", type=str, default="Ethiopia", help="Country name")
    parser.add_argument("--version", type=str, default="V6", help="Model version")
    
    args = parser.parse_args()
    verify_fao(args.country, version=args.version)
