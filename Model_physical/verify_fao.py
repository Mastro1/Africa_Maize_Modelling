import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg

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

def verify_fao(country_name, input_results_dir=None, version='v1'):
    """
    Verifies model results against FAO data for a specific country.
    
    Args:
        country_name (str): Name of the country to verify (e.g., 'Kenya', 'Malawi').
        input_results_dir (str, optional): Directory containing model results. 
                                           Defaults to 'Model_physical/Input/Results'.
    """
    
    # Defaults
    if input_results_dir is None:
        # Based on user's recent file structure changes
        input_results_dir = r'Model_physical/Results'
    
    # 1. Load Model Results
    model_results_path = os.path.join(input_results_dir, f'maize_yield_estimates_{version}.csv')
    if not os.path.exists(model_results_path):
        print(f"Error: Model results not found at {model_results_path}")
        return

    print(f"Loading model results from {model_results_path}...")
    df_model = pd.read_csv(model_results_path)
    
    # 2. Load Crop Area Data (for weighting)
    crop_area_path = r'GADM/crop_areas/africa_crop_areas_glad_filtered.csv'
    if not os.path.exists(crop_area_path):
        print(f"Error: Crop area data not found at {crop_area_path}")
        return

    print("Loading crop area data...")
    df_area = pd.read_csv(crop_area_path)
    
    # Filter for the specific country in crop area data
    print(f"Filtering crop area data for: {country_name}")
    df_area_country = df_area[df_area['country'] == country_name].copy()
    
    if df_area_country.empty:
        print(f"Error: No crop area data found for country '{country_name}'. Check spelling.")
        print(f"Available countries: {df_area['country'].unique()[:10]}...")
        return
    
    # Merge Model Results with Crop Area
    # Ensure consistent column names for merge
    # PCODE usually matches between GADM based files
    if 'PCODE' not in df_model.columns:
         # Try to infer if PCODE has a different name
         if 'District' in df_model.columns:
             df_model = df_model.rename(columns={'District': 'PCODE'})
    
    if 'Year' in df_model.columns:
        df_model = df_model.rename(columns={'Year': 'year'})
        
    cols_needed = ['PCODE', 'year', 'crop_area_ha']
    # Check if crop_area_ha exists
    if 'crop_area_ha' not in df_area_country.columns:
        print("Error: 'crop_area_ha' column missing in crop area file.")
        return

    df_area_subset = df_area_country[cols_needed]
    
    print("Merging model results with crop area...")
    df_merged = pd.merge(df_model, df_area_subset, on=['PCODE', 'year'], how='inner')
    
    if df_merged.empty:
        print(f"Warning: Merged dataframe is empty for {country_name}.")
        print("Possible reasons: PCODE mismatch, or model has not been run for this country yet.")
        print(f"Model PCODEs sample: {df_model['PCODE'].unique()[:5]}")
        print(f"Area PCODEs sample: {df_area_subset['PCODE'].unique()[:5]}")
        return

    # 3. Aggregate to National Level (Weighted Mean)
    print("Aggregating to national level...")
    
    if 'Yield_Estimated_t_ha' not in df_merged.columns:
        print("Error: 'Yield_Estimated_t_ha' column missing in model results.")
        return

    # Weighted Mean Yield = Sum(Yield * Area) / Sum(Area)
    df_merged['production_proxy'] = df_merged['Yield_Estimated_t_ha'] * df_merged['crop_area_ha']
    
    national_stats = df_merged.groupby('year').apply(
        lambda x: pd.Series({
            'Model_Yield_t_ha': x['production_proxy'].sum() / x['crop_area_ha'].sum(),
            'Total_Area_ha': x['crop_area_ha'].sum(),
            'Sample_Size': len(x)
        })
    ).reset_index()
    
    print(f"Aggregated data for {len(national_stats)} years.")

    # 4. Load FAO Data
    fao_path = r'FAOSTAT/faostat_maize.csv'
    if not os.path.exists(fao_path):
        print(f"Error: FAO data not found at {fao_path}")
        return

    print("Loading FAO data...")
    df_fao = pd.read_csv(fao_path)
    
    # Filter for Country, Yield, Maize
    # FAO names might differ sightly (e.g. "Cote d'Ivoire")
    # We use simple matching for now
    
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
        return
    
    # Convert kg/ha or hg/ha to t/ha
    # Check Unit
    unit = df_fao_country['Unit'].iloc[0]
    print(f"FAO Unit found: {unit}")
    
    if unit == 'kg/ha':
        df_fao_country['FAO_Yield_t_ha'] = df_fao_country['Value'] / 1000.0
    elif unit == 'hg/ha':
        df_fao_country['FAO_Yield_t_ha'] = df_fao_country['Value'] / 10000.0
    elif unit == 't/ha':
        df_fao_country['FAO_Yield_t_ha'] = df_fao_country['Value']
    else:
        print(f"Warning: Unknown unit {unit}, assuming kg/ha")
        df_fao_country['FAO_Yield_t_ha'] = df_fao_country['Value'] / 1000.0

    df_fao_country = df_fao_country[['Year', 'FAO_Yield_t_ha']].rename(columns={'Year': 'year'})
    
    # 5. Merge and Compare
    print("Comparing data...")
    comparison = pd.merge(national_stats, df_fao_country, on='year', how='inner')
    
    # Exclude year 2000 as requested
    comparison = comparison[comparison['year'] != 2000]
    
    if comparison.empty:
        print("No overlapping years found between Model and FAO data.")
        return

    print("\nComparison (t/ha):")
    print(comparison[['year', 'Model_Yield_t_ha', 'FAO_Yield_t_ha']])
    
    # --- Detrending Analysis ---
    
    # 1. Linear Trend
    # Fit linear regression: y = mx + c
    
    # Model
    model_coeffs = np.polyfit(comparison['year'], comparison['Model_Yield_t_ha'], 1)
    model_linear_trend = np.polyval(model_coeffs, comparison['year'])
    comparison['Model_Yield_Linear_Trend'] = model_linear_trend
    comparison['Model_Dev_Linear'] = (comparison['Model_Yield_t_ha'] - model_linear_trend) / model_linear_trend
    
    # FAO
    fao_coeffs = np.polyfit(comparison['year'], comparison['FAO_Yield_t_ha'], 1)
    fao_linear_trend = np.polyval(fao_coeffs, comparison['year'])
    comparison['FAO_Yield_Linear_Trend'] = fao_linear_trend
    comparison['FAO_Dev_Linear'] = (comparison['FAO_Yield_t_ha'] - fao_linear_trend) / fao_linear_trend
    
    # 2. Kernel Trend
    # Model
    comparison['Model_Yield_Kernel_Trend'] = smooth_data_kernel_regression(comparison['Model_Yield_t_ha'].values, comparison['year'].values)
    comparison['Model_Dev_Kernel'] = (comparison['Model_Yield_t_ha'] - comparison['Model_Yield_Kernel_Trend']) / comparison['Model_Yield_Kernel_Trend']
    
    # FAO
    comparison['FAO_Yield_Kernel_Trend'] = smooth_data_kernel_regression(comparison['FAO_Yield_t_ha'].values, comparison['year'].values)
    comparison['FAO_Dev_Kernel'] = (comparison['FAO_Yield_t_ha'] - comparison['FAO_Yield_Kernel_Trend']) / comparison['FAO_Yield_Kernel_Trend']

    # --- Metrics ---
    
    # Original
    corr_orig = comparison['Model_Yield_t_ha'].corr(comparison['FAO_Yield_t_ha'])
    rmse_orig = ((comparison['Model_Yield_t_ha'] - comparison['FAO_Yield_t_ha']) ** 2).mean() ** 0.5
    
    # Linear Detrended
    corr_lin = comparison['Model_Dev_Linear'].corr(comparison['FAO_Dev_Linear'])
    rmse_lin = ((comparison['Model_Dev_Linear'] - comparison['FAO_Dev_Linear']) ** 2).mean() ** 0.5
    
    # Kernel Detrended
    corr_ker = comparison['Model_Dev_Kernel'].corr(comparison['FAO_Dev_Kernel'])
    rmse_ker = ((comparison['Model_Dev_Kernel'] - comparison['FAO_Dev_Kernel']) ** 2).mean() ** 0.5
    
    print(f"\nOriginal Correlation: {corr_orig:.4f}, RMSE: {rmse_orig:.4f}")
    print(f"Linear Detrended Correlation: {corr_lin:.4f}")
    print(f"Kernel Detrended Correlation: {corr_ker:.4f}")
    
    # --- Plotting ---
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # Subplot 1: Original Data with Trends
    ax1 = axes[0]
    ax1.plot(comparison['year'], comparison['Model_Yield_t_ha'], label='Model Data', marker='o', color='blue', alpha=0.6)
    ax1.plot(comparison['year'], comparison['Model_Yield_Linear_Trend'], label='Model Linear Trend', linestyle='--', color='blue')
    ax1.plot(comparison['year'], comparison['Model_Yield_Kernel_Trend'], label='Model Kernel Trend', linestyle=':', color='blue', linewidth=2)
    
    ax1.plot(comparison['year'], comparison['FAO_Yield_t_ha'], label='FAO Data', marker='x', color='orange', alpha=0.6)
    ax1.plot(comparison['year'], comparison['FAO_Yield_Linear_Trend'], label='FAO Linear Trend', linestyle='--', color='orange')
    ax1.plot(comparison['year'], comparison['FAO_Yield_Kernel_Trend'], label='FAO Kernel Trend', linestyle=':', color='orange', linewidth=2)
    
    ax1.set_title(f'{country_name}: Original Data and Trends\nRaw Corr: {corr_orig:.2f}')
    ax1.set_ylabel('Yield (t/ha)')
    ax1.legend(ncol=2)
    ax1.grid(True)
    
    # Subplot 2: Linear Detrended (Relative Deviation)
    ax2 = axes[1]
    ax2.plot(comparison['year'], comparison['Model_Dev_Linear'], label='Model Dev (Linear)', marker='o', color='blue')
    ax2.plot(comparison['year'], comparison['FAO_Dev_Linear'], label='FAO Dev (Linear)', marker='x', color='orange')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title(f'Linear Detrended (Relative Deviation)\nCorrelation: {corr_lin:.2f}')
    ax2.set_ylabel('Relative Deviation')
    ax2.legend()
    ax2.grid(True)
    
    # Subplot 3: Kernel Detrended (Relative Deviation)
    ax3 = axes[2]
    ax3.plot(comparison['year'], comparison['Model_Dev_Kernel'], label='Model Dev (Kernel)', marker='o', color='blue')
    ax3.plot(comparison['year'], comparison['FAO_Dev_Kernel'], label='FAO Dev (Kernel)', marker='x', color='orange')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_title(f'Kernel Detrended (Relative Deviation)\nCorrelation: {corr_ker:.2f}')
    ax3.set_ylabel('Relative Deviation')
    ax3.set_xlabel('Year')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save Plot
    output_plot = os.path.join(input_results_dir, f'{country_name}_fao_verification_plot.png')
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # Save comparison to CSV
    output_csv = os.path.join(input_results_dir, f'{country_name}_fao_verification_table.csv')
    comparison.to_csv(output_csv, index=False)
    print(f"Comparison table saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Maize Model against FAO data")
    parser.add_argument("--country", type=str, default="Kenya", help="Country name to verify (default: Kenya)")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing model results")
    parser.add_argument("--version", type=str, default="v1", help="Version of the model results")
    
    args = parser.parse_args()
    
    print(f"Starting verification for {args.country}...")
    verify_fao(args.country, args.results_dir, args.version)
