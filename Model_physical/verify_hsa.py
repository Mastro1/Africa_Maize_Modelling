import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import random
import json

# Using statsmodels for Kernel Regression as requested
try:
    from statsmodels.nonparametric.kernel_regression import KernelReg
except ImportError:
    print("Warning: statsmodels not found. Kernel detrending will be disabled.")
    KernelReg = None

def smooth_data_kernel_regression(arr: np.array, years: np.array) -> np.ndarray:
    """
    Smooth the data using kernel regression with a bandwith of 3.
    Borrowed from verify_fao.py
    """
    if KernelReg is None:
        return arr # Fallback
        
    # Perform kernel regression on the input array
    # bw=[3] is the bandwidth used in verify_fao
    kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[3])

    # Obtain the smoothed data
    smoothed_data = kr.fit()[0]

    return smoothed_data

def manual_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def get_parent_gadm_id_v1(pcode):
    """
    Tries to derive the Level 1 Parent ID from a Level 2 PCODE.
    Assumes format: "AGO.1.2_1" -> "AGO.1_1"
    """
    parts = pcode.split('.')
    if len(parts) >= 3:
        # e.g. AGO, 1, 2_1 or KEN, 1, 2_1
        # Reconstruct: parts[0] + "." + parts[1] + "_1"
        # Handle cases where parts[1] might already have _1 if it's mixed
        # But usually Level 2 is Country.Admin1.Admin2_Version
        
        # Check standard format "XXX.A.B_1"
        return f"{parts[0]}.{parts[1]}_1"
    return pcode

def get_main_seasons(country_name):
    """
    Retrieves the list of 'Main' seasons (Maize 1) for a country from cropCalendarMatching.json
    """
    json_path = r'HarvestStatAfrica/crop_calendar/cropCalendarMatching.json'
    if not os.path.exists(json_path):
        print(f"Warning: Calendar matching file not found at {json_path}")
        return None
        
    try:
        with open(json_path, 'r') as f:
            calendar_map = json.load(f)
            
        # Case insensitive lookup
        matched_country = next((c for c in calendar_map.keys() if c.lower() == country_name.lower()), None)
        
        if not matched_country:
            print(f"Warning: Country {country_name} not found in calendar matching file.")
            return None
            
        seasons = calendar_map[matched_country]
        # Filter for "Maize 1"
        main_seasons = [s_name for s_name, type_name in seasons.items() if type_name == "Maize 1"]
        
        if not main_seasons:
            print(f"Warning: No 'Maize 1' season found for {country_name} in calendar file.")
            return None
            
        print(f"Identified Main Seasons for {matched_country}: {main_seasons}")
        return main_seasons
        
    except Exception as e:
        print(f"Error reading calendar matching file: {e}")
        return None

def verify_hsa(country_name, input_results_dir=None, version='v3', output_dir=None):
    """
    Verifies model results against HarvestStat Africa (HSA) data for a specific country.
    Includes detrending and admin-level aggregation.
    """
    
    # Defaults
    if input_results_dir is None:
        input_results_dir = r'Model_physical/Results'
    
    if output_dir is None:
        output_dir = input_results_dir

    # 1. Load Model Results
    global_path = os.path.join(input_results_dir, f'Global_Maize_Yield_{version.upper()}.csv')
    country_path = os.path.join(input_results_dir, f'maize_yield_estimates_{version.upper()}_{country_name.replace(" ", "_")}.csv')
    old_path = os.path.join(input_results_dir, f'maize_yield_estimates_{version}.csv')

    if os.path.exists(global_path):
        print(f"Loading results from global file: {global_path}")
        df_model_all = pd.read_csv(global_path)
        df_model = df_model_all[df_model_all['Country'].str.strip().str.lower() == country_name.strip().lower()]
    elif os.path.exists(country_path):
        print(f"Loading results from country file: {country_path}")
        df_model = pd.read_csv(country_path)
    elif os.path.exists(old_path):
        print(f"Loading results from: {old_path}")
        df_model = pd.read_csv(old_path)
    else:
        print(f"Error: Model results not found in {input_results_dir}")
        return

    # Handle multiple seasons (V6) - for now just take Season 1
    if 'Season' in df_model.columns:
        print("Filtering for Season 1 for HSA comparison...")
        df_model = df_model[df_model['Season'] == 1]
    
    # Standardize Model Columns
    if 'District' in df_model.columns:
        df_model = df_model.rename(columns={'District': 'PCODE'})
    if 'Year' in df_model.columns:
        df_model = df_model.rename(columns={'Year': 'year'})
        
    # 2. Load Crop Area Data (For Aggregation)
    crop_area_path = r'GADM/crop_areas/africa_crop_areas_glad_filtered.csv'
    if not os.path.exists(crop_area_path):
        print(f"Warning: Crop area file not found at {crop_area_path}. Aggregation might be inaccurate.")
        df_area_country = None
    else:
        df_area = pd.read_csv(crop_area_path)
        df_area_country = df_area[df_area['country'] == country_name].copy()
        
    # Merge Model with Area
    if df_area_country is not None and not df_area_country.empty:
        # subset area
        df_area_sub = df_area_country[['PCODE', 'crop_area_ha']]
        df_model = pd.merge(df_model, df_area_sub, on='PCODE', how='inner')
    else:
        df_model['crop_area_ha'] = 1 # Fallback equal weights
    
    # Pre-calculate Level 1 Aggregations for the Model
    # 1. Create Parent ID column
    df_model['PCODE_L1'] = df_model['PCODE'].apply(get_parent_gadm_id_v1)
    
    # 2. Group by L1 + Year
    df_model['prod_proxy'] = df_model['Yield_Estimated_t_ha'] * df_model['crop_area_ha']
    
    # Safer aggregation: Sum proxy and area, then divide
    df_model_l1 = df_model.groupby(['PCODE_L1', 'year'])[['prod_proxy', 'crop_area_ha']].sum().reset_index()
    df_model_l1['Yield_Estimated_t_ha'] = df_model_l1['prod_proxy'] / df_model_l1['crop_area_ha']
    
    # Keep only relevant columns
    df_model_l1 = df_model_l1[['PCODE_L1', 'year', 'Yield_Estimated_t_ha']]
    
    
    # 3. Load HSA Data and Mapping
    hsa_data_path = r'HarvestStatAfrica/data/hvstat_africa_data_v1.0.csv'
    mapping_path = r'HarvestStatAfrica/GADM_matching/HSA_to_GADM_mapping.csv'
    
    if not os.path.exists(hsa_data_path) or not os.path.exists(mapping_path):
        print("Error: HSA data or mapping file not found.")
        return

    print("Loading HSA data and Mapping...")
    df_hsa = pd.read_csv(hsa_data_path)
    df_map = pd.read_csv(mapping_path)
    
    # Filter HSA for Country and Product
    hsa_countries = df_hsa['country'].unique()
    # Simple lookup
    matched_country = next((c for c in hsa_countries if c.lower() == country_name.lower()), None)
    
    if not matched_country:
        print(f"Error: Country '{country_name}' not found in HSA data.")
        return
        
    print(f"Filtering HSA data for {matched_country} and Maize...")
    df_hsa_country = df_hsa[
        (df_hsa['country'] == matched_country) & 
        (df_hsa['product'].isin(['Maize', 'Maize (corn)']))
    ].copy()
    
    if df_hsa_country.empty:
        print(f"Warning: No Maize data found in HSA for {matched_country}.")
        return

    # Filter for Main Season
    main_seasons = get_main_seasons(matched_country)
    if main_seasons:
        print(f"Filtering for seasons: {main_seasons}")
        df_hsa_country = df_hsa_country[df_hsa_country['season_name'].isin(main_seasons)]
        if df_hsa_country.empty:
            print("Warning: No data left after filtering for main season.")
            return
    else:
        print("Using all seasons (no match found or error).")

    # Map HSA data
    print("Mapping HSA data to GADM IDs...")
    df_hsa_mapped = pd.merge(df_hsa_country, df_map[['FNID', 'gadm_id', 'gadm_name', 'gadm_level']], 
                             left_on='fnid', right_on='FNID', how='inner')
    
    df_hsa_mapped = df_hsa_mapped.rename(columns={
        'harvest_year': 'year',
        'yield': 'HSA_Yield_t_ha'
    })
    
    # 4. Compare
    # Strategy: Iterate through mapped HSA records and find matching model data
    # either at L1 or L2 depending on gadm_level
    
    merged_rows = []
    
    # Split HSA by level
    hsa_l1 = df_hsa_mapped[df_hsa_mapped['gadm_level'] == 1.0]
    hsa_l2 = df_hsa_mapped[df_hsa_mapped['gadm_level'] == 2.0]
    
    print(f"HSA Data points: {len(df_hsa_mapped)} (L1: {len(hsa_l1)}, L2: {len(hsa_l2)})")
    
    # Match L2 directly
    if not hsa_l2.empty:
        # Ensure types using .loc to avoid SettingWithCopyWarning
        hsa_l2 = hsa_l2.copy()
        hsa_l2.loc[:, 'year'] = hsa_l2['year'].astype(int)
        df_model['year'] = df_model['year'].astype(int)
        
        merged_l2 = pd.merge(hsa_l2[['gadm_id', 'gadm_name', 'year', 'HSA_Yield_t_ha']],
                             df_model[['PCODE', 'year', 'Yield_Estimated_t_ha']],
                             left_on=['gadm_id', 'year'],
                             right_on=['PCODE', 'year'],
                             how='inner')
        merged_rows.append(merged_l2)
        
    # Match L1 with Aggregated Model
    if not hsa_l1.empty:
        hsa_l1 = hsa_l1.copy()
        hsa_l1.loc[:, 'year'] = hsa_l1['year'].astype(int)
        df_model_l1['year'] = df_model_l1['year'].astype(int)
        
        merged_l1 = pd.merge(hsa_l1[['gadm_id', 'gadm_name', 'year', 'HSA_Yield_t_ha']],
                             df_model_l1[['PCODE_L1', 'year', 'Yield_Estimated_t_ha']],
                             left_on=['gadm_id', 'year'],
                             right_on=['PCODE_L1', 'year'],
                             how='inner')
        # Rename PCODE_L1 to PCODE for consistency
        merged_l1 = merged_l1.rename(columns={'PCODE_L1': 'PCODE'})
        merged_rows.append(merged_l1)
        
    if not merged_rows:
        print("No matches found given the Admin Levels.")
        return
        
    df_final = pd.concat(merged_rows, ignore_index=True)
    print(f"Total Comparison Points: {len(df_final)}")
    
    if len(df_final) < 5:
        print("Not enough points for reliable statistics.")
        return
        
    # 5. Detrending and Stats
    # We apply detrending PER LOCATION (PCODE) to remove local biases/trends 
    # OR we apply it to the whole dataset if we are just looking at overall variability?
    # Usually, detrending is done per time-series.
    
    # However, verify_fao.py does "National" detrending.
    # Here we have distinct locations.
    # The user said "detrend the values of the timeseries". This implies per-series.
    # But doing kernel regression on short series (N=5-10) is unstable.
    
    # Let's try to calculate relative deviation from the mean/trend per location.
    
    # For now, let's attempt per-PCODE detrending if N is sufficient, else skip those.
    
    df_final['Model_Dev'] = np.nan
    df_final['HSA_Dev'] = np.nan
    
    pcodes = df_final['PCODE'].unique()
    valid_pcodes = []
    
    for pcode in pcodes:
        sub = df_final[df_final['PCODE'] == pcode].sort_values('year')
        if len(sub) < 5: # Need standard amount of points for kernel/trend
            continue
            
        years = sub['year'].values
        model_y = sub['Yield_Estimated_t_ha'].values
        hsa_y = sub['HSA_Yield_t_ha'].values
        
        # Kernel Trend
        if KernelReg is not None:
            try:
                model_trend = smooth_data_kernel_regression(model_y, years)
                hsa_trend = smooth_data_kernel_regression(hsa_y, years)
                
                # Deviations
                # Avoid division by zero
                model_dev = (model_y - model_trend) / np.where(model_trend == 0, 1, model_trend)
                hsa_dev = (hsa_y - hsa_trend) / np.where(hsa_trend == 0, 1, hsa_trend)
                
                df_final.loc[sub.index, 'Model_Dev'] = model_dev
                df_final.loc[sub.index, 'HSA_Dev'] = hsa_dev
                valid_pcodes.append(pcode)
            except Exception as e:
                print(f"Detrending failed for {pcode}: {e}")
        else:
             # Fallback to simple linear detrend or mean subtraction? 
             # Let's just use mean subtraction for "variability" if kernel fails
             model_dev = (model_y - np.mean(model_y)) / np.mean(model_y)
             hsa_dev = (hsa_y - np.mean(hsa_y)) / np.mean(hsa_y)
             df_final.loc[sub.index, 'Model_Dev'] = model_dev
             df_final.loc[sub.index, 'HSA_Dev'] = hsa_dev
             valid_pcodes.append(pcode)

    # 6. Global Stats on Validity
    df_valid = df_final.dropna(subset=['Model_Dev', 'HSA_Dev'])
    
    if df_valid.empty:
        print("No valid data after detrending.")
        return
        
    y_true_dev = df_valid['HSA_Dev']
    y_pred_dev = df_valid['Model_Dev']
    
    r2 = manual_r2(y_true_dev, y_pred_dev)
    corr = y_true_dev.corr(y_pred_dev)
    rmse = np.sqrt(((y_true_dev - y_pred_dev) ** 2).mean())
    
    print("\n--- Verification Statistics (Detrended Variability) ---")
    print(f"Country: {matched_country}")
    print(f"R² Score: {r2:.4f}")
    print(f"Correlation: {corr:.4f}")
    print(f"RMSE (Relative): {rmse:.4f}")
    print(f"N (Points): {len(df_valid)}")
    print("-------------------------------------------------------\n")
    
    # 7. Plots
    
    # Scatter of Deviations
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_dev, y_pred_dev, alpha=0.6, edgecolors='w')
    
    min_val = min(y_true_dev.min(), y_pred_dev.min())
    max_val = max(y_true_dev.max(), y_pred_dev.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    
    plt.xlabel(f'HSA Yield Deviation')
    plt.ylabel(f'Model Yield Deviation')
    plt.title(f'{matched_country}: Detrended Comparison (Main Season)\nR²={r2:.2f}, Corr={corr:.2f}, N={len(df_valid)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    scatter_path = os.path.join(output_dir, f'{country_name}_HSA_detrended_scatter.png')
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to: {scatter_path}")
    plt.close()
    
    # Time Series (showing Trends + Points) for a few locations
    n_plots = min(4, len(valid_pcodes))
    if n_plots > 0:
        selected_pcodes = random.sample(valid_pcodes, n_plots)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=False)
        if n_plots == 1: axes = [axes]
        
        for i, pcode in enumerate(selected_pcodes):
            subset = df_final[df_final['PCODE'] == pcode].sort_values('year')
            
            years = subset['year']
            loc_name = subset['gadm_name'].iloc[0]
            
            ax = axes[i]
            # Plot Detrended Deviations
            ax.plot(years, subset['HSA_Dev'], 'o-', color='orange', label='HSA Dev')
            ax.plot(years, subset['Model_Dev'], 'x--', color='blue', label='Model Dev')
            
            # Add zero line for reference
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f"Time Series (Detrended): {loc_name} ({pcode})")
            ax.set_ylabel("Relative Deviation")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        ts_path = os.path.join(output_dir, f'{country_name}_HSA_timeseries.png')
        plt.savefig(ts_path)
        print(f"Time series plots saved to: {ts_path}")
        plt.close()

    # Save Stats
    stats_file = os.path.join(output_dir, f'{country_name}_HSA_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Verification Statistics (HSA vs Model) - {country_name}\n")
        f.write(f"Metric: Detrended Relative Deviation\n")
        f.write(f"Season(s): {main_seasons}\n")
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"Correlation: {corr:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"N_Points: {len(df_valid)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Maize Model against HSA data")
    parser.add_argument("--country", type=str, default="Malawi", help="Country name to verify")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing model results")
    parser.add_argument("--version", type=str, default="v3", help="Version of the model results")
    
    args = parser.parse_args()
    
    verify_hsa(args.country, args.results_dir, args.version)
