import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import os
import scienceplots
from matplotlib.ticker import FuncFormatter
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.stats import pearsonr

plt.style.use(['science', 'ieee'])

PROJECT_ROOT = os.getcwd()

# ------ PARAMETERS ------
CLIMATE_INDICES = ['NAO', 'DMI', 'NINO34', 'TSA', 'EA']
MIN_YEARS_FOR_CORR = 8  # Minimum number of valid years to calculate correlation
# ------------------------

def smooth_data_kernel_regression(arr: np.array, years: np.array) -> np.ndarray:
    if len(arr) < 3: return arr
    try:
        kr = KernelReg(endog=arr, exog=years, var_type='c', reg_type="lc", bw=[3])
        return kr.fit()[0]
    except:
        return arr

def calculate_detrended_values(df):
    detrended_data = []
    for pcode in df['PCODE'].unique():
        loc_data = df[df['PCODE'] == pcode].sort_values('Year').copy()
        if len(loc_data) < 3: continue
        years = loc_data['Year'].values
        yields = loc_data['Yield_Estimated_t_ha'].values
        trend_values = smooth_data_kernel_regression(yields, years)
        detrended = (yields - trend_values) / trend_values
        loc_data['detrended_yield'] = detrended
        detrended_data.append(loc_data)
    return pd.concat(detrended_data, ignore_index=True)

def format_lon(x, pos):
    if x > 0: return f'{x:g}°E'
    if x < 0: return f'{abs(x):g}°W'
    return f'{x:g}°'

def format_lat(y, pos):
    if y > 0: return f'{y:g}°N'
    if y < 0: return f'{abs(y):g}°S'
    return f'{y:g}°'

def extract_climate_anomaly(climate_df, start_date, end_date):
    """
    Extracts the climate anomaly with the maximum absolute value within the 
    specified growing season Date Range to capture the strongest effect.
    """
    try:
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
    except Exception as e:
        return {ind: np.nan for ind in CLIMATE_INDICES}
        
    start_period = pd.Timestamp(sd.year, sd.month, 1)
    end_period = pd.Timestamp(ed.year, ed.month, 1)
    
    mask = (climate_df['date'] >= start_period) & (climate_df['date'] <= end_period)
    window_data = climate_df[mask]
    
    anomalies = {}
    for ind in CLIMATE_INDICES:
        values = window_data[ind].dropna()
        if len(values) > 0:
            max_val = values.max()
            min_val = values.min()
            # Absolute maximum retention with original sign
            if abs(max_val) > abs(min_val):
                anomalies[ind] = max_val
            else:
                anomalies[ind] = min_val
        else:
            anomalies[ind] = np.nan
            
    return anomalies

def main():
    print("Loading global maize yield data...")
    yield_path = os.path.join(PROJECT_ROOT, 'Model_physical', 'Results', 'V7_model', 'Global_Maize_Yield_V7.csv')
    df = pd.read_csv(yield_path)
    if 'Season' in df.columns:
        df = df[df['Season'] == 1].copy()
        
    print("Removing boundary years to avoid detrending artifacts...")
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df = df[~df['Year'].isin([min_year, max_year])].copy()
    
    print("Calculating kernel detrended values...")
    df = calculate_detrended_values(df)

    print("Loading and preparing climate data...")
    clim_path = os.path.join(PROJECT_ROOT, 'MacroClimate', 'climate_merged.csv')
    climate_df = pd.read_csv(clim_path)
    climate_df['date'] = pd.to_datetime(climate_df['date'])

    print("Aligning temporal data and calculating max absolute anomalies...")
    correlations = []
    
    # Process location by location
    for pcode, loc_data in df.groupby('PCODE'):
        climate_features = {ind: [] for ind in CLIMATE_INDICES}
        valid_yields = []
        
        for _, row in loc_data.iterrows():
            if pd.isna(row['detrended_yield']) or pd.isna(row['Start_Date']) or pd.isna(row['End_Date']):
                continue
                
            anomalies = extract_climate_anomaly(climate_df, row['Start_Date'], row['End_Date'])
            
            # Skip if missing any climate data in the target window
            if any(pd.isna(v) for v in anomalies.values()):
                continue
            
            valid_yields.append(row['detrended_yield'])
            for ind in CLIMATE_INDICES:
                climate_features[ind].append(anomalies[ind])
                
        # Calculate Pearson correlations
        if len(valid_yields) >= MIN_YEARS_FOR_CORR:
            corr_dict = {'PCODE': pcode}
            for ind in CLIMATE_INDICES:
                r, p = pearsonr(climate_features[ind], valid_yields)
                corr_dict[ind] = r
            correlations.append(corr_dict)
            
    corr_df = pd.DataFrame(correlations)
    
    print("Loading shapefiles...")
    admin2_path = os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_2_processed.shp')
    admin1_path = os.path.join(PROJECT_ROOT, 'GADM', 'gadm41_AFR_shp', 'gadm41_AFR_1_processed.shp')
    
    admin2_gdf = gpd.read_file(admin2_path)
    admin1_gdf = gpd.read_file(admin1_path)
    
    admin1_pcodes = set(admin1_gdf['FNID'])
    admin2_pcodes = set(admin2_gdf['FNID'])
    africa_outline = admin1_gdf.dissolve()

    data_admin1 = corr_df[corr_df['PCODE'].isin(admin1_pcodes)]
    data_admin2 = corr_df[corr_df['PCODE'].isin(admin2_pcodes)]

    print("Plotting maps...")
    print("Calculating climate index correlation matrix for analysis years...")
    analysis_years = df['Year'].unique()
    climate_analysis_subset = climate_df[climate_df['date'].dt.year.isin(analysis_years)]
    climate_corr = climate_analysis_subset[CLIMATE_INDICES].corr()

    print("Plotting maps and correlation matrix...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 10)) # Slightly taller for bottom legend
    axes_flat = axes.flatten()
    
    cmap = plt.cm.get_cmap('RdBu_r') # Blue=Negative Correlation, Red=Positive
    vmin, vmax = -0.6, 0.6 
    
    for idx, indicator in enumerate(CLIMATE_INDICES):
        ax = axes_flat[idx]
        
        africa_outline.plot(ax=ax, color='#f0f0f0', edgecolor='none')
        admin2_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.05)
        
        if not data_admin2.empty:
            merged2 = admin2_gdf.merge(data_admin2, left_on='FNID', right_on='PCODE')
            merged2.plot(ax=ax, column=indicator, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
            
        if not data_admin1.empty:
            merged1 = admin1_gdf.merge(data_admin1, left_on='FNID', right_on='PCODE')
            merged1.plot(ax=ax, column=indicator, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='none')
            
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')

        ax.set_xlim(-20, 55)
        ax.set_ylim(-35, 40)
        
        lon_ticks = np.arange(-20, 61, 20)
        lat_ticks = np.arange(-30, 41, 15)
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
        
        ax.tick_params(axis='both', which='major', labelsize=8, direction='in', 
                       length=3, width=0.6, colors='black',
                       top=True, right=True, bottom=True, left=True,
                       labeltop=False, labelleft=True, 
                       labelbottom=True, labelright=False)
        
        ax.grid(True, linestyle='--', linewidth=0.3, color='#cccccc', alpha=0.5, zorder=0)
        ax.set_title(f'{indicator}', fontsize=12, fontweight='bold', pad=8) # Keep only index name

    # Convert the 6th subplot into the Climate Correlation Matrix
    ax_mat = axes_flat[5]
    
    # Mask upper triangle (user feedback: values only below diagonal)
    mask = np.triu(np.ones_like(climate_corr, dtype=bool))
    climate_corr_masked = climate_corr.mask(mask)
    
    im_mat = ax_mat.imshow(climate_corr_masked, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Add text annotations for the matrix (only below diagonal)
    for i in range(len(CLIMATE_INDICES)):
        for j in range(len(CLIMATE_INDICES)):
            if i > j: # Only below diagonal
                val = climate_corr.iloc[i, j]
                text_color = "white" if abs(val) > 0.5 else "black"
                ax_mat.text(j, i, f"{val:.2f}", ha="center", va="center", 
                            color=text_color, fontsize=10, fontweight='bold')
    
    ax_mat.set_xticks(np.arange(len(CLIMATE_INDICES)))
    ax_mat.set_yticks(np.arange(len(CLIMATE_INDICES)))
    ax_mat.set_xticklabels(CLIMATE_INDICES, fontsize=10, fontweight='bold')
    ax_mat.set_yticklabels(CLIMATE_INDICES, fontsize=10, fontweight='bold')
    ax_mat.set_title('Index Correlation Matrix', fontsize=12, fontweight='bold', pad=12) # Re-added matrix title
    
    # Clean up matrix axes
    ax_mat.tick_params(axis='both', which='both', length=0)
    # Hide top and right spines for a lower-triangle feel
    ax_mat.spines['top'].set_visible(False)
    ax_mat.spines['right'].set_visible(False)
    ax_mat.spines['bottom'].set_linewidth(0.8)
    ax_mat.spines['left'].set_linewidth(0.8)

    # Place the global colorbar at the bottom of the figure
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Pearson Correlation ($r$)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Adjust layout to minimize spacing
    plt.tight_layout(rect=[0, 0.08, 1, 1.0])
    fig.subplots_adjust(hspace=0.05, wspace=0.15) # Increased horizontal spacing
    
    out_dir = os.path.join(PROJECT_ROOT, 'Model_physical', 'plots', 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    save_path_pdf = os.path.join(out_dir, 'africa_climate_correlation.pdf')
    save_path_png = os.path.join(out_dir, 'africa_climate_correlation.png')
    
    # Must wait for tight_layout before saving
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Figures saved to {out_dir}")

if __name__ == "__main__":
    main()
