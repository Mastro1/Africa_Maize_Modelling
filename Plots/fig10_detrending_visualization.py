import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys
import os

# Set font family to match the plotly version
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def load_admin0_data(data_dir, country_name):
    """Load national yield data for a specific country."""
    admin0_path = data_dir / "all_admin0_merged_data.csv"
    admin0_df = pd.read_csv(admin0_path)
    admin0_df['national_yield'] = admin0_df['production'] / admin0_df['area']
    country_data = admin0_df[admin0_df['country'] == country_name].copy()
    return country_data[['country', 'year', 'national_yield']].dropna()

def load_admin_gt0_data(data_dir, country_name):
    """Load admin level yield data for a specific country (individual PCODEs)."""
    admin_gt0_path = data_dir / "all_admin_gt0_merged_data.csv"
    admin_gt0_df = pd.read_csv(admin_gt0_path)
    
    # Filter for the specific country
    country_data = admin_gt0_df[admin_gt0_df['country'] == country_name].copy()
    
    # Calculate yield for each PCODE
    country_data['yield'] = country_data['production'] / country_data['area']
    
    return country_data[['country', 'year', 'PCODE', 'yield']].dropna()

def compute_national_trend_single_country(admin0_df: pd.DataFrame) -> pd.DataFrame:
    """Compute linear trend for a single country's national yield data."""
    trend_rows = []
    for country, g in admin0_df.groupby('country'):
        g_clean = g.dropna(subset=['year', 'national_yield'])
        if g_clean.empty:
            continue
        X = g_clean['year'].values.reshape(-1, 1)
        y = g_clean['national_yield'].values
        model = LinearRegression()
        try:
            model.fit(X, y)
            yhat = model.predict(X)
        except Exception:
            yhat = np.full_like(y, fill_value=np.nanmean(y), dtype=float)
        out = pd.DataFrame({
            'country': g_clean['country'].values,
            'year': g_clean['year'].values,
            'national_trend': yhat,
            'national_yield': g_clean['national_yield'].values
        })
        trend_rows.append(out)
    if not trend_rows:
        return pd.DataFrame(columns=['country', 'year', 'national_trend', 'national_yield'])
    trend_df = pd.concat(trend_rows, ignore_index=True)
    return trend_df

def create_detrending_visualization(country_name='Kenya'):
    """Create a matplotlib visualization showing FAO data and trend line for detrending."""
    # Load data
    PROJECT_ROOT = Path(os.getcwd())
    data_dir = PROJECT_ROOT / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat"
    
    # Load national yield data (FAO)
    admin0_df = load_admin0_data(data_dir, country_name)
    
    # Load admin level data (individual PCODEs)
    admin_gt0_data = load_admin_gt0_data(data_dir, country_name)
    
    # Compute trend
    trend_df = compute_national_trend_single_country(admin0_df)
    
    if trend_df.empty:
        print(f"No data available for {country_name}")
        return None
    
    # Sort data by year to ensure proper line connections
    trend_df = trend_df.sort_values('year')
    
    # Merge ground data with trend data for detrending calculation
    # Rename columns to avoid conflicts
    trend_df_renamed = trend_df.rename(columns={'national_yield': 'fao_yield'})
    
    # Get unique PCODEs
    pcodes = admin_gt0_data['PCODE'].unique()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Detrending Approach for {country_name}', fontfamily='Times New Roman', fontsize=14, x=0.5)
    
    # Left subplot - National yield data (simplified)
    # Add actual yield data (FAO) as lines connecting the points
    ax1.plot(trend_df_renamed['year'], trend_df_renamed['fao_yield'], 
             color='#24245c', linewidth=2, marker='o', markersize=8, label='FAO National Yield', 
             markerfacecolor='#24245c', markeredgecolor='#24245c', markeredgewidth=1)
    
    # Add trend line (using the specified color)
    ax1.plot(trend_df_renamed['year'], trend_df_renamed['national_trend'], 
             color='#fccd32', linewidth=3, label='Linear Trend')
    
    ax1.set_title(f'National Yield Data for {country_name}', fontfamily='Times New Roman', fontsize=12)
    ax1.set_xlabel('Year', fontfamily='Times New Roman', fontsize=10)
    ax1.set_ylabel('Yield (tonnes/hectare)', fontfamily='Times New Roman', fontsize=10)
    ax1.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Collect all detrended data for the right subplot
    all_detrended_data = []
    all_fao_detrended_data = []
    
    # Calculate FAO detrended values
    trend_df_for_fao = trend_df_renamed[['year', 'fao_yield', 'national_trend']].copy()
    trend_df_for_fao['detrended_yield'] = (trend_df_for_fao['fao_yield'] - trend_df_for_fao['national_trend']) / trend_df_for_fao['national_trend']
    all_fao_detrended_data.append(trend_df_for_fao)
    
    # Calculate detrended values for individual PCODEs
    for pcode in pcodes:
        pcode_data = admin_gt0_data[admin_gt0_data['PCODE'] == pcode].copy()
        if len(pcode_data) > 1:  # Only show PCODEs with multiple years of data
            pcode_data = pcode_data.sort_values('year')
            
            # Calculate detrended values for this PCODE
            pcode_with_trend = pd.merge(pcode_data, trend_df_renamed[['year', 'national_trend']], on='year', how='inner')
            pcode_with_trend['detrended_yield'] = (pcode_with_trend['yield'] - pcode_with_trend['national_trend']) / pcode_with_trend['national_trend']
            pcode_with_trend['PCODE'] = pcode
            
            all_detrended_data.append(pcode_with_trend)
    
    # Combine all detrended data
    if all_detrended_data:
        combined_detrended = pd.concat(all_detrended_data, ignore_index=True)
        
        # Right subplot - Detrended yield data for individual PCODEs
        # Plot each PCODE with a different color (but consistent across years for the same PCODE)
        unique_pcodes = combined_detrended['PCODE'].unique()
        
        # Generate a distinct color for each PCODE
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab20', len(unique_pcodes))
        
        for i, pcode in enumerate(unique_pcodes):
            pcode_detrended = combined_detrended[combined_detrended['PCODE'] == pcode]
            ax2.scatter(pcode_detrended['year'], pcode_detrended['detrended_yield'], 
                       s=36, alpha=0.7, color=colors(i), marker='o', zorder=3, label='_nolegend_')  # Don't show in legend
    
    # Add FAO detrended data (thicker line)
    if all_fao_detrended_data:
        fao_detrended = pd.concat(all_fao_detrended_data, ignore_index=True)
        ax2.plot(fao_detrended['year'], fao_detrended['detrended_yield'], 
                 color='#24245c', linewidth=2, marker='o', markersize=8, 
                 label='FAO Detrended', markerfacecolor='#24245c', markeredgecolor='#24245c', markeredgewidth=1)
    
    # Add a grey point to represent individual locations in the legend (like in the original)
    ax2.scatter([], [], color='lightgray', s=36, alpha=0.7, label='Locations', zorder=3)
    
    # Add trend line at zero for the detrended data (always zero by definition)
    if not trend_df_renamed.empty:
        ax2.axhline(y=0, color='#fccd32', linewidth=3, label='Zero Trend')
    
    ax2.set_title(f'Detrended Yield Data for {country_name}', fontfamily='Times New Roman', fontsize=12)
    ax2.set_xlabel('Year', fontfamily='Times New Roman', fontsize=10)
    ax2.set_ylabel('Detrended Yield', fontfamily='Times New Roman', fontsize=10)
    
    # Add legend with custom positioning
    ax2.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.25)
    
    return fig

if __name__ == "__main__":
    # Get country name from command line argument, default to Kenya
    if len(sys.argv) > 1:
        country_name = sys.argv[1]
    else:
        country_name = 'Zimbabwe'
    
    # Check if data files exist, if not, create sample data for testing
    PROJECT_ROOT = Path(os.getcwd())
    data_dir = PROJECT_ROOT / "RemoteSensing" / "HarvestStatAfrica" / "preprocessed_concat"
    admin0_path = data_dir / "all_admin0_merged_data.csv"
    admin_gt0_path = data_dir / "all_admin_gt0_merged_data.csv"
    
    if os.path.exists(admin0_path) and os.path.exists(admin_gt0_path):
        print(f"Creating detrending visualization for {country_name} using real data...")
        fig = create_detrending_visualization(country_name)
    else:
        print(f"Creating detrending visualization for {country_name} using sample data...")
        # Create sample data for testing
        import numpy as np
        np.random.seed(42)  # For reproducible results
        
        # Create sample data to simulate the structure expected by the functions
        n_years = 20
        years = np.arange(2000, 2000 + n_years)
        
        # Sample admin0 data (national level)
        national_yield = 2.0 + 0.05 * (years - 2000) + np.random.normal(0, 0.2, n_years)  # Slight positive trend
        admin0_df = pd.DataFrame({
            'country': [country_name] * n_years,
            'year': years,
            'production': national_yield * 100000,  # Simulated production
            'area': [100000] * n_years  # Simulated area
        })
        
        # Sample admin_gt0 data (subnational level with PCODEs)
        n_pcodes = 5
        pcode_data_list = []
        for i in range(n_pcodes):
            pcode_years = np.random.choice(years, size=np.random.randint(5, n_years), replace=False)
            for year in pcode_years:
                # Each PCODE has slightly different yield characteristics
                yield_val = national_yield[year-2000] * (0.8 + np.random.random() * 0.4)  # 80-120% of national
                pcode_data_list.append({
                    'country': country_name,
                    'year': year,
                    'PCODE': f'PCODE_{i+1}',
                    'production': yield_val * 10000,  # Simulated production for this region
                    'area': [10000]  # Simulated area for this region
                })
        
        admin_gt0_df = pd.DataFrame(pcode_data_list)
        
        # Create a custom function that uses the sample data
        def create_detrending_visualization_with_sample(country_name='Kenya'):
            """Same function but using sample data instead of loading from files."""
            
            # Get national data for the specified country
            country_admin0 = admin0_df[admin0_df['country'] == country_name].copy()
            country_admin0['national_yield'] = country_admin0['production'] / country_admin0['area']
            
            # Get admin level data for the specified country
            country_admin_gt0 = admin_gt0_df[admin_gt0_df['country'] == country_name].copy()
            country_admin_gt0['yield'] = country_admin_gt0['production'] / country_admin_gt0['area']
            
            # Compute trend
            trend_df = compute_national_trend_single_country(country_admin0)
            
            if trend_df.empty:
                print(f"No data available for {country_name}")
                return None
            
            # Sort data by year to ensure proper line connections
            trend_df = trend_df.sort_values('year')
            
            # Merge ground data with trend data for detrending calculation
            # Rename columns to avoid conflicts
            trend_df_renamed = trend_df.rename(columns={'national_yield': 'fao_yield'})
            
            # Get unique PCODEs
            pcodes = country_admin_gt0['PCODE'].unique()
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(f'Detrending Approach for {country_name}', fontfamily='Times New Roman', fontsize=14, x=0.5)
            
            # Left subplot - National yield data (simplified)
            # Add actual yield data (FAO) as lines connecting the points
            ax1.plot(trend_df_renamed['year'], trend_df_renamed['fao_yield'], 
                     color='#24245c', linewidth=2, marker='o', markersize=8, label='FAO National Yield', 
                     markerfacecolor='#24245c', markeredgecolor='#24245c', markeredgewidth=1)
            
            # Add trend line (using the specified color)
            ax1.plot(trend_df_renamed['year'], trend_df_renamed['national_trend'], 
                     color='#fccd32', linewidth=3, label='Linear Trend')
            
            ax1.set_title(f'National Yield Data for {country_name}', fontfamily='Times New Roman', fontsize=12)
            ax1.set_xlabel('Year', fontfamily='Times New Roman', fontsize=10)
            ax1.set_ylabel('Yield (tonnes/hectare)', fontfamily='Times New Roman', fontsize=10)
            ax1.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # Collect all detrended data for the right subplot
            all_detrended_data = []
            all_fao_detrended_data = []
            
            # Calculate FAO detrended values
            trend_df_for_fao = trend_df_renamed[['year', 'fao_yield', 'national_trend']].copy()
            trend_df_for_fao['detrended_yield'] = (trend_df_for_fao['fao_yield'] - trend_df_for_fao['national_trend']) / trend_df_for_fao['national_trend']
            all_fao_detrended_data.append(trend_df_for_fao)
            
            # Calculate detrended values for individual PCODEs
            for pcode in pcodes:
                pcode_data = country_admin_gt0[country_admin_gt0['PCODE'] == pcode].copy()
                if len(pcode_data) > 1:  # Only show PCODEs with multiple years of data
                    pcode_data = pcode_data.sort_values('year')
                    
                    # Calculate detrended values for this PCODE
                    pcode_with_trend = pd.merge(pcode_data, trend_df_renamed[['year', 'national_trend']], on='year', how='inner')
                    if not pcode_with_trend.empty:
                        pcode_with_trend['detrended_yield'] = (pcode_with_trend['yield'] - pcode_with_trend['national_trend']) / pcode_with_trend['national_trend']
                        pcode_with_trend['PCODE'] = pcode
                        
                        all_detrended_data.append(pcode_with_trend)
            
            # Combine all detrended data
            if all_detrended_data:
                combined_detrended = pd.concat(all_detrended_data, ignore_index=True)
                
                # Right subplot - Detrended yield data for individual PCODEs
                # Plot each PCODE with a different color (but consistent across years for the same PCODE)
                unique_pcodes = combined_detrended['PCODE'].unique()
                
                # Generate a distinct color for each PCODE
                import matplotlib.cm as cm
                colors = cm.get_cmap('tab20', len(unique_pcodes))
                
                for i, pcode in enumerate(unique_pcodes):
                    pcode_detrended = combined_detrended[combined_detrended['PCODE'] == pcode]
                    ax2.scatter(pcode_detrended['year'], pcode_detrended['detrended_yield'], 
                               s=36, alpha=0.7, color=colors(i), marker='o', zorder=3, label='_nolegend_')  # Don't show in legend
            
            # Add FAO detrended data (thicker line)
            if all_fao_detrended_data:
                fao_detrended = pd.concat(all_fao_detrended_data, ignore_index=True)
                ax2.plot(fao_detrended['year'], fao_detrended['detrended_yield'], 
                         color='#24245c', linewidth=2, marker='o', markersize=8, 
                         label='FAO Detrended', markerfacecolor='#24245c', markeredgecolor='#24245c', markeredgewidth=1)
            
            # Add a grey point to represent individual locations in the legend (like in the original)
            ax2.scatter([], [], color='lightgray', s=36, alpha=0.7, label='Locations', zorder=3)
            
            # Add trend line at zero for the detrended data (always zero by definition)
            if not trend_df_renamed.empty:
                ax2.axhline(y=0, color='#fccd32', linewidth=3, label='Zero Trend')
            
            ax2.set_title(f'Detrended Yield Data for {country_name}', fontfamily='Times New Roman', fontsize=12)
            ax2.set_xlabel('Year', fontfamily='Times New Roman', fontsize=10)
            ax2.set_ylabel('Detrended Yield', fontfamily='Times New Roman', fontsize=10)
            
            # Add legend with custom positioning
            ax2.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)
            
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # Adjust layout to prevent overlap
            plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.25)
            
            return fig
        
        fig = create_detrending_visualization_with_sample(country_name)
    
    if fig:
        output_dir = Path(__file__).parent / "output"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"detrending_visualization_{country_name.lower().replace(' ', '_')}.png"
        output_path = output_dir / output_filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as {output_path}")
        
        # Show the plot
        plt.show()
    else:
        print(f"Failed to create visualization for {country_name}")