import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def get_season_length(p, e):
    """Calculates season length handling year-crossing."""
    if pd.isna(p) or pd.isna(e):
        return np.nan
    if p < e:
        return e - p
    else:
        return (365 - p) + e

def analyze_all_calendars():
    calendar_path = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling\GADM\crop_calendar\maize_crop_calendar_extraction.csv"
    output_dir = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling\Model_physical\Results\Calendar_Analysis"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading Global Calendar Data...")
    df = pd.read_csv(calendar_path)
    
    summary_list = []
    all_lengths_data = []

    # Process each season
    for s_idx in [1, 2]:
        p_col = f'Maize_{s_idx}_planting'
        e_col = f'Maize_{s_idx}_endofseaso'
        h_col = f'Maize_{s_idx}_harvest'
        
        if p_col not in df.columns or e_col not in df.columns:
            continue
            
        print(f"Analyzing Season {s_idx}...")
        
        # Calculate lengths for the whole dataframe
        df[f'S{s_idx}_Length'] = df.apply(lambda x: get_season_length(x[p_col], x[e_col]), axis=1)
        
        # Aggregate by Country
        countries = df['ADMIN0'].unique()
        for country in countries:
            country_df = df[(df['ADMIN0'] == country) & df[f'S{s_idx}_Length'].notna()]
            if country_df.empty:
                continue
                
            lengths = country_df[f'S{s_idx}_Length']
            
            # Record for summary CSV
            summary_list.append({
                'Country': country,
                'Season': s_idx,
                'Count': len(lengths),
                'Mean_Length': np.mean(lengths),
                'Min_Length': np.min(lengths),
                'Max_Length': np.max(lengths),
                'Std_Length': np.std(lengths),
                'Outliers_Long (>270d)': (lengths > 270).sum(),
                'Outliers_Short (<60d)': (lengths < 60).sum()
            })
            
            # Collect for plotting
            temp_plot_df = pd.DataFrame({
                'Length': lengths,
                'Country': country,
                'Season': f'Season {s_idx}'
            })
            all_lengths_data.append(temp_plot_df)

    # 1. Export Summary CSV
    df_summary = pd.DataFrame(summary_list)
    summary_path = os.path.join(output_dir, "Maize_Calendar_Global_Summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Global summary saved to {summary_path}")

    # 2. Visualizations
    if not all_lengths_data:
        print("No data to plot.")
        return
        
    df_plot = pd.concat(all_lengths_data)
    
    # --- Plot A: Global Overview (Boxplot using Matplotlib) ---
    plt.figure(figsize=(16, 10))
    
    # Order countries by mean length of Season 1
    order_df = df_summary[df_summary['Season'] == 1].sort_values('Mean_Length', ascending=False)
    order = order_df['Country'].tolist()
    
    # Prepare data for boxplot
    plot_data = []
    labels = []
    for country in order:
        # S1
        s1_vals = df_plot[(df_plot['Country'] == country) & (df_plot['Season'] == 'Season 1')]['Length'].values
        if len(s1_vals) > 0:
            plot_data.append(s1_vals)
            labels.append(f"{country} (S1)")
        # S2
        s2_vals = df_plot[(df_plot['Country'] == country) & (df_plot['Season'] == 'Season 2')]['Length'].values
        if len(s2_vals) > 0:
            plot_data.append(s2_vals)
            labels.append(f"{country} (S2)")

    plt.boxplot(plot_data, labels=labels)
    plt.xticks(rotation=90, fontsize=8)
    plt.title("Distribution of Maize Season Lengths by Country and Season", fontsize=16)
    plt.ylabel("Season Length (Days)", fontsize=12)
    plt.axhline(180, color='gray', linestyle='--', alpha=0.5, label='180 Days')
    plt.axhline(270, color='red', linestyle='--', alpha=0.5, label='270 Days (High)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "Global_Season_Length_Distributions.png")
    plt.savefig(plot_path)
    print(f"Global plot saved to {plot_path}")

    # --- Plot B: "Weird Data" Summary (Bar Chart instead of Heatmap) ---
    outlier_df = df_summary.groupby('Country')[['Outliers_Long (>270d)', 'Outliers_Short (<60d)']].sum().sort_values('Outliers_Long (>270d)', ascending=False).head(20)
    
    if not outlier_df.empty:
        plt.figure(figsize=(12, 8))
        outlier_df.plot(kind='bar', stacked=True, color=['red', 'orange'], ax=plt.gca())
        plt.title("Top 20 Countries with Potential Outlier Season Lengths", fontsize=14)
        plt.ylabel("Number of Admin Regions")
        plt.tight_layout()
        plot_path_outliers = os.path.join(output_dir, "Potential_Outliers_Summary.png")
        plt.savefig(plot_path_outliers)
        print(f"Outlier plot saved to {plot_path_outliers}")

    # Print interesting finding
    print("\nTop 5 Countries with longest mean Season 1:")
    print(df_summary[df_summary['Season'] == 1].sort_values('Mean_Length', ascending=False).head(5))

    print("\nTop 5 Countries with most Season 1 outliers (>270 days):")
    print(df_summary[df_summary['Season'] == 1].sort_values('Outliers_Long (>270d)', ascending=False).head(5))

if __name__ == "__main__":
    analyze_all_calendars()
