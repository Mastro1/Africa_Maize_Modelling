import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
INPUT_PATH = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis", "Global_Dynamic_Calendar_Dates_V5.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis")

def analyze_dynamic_results():
    if not os.path.exists(INPUT_PATH):
        print(f"File not found: {INPUT_PATH}")
        return
        
    df = pd.read_csv(INPUT_PATH)
    
    # 1. Completion Rate by Country and Season
    df['Success'] = df['Dynamic_SOS'].notna().astype(int)
    stats = df.groupby(['Country', 'Season']).agg({
        'Success': ['count', 'sum', 'mean'],
        'Season_Length': ['mean', 'std']
    })
    
    # Flatten columns
    stats.columns = ['Total_Attempts', 'Successful_Extractions', 'Success_Rate', 'Avg_Season_Length', 'Std_Season_Length']
    stats = stats.reset_index()
    
    # 2. Planting Shift (Dynamic - Static)
    df['Dynamic_SOS_Date'] = pd.to_datetime(df['Dynamic_SOS'])
    
    def get_doy_relative(row):
        if pd.isna(row['Dynamic_SOS_Date']): return None
        ref_date = pd.Timestamp(year=row['Year'], month=1, day=1)
        return (row['Dynamic_SOS_Date'] - ref_date).days + 1
        
    df['Dynamic_SOS_Rel_DOY'] = df.apply(get_doy_relative, axis=1)
    df['Planting_Shift'] = df['Dynamic_SOS_Rel_DOY'] - df['Static_Planting']
    
    # Handle wrap-around
    df.loc[df['Planting_Shift'] > 180, 'Planting_Shift'] -= 365
    df.loc[df['Planting_Shift'] < -180, 'Planting_Shift'] += 365
    
    shift_stats = df.groupby(['Country', 'Season'])['Planting_Shift'].mean().reset_index()
    stats = pd.merge(stats, shift_stats, on=['Country', 'Season'])
    
    # Export Stats
    stats_path = os.path.join(OUTPUT_DIR, "Global_Dynamic_Summary_Stats_by_Season.csv")
    stats.to_csv(stats_path, index=False)
    print(f"Summary stats saved to {stats_path}")
    
    # 3. Visualizations
    plt.figure(figsize=(16, 15))
    countries = sorted(df['Country'].unique())
    x = np.arange(len(countries))
    width = 0.35
    
    # Success Rate Plot
    ax1 = plt.subplot(3, 1, 1)
    s1_success = [stats[(stats['Country'] == c) & (stats['Season'] == 1)]['Success_Rate'].values[0] * 100 if not stats[(stats['Country'] == c) & (stats['Season'] == 1)].empty else 0 for c in countries]
    s2_success = [stats[(stats['Country'] == c) & (stats['Season'] == 2)]['Success_Rate'].values[0] * 100 if not stats[(stats['Country'] == c) & (stats['Season'] == 2)].empty else 0 for c in countries]
    
    ax1.bar(x - width/2, s1_success, width, label='Season 1', color='skyblue')
    ax1.bar(x + width/2, s2_success, width, label='Season 2', color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(countries, rotation=90, fontsize=8)
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Success Rate by Country and Season")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Planting Shift Plot
    ax2 = plt.subplot(3, 1, 2)
    s1_shift = [stats[(stats['Country'] == c) & (stats['Season'] == 1)]['Planting_Shift'].values[0] if not stats[(stats['Country'] == c) & (stats['Season'] == 1)].empty else 0 for c in countries]
    s2_shift = [stats[(stats['Country'] == c) & (stats['Season'] == 2)]['Planting_Shift'].values[0] if not stats[(stats['Country'] == c) & (stats['Season'] == 2)].empty else 0 for c in countries]
    
    ax2.bar(x - width/2, s1_shift, width, label='Season 1', color='salmon')
    ax2.bar(x + width/2, s2_shift, width, label='Season 2', color='darkred')
    ax2.set_xticks(x)
    ax2.set_xticklabels(countries, rotation=90, fontsize=8)
    ax2.set_ylabel("Avg Shift (Days)")
    ax2.set_title("Average Planting Date Shift by Country and Season")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Season Length Boxplot
    ax3 = plt.subplot(3, 1, 3)
    # Prepare data for group boxplot
    boxplot_data = []
    positions = []
    labels = []
    colors = []
    
    for i, c in enumerate(countries):
        s1_data = df[(df['Country'] == c) & (df['Season'] == 1) & (df['Season_Length'].notna())]['Season_Length'].tolist()
        s2_data = df[(df['Country'] == c) & (df['Season'] == 2) & (df['Season_Length'].notna())]['Season_Length'].tolist()
        
        if s1_data:
            boxplot_data.append(s1_data)
            positions.append(i - 0.2)
            colors.append('lightgreen')
        if s2_data:
            boxplot_data.append(s2_data)
            positions.append(i + 0.2)
            colors.append('forestgreen')
        labels.append(c)

    bp = ax3.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=0.3)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    ax3.set_xticks(range(len(countries)))
    ax3.set_xticklabels(countries, rotation=90, fontsize=8)
    ax3.set_ylabel("Season Length (Days)")
    ax3.set_title("Season Length Distribution by Country and Season")
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Custom legend for boxplot
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='lightgreen', lw=4, label='Season 1'),
                       Line2D([0], [0], color='forestgreen', lw=4, label='Season 2')]
    ax3.legend(handles=legend_elements)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "Global_Dynamic_Analysis_Summary.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Summary plot saved to {plot_path}")

if __name__ == "__main__":
    analyze_dynamic_results()
