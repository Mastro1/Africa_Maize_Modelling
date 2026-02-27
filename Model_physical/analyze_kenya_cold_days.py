import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
INPUT_PATH = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis", "Global_Dynamic_Calendar_Dates_V5.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis")

def analyze_kenya_cold_days():
    if not os.path.exists(INPUT_PATH):
        print(f"File not found: {INPUT_PATH}")
        return
        
    df = pd.read_csv(INPUT_PATH)
    
    # Filter for Kenya and valid Season Length
    df_kenya = df[(df['Country'] == 'Kenya') & (df['Season_Length'].notna())].copy()
    
    if df_kenya.empty:
        print("No data found for Kenya.")
        return

    plt.figure(figsize=(10, 7))
    
    seasons = [1, 2]
    colors = ['skyblue', 'steelblue']
    markers = ['o', 's']
    
    for s_idx, color, marker in zip(seasons, colors, markers):
        subset = df_kenya[df_kenya['Season'] == s_idx]
        if subset.empty: continue
        
        x = subset['Season_Length']
        y = subset['Cold_Days']
        
        # Scatter
        plt.scatter(x, y, alpha=0.5, color=color, label=f'Season {s_idx}', marker=marker)
        
        # Regression
        if len(subset) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            line = slope * x + intercept
            plt.plot(x, line, color=color, linestyle='--', linewidth=2, 
                     label=f'S{s_idx} Trend (R²={r_value**2:.2f})')

    plt.xlabel("Season Length (Days between SOS and EOS)")
    plt.ylabel("Number of Cold Days (< 10°C)")
    plt.title("Kenya: Relationship between Season Length and Cold Days")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = os.path.join(OUTPUT_DIR, "Kenya_ColdDays_vs_SeasonLength.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Plot saved to: {plot_path}")
    
    # Print Correlation Summary
    print("\n--- Correlation Summary (Kenya) ---")
    for s_idx in seasons:
        sub = df_kenya[df_kenya['Season'] == s_idx]
        if not sub.empty:
            corr = sub['Season_Length'].corr(sub['Cold_Days'])
            print(f"Season {s_idx}: Correlation (r) = {corr:.3f}")

if __name__ == "__main__":
    analyze_kenya_cold_days()
