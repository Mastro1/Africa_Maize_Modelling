import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def analyze_calendar(country="Kenya"):
    calendar_path = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling\GADM\crop_calendar\maize_crop_calendar_extraction.csv"
    output_dir = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling\Model_physical\Results"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading Calendar for {country}...")
    df = pd.read_csv(calendar_path)
    
    # Filter by Country
    if 'ADMIN0' in df.columns:
        df = df[df['ADMIN0'] == country]
    else:
        print("Warning: ADMIN0 column not found. Analyzing all data.")
        
    if df.empty:
        print(f"No data found for country: {country}")
        return

    # Filter for Maize_1 relevant columns
    cols = ['Maize_1_planting', 'Maize_1_endofseaso', 'Maize_1_harvest']
    df = df.dropna(subset=cols)
    
    print(f"Loaded {len(df)} rows with valid dates.")
    
    planting = df['Maize_1_planting']
    end_season = df['Maize_1_endofseaso']
    harvest = df['Maize_1_harvest'] # Just for reference
    
    # Calculate Season Lengths
    lengths = []
    cross_year_count = 0
    
    for p, e in zip(planting, end_season):
        if p < e:
            length = e - p
        else:
            # Crosses year
            length = (365 - p) + e
            cross_year_count += 1
        lengths.append(length)
        
    df['Season_Length'] = lengths
    
    print(f"Cross-year seasons: {cross_year_count} / {len(df)}")
    print(f"Mean Length: {np.mean(lengths):.2f}")
    print(f"Max Length: {np.max(lengths):.2f}")
    print(f"Min Length: {np.min(lengths):.2f}")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Planting Date Distribution
    axes[0,0].hist(planting, bins=36, color='green', alpha=0.7)
    axes[0,0].set_title("Distribution of Planting DOY")
    axes[0,0].set_xlabel("Day of Year")
    axes[0,0].set_ylabel("Count")
    
    # 2. End of Season Date Distribution
    axes[0,1].hist(end_season, bins=36, color='red', alpha=0.7)
    axes[0,1].set_title("Distribution of End of Season DOY")
    axes[0,1].set_xlabel("Day of Year")
    
    # 3. Season Length Distribution
    axes[1,0].hist(lengths, bins=30, color='blue', alpha=0.7)
    axes[1,0].set_title("Distribution of Season Length (Days)")
    axes[1,0].set_xlabel("Days")
    axes[1,0].axvline(180, color='k', linestyle='--', label='180 days')
    axes[1,0].axvline(250, color='r', linestyle='--', label='250 days')
    axes[1,0].legend()
    
    # 4. Window Length (Season + 60 days buffer)
    window_lengths = [l + 60 for l in lengths]
    axes[1,1].hist(window_lengths, bins=30, color='orange', alpha=0.7)
    axes[1,1].set_title("Distribution of Search Window Size (+/- 30 days)")
    axes[1,1].set_xlabel("Days")
    axes[1,1].axvline(365, color='r', linestyle='--', label='365 days')
    axes[1,1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{country}_calendar_distribution_analysis.png")
    plt.savefig(plot_path)
    print(f"Analysis plot saved to {plot_path}")
    
    # Print some examples of very long seasons
    long_seasons = df[df['Season_Length'] > 250]
    if not long_seasons.empty:
        print("\nExamples of very long seasons (>250 days):")
        print(long_seasons[['ADMIN0', 'ADMIN1', 'Maize_1_planting', 'Maize_1_endofseaso', 'Season_Length']].head(10))

if __name__ == "__main__":
    analyze_calendar("Kenya")
