import pandas as pd
import os
from pathlib import Path

def create_africa_maize_summary():
    """Create summary statistics for Africa maize yield predictions."""

    # Define paths
    data_dir = Path(__file__).parent
    input_file = data_dir / "all_africa_maize_yield_predictions.csv"
    output_file = data_dir / "africa_maize_summary.txt"

    print(f"Reading data from: {input_file}")

    # Read the CSV file
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df):,} records from {df['country'].nunique()} countries")

    # Initialize results list
    summary_data = []

    # Process each country
    for country in sorted(df['country'].unique()):
        country_df = df[df['country'] == country].copy()

        # Season 1 statistics
        season1_df = country_df[country_df['season_index'] == 1]
        season1_admin_units = season1_df['PCODE'].nunique() if not season1_df.empty else 0
        season1_yield_min = season1_df['final_yield'].min() if not season1_df.empty else None
        season1_yield_max = season1_df['final_yield'].max() if not season1_df.empty else None

        # Season 2 statistics
        season2_df = country_df[country_df['season_index'] == 2]
        season2_admin_units = season2_df['PCODE'].nunique() if not season2_df.empty else 0
        season2_yield_min = season2_df['final_yield'].min() if not season2_df.empty else None
        season2_yield_max = season2_df['final_yield'].max() if not season2_df.empty else None

        # Overall statistics
        total_admin_units = country_df['PCODE'].nunique()

        summary_data.append({
            'country': country,
            'season1_admin_units': season1_admin_units,
            'season1_yield_min': season1_yield_min,
            'season1_yield_max': season1_yield_max,
            'season2_admin_units': season2_admin_units,
            'season2_yield_min': season2_yield_min,
            'season2_yield_max': season2_yield_max
        })

    # Calculate totals
    total_season1_units = sum(item['season1_admin_units'] for item in summary_data)
    total_season2_units = sum(item['season2_admin_units'] for item in summary_data)

    # Create output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("AFRICA MAIZE YIELD PREDICTIONS - COUNTRY SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Double header
    output_lines.append(f"{'Country':<25} {'Maize Season 1':<30} {'Maize Season 2':<30}")
    output_lines.append(f"{'':<25} {'Units':<8} {'Yield Range':<21} {'Units':<8} {'Yield Range':<21}")
    output_lines.append("-" * 84)

    # Country data
    for item in summary_data:
        # Format yield ranges
        season1_range = f"{item['season1_yield_min']:.3f}-{item['season1_yield_max']:.3f}" if item['season1_yield_min'] is not None else "-"
        season2_range = f"{item['season2_yield_min']:.3f}-{item['season2_yield_max']:.3f}" if item['season2_yield_min'] is not None else "-"

        output_lines.append(f"{item['country']:<25} {item['season1_admin_units']:<8} {season1_range:<21} {item['season2_admin_units']:<8} {season2_range:<21}")

    # Totals
    output_lines.append("-" * 84)
    output_lines.append(f"{'TOTALS':<25} {total_season1_units:<8} {'-':<21} {total_season2_units:<8} {'-':<21}")
    output_lines.append("=" * 84)

    # Additional statistics
    output_lines.append("")
    output_lines.append("ADDITIONAL STATISTICS:")
    output_lines.append(f"Total countries: {len(summary_data)}")
    output_lines.append(f"Countries with both maize seasons: {sum(1 for item in summary_data if item['season1_admin_units'] > 0 and item['season2_admin_units'] > 0)}")
    output_lines.append(f"Countries with only maize season 1: {sum(1 for item in summary_data if item['season1_admin_units'] > 0 and item['season2_admin_units'] == 0)}")
    output_lines.append(f"Countries with only maize season 2: {sum(1 for item in summary_data if item['season1_admin_units'] == 0 and item['season2_admin_units'] > 0)}")

    # Overall yield statistics
    all_yields = df['final_yield'].dropna()
    output_lines.append("")
    output_lines.append("OVERALL YIELD STATISTICS:")
    output_lines.append(f"Global min yield: {all_yields.min():.3f} t/ha")
    output_lines.append(f"Global max yield: {all_yields.max():.3f} t/ha")
    output_lines.append(f"Global mean yield: {all_yields.mean():.3f} t/ha")
    output_lines.append(f"Global median yield: {all_yields.median():.3f} t/ha")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Summary saved to: {output_file}")
    print("\nSummary content:")
    print('\n'.join(output_lines))

    return output_file

if __name__ == "__main__":
    create_africa_maize_summary()