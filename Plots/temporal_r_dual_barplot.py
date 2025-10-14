import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def create_temporal_r_dual_barplot():
    """Create dual horizontal barplot for temporal r metrics."""

    # Load data
    metrics_dir = Path("Model/training_results/metrics")
    admin_metrics_path = metrics_dir / "per_admin_metrics.csv"
    temporal_agg_path = metrics_dir / "temporal_r_aggregated.csv"

    # Load the data
    admin_metrics = pd.read_csv(admin_metrics_path)
    temporal_agg = pd.read_csv(temporal_agg_path)

    # Clean country names (remove quotes if any)
    temporal_agg['country'] = temporal_agg['country'].str.strip('"')

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: National aggregated temporal r
    ax1 = axes[0]
    national_data = temporal_agg.dropna(subset=['temporal_r_national']).sort_values('temporal_r_national', ascending=True)
    y_pos = np.arange(len(national_data))

    # Use the specified blue color
    ax1.barh(y_pos, national_data['temporal_r_national'], color='#24245c', alpha=0.8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(national_data['country'])
    ax1.set_xlabel('Temporal Correlation (r)')
    ax1.set_title('National Aggregated Temporal Correlation', pad=20)

    # Add overall median line
    overall_median = national_data['temporal_r_national'].median()
    ax1.axvline(overall_median, color='green', linestyle=':', linewidth=1.5,
                label=f'Overall Median: {overall_median:.3f}')

    # Add value labels
    for i, v in enumerate(national_data['temporal_r_national'].values):
        if np.isfinite(v):
            ax1.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=10)

    ax1.legend(loc='lower right')
    ax1.set_xlim(-0.2, 1.0)
    ax1.grid(axis='x', alpha=0.3)

    # Right subplot: Distribution of temporal r across locations (violin plot)
    ax2 = axes[1]
    # Use the same country order as the left subplot
    countries = national_data['country'].tolist()

    # Prepare data for violin plot
    data_for_violin = []
    valid_countries = []

    for country in countries:
        country_data = admin_metrics[admin_metrics['country'] == country]['temporal_r'].dropna()
        if len(country_data) > 0:
            data_for_violin.append(country_data.values)
            valid_countries.append(country)
        else:
            # If no data for this country, add empty array to maintain position
            data_for_violin.append(np.array([]))
            valid_countries.append(country)

    if data_for_violin:
        # Create violin plot with original steelblue color
        parts = ax2.violinplot(data_for_violin, positions=range(len(valid_countries)),
                             vert=False, showmedians=True, widths=0.7)

        # Use original steelblue color like in the combined_analysis.py
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1)

        ax2.set_yticks(range(len(valid_countries)))
        ax2.set_yticklabels([])  # Remove y-axis labels since they match the left plot
        ax2.set_xlabel('Temporal Correlation (r)')
        ax2.set_title('Distribution of Temporal r Across Locations', pad=20)

        # Add median line
        all_temporal_r = admin_metrics['temporal_r'].dropna()
        overall_median = np.median(all_temporal_r)
        ax2.axvline(overall_median, color='green', linestyle=':', linewidth=1.5,
                    label=f'Overall Median: {overall_median:.3f}')

        ax2.grid(axis='x', alpha=0.3)
        ax2.legend(loc='lower right')

    # Adjust layout
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = Path("Plots/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the figure
    output_path = output_dir / "temporal_r_dual_barplot.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Dual temporal r barplot saved to: {output_path}")

if __name__ == "__main__":
    create_temporal_r_dual_barplot()
