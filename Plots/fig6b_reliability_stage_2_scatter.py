import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import stats
import os

# Set font family to match the plotly version
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

PROJECT_ROOT = os.getcwd()

def plot_comparison(ethiopia_df, burkina_faso_df):
    """
    Create a comparison plot between Ethiopia and Burkina Faso showing NDVI integral, 
    plant harvest, and GDD for both countries in a 2x2 subplot layout using Matplotlib.
    
    Parameters:
    - ethiopia_df: DataFrame with Ethiopia data
    - burkina_faso_df: DataFrame with Burkina Faso data
    """
    # Filter data to remove rows with missing values
    eth_df = ethiopia_df.dropna(subset=['ndvi_integral_plant_harv', 'gdd_plant_harv', 'yield']).copy()
    bfa_df = burkina_faso_df.dropna(subset=['ndvi_integral_plant_harv', 'gdd_plant_harv', 'yield']).copy()
    
    # Create figure and subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    fig.suptitle('Comparison of Ethiopia and Burkina Faso', fontsize=14, fontfamily='Times New Roman', x=0.5, fontweight='bold')
    
    # Helper function to add trend line and calculate R²
    def add_trend_line(ax, x_data, y_data, color='red'):
        # Remove any NaN values
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) > 1:
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            # Create trend line
            x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_trend = slope * x_trend + intercept
            
            # Add trend line
            ax.plot(x_trend, y_trend, color=color, linewidth=2, linestyle='--')
            
            # Add R² annotation
            ax.text(x_clean.max() * 0.95, y_clean.max() * 0.95, 
                   f"R² = {r_value**2:.3f}\np = {p_value:.3f}", 
                   fontsize=10, ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))
    
    # set the axis size for the font
    axis_size = 9
    
    # Plot 1: Ethiopia NDVI integral vs yield (top left)
    axes[0,0].scatter(eth_df['ndvi_integral_plant_harv'], eth_df['yield'], 
                      c='green', s=30, alpha=0.6)
    add_trend_line(axes[0,0], eth_df['ndvi_integral_plant_harv'].values, eth_df['yield'].values, color='darkgreen')
    axes[0,0].set_title('Ethiopia - NDVI Integral vs Yield', fontfamily='Times New Roman', fontsize=axis_size + 1)
    #axes[0,0].set_xlabel('NDVI Integral (Plant-Harvest)', fontfamily='Times New Roman', fontsize=axis_size)
    axes[0,0].set_ylabel('Yield (t/ha)', fontfamily='Times New Roman',fontsize=axis_size)
    
    # Plot 3: Burkina Faso NDVI integral vs yield (top right)
    axes[1,0].scatter(bfa_df['ndvi_integral_plant_harv'], bfa_df['yield'], 
                      c='green', s=30, alpha=0.6)
    add_trend_line(axes[1,0], bfa_df['ndvi_integral_plant_harv'].values, bfa_df['yield'].values, color='darkgreen')
    axes[1,0].set_title('Burkina Faso - NDVI Integral vs Yield', fontfamily='Times New Roman', fontsize=axis_size + 1)
    axes[1,0].set_xlabel('NDVI Integral (Plant-Harvest)', fontfamily='Times New Roman', fontsize=axis_size)
    axes[1,0].set_ylabel('Yield (t/ha)', fontfamily='Times New Roman', fontsize=axis_size)
    
    # Plot 2: Ethiopia GDD vs yield (bottom left)
    axes[0,1].scatter(eth_df['gdd_plant_harv'], eth_df['yield'], 
                      c='red', s=30, alpha=0.6)
    add_trend_line(axes[0,1], eth_df['gdd_plant_harv'].values, eth_df['yield'].values, color='darkred')
    axes[0,1].set_title('Ethiopia - GDD vs Yield', fontfamily='Times New Roman', fontsize=axis_size + 1)
    # axes[0,1].set_xlabel('GDD (Plant-Harvest) (K·days)', fontfamily='Times New Roman', fontsize=axis_size)
    #axes[0,1].set_ylabel('Yield (t/ha)', fontfamily='Times New Roman', fontsize=axis_size)
    
    # Plot 4: Burkina Faso GDD vs yield (bottom right)
    axes[1,1].scatter(bfa_df['gdd_plant_harv'], bfa_df['yield'], 
                      c='red', s=30, alpha=0.6)
    add_trend_line(axes[1,1], bfa_df['gdd_plant_harv'].values, bfa_df['yield'].values, color='darkred')
    axes[1,1].set_title('Burkina Faso - GDD vs Yield', fontfamily='Times New Roman', fontsize=axis_size + 1)
    axes[1,1].set_xlabel('GDD (Plant-Harvest) (K·days)', fontfamily='Times New Roman', fontsize=axis_size)
    #axes[1,1].set_ylabel('Yield (t/ha)', fontfamily='Times New Roman', fontsize=axis_size)
    
    # Adjust layout to prevent overlap with proper spacing
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95, hspace=0.3, wspace=0.2)
    
    # Show the plot
    plt.show()
    
    # Save the figure with 300 DPI
    output_dir = os.path.join(PROJECT_ROOT, "Plots", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reliability_stage_2_scatter.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    # Example usage of plot_comparison function:
    # Load data for Ethiopia and Burkina Faso
    ethiopia_df = pd.read_csv("RemoteSensing/HarvestStatAfrica/preprocessed/Ethiopia_admin2_merged_data_GLAM.csv")
    burkina_faso_df = pd.read_csv("RemoteSensing/HarvestStatAfrica/preprocessed/Burkina_Faso_admin2_merged_data_GLAM.csv")
    plot_comparison(ethiopia_df, burkina_faso_df)