import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from scipy import stats

# Set plotly renderer for better display
pio.renderers.default = "browser"

PROJECT_ROOT = os.getcwd()

def plot_comparison(ethiopia_df, burkina_faso_df):
    """
    Create a comparison plot between Ethiopia and Burkina Faso showing NDVI integral, 
    plant harvest, and GDD for both countries in a 2x2 subplot layout using Plotly.
    
    Parameters:
    - ethiopia_df: DataFrame with Ethiopia data
    - burkina_faso_df: DataFrame with Burkina Faso data
    """
    # Filter data to remove rows with missing values
    eth_df = ethiopia_df.dropna(subset=['ndvi_integral_plant_harv', 'gdd_plant_harv', 'yield']).copy()
    bfa_df = burkina_faso_df.dropna(subset=['ndvi_integral_plant_harv', 'gdd_plant_harv', 'yield']).copy()
    
    # Create subplots (2 rows, 2 columns) - restructured so same country is in same column
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Ethiopia - NDVI Integral vs Yield',
            'Burkina Faso - NDVI Integral vs Yield',
            'Ethiopia - GDD vs Yield',
            'Burkina Faso - GDD vs Yield'
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.06
    )
    
    # Helper function to add trend line and calculate R²
    def add_trend_line(fig, x_data, y_data, row, col, color='red'):
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
            fig.add_trace(
                go.Scatter(x=x_trend, y=y_trend,
                          mode='lines', name='Trend',
                          line=dict(color=color, width=2, dash='dash'),
                          showlegend=False),
                row=row, col=col
            )
            
            # Add R² annotation
            fig.add_annotation(
                x=x_clean.max() * 0.95, y=y_clean.max() * 0.95,
                text=f"R² = {r_value**2:.3f}<br>p = {p_value:.3f}",
                showarrow=False,
                font=dict(size=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                row=row, col=col
            )
    
    # Plot 1: Ethiopia NDVI integral vs yield (top left)
    fig.add_trace(
        go.Scatter(
            x=eth_df['ndvi_integral_plant_harv'],
            y=eth_df['yield'],
            mode='markers',
            marker=dict(color='green', size=6, opacity=0.6),
            name='Ethiopia - NDVI',
            text=[f"Year: {year}" for year in eth_df['year']],
            hovertemplate="NDVI Integral: %{x:.2f}<br>Yield: %{y:.2f} t/ha<br>%{text}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add trend line for Ethiopia NDVI
    add_trend_line(fig, eth_df['ndvi_integral_plant_harv'].values, eth_df['yield'].values, 1, 1, 'darkgreen')
    
    # Plot 2: Burkina Faso NDVI integral vs yield (top right)
    fig.add_trace(
        go.Scatter(
            x=bfa_df['ndvi_integral_plant_harv'],
            y=bfa_df['yield'],
            mode='markers',
            marker=dict(color='green', size=6, opacity=0.6),
            name='Burkina Faso - NDVI',
            text=[f"Year: {year}" for year in bfa_df['year']],
            hovertemplate="NDVI Integral: %{x:.2f}<br>Yield: %{y:.2f} t/ha<br>%{text}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add trend line for Burkina Faso NDVI
    add_trend_line(fig, bfa_df['ndvi_integral_plant_harv'].values, bfa_df['yield'].values, 1, 2, 'darkgreen')
    
    # Plot 3: Ethiopia GDD vs yield (bottom left)
    fig.add_trace(
        go.Scatter(
            x=eth_df['gdd_plant_harv'],
            y=eth_df['yield'],
            mode='markers',
            marker=dict(color='red', size=6, opacity=0.6),
            name='Ethiopia - GDD',
            text=[f"Year: {year}" for year in eth_df['year']],
            hovertemplate="GDD: %{x:.1f} K·days<br>Yield: %{y:.2f} t/ha<br>%{text}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Add trend line for Ethiopia GDD
    add_trend_line(fig, eth_df['gdd_plant_harv'].values, eth_df['yield'].values, 2, 1, 'darkred')
    
    # Plot 4: Burkina Faso GDD vs yield (bottom right)
    fig.add_trace(
        go.Scatter(
            x=bfa_df['gdd_plant_harv'],
            y=bfa_df['yield'],
            mode='markers',
            marker=dict(color='red', size=6, opacity=0.6),
            name='Burkina Faso - GDD',
            text=[f"Year: {year}" for year in bfa_df['year']],
            hovertemplate="GDD: %{x:.1f} K·days<br>Yield: %{y:.2f} t/ha<br>%{text}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # Add trend line for Burkina Faso GDD
    add_trend_line(fig, bfa_df['gdd_plant_harv'].values, bfa_df['yield'].values, 2, 2, 'darkred')
    
    # Update layout to match the style of other plots
    fig.update_layout(
        title='Comparison of Ethiopia and Burkina Faso',
        title_x=0.5,
        height=600,
        width=800,
        showlegend=False,
        font=dict(family="Times New Roman", size=12)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="NDVI Integral (Plant-Harvest)", row=1, col=1)
    fig.update_xaxes(title_text="NDVI Integral (Plant-Harvest)", row=1, col=2)
    fig.update_xaxes(title_text="GDD (Plant-Harvest) (K·days)", row=2, col=1)
    fig.update_xaxes(title_text="GDD (Plant-Harvest) (K·days)", row=2, col=2)
    
    fig.update_yaxes(title_text="Yield (t/ha)", row=1, col=1)
    fig.update_yaxes(title_text="Yield (t/ha)", row=1, col=2)
    fig.update_yaxes(title_text="Yield (t/ha)", row=2, col=1)
    fig.update_yaxes(title_text="Yield (t/ha)", row=2, col=2)
    
    fig.show()

    fig.write_html(os.path.join(PROJECT_ROOT, "Plots", "output", "reliability_stage_2_scatter.html"))


if __name__ == "__main__":
    # Example usage of plot_comparison function:
    # Load data for Ethiopia and Burkina Faso
    ethiopia_df = pd.read_csv("RemoteSensing/HarvestStatAfrica/preprocessed/Ethiopia_admin2_merged_data_GLAM.csv")
    burkina_faso_df = pd.read_csv("RemoteSensing/HarvestStatAfrica/preprocessed/Burkina_Faso_admin2_merged_data_GLAM.csv")
    plot_comparison(ethiopia_df, burkina_faso_df)