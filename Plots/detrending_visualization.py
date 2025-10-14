import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys
import os

# Set the default renderer to open in browser
pio.renderers.default = "browser"

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

def get_subplot_topright_corner(fig, isubplot):
    """Get the top-right corner position of a subplot in normalized coordinates."""
    x_right = fig.layout[f'xaxis{isubplot}']['domain'][1]  # 1=right
    y_top = fig.layout[f'yaxis{isubplot}']['domain'][1]  # 1=top
    return x_right, y_top

def create_detrending_visualization(country_name='Kenya'):
    """Create a plotly visualization showing FAO data and trend line for detrending."""
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
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'National Yield Data for {country_name}',
            f'Detrended Yield Data for {country_name}'
        ),
        horizontal_spacing=0.1
    )
    
    # Left subplot - National yield data (simplified)
    # Add actual yield data (FAO) as lines connecting the points
    fig.add_trace(
        go.Scatter(
            x=trend_df_renamed['year'],
            y=trend_df_renamed['fao_yield'],
            mode='lines+markers',
            name='FAO National Yield',
            marker=dict(size=8, color='#24245c'),  # Using the specified blue color
            line=dict(width=2, color='#24245c'),
            hovertemplate='<b>Year</b>: %{x}<br><b>FAO Yield</b>: %{y:.2f} t/ha<extra></extra>',
            legend="legend1"
        ),
        row=1, col=1
    )
    
    # Add trend line (using the specified color)
    fig.add_trace(
        go.Scatter(
            x=trend_df_renamed['year'],
            y=trend_df_renamed['national_trend'],
            mode='lines',
            name='Linear Trend',
            line=dict(width=3, color='#fccd32'),  # Using the specified color
            hovertemplate='<b>Year</b>: %{x}<br><b>Trend</b>: %{y:.2f} t/ha<extra></extra>',
            legend="legend1"
        ),
        row=1, col=1
    )
    
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
        for pcode in combined_detrended['PCODE'].unique():
            pcode_detrended = combined_detrended[combined_detrended['PCODE'] == pcode]
            fig.add_trace(
                go.Scatter(
                    x=pcode_detrended['year'],
                    y=pcode_detrended['detrended_yield'],
                    mode='markers',
                    name=f'PCODE: {pcode}',
                    marker=dict(size=6),
                    hovertemplate=f'<b>PCODE</b>: {pcode}<br><b>Year</b>: %{{x}}<br><b>Detrended Yield</b>: %{{y:.2f}}<extra></extra>',
                    legend="legend2",
                    showlegend=False  # Don't show in legend as requested
                ),
                row=1, col=2
            )
    
    # Add FAO detrended data (thicker line)
    if all_fao_detrended_data:
        fao_detrended = pd.concat(all_fao_detrended_data, ignore_index=True)
        fig.add_trace(
            go.Scatter(
                x=fao_detrended['year'],
                y=fao_detrended['detrended_yield'],
                mode='lines+markers',
                name='FAO Detrended',
                marker=dict(size=8, color='#24245c'),
                line=dict(width=2, color='#24245c'),  # Same thickness as the other lines
                hovertemplate='<b>Year</b>: %{x}<br><b>FAO Detrended Yield</b>: %{y:.2f}<extra></extra>',
                legend="legend2"
            ),
            row=1, col=2
        )
    
    # Add a grey point to the legend to represent individual locations
    fig.add_trace(
        go.Scatter(
            x=[None],  # No actual data points
            y=[None],
            mode='markers',
            name='Locations',
            marker=dict(size=6, color='grey'),
            legend="legend2",
            showlegend=True
        ),
        row=1, col=2
    )
    
    # Add trend line at zero for the detrended data (always zero by definition)
    if not trend_df_renamed.empty:
        fig.add_trace(
            go.Scatter(
                x=[trend_df_renamed['year'].min(), trend_df_renamed['year'].max()],
                y=[0, 0],
                mode='lines',
                name='Zero Trend',
                line=dict(width=3, color='#fccd32'),
                hovertemplate='<b>Zero Trend</b><extra></extra>',
                legend="legend2"
            ),
            row=1, col=2
        )
    
    # Configure legend position for each subplot
    leg_dict = {}
    for isp in range(1, 3):  # 2 subplots
        # Get subplot top right corner position
        xl, yl = get_subplot_topright_corner(fig, isp)
        # note: legend or legend1 are both valid for first legend
        leg_dict[f'legend{isp if isp > 1 else ""}'] = {
            'x': xl - 0.01,  # Slightly inset from the right edge
            'y': yl - 0.03,  # Slightly inset from the top edge
            'xanchor': 'right',  # anchor of legend at its right
            'yanchor': 'top',  # anchor of legend at its top
            'font': dict(family="Times New Roman", size=10)
        }
    
    # Update layout to follow design rules
    fig.update_layout(
        title=f'Detrending Approach for {country_name}',
        title_x=0.5,  # Center the title
        hovermode='x unified',
        width=1200,  # Wider to accommodate two subplots
        height=400,
        font=dict(
            family="Times New Roman",
            size=12
        )
    )
    
    # Apply legend configuration
    fig.update_layout(leg_dict)
    
    # Update axis labels
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_yaxes(title_text="Yield (tonnes/hectare)", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="Detrended Yield", row=1, col=2)
    
    return fig

if __name__ == "__main__":
    # Get country name from command line argument, default to Kenya
    if len(sys.argv) > 1:
        country_name = sys.argv[1]
    else:
        country_name = 'Zimbabwe'
    
    print(f"Creating detrending visualization for {country_name}...")
    
    # Create visualization
    fig = create_detrending_visualization(country_name)
    if fig:
        os.makedirs(Path(__file__).parent / "output", exist_ok=True)
        output_filename = f"detrending_visualization_{country_name.lower().replace(' ', '_')}.html"
        output_path = Path(__file__).parent / "output" / output_filename
        fig.write_html(output_path)
        print(f"Visualization saved as {output_path}")
        
        # Show in browser
        fig.show()
    else:
        print(f"Failed to create visualization for {country_name}")