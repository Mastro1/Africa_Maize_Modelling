import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import helper functions from the processing script
try:
    from process_all_dynamic_calendars_v3_4 import preprocess_and_merge, refine_smoothing, USE_STSG_SMOOTHING
except ImportError:
    from scipy.signal import savgol_filter
    def refine_smoothing(df, iters=3, window=31, poly=2, input_col='NDVI_mean'):
        if df.empty: return df
        y = df[input_col].values
        if len(y) == 0: return df
        w = window if len(y) > window else (len(y) // 2 * 2 - 1)
        if w < 5: w = 5 if len(y) >= 5 else 3
        if w <= poly: w = poly + 1
        if w % 2 == 0: w += 1
        for _ in range(iters):
            try:
                y = savgol_filter(y, window_length=w, polyorder=poly)
            except ValueError: pass
        df['NDVI_smooth'] = y
        return df

    def preprocess_and_merge(df_vi, df_era5):
        valid_pcodes = df_vi['PCODE'].unique()
        df_era5 = df_era5[df_era5['PCODE'].isin(valid_pcodes)]
        if df_vi['date'].dtype == 'O': df_vi['date'] = pd.to_datetime(df_vi['date'])
        if df_era5['date'].dtype == 'O': df_era5['date'] = pd.to_datetime(df_era5['date'])
        df_merged = pd.merge(df_era5, df_vi[['date', 'PCODE', 'NDVI_mean']], on=['date', 'PCODE'], how='left')
        df_merged['NDVI_mean'] = df_merged.groupby('PCODE')['NDVI_mean'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
        if not df_merged.empty and 'temperature_2m' in df_merged.columns and df_merged['temperature_2m'].iloc[0] > 100:
            for col in ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max']:
                if col in df_merged.columns: df_merged[col] = df_merged[col] - 273.15
        return df_merged
    USE_STSG_SMOOTHING = True

# ---------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------
BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
RESULTS_FILE = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis_V3_4", "Global_Calendar_V3_4.csv")
EXTRACTIONS_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
print(f"Loading Results CSV from {RESULTS_FILE}...", flush=True)
if os.path.exists(RESULTS_FILE):
    df_results = pd.read_csv(RESULTS_FILE)
    for col in ['SOS', 'Silking', 'EOS', 'Static_SOS', 'Static_EOS', 'Fence_Start', 'Fence_End']:
        if col in df_results.columns:
            df_results[col] = pd.to_datetime(df_results[col], errors='coerce')
else:
    print(f"ERROR: Results file not found at {RESULTS_FILE}")
    df_results = pd.DataFrame(columns=['Country', 'PCODE', 'Year', 'Season', 'Length_Days', 'GDD_Total'])

countries = sorted(df_results['Country'].unique())

# ---------------------------------------------------------
# INITIALIZE APP
# ---------------------------------------------------------
app = dash.Dash(__name__, title="Dynamic Calendar V3.4 Analysis")

app.layout = html.Div([
    html.H1("Dynamic Calendar V3.4 Analysis Dashboard (Strict Fenced)", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select Country:"),
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': c, 'value': c} for c in countries],
            value=countries[0] if countries else None,
            clearable=False
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
    
    html.Div([
        html.Label("Status:"),
        html.Div(id='status-text', children="Ready", style={'color': 'blue'})
    ], style={'width': '60%', 'display': 'inline-block', 'float': 'right', 'margin': '10px'}),

    dcc.Tabs([
        # TAB 1: STABILITY & ANOMALIES
        dcc.Tab(label='Stability & Anomalies', children=[
            html.Div([
                html.P(children="Analyze stability of Season Length over years. Click a point to see the seasonal graph."),
                dcc.Graph(id='stability-scatter-plot'),
                html.Hr(),
                html.Div([
                    html.H4("Yearly Dynamic Calendar Graph (Click on Scatter Plot above)"),
                    dcc.Graph(id='yearly-detail-graph')
                ])
            ], style={'padding': '20px'})
        ]),
        
        # TAB 2: PEER COMPARISON
        dcc.Tab(label='Peer Comparison (Outliers)', children=[
            html.Div([
                html.P(children="Compare locations within the country. Boxplot shows distribution of season lengths."),
                dcc.Graph(id='peer-boxplot'),
                html.Hr(),
                html.H4("Average Season Length per Location (Comparison)"),
                dcc.Graph(id='avg-length-bar'),
                html.Hr(),
                html.P(children="GDD vs Season Length. Points far from the cluster might be issues."),
                dcc.Graph(id='gdd-length-scatter'),
                html.Hr(),
                html.Div([
                    html.H4("Yearly Dynamic Calendar Graph (Click on GDD Plot above)"),
                    dcc.Graph(id='yearly-detail-graph-gdd')
                ])
            ], style={'padding': '20px'})
        ]),
        
        # TAB 3: SEASON COMPARISON
        dcc.Tab(label='Season 1 vs Season 2', children=[
            html.Div([
                html.P("Comparison of Season 1 vs Season 2 lengths for locations with two seasons."),
                dcc.Graph(id='s1-s2-scatter')
            ], style={'padding': '20px'})
        ]),

        # TAB 4: SINGLE TIMESERIES
        dcc.Tab(label='Single Timeseries', children=[
            html.Div([
                html.P("View single location timeseries for a specific year."),
                html.Div([
                    html.Label("Select Location (PCODE):"),
                    dcc.Dropdown(id='single-pcode-dropdown', placeholder="Select a PCODE"),
                ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '10px'}),
                html.Div([
                    html.Label("Select Year:"),
                    dcc.Dropdown(id='single-year-dropdown', placeholder="Select a Year"),
                ], style={'width': '45%', 'display': 'inline-block'}),
                html.Hr(),
                dcc.Graph(id='single-ts-graph')
            ], style={'padding': '20px'})
        ])
    ])
])

# ---------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------

data_cache = {}

def get_country_data(country):
    if country in data_cache:
        return data_cache[country]
    
    print(f"Loading data for {country}...", flush=True)
    vi_path = os.path.join(EXTRACTIONS_DIR, f"{country}_admin2_VI_timeseries_GADM.csv")
    era5_path = os.path.join(EXTRACTIONS_DIR, f"{country}_admin2_ERA5_timeseries_GADM.csv")
    
    if os.path.exists(vi_path) and os.path.exists(era5_path):
        df_vi = pd.read_csv(vi_path)
        df_era5 = pd.read_csv(era5_path)
        df_main = preprocess_and_merge(df_vi, df_era5)
        df_main = df_main[df_main['date'].dt.year >= 2000].copy()
        
        stsg_loaded = False
        if USE_STSG_SMOOTHING:
            country_clean = country.replace(' ', '_')
            stsg_path = os.path.join(BASE_DIR, "Model_physical", "Results", "STSG", f"{country_clean}_NDVI_STSG.csv")
            if os.path.exists(stsg_path):
                print(f"  Loading STSG for {country}...", flush=True)
                try:
                    df_stsg = pd.read_csv(stsg_path)
                    if 'date' in df_stsg.columns:
                        df_stsg['date'] = pd.to_datetime(df_stsg['date'])
                    
                    if 'NDVI_STSG' in df_stsg.columns:
                        df_stsg = df_stsg[['date', 'PCODE', 'NDVI_STSG']]
                        if df_main['date'].dtype == 'O': df_main['date'] = pd.to_datetime(df_main['date'])
                        df_main = pd.merge(df_main, df_stsg, on=['date', 'PCODE'], how='left')
                        stsg_loaded = True
                    else:
                        print(f"  STSG file found but 'NDVI_STSG' column missing.", flush=True)
                except Exception as e:
                    print(f"  Error loading STSG file: {e}", flush=True)
            else:
                 print(f"  STSG file not found for {country}. Using refined smoothing on-the-fly.", flush=True)

        data_cache[country] = (df_main, stsg_loaded)
        return df_main, stsg_loaded
    return None, False

@app.callback(
    Output('status-text', 'children'),
    Input('country-dropdown', 'value')
)
def update_status(country):
    return f"Selected: {country}"

@app.callback(
    [Output('stability-scatter-plot', 'figure'),
     Output('peer-boxplot', 'figure'),
     Output('avg-length-bar', 'figure'),
     Output('gdd-length-scatter', 'figure'),
     Output('s1-s2-scatter', 'figure')],
    [Input('country-dropdown', 'value')]
)
def update_main_graphs(country):
    if not country:
        empty = go.Figure()
        return empty, empty, empty, empty, empty
    
    df_c = df_results[df_results['Country'] == country].copy()
    if df_c.empty:
        empty = go.Figure()
        return empty, empty, empty, empty, empty

    # 1. Stability Scatter
    fig_stab = go.Figure()
    for season in sorted(df_c['Season'].unique()):
        dfs = df_c[df_c['Season'] == season]
        fig_stab.add_trace(go.Scatter(
            x=dfs['Year'], y=dfs['Length_Days'],
            mode='markers',
            name=f"Season {season}",
            marker=dict(size=8, opacity=0.7),
            text=dfs['PCODE'],
            customdata=dfs[['PCODE']].values,
            hovertemplate="PCODE: %{text}<br>Year: %{x}<br>Length: %{y} days<extra></extra>"
        ))
    fig_stab.update_layout(title=f"Season Length Stability: {country}", xaxis_title="Year", yaxis_title="Length (days)", clickmode='event+select')
    
    # 2. Peer Boxplot
    fig_peer = go.Figure()
    for season in sorted(df_c['Season'].unique()):
        dfs = df_c[df_c['Season'] == season]
        fig_peer.add_trace(go.Box(
            y=dfs['Length_Days'],
            name=f"Season {season}",
            boxpoints='all',
            jitter=0.3, pointpos=-1.8, text=dfs['PCODE'] + " (" + dfs['Year'].astype(str) + ")"
        ))
    fig_peer.update_layout(title=f"Distribution of Season Lengths: {country}", yaxis_title="Length (days)")
    
    # 3. Avg Length Bar
    df_avg = df_c.groupby(['PCODE', 'Season'])['Length_Days'].mean().reset_index().sort_values('Length_Days')
    fig_avg = go.Figure()
    for season in sorted(df_avg['Season'].unique()):
        dfa = df_avg[df_avg['Season'] == season]
        fig_avg.add_trace(go.Bar(x=dfa['PCODE'], y=dfa['Length_Days'], name=f"Season {season}"))
    fig_avg.update_layout(title=f"Average Season Length by PCODE: {country}", barmode='group', xaxis_title="PCODE", yaxis_title="Avg Length (days)")
    
    # 4. GDD vs Length
    fig_gdd = go.Figure()
    for season in sorted(df_c['Season'].unique()):
        dfs = df_c[df_c['Season'] == season]
        fig_gdd.add_trace(go.Scatter(
            x=dfs['GDD_Total'], y=dfs['Length_Days'],
            mode='markers',
            name=f"Season {season}",
            text=dfs['PCODE'] + " (" + dfs['Year'].astype(str) + ")",
            customdata=dfs[['PCODE', 'Year']].values,
            hovertemplate="PCODE: %{text}<br>GDD: %{x}<br>Length: %{y} days<extra></extra>"
        ))
    fig_gdd.update_layout(title="GDD vs Season Length", xaxis_title="Total GDD", yaxis_title="Length (days)", clickmode='event+select')
    
    # 5. S1 vs S2 Comparison
    s1 = df_c[df_c['Season'] == 1][['PCODE', 'Year', 'Length_Days']].rename(columns={'Length_Days': 'S1_Length'})
    s2 = df_c[df_c['Season'] == 2][['PCODE', 'Year', 'Length_Days']].rename(columns={'Length_Days': 'S2_Length'})
    fig_vs = go.Figure()
    if not s2.empty:
        df_comp = pd.merge(s1, s2, on=['PCODE', 'Year'])
        fig_vs.add_trace(go.Scatter(x=df_comp['S1_Length'], y=df_comp['S2_Length'], mode='markers', name="S1 vs S2", text=df_comp['PCODE'] + " (" + df_comp['Year'].astype(str) + ")"))
        if not df_comp.empty:
            max_v = max(df_comp['S1_Length'].max(), df_comp['S2_Length'].max())
            fig_vs.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(color="Gray", dash="dash"))
    fig_vs.update_layout(title="Season 1 vs Season 2 Length Comparison", xaxis_title="S1 Length", yaxis_title="S2 Length")
    
    return fig_stab, fig_peer, fig_avg, fig_gdd, fig_vs

def generate_yearly_detail_figure(country, pcode, year):
    if not country or not pcode or not year:
        fig = go.Figure()
        fig.update_layout(title="Select a PCODE and Year to see details.")
        return fig

    try: year = int(float(year))
    except: pass

    print(f"Generating detail plot for {country} - {pcode} - {year}", flush=True)
    
    df_main_tuple = get_country_data(country)
    if df_main_tuple is None or df_main_tuple[0] is None:
        fig = go.Figure()
        fig.update_layout(title="Error loading raw data.")
        return fig
    
    df_main, stsg_loaded = df_main_tuple
    df_pcode = df_main[df_main['PCODE'] == pcode].copy()
    if df_pcode.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data found for {pcode}")
        return fig
    
    applied_stsg = False
    if stsg_loaded and 'NDVI_STSG' in df_pcode.columns:
         df_pcode['NDVI_smooth'] = df_pcode['NDVI_STSG'].interpolate(method='linear', limit_direction='both')
         applied_stsg = True
    
    # Always apply the refinement filter
    input_col_to_smooth = 'NDVI_smooth' if applied_stsg else 'NDVI_mean'
    df_pcode = refine_smoothing(df_pcode, input_col=input_col_to_smooth)
    
    df_pcode['Year'] = df_pcode['date'].dt.year
    start_date = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=120)
    end_date = pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=360)
    subset = df_pcode[(df_pcode['date'] >= start_date) & (df_pcode['date'] <= end_date)].copy()
    
    res_rows = df_results[(df_results['Country'] == country) & (df_results['PCODE'] == pcode) & (df_results['Year'] == year)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset['date'], y=subset['NDVI_mean'], mode='lines', name='Raw NDVI', line=dict(color='#32CD32', width=1), opacity=0.4))
    fig.add_trace(go.Scatter(x=subset['date'], y=subset['NDVI_smooth'], mode='lines', name='Smoothed NDVI', line=dict(color='darkgreen', width=3)))
    
    colors = {1: 'blue', 2: 'orange'}
    for _, row in res_rows.iterrows():
        if pd.isna(row['Season']): continue
        s_idx = int(row['Season'])
        c = colors.get(s_idx, 'black')
        sos, eos, silk = row['SOS'], row['EOS'], row['Silking']
        f_start, f_end = row.get('Fence_Start'), row.get('Fence_End')
        
        # Plot Fence
        if f_start and f_end and not pd.isnull(f_start) and not pd.isnull(f_end):
             fig.add_vrect(x0=f_start.timestamp()*1000, x1=f_end.timestamp()*1000, 
                           fillcolor=c, opacity=0.05, layer="below", line_width=0, annotation_text=f"S{s_idx} Fence")

        sos_pl = sos.timestamp() * 1000 if pd.notnull(sos) else None
        eos_pl = eos.timestamp() * 1000 if pd.notnull(eos) else None
        silk_pl = silk.timestamp() * 1000 if pd.notnull(silk) else None
        
        if sos_pl: fig.add_vline(x=sos_pl, line_dash="solid", line_color=c, annotation_text=f"S{s_idx} SOS")
        if eos_pl: fig.add_vline(x=eos_pl, line_dash="dashdot", line_color=c, annotation_text=f"S{s_idx} EOS")
        if silk_pl: fig.add_vline(x=silk_pl, line_dash="dot", line_color=c, annotation_text=f"S{s_idx} Silk")
        
        if sos_pl and eos_pl:
            fig.add_vrect(x0=sos_pl, x1=eos_pl, fillcolor=c, opacity=0.1, layer="below", line_width=0)
            midpoint = sos + (eos - sos) / 2
            midpoint_pl = midpoint.timestamp() * 1000
            
            # Static Window comparison if available
            st_sos, st_eos = row.get('Static_SOS'), row.get('Static_EOS')
            if st_sos and st_eos and not pd.isnull(st_sos) and not pd.isnull(st_eos):
                 fig.add_vrect(x0=st_sos.timestamp()*1000, x1=st_eos.timestamp()*1000, 
                               fillcolor="#888", opacity=0.05, layer="below", line_dash="dash")

            length_days = row.get('Length_Days', 0)
            gdd_total = row.get('GDD_Total', 0)
            method = row.get('Method', 'Unknown')
            fig.add_annotation(x=midpoint_pl, y=subset['NDVI_smooth'].min(), 
                               text=f"S{s_idx}: {length_days:.0f}d | {gdd_total:.0f} GDD | {method}", 
                               showarrow=False, yshift=10, bgcolor="white", opacity=0.8, font=dict(color=c))

    fig.update_layout(title=f"Calendar Verification V3.4: {pcode} ({year})", 
                      xaxis_title="Date", yaxis_title="NDVI", 
                      hovermode="x unified", template="plotly_white", height=500)
    return fig

def create_yearly_detail_fig(clickData, country):
    if not clickData or not country:
        fig = go.Figure()
        fig.update_layout(title="Click a point on the plot above to see details.")
        return fig
    
    point = clickData['points'][0]
    try:
        if 'customdata' in point:
            pcode = point['customdata'][0]
            if len(point['customdata']) > 1:
                year = point['customdata'][1]
            else:
                year = point['x']
        elif 'text' in point:
            pcode = point['text'].split(' ')[0]
            year = point['x']
        else:
             raise ValueError("No identifying info found")
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error: Could not retrieve info from click data. ({str(e)})")
        return fig
    
    return generate_yearly_detail_figure(country, pcode, year)

@app.callback(
    Output('yearly-detail-graph', 'figure'),
    [Input('stability-scatter-plot', 'clickData'), Input('country-dropdown', 'value')]
)
def display_yearly_detail(clickData, country):
    return create_yearly_detail_fig(clickData, country)

@app.callback(
    Output('yearly-detail-graph-gdd', 'figure'),
    [Input('gdd-length-scatter', 'clickData'), Input('country-dropdown', 'value')]
)
def display_yearly_detail_gdd(clickData, country):
    return create_yearly_detail_fig(clickData, country)

# CALLBACKS FOR TAB 4: SINGLE TIMESERIES
@app.callback(
    [Output('single-pcode-dropdown', 'options'),
     Output('single-pcode-dropdown', 'value')],
    [Input('country-dropdown', 'value')]
)
def update_pcode_dropdown(country):
    if not country:
        return [], None
    df_c = df_results[df_results['Country'] == country]
    pcodes = sorted(df_c['PCODE'].unique())
    options = [{'label': p, 'value': p} for p in pcodes]
    value = pcodes[0] if pcodes else None
    return options, value

@app.callback(
    [Output('single-year-dropdown', 'options'),
     Output('single-year-dropdown', 'value')],
    [Input('single-pcode-dropdown', 'value')],
    [State('country-dropdown', 'value')]
)
def update_year_dropdown(pcode, country):
    if not country or not pcode:
        return [], None
    df_c = df_results[(df_results['Country'] == country) & (df_results['PCODE'] == pcode)]
    years = sorted(df_c['Year'].unique(), reverse=True)
    options = [{'label': str(y), 'value': y} for y in years]
    value = years[0] if years else None
    return options, value

@app.callback(
    Output('single-ts-graph', 'figure'),
    [Input('single-pcode-dropdown', 'value'),
     Input('single-year-dropdown', 'value')],
    [State('country-dropdown', 'value')]
)
def update_single_ts_graph(pcode, year, country):
    return generate_yearly_detail_figure(country, pcode, year)

if __name__ == '__main__':
    app.run(debug=True)
