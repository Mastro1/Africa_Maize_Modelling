import json
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_sources():
    with open('ASR/macroClimate/sources.json', 'r') as f:
        return json.load(f)

def load_nao(source):
    url = source['data_file']
    response = requests.get(url)
    data = response.text.split('\n')
    nao_data = []
    for line in data[1:]:  # Skip header
        if line.strip():
            parts = line.split()
            if len(parts) >= 13:
                year = int(parts[0])
                values = [float(x) for x in parts[1:13]]
                for month, value in enumerate(values, 1):
                    nao_data.append({
                        'date': pd.Timestamp(year=year, month=month, day=1),
                        'NAO': value
                    })
    df = pd.DataFrame(nao_data)
    return df

def load_local_csv(source, indicator, missing_value):
    # Read, skip first row (header with extra info), assign columns
    df = pd.read_csv(source['data_file'], skiprows=1, header=None)
    df.columns = ['date', indicator]
    # Clean up whitespace and replace missing values (as string and float)
    df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), errors='coerce')
    df[indicator] = df[indicator].astype(str).str.strip()
    df[indicator] = df[indicator].replace([str(missing_value), missing_value], np.nan)
    df[indicator] = pd.to_numeric(df[indicator], errors='coerce')
    # Drop any value below -99
    df.loc[df[indicator] < -99, indicator] = np.nan
    return df[['date', indicator]]

def main():
    sources = load_sources()
    dfs = []
    for indicator, source in sources.items():
        if indicator == 'NAO':
            df = load_nao(source)
        elif indicator == 'DMI':
            df = load_local_csv(source, 'DMI', -9999)
        elif indicator == 'NINO34':
            df = load_local_csv(source, 'NINO34', -99.99)
        elif indicator == 'TSA':
            df = load_local_csv(source, 'TSA', -99.99)
        elif indicator == 'EA':
            df = load_local_csv(source, 'EA', -99.99)
        else:
            continue
        dfs.append(df)
    # Merge all on 'date'
    from functools import reduce
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    # Filter to only data from 1999 onwards
    df_merged = df_merged[df_merged['date'] >= pd.Timestamp('1999-01-01')].reset_index(drop=True)
    print("\nMerged dataset shape:", df_merged.shape)
    print("Columns:", df_merged.columns.tolist())
    print(df_merged.head())
    # Export
    df_merged.to_csv('ASR/macroClimate/climate_merged.csv', index=False)
    print("\nExported merged dataset to ASR/macroClimate/climate_merged.csv")

def check_missing_data():
    df = pd.read_csv('ASR/macroClimate/climate_merged.csv')
    df['date'] = pd.to_datetime(df['date'])
    indicators = ['NAO', 'DMI', 'NINO34', 'TSA', 'EA']
    for indicator in indicators:
        missing = df[indicator].isna().sum()
        print(f"{indicator} missing values: {missing}")

def create_yearly_reduced():
    df = pd.read_csv('ASR/macroClimate/climate_merged.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    indicators = ['NAO', 'DMI', 'NINO34', 'TSA', 'EA']
    yearly_data = []
    
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        yearly_row = {'year': year}
        
        for indicator in indicators:
            values = year_data[indicator].dropna()
            if len(values) > 0:
                max_val = values.max()
                min_val = values.min()
                # Choose max if abs(max) > abs(min), otherwise choose min
                if abs(max_val) > abs(min_val):
                    yearly_row[indicator] = max_val
                else:
                    yearly_row[indicator] = min_val
            else:
                yearly_row[indicator] = np.nan
        
        yearly_data.append(yearly_row)
    
    df_yearly = pd.DataFrame(yearly_data)
    df_yearly = df_yearly.sort_values('year').reset_index(drop=True)
    
    # Export yearly reduced data
    df_yearly.to_csv('ASR/macroClimate/climate_yearly_reduced.csv', index=False)
    print(f"\nYearly reduced dataset exported to ASR/macroClimate/climate_yearly_reduced.csv")
    print(f"Shape: {df_yearly.shape}")
    print(df_yearly.head())
    
    return df_yearly

def plot_indicators():
    df = pd.read_csv('ASR/macroClimate/climate_merged.csv')
    df['date'] = pd.to_datetime(df['date'])
    indicators = ['NAO', 'DMI', 'NINO34', 'TSA', 'EA']
    fig = make_subplots(rows=5, cols=1, subplot_titles=indicators)
    for i, indicator in enumerate(indicators, 1):
        fig.add_trace(go.Scatter(x=df['date'], y=df[indicator], mode='lines', name=indicator), row=i, col=1)
    fig.update_layout(height=1500, title_text="Climate Indicators Over Time", showlegend=False)
    fig.write_html('ASR/macroClimate/all_indicators_plot.html')
    print("Interactive plot with all indicators saved as ASR/macroClimate/all_indicators_plot.html")

if __name__ == "__main__":
    main()
    check_missing_data()
    plot_indicators()
    create_yearly_reduced()
