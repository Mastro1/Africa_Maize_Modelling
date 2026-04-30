import pandas as pd

COUNTRY_NAME = "Zambia"

macroClimate = pd.read_csv("ASR/macroClimate/climate_merged.csv")
macroClimate['date'] = pd.to_datetime(macroClimate['date'])
macroClimate['year'] = macroClimate['date'].dt.year

# Calculate max and min for each year
macroClimate_max = macroClimate.groupby(['year']).max().reset_index()
macroClimate_min = macroClimate.groupby(['year']).min().reset_index()

# Create a new dataframe to store the extreme values
macroClimate_extreme = macroClimate_max.copy()

# For each climate index column, compare abs(max) and abs(min)
climate_columns = ['NAO', 'DMI', 'NINO34', 'TSA', 'EA']
for col in climate_columns:
    # Compare absolute values of max and min
    mask = macroClimate_max[col].abs() < macroClimate_min[col].abs()
    # Where abs(min) is larger, use the min value
    macroClimate_extreme.loc[mask, col] = macroClimate_min.loc[mask, col]

macroClimate = macroClimate_extreme.merge(macroClimate_min, on=['year'] + climate_columns, how='left')
print("macroClimate_extreme")
print(macroClimate.columns)
print(macroClimate.head())

fao = pd.read_csv("data/faostat/FAOSTAT_maize.csv")
fao = fao[fao['Area'] == COUNTRY_NAME]
fao = fao.pivot(index='Year', columns='Element', values='Value').reset_index()
fao["PCODE"] = f"{COUNTRY_NAME}_COUNTRY"
fao.rename(columns={'Production': 'production', 'Yield': 'yield', 'Area harvested': 'area_harvested'}, inplace=True)
fao["yield"] = fao["yield"]/1000
# For fao, ensure 'Year' is renamed to 'year' and is int
fao = fao.rename(columns={'Year': 'year'})
fao['year'] = fao['year'].astype(int)

admin2 = pd.read_csv("ASR/zambia/downscaling/data/admin2_merged_data.csv")

# correlation between macroClimate and fao
macroClimate_data = macroClimate.merge(fao, on=['year'], how='right')
print("merged with fao")
print(macroClimate_data.columns)
print(macroClimate_data.head())

correlation = macroClimate_data[['yield', 'NAO', 'DMI', 'NINO34', 'TSA', 'EA']].corr()
print(correlation)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
# take values of correlation excluding yield
# Create a bar chart using plotly
fig = px.bar(
    x=correlation['yield'].iloc[1:].index,
    y=correlation['yield'].iloc[1:].values,
    text=correlation['yield'].iloc[1:].round(3),
    title='Correlation of Climate Indices with FAO Yield',
    labels={'x': 'Climate Indices', 'y': 'Correlation Coefficient'}
)

fig.update_layout(
    yaxis_range=[-0.5, 0.5],
    showlegend=False
)

fig.add_hline(y=0, line_dash="dash", line_color="gray")

fig.show()











