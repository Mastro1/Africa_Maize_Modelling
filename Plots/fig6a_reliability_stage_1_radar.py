import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import os

# Set font family to match the plotly version
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

markdown_table_string = """
| Rank         | Country                      | Score | Years | Temporal | Locations | Loc. Comp. | Maize% | Yield CV | Flag%    | Assessment       |
| ------------ | ---------------------------- | ----- | ----- | -------- | --------- | ---------- | ------ | -------- | -------- | ---------------- |
| 1            | **Zimbabwe**                 | 0.853 | 21/24 | 87.5%    | 7         | 100.0%     | 48.5%  | 0.450    | 0.0%     | **EXCELLENT**    |
| 2            | **Lesotho**                  | 0.848 | 18/24 | 75.0%    | 8         | 100.0%     | 59.9%  | 0.439    | 0.0%     | **EXCELLENT**    |
| 3            | **Zambia**                   | 0.843 | 17/24 | 70.8%    | 48        | 100.0%     | 45.7%  | 0.352    | 0.1%     | **EXCELLENT**    |
| 4            | **Malawi**                   | 0.808 | 20/24 | 83.3%    | 21        | 96.2%      | 42.2%  | 0.333    | 0.0%     | **VERY GOOD**    |
| 5            | **Benin**                    | 0.795 | 21/24 | 87.5%    | 45        | 100.0%     | 31.0%  | 0.210    | 0.0%     | **VERY GOOD**    |
| 6            | **South Africa**             | 0.790 | 22/24 | 91.7%    | 6         | 97.7%      | 49.7%  | 0.233    | 0.0%     | **VERY GOOD**    |
| 7            | **Mozambique**               | 0.787 | 21/24 | 87.5%    | 8         | 94.6%      | 28.0%  | 0.422    | 0.6%     | **GOOD**         |
| 8            | **Angola**                   | 0.753 | 17/24 | 70.8%    | 8         | 100.0%     | 36.1%  | 0.365    | 0.0%     | **GOOD**         |
| 9            | **Mali**                     | 0.748 | 22/24 | 91.7%    | 4         | 100.0%     | 11.4%  | 0.408    | 1.1%     | **GOOD**         |
| 10           | **Kenya**                    | 0.745 | 14/24 | 58.3%    | 33        | 100.0%     | 36.9%  | 0.298    | 0.2%     | **GOOD**         |
| 11           | **Ethiopia**                 | 0.684 | 13/24 | 54.2%    | 39        | 100.0%     | 16.5%  | 0.268    | 0.0%     | **AVERAGE**      |
| 12           | **Togo**                     | 0.683 | 15/24 | 62.5%    | 24        | 93.6%      | 30.1%  | 0.198    | 0.0%     | **AVERAGE**      |
| 13           | **Burkina Faso**             | 0.679 | 15/24 | 62.5%    | 32        | 93.3%      | 10.7%  | 0.290    | 0.0%     | **AVERAGE**      |
| 14           | **Madagascar**               | 0.672 | 18/24 | 75.0%    | 16        | 100.0%     | 6.8%   | 0.268    | 0.0%     | **AVERAGE**      |
| 15           | **Senegal**                  | 0.667 | 15/24 | 62.5%    | 23        | 73.9%      | 6.2%   | 0.398    | 0.8%     | **AVERAGE**      |
| 16           | **Chad**                     | 0.658 | 17/24 | 70.8%    | 11        | 96.8%      | 5.7%   | 0.300    | 0.0%     | **AVERAGE**      |
| 17           | **Nigeria_cleaned**          | 0.656 | 17/24 | 70.8%    | 18        | 100.0%     | 0.0%   | 0.319    | 3.6%     | **AVERAGE**      |
| 18           | **Somalia**                  | 0.643 | 11/24 | 45.8%    | 9         | 87.9%      | 18.2%  | 0.389    | 0.0%     | **BELOW AVERAGE**|
| 19           | **Mauritania**               | 0.611 | 15/24 | 62.5%    | 3         | 68.9%      | 5.1%   | 0.355    | 0.0%     | **BELOW AVERAGE**|
| 20           | **Niger**                    | 0.596 | 17/24 | 70.8%    | 17        | 47.1%      | 0.1%   | 0.299    | 2.9%     | **BELOW AVERAGE**|
| 21           | **Tanzania, United Republic of** | 0.580 | 11/24 | 45.8%    | 17        | 93.6%      | 0.0%   | 0.277    | 0.0%     | **POOR**         |
| 22           | **DRC**                      | 0.565 | 11/24 | 45.8%    | 15        | 94.5%      | 0.0%   | 0.240    | 0.0%     | **POOR**         |
| 23           | **Rwanda**                   | 0.560 | 5/24  | 20.8%    | 26        | 83.8%      | 10.6%  | 0.306    | 0.0%     | **POOR**         |
| 24           | **Uganda**                   | 0.500 | 1/24  | 4.2%     | 57        | 100.0%     | 15.9%  | 0.000    | 0.0%     | **POOR**         |
| 25           | **Guinea**                   | 0.469 | 6/24  | 25.0%    | 12        | 100.0%     | 14.1%  | 0.000    | 0.0%     | **POOR**         |
| 26           | **Central African Republic** | 0.437 | 3/24  | 12.5%    | 10        | 100.0%     | 12.5%  | 0.021    | 0.0%     | **VERY POOR**    |
| 27           | **Sierra Leone**             | 0.405 | 4/24  | 16.7%    | 5         | 100.0%     | 1.7%   | 0.000    | 0.0%     | **VERY POOR**    |
| **REJECTED** | **Nigeria**                  | 0.707 | 23/24 | 95.8%    | 26        | 78.4%      | 10.1%  | 0.249    | 8.0% | **UNACCEPTABLE** |
| **REJECTED** | **Ghana**                    | 0.655 | 16/24 | 66.7%    | 12        | 98.4%      | 13.9%  | 0.320    | 24.0% | **UNACCEPTABLE** |
"""

# Process the markdown string to be readable by pandas
# 1. Remove leading/trailing whitespace and pipes
# 2. Handle the separator line
lines = markdown_table_string.strip().split('\n')
# Clean each line by removing leading/trailing pipes and whitespace
processed_lines = [line.strip().strip('|').strip() for line in lines]
# Reconstruct the table string, but this time it's cleaner
clean_table_string = "\n".join(processed_lines)

# Use StringIO to treat the string as a file for pandas
data = io.StringIO(clean_table_string)

# Read the data, using the pipe as a separator and skipping the markdown separator line
df = pd.read_csv(data, sep='|', skiprows=[1])

# Clean up column names and data by stripping whitespace
df.columns = df.columns.str.strip()
for col in df.columns:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

# Remove markdown formatting (bolding with **)
df['Country'] = df['Country'].str.replace(r'\*\*', '', regex=True)
df['Flag%'] = df['Flag%'].str.replace(r'\*\*', '', regex=True)
df['Assessment'] = df['Assessment'].str.replace(r'\*\*', '', regex=True)

# --- Data processing for radar plot ---

# Convert relevant columns to numeric types
df['Years'] = df['Years'].str.split('/').str[0].astype(float)
df['Locations'] = df['Locations'].astype(float)
df['Loc. Comp.'] = df['Loc. Comp.'].str.replace('%', '').astype(float)
df['Maize%'] = df['Maize%'].str.replace('%', '').astype(float)
df['Yield CV'] = df['Yield CV'].astype(float)
df['Flag%'] = df['Flag%'].str.replace('%', '').astype(float)

# Invert Flag% as requested (less is better, so we want higher scores for lower flags)
# We will convert percentage to a rate and subtract from 1.
df['Flag% Inverted'] = 1 - (df['Flag%'] / 100)


# Define countries and variables for the radar plot
countries_to_plot = ['Zimbabwe', 'Zambia', 'Ghana', 'Uganda']
variables = ['Years', 'Locations', 'Loc. Comp.', 'Maize%', 'Yield CV', 'Flag% Inverted']
variable_names = ['Years', 'Locations', 'Location \nCompleteness', 'Maize Area %', 'Yield CV', 'Flag Rate \n(Inverted)']


# Filter data for the selected countries
df_plot = df[df['Country'].isin(countries_to_plot)].copy()

# Normalize data for the radar plot
# For each variable, we divide by the max value in the whole dataset to keep proportions
for var in variables:
    max_val = df[var].max()
    if max_val > 0:
        df_plot[f'{var}_norm'] = df_plot[var] / max_val
    else:
        df_plot[f'{var}_norm'] = 0


def create_radar_plot(df_plot, countries_to_plot, variables, variable_names):
    # Number of variables
    N = len(variables)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Define custom colors to match the plotly default colors
    # These are the default colors from Plotly's qualitative.Plotly
    custom_colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa']  # Blue, Red, Green, Purple
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    # Set font properties
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    ax.set_xticks(angles[:-1])
    # Use full variable names but with custom alignment
    labels = ax.set_xticklabels(variable_names, fontsize=12)
    
    # Custom alignment based on position:
    # Top (0 degrees) and bottom (180 degrees) - center
    # Right side (0-90 and 270-360 degrees) - left align
    # Left side (90-270 degrees) - right align
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        # Convert angle to degrees for easier comparison
        angle_deg = angle * 180 / pi
        
        # Normalize angle to be between 0 and 360
        if angle_deg < 0:
            angle_deg += 360
        
        # Adjust alignment based on position
        if angle_deg == 0 or angle_deg == 180:  # Top and bottom
            label.set_horizontalalignment('center')
        elif 0 < angle_deg < 180:  # Right side (including top-right and bottom-right)
            label.set_horizontalalignment('left')
        else:  # Left side (including top-left and bottom-left)
            label.set_horizontalalignment('right')

    # Draw ylabels
    ax.set_ylim(0, 1)
    
    # Plot data and fill area for each country
    for i, country in enumerate(countries_to_plot):
        country_data = df_plot[df_plot['Country'] == country]
        values = country_data[[f'{var}_norm' for var in variables]].iloc[0].tolist()
        values += values[:1]  # Complete the circle
        
        color = custom_colors[i % len(custom_colors)]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=country, color=color)
        
        # Fill area with transparency
        ax.fill(angles, values, color=color, alpha=0.1)  # 10% transparency as in original
    
    # Add title
    plt.title('Country Reliability Radar Plot\n', size=14, fontweight='bold', pad=20, fontfamily='Times New Roman')
    
    # Add legend positioned closer to the radar
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    
    return fig

# Create the radar plot
fig = create_radar_plot(df_plot, countries_to_plot, variables, variable_names)

# Ensure output directory exists
output_dir = os.path.join(os.getcwd(), "Plots", "output")
os.makedirs(output_dir, exist_ok=True)

# Save the plot with high DPI
output_path = os.path.join(output_dir, "reliability_radar_plot.png")
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Radar plot saved to '{output_path}'")

# Show the plot
plt.show()