import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

PROJECT_ROOT = os.getcwd()

# List of countries to highlight
golden_cohort = ['Angola', 'Lesotho', 'Zimbabwe', 'Ethiopia', 'Zambia', 
                 'Kenya', 'South Africa', 'Tanzania', 'Nigeria']

# Read the shapefile
shapefile_path = os.path.join(PROJECT_ROOT, "GADM", "gadm41_AFR_shp", "gadm41_AFR_0_processed.shp")
africa_map = gpd.read_file(shapefile_path)

# Create a new column to identify which countries to highlight
africa_map['highlight'] = africa_map['ADMIN0'].apply(lambda x: x in golden_cohort)

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# Plot all countries in a base color
africa_map.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)

# Plot the golden cohort countries in the specified color
africa_map[africa_map['highlight']].plot(ax=ax, color='#251E5D', edgecolor='white', linewidth=0.5)

# Function to add callout labels with leader lines
def add_callout_label(ax, country_name, centroid_x, centroid_y, text_x, text_y):
    # Draw leader line from centroid to text
    ax.plot([centroid_x, text_x], [centroid_y, text_y], 
            color='black', linewidth=0.8, alpha=0.7)
    
    # Add a small circle at the centroid connection point
    ax.plot(centroid_x, centroid_y, 'o', color='#251E5D', markersize=3)
    
    # Add text in a box
    ax.text(text_x, text_y, country_name, 
            fontsize=16, 
            ha='center', 
            va='center',
            color='white',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#251E5D', edgecolor='white', alpha=0.9))

# Add callout labels for the golden cohort countries
golden_cohort_data = africa_map[africa_map['highlight']]

for idx, row in golden_cohort_data.iterrows():
    country_name = row['ADMIN0']
    centroid = row['geometry'].centroid

    # Use centroid position for labels, with a small offset for Lesotho
    if country_name == 'Lesotho':
        # Move Lesotho label slightly right and down from centroid
        text_x = centroid.x + 3
        text_y = centroid.y - 3
    else:
        # Use centroid position for all other countries
        text_x = centroid.x
        text_y = centroid.y

    add_callout_label(ax, country_name, centroid.x, centroid.y, text_x, text_y)

# Remove axes for a cleaner look
ax.set_axis_off()

# Add a title
plt.title('Golden Cohort Countries', fontsize=18, pad=20, fontweight='bold')

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(PROJECT_ROOT, "Plots", "output", "golden_cohort_map.png"), dpi=300, bbox_inches='tight')

print("Map saved as 'golden_cohort_map.png'")