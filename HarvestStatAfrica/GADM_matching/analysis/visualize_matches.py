import geopandas as gpd
import pandas as pd
import folium
import os
import random
import matplotlib.pyplot as plt
# import seaborn as sns # Removed due to missing dependency

# --- Configuration ---
# Input Files (Relative to Project Root)
HSA_SHAPEFILE = 'HarvestStatAfrica/data/hvstat_africa_boundary_v1.0.gpkg'
GADM_SHAPEFILE_ADMIN1 = 'GADM/gadm41_AFR_shp/gadm41_AFR_1.shp'
GADM_SHAPEFILE_ADMIN2 = 'GADM/gadm41_AFR_shp/gadm41_AFR_2.shp'

# Working files (Relative to Project Root, inside analysis folder)
RESULTS_CSV = 'HarvestStatAfrica/GADM_matching/analysis/HSA_GADM_comparison.csv'
OUTPUT_MAP = 'HarvestStatAfrica/GADM_matching/analysis/verification_map_threshold.html'
OUTPUT_PLOT = 'HarvestStatAfrica/GADM_matching/analysis/overlap_distribution.png'

def generate_plots(df):
    """Generate histograms and boxplots using matplotlib."""
    if 'Spatial_Overlap_Pct' not in df.columns:
        print("Warning: Spatial_Overlap_Pct column not found in results.")
        return

    # Filter for matched spatial results
    df_spatial = df[df['Spatial_GID'].notna()]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Global Histogram
    plt.subplot(2, 1, 1)
    plt.hist(df_spatial['Spatial_Overlap_Pct'], bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=80, color='red', linestyle='--', label='Threshold (80%)')
    plt.title('Global Distribution of Spatial Overlap %')
    plt.xlabel('Overlap Percentage')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    # 2. Boxplot per Country
    # Sort countries by median overlap to highlight problematic ones
    country_medians = df_spatial.groupby('Country')['Spatial_Overlap_Pct'].median().sort_values()
    
    plt.subplot(2, 1, 2)
    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    for country in country_medians.index:
        country_data = df_spatial[df_spatial['Country'] == country]['Spatial_Overlap_Pct']
        data_to_plot.append(country_data)
        labels.append(country)
        
    plt.boxplot(data_to_plot, labels=labels, vert=True, patch_artist=True)
    plt.xticks(rotation=90)
    plt.title('Spatial Overlap Distribution by Country (Sorted by Median)')
    plt.xlabel('Country')
    plt.ylabel('Overlap Percentage')
    plt.axhline(y=80, color='red', linestyle='--', label='Threshold (80%)')
    plt.grid(axis='y', alpha=0.75)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    print(f"Distribution plot saved to: {OUTPUT_PLOT}")

def main():
    print("Loading datasets for visualization...")
    
    # Load Results
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: Results file not found: {RESULTS_CSV}")
        return
    df_results = pd.read_csv(RESULTS_CSV)
    
    # --- Generate Plots ---
    print("Generating distribution plots...")
    generate_plots(df_results)
    
    # Load Geometries 
    print("Loading geometry files (this may take a moment)...")
    gdf_hsa = gpd.read_file(HSA_SHAPEFILE)
    gdf_gadm1 = gpd.read_file(GADM_SHAPEFILE_ADMIN1).set_index('GID_1')
    gdf_gadm2 = gpd.read_file(GADM_SHAPEFILE_ADMIN2).set_index('GID_2')
    
    # Create Map centered on Africa
    print("Creating map...")
    m = folium.Map(location=[0, 20], zoom_start=4, tiles='CartoDB positron')
    
    # Create Feature Groups
    fg_strong = folium.FeatureGroup(name="Strong Matches (>= 80%)", show=True)
    fg_weak = folium.FeatureGroup(name="Weak Matches (< 80%)", show=True)
    
    total_count = 0
    weak_count = 0
    
    for _, row in df_results.iterrows():
        hsa_idx = row['HSA_Index']
        if hsa_idx not in gdf_hsa.index: continue
        
        is_strong = row.get('Is_Strong_Spatial', False)
        target_fg = fg_strong if is_strong else fg_weak
        
        # Color Coding
        hsa_color = 'green' if is_strong else 'orange'
        gadm_color = 'darkgreen' if is_strong else 'red'
        
        # 1. HSA Polygon
        hsa_geom = gdf_hsa.loc[hsa_idx].geometry
        hsa_name = f"{row['Country']} - {row['HSA_Admin1']}"
        if pd.notna(row['HSA_Admin2']):
            hsa_name += f" - {row['HSA_Admin2']}"
            
        folium.GeoJson(
            hsa_geom,
            name=f"HSA: {hsa_name}",
            style_function=lambda x, col=hsa_color: {'color': col, 'weight': 2, 'fillOpacity': 0.1},
            tooltip=f"HSA: {hsa_name} (Overlap: {row['Spatial_Overlap_Pct']:.1f}%)"
        ).add_to(target_fg)
        
        # 2. GADM Spatial Match
        gadm_gid = row['Spatial_GID']
        gadm_geom = None
        gadm_name = gadm_gid
        
        if pd.notna(gadm_gid):
            if gadm_gid in gdf_gadm1.index:
                gadm_geom = gdf_gadm1.loc[gadm_gid].geometry
                gadm_name = gdf_gadm1.loc[gadm_gid]['NAME_1']
            elif gadm_gid in gdf_gadm2.index:
                gadm_geom = gdf_gadm2.loc[gadm_gid].geometry
                gadm_name = gdf_gadm2.loc[gadm_gid]['NAME_2']
                
            if gadm_geom is not None:
                folium.GeoJson(
                    gadm_geom,
                    name=f"GADM Match: {gadm_name}",
                    style_function=lambda x, col=gadm_color: {'color': col, 'weight': 2, 'dashArray': '5, 5', 'fillOpacity': 0.05},
                    tooltip=f"Matched: {gadm_name} ({gadm_gid})"
                ).add_to(target_fg)
        
        total_count += 1
        if not is_strong:
            weak_count += 1
            
        if total_count % 100 == 0:
            print(f"Processed {total_count} records...")

    fg_strong.add_to(m)
    fg_weak.add_to(m)
    folium.LayerControl().add_to(m)
    
    m.save(OUTPUT_MAP)
    print(f"Map saved to: {OUTPUT_MAP}")
    print(f"Visualized {total_count} records ({weak_count} weak matches).")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
