#!/usr/bin/env python3
"""
Extract GAEZ v5 Maize Yield Values from Direct GeoTIFF
=====================================================

Extracts agro-climatic potential yield values for maize from the GAEZ v5 GeoTIFF
downloaded from Google Cloud and creates visualizations with administrative boundaries.

File: DATA_GAEZ-V5_MAPSET_RES02-YLD_GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif
- GAEZ v5: Version 5
- RES02-YLD: Resolution 2 Yield data  
- HP0120: High Potential 2001-2020 period
- AGERA5: AgERA5 climate data
- HIST: Historical scenario
- MAIZ: Maize crop
- HRLM: High Rainfed Low-Medium management
"""

import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class GAEZYieldExtractor:
    def __init__(self, yield_file, boundaries_path, output_dir, output_file):
        self.yield_file = yield_file
        self.boundaries_path = boundaries_path
        self.output_dir = output_dir
        self.output_file = output_file
        os.makedirs(self.output_dir, exist_ok=True)
        
        # GAEZ yield data characteristics  
        self.no_data_value = -9999  # Common GAEZ no-data value
        self.units = "t_ha"  # Yield in tonnes per hectare
        self.conversion_factor = 1000  # Convert kg/ha to t/ha
        
    def verify_files(self):
        """Verify that required files exist"""
        print("📂 Verifying input files...")
        
        if not os.path.exists(self.yield_file):
            print(f"❌ GAEZ yield file not found: {self.yield_file}")
            return False
        
        if not os.path.exists(self.boundaries_path):
            print(f"❌ Boundaries file not found: {self.boundaries_path}")
            return False
        
        print(f"✅ GAEZ yield file found: {os.path.basename(self.yield_file)}")
        print(f"✅ Boundaries file found: {os.path.basename(self.boundaries_path)}")
        return True
    
    def analyze_raster(self):
        """Analyze the GAEZ raster properties and data"""
        print(f"\n🔍 Analyzing GAEZ raster data...")
        
        try:
            with rasterio.open(self.yield_file) as src:
                print(f"📊 Raster Properties:")
                print(f"  CRS: {src.crs}")
                print(f"  Transform: {src.transform}")
                print(f"  Shape: {src.width} x {src.height}")
                print(f"  Bounds: {src.bounds}")
                print(f"  Data type: {src.dtypes[0]}")
                print(f"  No data value: {src.nodata}")
                
                # Read full data for analysis
                print(f"  Reading raster data...")
                data = src.read(1)
                
                # Handle different possible no-data values
                if src.nodata is not None:
                    self.no_data_value = src.nodata
                    valid_data = data[data != src.nodata]
                else:
                    # Check for common no-data values
                    possible_nodata = [-9999, -32768, 0]
                    for nd_val in possible_nodata:
                        test_data = data[data != nd_val]
                        if len(test_data) < len(data) * 0.9:  # If removing this value removes <90% of data
                            self.no_data_value = nd_val
                            valid_data = test_data
                            break
                    else:
                        valid_data = data[data >= 0]  # Assume negative values are no-data
                        self.no_data_value = -9999
                
                print(f"  Detected no-data value: {self.no_data_value}")
                
                if len(valid_data) > 0:
                    print(f"📈 Data Statistics:")
                    print(f"  Valid pixels: {len(valid_data):,} ({100*len(valid_data)/data.size:.1f}%)")
                    print(f"  Min yield: {np.min(valid_data)/self.conversion_factor:.1f} t/ha")
                    print(f"  Max yield: {np.max(valid_data)/self.conversion_factor:.1f} t/ha")
                    print(f"  Mean yield: {np.mean(valid_data)/self.conversion_factor:.1f} t/ha")
                    print(f"  Median yield: {np.median(valid_data)/self.conversion_factor:.1f} t/ha")
                    print(f"  Std deviation: {np.std(valid_data)/self.conversion_factor:.1f} t/ha")
                    
                    # Check for reasonable maize yield values
                    if np.max(valid_data) > 20000:
                        print(f"  ⚠️  Very high yields detected - check if units are correct")
                    if np.min(valid_data) < 0:
                        print(f"  ⚠️  Negative yields detected - may indicate data issues")
                    
                    return True
                else:
                    print(f"  ❌ No valid data found in raster!")
                    return False
                
        except Exception as e:
            print(f"❌ Error reading raster: {e}")
            return False
    
    def load_boundaries(self):
        """Load administrative boundaries"""
        print(f"\n📍 Loading administrative boundaries...")
        
        try:
            boundaries = gpd.read_file(self.boundaries_path)
            print(f"✅ Loaded {len(boundaries)} administrative regions")
            print(f"Columns: {list(boundaries.columns)}")
            
            # Fix CRS issue - set to WGS84 if None
            if boundaries.crs is None:
                print(f"Setting CRS to EPSG:4326 (was None)")
                boundaries = boundaries.set_crs('EPSG:4326')
            elif boundaries.crs != 'EPSG:4326':
                print(f"Converting CRS from {boundaries.crs} to EPSG:4326")
                boundaries = boundaries.to_crs('EPSG:4326')
            
            return boundaries
            
        except Exception as e:
            print(f"❌ Error loading boundaries: {e}")
            return None
    
    def extract_yield_values(self, boundaries):
        """Extract yield values for each administrative region"""
        print(f"\n📊 Extracting yield values for {len(boundaries)} regions...")
        print(f"Using zonal statistics to calculate average yield per administrative region...")
        
        try:
            # Perform comprehensive zonal statistics for continuous data
            stats = zonal_stats(
                boundaries,
                self.yield_file,
                stats=['count', 'min', 'max', 'mean', 'median', 'std', 'sum'],
                nodata=self.no_data_value,
                all_touched=False,  # More accurate for continuous data
                categorical=False,  # We want continuous statistics
                raster_out=False   # Don't return full raster arrays
            )
            
            print(f"✅ Completed zonal statistics")
            
            # Create results dataframe
            results = []
            
            for i, (region, stat) in enumerate(zip(boundaries.itertuples(), stats)):
                
                # Get basic region info using actual boundary file columns
                result_row = {
                    'FNID': getattr(region, 'FNID', 'Unknown'),
                    'ADMIN0': getattr(region, 'ADMIN0', 'Unknown'),
                    'ADMIN1': getattr(region, 'ADMIN1', 'Unknown'), 
                    'ADMIN2': getattr(region, 'ADMIN2', 'Unknown'),
                }
                
                # Add comprehensive yield statistics for continuous data (convert kg/ha to t/ha)
                if stat and stat['count'] > 0:
                    result_row.update({
                        # Primary yield metrics (converted to t/ha)
                        'yield_mean_t_ha': round(stat['mean'] / self.conversion_factor, 2) if stat['mean'] is not None else None,
                        'yield_median_t_ha': round(stat['median'] / self.conversion_factor, 2) if stat['median'] is not None else None,
                        'yield_min_t_ha': round(stat['min'] / self.conversion_factor, 2) if stat['min'] is not None else None,
                        'yield_max_t_ha': round(stat['max'] / self.conversion_factor, 2) if stat['max'] is not None else None,
                        'yield_std_t_ha': round(stat['std'] / self.conversion_factor, 2) if stat['std'] is not None else None,
                        'yield_sum_t': round(stat['sum'] / self.conversion_factor, 0) if stat['sum'] is not None else None,
                        # Data quality metrics
                        'pixel_count': stat['count'],
                        'cv_percent': round(100 * stat['std'] / stat['mean'], 1) if stat['std'] and stat['mean'] and stat['mean'] > 0 else None,
                        'has_data': True,
                        'data_quality': 'Good' if stat['count'] >= 10 else 'Limited' if stat['count'] >= 5 else 'Poor'
                    })
                else:
                    result_row.update({
                        'yield_mean_t_ha': None,
                        'yield_median_t_ha': None,
                        'yield_min_t_ha': None,
                        'yield_max_t_ha': None,
                        'yield_std_t_ha': None,
                        'yield_sum_t': None,
                        'pixel_count': 0,
                        'cv_percent': None,
                        'has_data': False,
                        'data_quality': 'No Data'
                    })
                
                results.append(result_row)
                
                # Progress indicator
                if (i + 1) % 200 == 0:
                    print(f"  Processed {i + 1}/{len(boundaries)} regions...")
            
            results_df = pd.DataFrame(results)
            
            # Summary statistics
            valid_regions = results_df[results_df['has_data'] == True]
            print(f"\n📈 Extraction Summary:")
            print(f"Total regions: {len(results_df)}")
            print(f"Regions with yield data: {len(valid_regions)} ({100*len(valid_regions)/len(results_df):.1f}%)")
            
            if len(valid_regions) > 0:
                yield_stats = valid_regions['yield_mean_t_ha'].describe()
                print(f"\nYield Statistics (t/ha):")
                print(f"  Min: {yield_stats['min']:.2f}")
                print(f"  Max: {yield_stats['max']:.2f}")
                print(f"  Mean: {yield_stats['mean']:.2f}")
                print(f"  Median: {yield_stats['50%']:.2f}")
            
            return results_df
            
        except Exception as e:
            print(f"❌ Error in extraction: {e}")
            return None
    
    def save_results(self, results_df):
        """Save extraction results to CSV"""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_file)
        
        try:
            results_df.to_csv(output_path, index=False)
            print(f"💾 Results saved: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None
    
    def create_raster_visualization(self):
        """Create a visualization of the raw raster data with boundaries"""
        print("🗺️ Creating raster visualization with boundaries...")
        
        try:
            # Read raster data
            with rasterio.open(self.yield_file) as src:
                raster_data = src.read(1)
                raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                
                # Mask no-data values
                masked_data = np.ma.masked_where(
                    raster_data == self.no_data_value, 
                    raster_data
                )
            
            # Load boundaries
            boundaries = self.load_boundaries()
            if boundaries is None:
                return None
            
            # Create matplotlib visualization
            fig, ax = plt.subplots(1, 1, figsize=(15, 12))
            
            # Plot raster
            im = ax.imshow(
                masked_data, 
                extent=raster_extent,
                cmap='YlOrRd',
                alpha=0.8,
                aspect='auto'
            )
            
            # Plot boundaries
            boundaries.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Maize Yield (t/ha)')
            
            # Convert colorbar labels to t/ha
            tick_locs = cbar.get_ticks()
            tick_labels = [f'{tick/self.conversion_factor:.1f}' for tick in tick_locs]
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(tick_labels)
            
            # Formatting
            ax.set_title(
                'GAEZ v5 Maize Agro-climatic Potential Yield\n'
                'AgERA5, Historical, High Rainfed Low-Medium Management',
                fontsize=14, pad=20
            )
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Set extent to Africa
            ax.set_xlim(-20, 55)
            ax.set_ylim(-35, 38)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "gaez_maize_yield_raster_map.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🗺️ Raster map saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"❌ Error creating raster visualization: {e}")
            return None
    
    def create_yield_map(self, results_df, boundaries):
        """Create an interactive map of yield values by administrative region"""
        print("🗺️ Creating interactive yield map...")
        
        try:
            # Merge results with boundaries for mapping (only yield columns, ADMIN columns already in boundaries)
            boundaries_with_yield = boundaries.merge(
                results_df[['FNID', 'yield_mean_t_ha', 'data_quality']],
                left_on='FNID',
                right_on='FNID',
                how='left'
            )
            
            # Create plotly map (using choropleth instead of deprecated choropleth_mapbox)
            fig = px.choropleth(
                boundaries_with_yield,
                geojson=boundaries_with_yield.geometry,
                locations=boundaries_with_yield.index,
                color='yield_mean_t_ha',
                hover_name='ADMIN2',
                hover_data={
                    'FNID': True,
                    'ADMIN0': True,
                    'ADMIN1': True,
                    'yield_mean_t_ha': ':.2f',
                    'data_quality': True
                },
                color_continuous_scale='YlOrRd',
                title="GAEZ v5 Maize Yield Potential by Administrative Region<br>AgERA5, Historical, High Rainfed Low-Medium Management (t/ha)"
            )
            
            fig.update_layout(
                height=700,
                width=1200,
                coloraxis_colorbar_title_text="Mean Yield (t/ha)"
            )
            
            # Update geo layout for better Africa view
            fig.update_geos(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular',
                center=dict(lat=0, lon=20),
                lataxis_range=[-35, 38],
                lonaxis_range=[-20, 55]
            )
            
            # Save map
            map_path = os.path.join(self.output_dir, "gaez_maize_yield_admin_map.html")
            fig.write_html(map_path)
            print(f"🗺️ Interactive admin map saved: {map_path}")
            
            return map_path
            
        except Exception as e:
            print(f"❌ Error creating yield map: {e}")
            return None
    
    def create_summary_plots(self, results_df):
        """Create summary plots of the yield data"""
        print("📊 Creating summary plots...")
        
        try:
            valid_data = results_df[results_df['has_data'] == True]
            
            if len(valid_data) == 0:
                print("⚠️ No valid data for plotting")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Yield distribution histogram
            ax1 = axes[0, 0]
            valid_data['yield_mean_t_ha'].hist(bins=40, ax=ax1, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Mean Yield (t/ha)')
            ax1.set_ylabel('Number of Regions')
            ax1.set_title('Distribution of Mean Yield Values')
            ax1.grid(True, alpha=0.3)
            
            # 2. Top 20 regions by yield
            ax2 = axes[0, 1]
            top_regions = valid_data.nlargest(20, 'yield_mean_t_ha')
            y_pos = range(len(top_regions))
            ax2.barh(y_pos, top_regions['yield_mean_t_ha'], color='orange')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f"{row['ADMIN2'][:15]}, {row['ADMIN0'][:10]}" 
                                for _, row in top_regions.iterrows()], fontsize=8)
            ax2.set_xlabel('Mean Yield (t/ha)')
            ax2.set_title('Top 20 Regions by Yield')
            
            # 3. Yield by country (top 15)
            ax3 = axes[1, 0]
            country_yields = valid_data.groupby('ADMIN0')['yield_mean_t_ha'].agg(['mean', 'count']).reset_index()
            country_yields = country_yields[country_yields['count'] >= 3]  # At least 3 regions
            top_countries = country_yields.nlargest(15, 'mean')
            
            bars = ax3.bar(range(len(top_countries)), top_countries['mean'], color='green', alpha=0.7)
            ax3.set_xticks(range(len(top_countries)))
            ax3.set_xticklabels(top_countries['ADMIN0'], rotation=45, ha='right')
            ax3.set_ylabel('Mean Yield (t/ha)')
            ax3.set_title('Average Yield by Country (Top 15)')
            
            # 4. Data quality distribution
            ax4 = axes[1, 1]
            quality_counts = results_df['data_quality'].value_counts()
            colors = {'Good': 'green', 'Limited': 'orange', 'Poor': 'red', 'No Data': 'gray'}
            bar_colors = [colors.get(x, 'blue') for x in quality_counts.index]
            
            ax4.bar(quality_counts.index, quality_counts.values, color=bar_colors, alpha=0.7)
            ax4.set_ylabel('Number of Regions')
            ax4.set_title('Data Quality Distribution')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plots
            plot_path = os.path.join(self.output_dir, "gaez_yield_summary_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Summary plots saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"❌ Error creating plots: {e}")
            return None
    
    def run_extraction(self):
        """Main extraction function"""
        print("🌾 GAEZ v5 Direct Maize Yield Extraction")
        print("=" * 50)
        print("Source: GAEZ v5 Google Cloud GeoTIFF")
        print("Crop: Maize")
        print("Climate: AgERA5, Historical")
        print("Management: High Rainfed Low-Medium")
        
        # Verify files exist
        if not self.verify_files():
            return None
        
        # Analyze raster
        if not self.analyze_raster():
            return None
        
        # Load boundaries
        boundaries = self.load_boundaries()
        if boundaries is None:
            return None
        
        # Extract yield values
        results_df = self.extract_yield_values(boundaries)
        if results_df is None:
            return None
        
        # Save results
        output_path = self.save_results(results_df)
        if not output_path:
            return None
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        self.create_raster_visualization()
        self.create_yield_map(results_df, boundaries)
        self.create_summary_plots(results_df)
        
        print(f"\n✅ Extraction completed successfully!")
        print(f"Results: {output_path}")
        print(f"\nDataset Info:")
        print(f"- Source: GAEZ v5 from Google Cloud")
        print(f"- Crop: Maize")
        print(f"- Climate: AgERA5 Historical")
        print(f"- Management: High Rainfed Low-Medium")
        print(f"- Units: t/ha (tonnes per hectare)")
        print(f"- Extraction method: Zonal statistics (mean per admin region)")
        print(f"- Conversion: Original kg/ha divided by 1000")
        
        return output_path

if __name__ == "__main__":

    # GAEZ Yield Potential HSA
    extractor = GAEZYieldExtractor(
    yield_file = "GAEZ/DATA_GAEZ-V5_MAPSET_RES02-YLD_GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif",
    boundaries_path = "HarvestStatAfrica/data/hvstat_africa_boundary_v1.0.gpkg",
    output_dir = "HarvestStatAfrica/yield_potential",
    output_file = "gaez_maize_yield_potential_HSA.csv")
    extractor.run_extraction()

    # GAEZ Yield Potential Admin2
    extractor = GAEZYieldExtractor(
    yield_file = "GAEZ/DATA_GAEZ-V5_MAPSET_RES02-YLD_GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif",
    boundaries_path = "GADM/gadm41_AFR_shp/gadm41_AFR_2_processed.shp",
    output_dir = "GADM/yield_potential",
    output_file = "gaez_maize_yield_potential_admin2.csv")
    extractor.run_extraction()

    # GAEZ Yield Potential Admin1
    print("For prediction admin1")
    extractor = GAEZYieldExtractor(
    yield_file = "GAEZ/DATA_GAEZ-V5_MAPSET_RES02-YLD_GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif",
    boundaries_path = "GADM/gadm41_AFR_shp/gadm41_AFR_1_processed.shp",
    output_dir = "GADM/yield_potential",
    output_file = "gaez_maize_yield_potential_admin1.csv")
    extractor.run_extraction()

    # Concat the two csv files
    df_admin1 = pd.read_csv("GADM/yield_potential/gaez_maize_yield_potential_admin1.csv")
    df_admin2 = pd.read_csv("GADM/yield_potential/gaez_maize_yield_potential_admin2.csv")
    df_merged = pd.concat([df_admin1, df_admin2])
    df_merged.to_csv("GADM/yield_potential/gaez_maize_yield_potential.csv", index=False)