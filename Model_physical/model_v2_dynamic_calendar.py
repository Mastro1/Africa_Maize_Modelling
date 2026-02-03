
import pandas as pd
import numpy as np
import os
import sys
from scipy.signal import savgol_filter
import warnings
import warnings
warnings.filterwarnings("ignore")

# Import Dynamic Calendar Logic
# Assuming verify_dynamic_calendar.py is in the same directory (Model_physical)
try:
    from verify_dynamic_calendar import calculate_dynamic_dates
except ImportError:
    # If specific path needed or running from different cwd, adjust path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from verify_dynamic_calendar import calculate_dynamic_dates

class MaizeYieldModel:
    def __init__(self, data_dir, fpar_file, era5_new_file, era5_gadm_file, vi_file, calendar_file, output_dir, crop_area_file=None):
        """
        Initialize the model with file paths.
        """
        self.data_dir = data_dir
        self.fpar_path = os.path.join(data_dir, fpar_file)
        self.vi_path = os.path.join(data_dir, vi_file)
        self.era5_new_path = os.path.join(data_dir, era5_new_file)
        self.era5_gadm_path = os.path.join(data_dir, era5_gadm_file)
        self.calendar_path = calendar_file
        self.output_dir = output_dir
        self.crop_area_path = crop_area_file
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_and_merge_data(self):
        """
        Loads all data sources and merges them into a single daily dataframe.
        """
        print("Loading ERA5 Data Source 1 (New)...")
        # Load ERA5 New
        try:
            df_era5_new = pd.read_csv(self.era5_new_path, parse_dates=['date'])
            print(f"  - Loaded {len(df_era5_new)} rows from Source 1")
        except Exception as e:
            print(f"Error loading ERA5 Source 1: {e}")
            sys.exit(1)

        print("Loading ERA5 Data Source 2 (GADM)...")
        # Load ERA5 GADM
        try:
            df_era5_gadm = pd.read_csv(self.era5_gadm_path, parse_dates=['date'])
            print(f"  - Loaded {len(df_era5_gadm)} rows from Source 2")
        except Exception as e:
            print(f"Error loading ERA5 Source 2: {e}")
            sys.exit(1)

        print("Merging ERA5 datasets...")
        # Merge ERA5 on date and PCODE
        # Note: Both have 'volumetric_soil_water_layer_1' and 'ADMIN_NAME'. handling suffixes.
        # We need specific columns from each.
        
        # Select columns of interest from Source 1
        cols_new = ['date', 'PCODE', 'dewpoint_temperature_2m', 'surface_solar_radiation_downwards_sum']
        df_era5_new = df_era5_new[cols_new]
        
        # Select columns of interest from Source 2
        cols_gadm = ['date', 'PCODE', 'temperature_2m', 'total_precipitation_sum']
        df_era5_gadm = df_era5_gadm[cols_gadm]

        df_daily = pd.merge(df_era5_new, df_era5_gadm, on=['date', 'PCODE'], how='inner')
        print(f"  - Merged ERA5 dataframe size: {len(df_daily)}")

        print("Loading FPAR Data...")
        # Load FPAR
        try:
            df_fpar = pd.read_csv(self.fpar_path, parse_dates=['date'])
            print(f"  - Loaded {len(df_fpar)} rows from FPAR")
        except Exception as e:
            print(f"Error loading FPAR Data: {e}")
            sys.exit(1)
            
        print("Merging FPAR with ERA5...")
        # Merge FPAR with Daily Data
        # FPAR is 8-day, so we expect many NaNs after merge for the daily timescale if we merged on exact dates.
        # However, FPAR dates usually align with a schedule.
        # We merge essentially left on ERA5 (daily) logic.
        
        df_daily = pd.merge(df_daily, df_fpar[['date', 'PCODE', 'FPAR_mean']], on=['date', 'PCODE'], how='left')
        print(f"  - Combined dataframe size (FPAR): {len(df_daily)}")
        
        print("Loading VI Data (NDVI)...")
        try:
            df_vi = pd.read_csv(self.vi_path, parse_dates=['date'])
            print(f"  - Loaded {len(df_vi)} rows from VI Data")
            # Merge NDVI
            # VI data might have many columns, we need 'NDVI_mean' (or whatever is used in verify)
            # verify uses 'NDVI_mean'.
            if 'NDVI_mean' in df_vi.columns:
                 df_daily = pd.merge(df_daily, df_vi[['date', 'PCODE', 'NDVI_mean']], on=['date', 'PCODE'], how='left')
            else:
                 print("Warning: 'NDVI_mean' not found in VI data. Dynamic calendar might fail.")
            print(f"  - Combined dataframe size (NDVI): {len(df_daily)}")
            
        except Exception as e:
            print(f"Error loading VI Data: {e}")
            sys.exit(1)

        
        return df_daily

    def preprocess_fpar(self, df):
        """
        Interpolates and smooths the FPAR timeseries for each PCODE.
        """
        print("Preprocessing FPAR (Interpolation + Smoothing)...")
        
        # Sort by PCODE and date to ensure correct time series structure
        df = df.sort_values(by=['PCODE', 'date'])
        
        # We process each PCODE group
        results = []
        
        # Get unique PCODEs
        pcodes = df['PCODE'].unique()
        total_pcodes = len(pcodes)
        
        print(f"  - Processing {total_pcodes} PCODEs...")
        
        # Use groupby for efficiency? Or iterate?
        # Iteration is safer for complex operations like SavGol if not vectorized easily,
        # but groupby apply is better.
        
        def process_group(group):
            # 1. Interpolate (Linear)
            # Limit direction both to handle gaps.
            group['FPAR_interp'] = group['FPAR_mean'].interpolate(method='linear', limit_direction='both')
            
            # 2. Smooth (Savitzky-Golay)
            # Handle short series
            if len(group) > 31:
                # Fill remaining NaNs (e.g. at start/end if interpolate didn't catch them) with 0 or ffill/bfill
                # FPAR should be 0-1.
                series = group['FPAR_interp'].fillna(method='bfill').fillna(method='ffill').fillna(0)
                
                try:
                    group['FPAR_smooth'] = savgol_filter(series, window_length=31, polyorder=2)
                except Exception:
                    # Fallback if series too short for window (though check above attempts to avoid)
                    group['FPAR_smooth'] = series
                    
                # Clip to valid range [0, 1]
                group['FPAR_smooth'] = group['FPAR_smooth'].clip(0, 1)
            else:
                 group['FPAR_smooth'] = group['FPAR_interp'].fillna(0)
                 
            return group

        # Apply the function
        # This might be slow for many groups, but robust.
        # Groupby apply
        df_processed = df.groupby('PCODE', group_keys=False).apply(process_group)
        
        print("  - FPAR Preprocessing complete.")

        return df_processed
    
    def load_crop_calendar(self):
         print("Loading Crop Calendar...")
         try:
             df_cal = pd.read_csv(self.calendar_path)
             # Check if FNID exists and rename to PCODE if needed
             if 'FNID' in df_cal.columns:
                 df_cal = df_cal.rename(columns={'FNID': 'PCODE'})
                 
             print(f"  - Loaded {len(df_cal)} rows from Calendar")
             cols = ['PCODE', 'Maize_1_planting', 'Maize_1_endofseaso']
             # Ensure PCODE is unique or handle duplicates? 
             # Assuming one row per PCODE as per standard GADM calendar structure.
             return df_cal[cols]
         except Exception as e:
            print(f"Error loading Crop Calendar: {e}")
            sys.exit(1)


    def calculate_biophysical_variables(self, df):
        """
        Calculates PAR and VPD.
        """
        print("Calculating Biophysical Variables (PAR, VPD)...")
        from pyrealm.core import hygro
        
        # 1. PAR (Photosynthetically Active Radiation)
        # surface_solar_radiation_downwards_sum is in Joules/m2. Convert to MJ, then apply 0.48 factor.
        # Formula: ssrd * 1e-6 * 0.48
        df['PAR_MJ'] = df['surface_solar_radiation_downwards_sum'] * 1.0e-6 * 0.48
        
        # 2. VPD (Vapor Pressure Deficit)
        # Inputs: temperature_2m (K), dewpoint_temperature_2m (K)
        # Convert to Celsius
        temp_c = df['temperature_2m'] - 273.15
        dew_c = df['dewpoint_temperature_2m'] - 273.15
        
        # Calculate Vapor Pressure from Dewpoint using Magnus formula implementation in pyrealm
        # vp_sat at dewpoint is the actual vapor pressure
        vp_kpa = hygro.calc_vp_sat(dew_c)
        
        # Calculate VPD using actual VP and air temperature
        df['VPD_kPa'] = hygro.convert_vp_to_vpd(vp_kpa, temp_c)
        
        # Store Temp C for Stress calculation
        df['Temp_C'] = temp_c
        
        print("  - Biophysical variables calculated.")
        return df

    def calculate_yield_estimates(self, df_daily, df_calendar):
        """
        Iterates through PCODEs and Years to calculate Yield.
        """
        print("Calculating Yield estimates...")
        
        # Constants
        EPSILON_MAX = 2.8 # gC/MJ
        HI = 0.35
        RS = 0.18
        MC = 0.125
        C_FRAC = 0.45
        
        results = []
        
        # Merge calendar info into daily dataframe could be expensive/redundant since it's static per PCODE.
        # Better to iterate by PCODE and look up calendar dates.
        
        pcodes = df_daily['PCODE'].unique()
        
        count = 0
        total = len(pcodes)
        
        for pcode in pcodes:
            count += 1
            if count % 10 == 0:
                print(f"  - Processing {count}/{total} PCODEs", end='\r')
                
            # Get calendar dates for this PCODE
            cal_row = df_calendar[df_calendar['PCODE'] == pcode]
            if cal_row.empty:
                continue
                
            planting_doy = cal_row.iloc[0]['Maize_1_planting']
            harvest_doy = cal_row.iloc[0]['Maize_1_endofseaso']
            
            if pd.isna(planting_doy) or pd.isna(harvest_doy):
                continue
                
            # Get data for this PCODE
            df_pcode = df_daily[df_daily['PCODE'] == pcode].copy()
            
            # Add DOY
            df_pcode['DOY'] = df_pcode['date'].dt.dayofyear
            df_pcode['Year'] = df_pcode['date'].dt.year
            
            # Identify Season
            # Case 1: Harvest > Planting (Same Year)
            # Case 2: Harvest < Planting (Cross Year - Harvest is in next year? Or Planting is previous?)
            # Instructions say: "Fixed dates... standard season."
            # Usually if Planting < Harvest, it's a simple range.
            # If Planting > Harvest, it implies season crosses into next year.
            
            # Simple approach: Iterate by Year.
            # If Planting < Harvest: Use current year for both.
            # If Planting > Harvest: Season starts in Year, ends in Year+1.
            
            years = df_pcode['Year'].unique()
            
            for year in years:
                # Define Start and End Dates
                start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=planting_doy - 1)
                
                if planting_doy < harvest_doy:
                    end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=harvest_doy - 1)
                    season_year = year
                else:
                    # Season crosses year boundary.
                    # e.g. Plant Oct (YEar), Harvest Feb (Year+1)
                    # We attribute yield to Harvest Year usually, or Planting Year. 
                    # "Year: The crop season year". Usually means harvest year.
                    end_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=harvest_doy - 1)
                    season_year = year + 1 # Or just year? Let's stick to start year as 'season year' for simplicity unless specified.
                    # Instructions say "Iterate through the data for each Year".
                    # Let's assume we construct the season STARTING in 'year'.
                
                # --- DYNAMIC CALENDAR INTEGRATION ---
                # Attempt to determine dynamic start/end dates for this specific season (Year + Pcode)
                # Pass the PCODE subset (df_pcode) which now has NDVI_mean joined.
                # calculate_dynamic_dates expects columns ['date', 'NDVI_mean'] in the dataframe passsed.
                
                # Optimization: df_pcode is ALL data for pcode. calculate_dynamic_dates slices internally.
                # This is efficient enough.
                
                dyn_sos, dyn_eos, used_dyn, search_start, search_end = calculate_dynamic_dates(
                    df_pcode, year, planting_doy, harvest_doy
                )
                
                if dyn_sos is not None and dyn_eos is not None:
                     start_date = dyn_sos
                     end_date = dyn_eos
                     # print(f"    -> Using Dynamic Dates: {start_date.date()} to {end_date.date()} (Season Year {year})")
                else:
                     # Fallback to Fixed if dynamic failed (e.g. missing data)
                     # print(f"    -> Dynamic failed/missing, using Fixed: {start_date.date()} to {end_date.date()}")
                     pass # start_date and end_date are already set to Fixed above

                
                # Filter daily data
                mask = (df_pcode['date'] >= start_date) & (df_pcode['date'] <= end_date)
                df_season = df_pcode[mask]
                
                if df_season.empty:
                    continue

                # Check for completeness of the season
                # If end_date is beyond the last available date in the dataset, skip this season (incomplete)
                if end_date > df_pcode['date'].max():
                    # print(f"  - Skipping incomplete season ending {end_date.date()}")
                    continue
                
                # Check for missing days (allow slight tolerance, e.g. 95%)
                expected_days = (end_date - start_date).days + 1
                if len(df_season) < expected_days * 0.95:
                    # print(f"  - Skipping season with insufficient data (Year {season_year}): {len(df_season)}/{expected_days} days")
                    continue

                
                # --- CALCULATIONS ---
                
                # 1. Temperature Stress (Ts)
                # Ts = exp( - (T - 30)^2 / (2 * 5^2) )
                df_season['Ts'] = np.exp( - (df_season['Temp_C'] - 30)**2 / (2 * 5**2) )
                
                # 2. Water Stress (Ws)
                # Ws = 1.5 / (1.5 + VPD)
                df_season['Ws'] = 1.5 / (1.5 + df_season['VPD_kPa'])
                
                # 3. NPP Daily
                # NPP = PAR * FPAR * epsilon * min(Ts, Ws) * 0.5
                # Ensure FPAR is not NaN (it should be smoothed/interpolated now)
                
                # Element-wise minimum of Ts and Ws
                min_stress = np.minimum(df_season['Ts'], df_season['Ws'])
                
                df_season['NPP_daily'] = (
                    df_season['PAR_MJ'] * 
                    df_season['FPAR_smooth'] * 
                    EPSILON_MAX * 
                    min_stress * 
                    0.5
                )
                
                # 4. Aggregation
                npp_total = df_season['NPP_daily'].sum()
                
                # Yield Formula
                # Yield (g/m2) = (Sum NPP / 0.45) * (HI / ((1+RS)*(1-MC)))
                biomass = npp_total / C_FRAC
                partitioning = HI / ((1 + RS) * (1 - MC))
                yield_gm2 = biomass * partitioning
                
                yield_tha = yield_gm2 * 0.01
                
                # Store results
                results.append({
                    'PCODE': pcode,
                    'Year': season_year,
                    'Yield_Estimated_t_ha': yield_tha,
                    'NPP_Total_gC': npp_total,
                    'Mean_FPAR_Season': df_season['FPAR_smooth'].mean(),
                    'Mean_VPD_Season': df_season['VPD_kPa'].mean()
                })
        
        print(f"\n  - Yield estimation complete for {len(results)} seasons.")
        return pd.DataFrame(results)

    def get_max_area_pcode(self, country):
        """
        Finds the PCODE with the highest crop area for the given country.
        """
        if self.crop_area_path is None or not os.path.exists(self.crop_area_path):
            print("Warning: Crop area file not provided or not found. Cannot determine max area PCODE.")
            return None
        
        print(f"Finding district with max crop area for {country}...")
        df_area = pd.read_csv(self.crop_area_path)
        df_country = df_area[df_area['country'] == country]
        df_country = df_country.dropna(subset=['admin_2'])
        
        if df_country.empty:
            print(f"Warning: No crop area data found for {country}.")
            return None
        
        # Determine max area PCODE (average across years if multiple records exist per PCODE)
        pcode_stats = df_country.groupby('PCODE')['crop_area_ha'].mean()
        max_pcode = pcode_stats.idxmax()
        
        print(f"  - System selected: {max_pcode} ({pcode_stats.max():.2f} ha)")
        return max_pcode

    def run_full_model(self, country, pcode=None):
        """
        Runs the full pipeline.
        """
        # 1. Load Data
        df_daily = self.load_and_merge_data()
        
        # 2. Preprocess FPAR
        df_daily = self.preprocess_fpar(df_daily)
        
        # 3. Calculate Biophysical metrics
        df_daily = self.calculate_biophysical_variables(df_daily)
        
        # 4. Load Calendar
        df_calendar = self.load_crop_calendar()
        
        # 5. Calculate Yield
        df_results = self.calculate_yield_estimates(df_daily, df_calendar)
        
        # 6. Save Results
        output_path = os.path.join(self.output_dir, "maize_yield_estimates.csv")
        df_results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # 7. Identify PCODE for plotting if not provided
        if pcode is None:
            pcode = self.get_max_area_pcode(country)
            
        if pcode:
            # 8. Plot Example (pcode)
            self.plot_example(df_daily, df_results, pcode)
        
        return df_results

    def plot_example(self, df_daily, df_results, pcode):
        import matplotlib.pyplot as plt
        
        print(f"Generating plots for {pcode}...")
        
        # Daily Data
        daily_subset = df_daily[df_daily['PCODE'] == pcode]
        if daily_subset.empty:
            print(f"No data found for {pcode} to plot.")
            return

        # Prepare directory
        plot_path = os.path.join(self.output_dir, f"{pcode}_timeseries.png")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot 1: FPAR
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_mean'], '.', label='Raw FPAR', alpha=0.3)
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_smooth'], '-', label='Smoothed FPAR', linewidth=1.5)
        axes[0].set_ylabel("FPAR")
        axes[0].legend()
        axes[0].set_title(f"FPAR Timeseries: {pcode}")
        
        # Plot 2: Biophysical Stress
        # We need to recalculate or store stress if it wasn't stored globally. 
        # In run_pipeline, calculate_biophysical_variables stores Temp and VPD. 
        # Stress Ts/Ws are calculated inside calculate_yield loop on the fly. 
        # Let's recalculate for plotting simple.
        
        daily_subset['Ts'] = np.exp( - (daily_subset['Temp_C'] - 30)**2 / (2 * 5**2) )
        daily_subset['Ws'] = 1.5 / (1.5 + daily_subset['VPD_kPa'])
        
        axes[1].plot(daily_subset['date'], daily_subset['Ts'], label='Temp Stress (Ts)', alpha=0.7)
        axes[1].plot(daily_subset['date'], daily_subset['Ws'], label='Water Stress (Ws)', alpha=0.7)
        axes[1].set_ylabel("Stress Factor (0-1)")
        axes[1].legend()
        axes[1].set_title("Biophysical Stress Factors")
        
        # Plot 3: Yield Estimates vs NPP
        res_subset = df_results[df_results['PCODE'] == pcode]
        
        # Create a date for plotting yield (center of year?)
        if not res_subset.empty:
            res_dates = pd.to_datetime(res_subset['Year'].astype(str) + "-07-01")
            
            axes[2].bar(res_dates, res_subset['Yield_Estimated_t_ha'], width=200, label='Yield (t/ha)', color='green', alpha=0.5)
            
            ax2 = axes[2].twinx()
            ax2.plot(res_dates, res_subset['NPP_Total_gC'], 'o-', label='Seasonal NPP', color='blue')
            ax2.set_ylabel("Total NPP (gC)")
            
            axes[2].set_ylabel("Yield (t/ha)")
            
            lines, labels = axes[2].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[2].legend(lines + lines2, labels + labels2, loc='upper left')
        
        axes[2].set_title("Estimated Yield")
        
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")



if __name__ == "__main__":
    # Configuration paths
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
    CALENDAR_FILE = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results")
    COUNTRY = "Kenya"
    # PCODE = "KEN.22.5_1" # Commented out to use default (max area)
    
    model = MaizeYieldModel(
        data_dir=DATA_DIR,
        fpar_file=f"{COUNTRY.replace(" ", "_")}_admin2_FPAR_timeseries_GLAD.csv",
        vi_file=f"{COUNTRY.replace(" ", "_")}_admin2_VI_timeseries_GADM.csv",
        era5_new_file=f"{COUNTRY.replace(" ", "_")}_admin2_new_ERA5_timeseries.csv",
        era5_gadm_file=f"{COUNTRY.replace(" ", "_")}_admin2_ERA5_timeseries_GADM.csv",
        calendar_file=CALENDAR_FILE,
        output_dir=OUTPUT_DIR,
        crop_area_file=CROP_AREA_FILE
    )
    

    # Run full model - pcode will be determined automatically
    model.run_full_model(country=COUNTRY)

    import verify_fao
    verify_fao.verify_fao(country_name=COUNTRY)