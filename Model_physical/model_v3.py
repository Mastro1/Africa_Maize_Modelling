
"""
MODEL V3
DIFFERENCES FROM V1:
1. Dynamic Crop Calendar: Uses NDVI-based Start of Season (SOS) and End of Season (EOS) 
   derived from 20% threshold and 50% drop logic (imported from verify_dynamic_calendar_v3).
2. STSG Smoothing for FPAR: Replaces simple Savitzky-Golay filter with Spatio-Temporal Savitzky-Golay (STSG)
   smoothing (imported from verify_fpar_timeseries).
3. Fallback: If dynamic dates cannot be found, it falls back to the fixed Global Yield Gap Atlas (GYGA) calendar logic.
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# STRESS FACTOR TUNING PARAMETERS
# ---------------------------------------------------------
T_OPT = 25.0   # Optimal Temperature (°C)
T_SIGMA = 7.0  # Sigma (Width) of the Gaussian Stress Curve
# ---------------------------------------------------------


# Add current directory to path to ensure imports work if run from root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import V3 logic
try:
    from verify_dynamic_calendar_v3 import calculate_dynamic_dates_v3, get_search_window
    from STSG_smoothing import run_stsg_ndvi, run_stsg_fpar, get_stsg_path
except ImportError as e:
    print(f"Error importing V3 modules: {e}")
    print("Ensure verify_dynamic_calendar_v3.py and STSG_smoothing.py are in Model_physical/")
    sys.exit(1)

class MaizeYieldModelV3:
    def __init__(self, data_dir, gadm_data_dir, fpar_file, era5_new_file, era5_gadm_file, calendar_file, output_dir, country, crop_area_file=None):
        self.data_dir = data_dir
        self.gadm_data_dir = gadm_data_dir
        self.fpar_path = os.path.join(data_dir, fpar_file)
        self.era5_new_path = os.path.join(data_dir, era5_new_file)
        self.era5_gadm_path = os.path.join(gadm_data_dir, era5_gadm_file)
        self.country = country
        
        self.calendar_path = calendar_file
        self.output_dir = output_dir
        self.crop_area_path = crop_area_file
        
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_merge_data(self):
        print("Loading ERA5 Data Source 1 (New)...")
        try:
            df_era5_new = pd.read_csv(self.era5_new_path, parse_dates=['date'])
        except Exception as e:
            print(f"Error loading ERA5 Source 1: {e}")
            sys.exit(1)

        print("Loading ERA5 Data Source 2 (GADM)...")
        try:
            df_era5_gadm = pd.read_csv(self.era5_gadm_path, parse_dates=['date'])
        except Exception as e:
            print(f"Error loading ERA5 Source 2: {e}")
            sys.exit(1)

        print("Merging ERA5 datasets...")
        cols_new = ['date', 'PCODE', 'dewpoint_temperature_2m', 'surface_solar_radiation_downwards_sum']
        df_era5_new = df_era5_new[cols_new]
        
        cols_gadm = ['date', 'PCODE', 'temperature_2m', 'total_precipitation_sum']
        df_era5_gadm = df_era5_gadm[cols_gadm]

        df_daily = pd.merge(df_era5_new, df_era5_gadm, on=['date', 'PCODE'], how='inner')
        print(f"  - Merged ERA5 dataframe size: {len(df_daily)}")

        print("Loading FPAR Data...")
        try:
            df_fpar = pd.read_csv(self.fpar_path, parse_dates=['date'])
        except Exception as e:
            print(f"Error loading FPAR Data: {e}")
            sys.exit(1)
            
        print("Merging FPAR with ERA5...")
        df_daily = pd.merge(df_daily, df_fpar[['date', 'PCODE', 'FPAR_mean']], on=['date', 'PCODE'], how='left')
        
        # Load NDVI STSG Data for Dynamic Dates (from Results/STSG/)
        print("Loading NDVI STSG Data for Dynamic Dates...")
        ndvi_stsg_path = get_stsg_path(self.country, "ndvi")
        if not os.path.exists(ndvi_stsg_path):
            print(f"  NDVI STSG file not found. Generating via STSG_smoothing module...")
            try:
                ndvi_stsg_path = run_stsg_ndvi(self.country)
            except Exception as e:
                print(f"  Error generating NDVI STSG: {e}")
                ndvi_stsg_path = None

        if ndvi_stsg_path and os.path.exists(ndvi_stsg_path):
            try:
                df_ndvi = pd.read_csv(ndvi_stsg_path, parse_dates=['date'])
                cols_ndvi = ['date', 'PCODE', 'NDVI_STSG']
                if 'NDVI_STSG' not in df_ndvi.columns:
                    if 'NDVI_mean' in df_ndvi.columns:
                        cols_ndvi = ['date', 'PCODE', 'NDVI_mean']
                    else:
                        print("Warning: NDVI STSG file missing expected columns.")
                df_daily = pd.merge(df_daily, df_ndvi[cols_ndvi], on=['date', 'PCODE'], how='left')
                print(f"  - Merged NDVI STSG data.")
            except Exception as e:
                print(f"Error loading NDVI STSG: {e}")
        else:
            print("Warning: NDVI STSG unavailable. Dynamic dates may fail.")

        return df_daily

    def preprocess_fpar(self, df):
        """
        Loads pre-computed FPAR STSG from Results/STSG/.
        If the file doesn't exist, generates it via STSG_smoothing module.
        Then interpolates to daily resolution.
        """
        print("Preprocessing FPAR (loading STSG from Results/STSG/)...")

        fpar_stsg_path = get_stsg_path(self.country, "fpar")

        if not os.path.exists(fpar_stsg_path):
            print(f"  FPAR STSG file not found. Generating via STSG_smoothing module...")
            try:
                fpar_stsg_path = run_stsg_fpar(self.country)
            except Exception as e:
                print(f"  Error generating FPAR STSG: {e}")
                print("  Falling back to raw FPAR.")
                df['FPAR_smooth'] = df['FPAR_mean'].fillna(0)
                return df

        # Load and merge
        print(f"  Loading: {fpar_stsg_path}")
        df_fpar_stsg = pd.read_csv(fpar_stsg_path, parse_dates=['date'])
        df = pd.merge(df, df_fpar_stsg[['date', 'PCODE', 'FPAR_STSG']],
                      on=['date', 'PCODE'], how='left')

        # Interpolate to daily per PCODE
        print("  Interpolating FPAR STSG to daily...")
        df['FPAR_smooth'] = df.groupby('PCODE')['FPAR_STSG'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
                       .fillna(method='bfill').fillna(method='ffill').fillna(0)
        )

        return df

    def calculate_biophysical_variables(self, df):
        """
        Calculates PAR and VPD. (Same as v1)
        """
        print("Calculating Biophysical Variables (PAR, VPD)...")
        from pyrealm.core import hygro
        
        df['PAR_MJ'] = df['surface_solar_radiation_downwards_sum'] * 1.0e-6 * 0.48
        
        temp_c = df['temperature_2m'] - 273.15
        dew_c = df['dewpoint_temperature_2m'] - 273.15
        
        vp_kpa = hygro.calc_vp_sat(dew_c)
        df['VPD_kPa'] = hygro.convert_vp_to_vpd(vp_kpa, temp_c)
        df['Temp_C'] = temp_c
        
        return df
    
    def load_crop_calendar(self):
         print("Loading Crop Calendar...")
         try:
             df_cal = pd.read_csv(self.calendar_path)
             if 'FNID' in df_cal.columns:
                 df_cal = df_cal.rename(columns={'FNID': 'PCODE'})
             cols = ['PCODE', 'Maize_1_planting', 'Maize_1_harvest', 'Maize_1_endofseaso'] # Added endofseason for dyn range
             return df_cal[cols]
         except Exception as e:
            print(f"Error loading Crop Calendar: {e}")
            sys.exit(1)

    def calculate_yield_estimates(self, df_daily, df_calendar):
        print("Calculating Yield estimates (V3 - Dynamic)...")
        
        EPSILON_MAX = 2.8
        HI = 0.35
        RS = 0.18
        MC = 0.125
        C_FRAC = 0.45
        
        results = []
        pcodes = df_daily['PCODE'].unique()
        
        count = 0
        total = len(pcodes)
        
        for pcode in pcodes:
            count += 1
            if count % 10 == 0: print(f"  - Processing {count}/{total} PCODEs", end='\r')
            
            cal_row = df_calendar[df_calendar['PCODE'] == pcode]
            if cal_row.empty: continue
            
            # Helper for getting search window requires: planting, end_of_season
            f_planting = cal_row.iloc[0]['Maize_1_planting']
            f_harvest = cal_row.iloc[0]['Maize_1_harvest']
            f_end = cal_row.iloc[0]['Maize_1_endofseaso'] # Needed for dynamic search window
            
            if pd.isna(f_planting) or pd.isna(f_harvest): continue
            
            # Prepare PCODE Data
            df_pcode = df_daily[df_daily['PCODE'] == pcode].copy()
            df_pcode['Year'] = df_pcode['date'].dt.year
            df_pcode = df_pcode.set_index('date', drop=False) # Important for dyn date lookup
            
            years = df_pcode['Year'].unique()
            
            for year in years:
                # -------------------------
                # V3 LOGIC: Dynamic Dates
                # -------------------------
                
                # 1. Determine Search Window based on Fixed Calendar
                # Using imported helper
                try:
                    s_start, s_end = get_search_window(year, f_planting, f_end)
                except Exception:
                    # Fallback logic if get_search_window fails or f_end missing
                    continue
                
                # 2. Find Dynamic Dates
                # Requires 'NDVI_STSG' or 'NDVI_mean' in df_pcode (merged in load_and_merge)
                sos, eos, col_used = calculate_dynamic_dates_v3(df_pcode, s_start, s_end)
                
                season_mode = "Dynamic"
                
                # 3. Fallback to Fixed if Dynamic Not Found
                if sos is None or eos is None:
                    season_mode = "Fixed (Fallback)"
                    # Use fixed dates logic from v1
                    start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_planting - 1)
                    if f_planting > f_planting: # Logic check - cross year?
                        # Use s_start/s_end approx or recalculate standard fixed
                        pass
                    
                    # More robust fixed:
                    # If planting < harvest, same year.
                    # If planting > harvest, cross year.
                    start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_planting - 1)
                    if f_planting < f_harvest:
                         end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
                    else:
                         end_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
                    
                    season_year = end_date.year
                else:
                    start_date = sos
                    end_date = eos
                    season_year = end_date.year # Attribute to harvest year (FAO Style)
                
                # 4. Filter Season Data
                mask = (df_pcode['date'] >= start_date) & (df_pcode['date'] <= end_date)
                df_season = df_pcode[mask]
                
                if df_season.empty: continue
                
                # Quality Checks
                if end_date > df_pcode['date'].max(): continue
                expected_days = (end_date - start_date).days + 1
                if len(df_season) < expected_days * 0.95: continue
                
                # 5. Calculate Yield (Same as v1)
                df_season['Ts'] = np.exp( - (df_season['Temp_C'] - T_OPT)**2 / (2 * T_SIGMA**2) )
                df_season['Ws'] = 1.5 / (1.5 + df_season['VPD_kPa'])
                min_stress = np.minimum(df_season['Ts'], df_season['Ws'])
                
                df_season['NPP_daily'] = (
                    df_season['PAR_MJ'] * 
                    df_season['FPAR_smooth'] * 
                    EPSILON_MAX * 
                    min_stress * 
                    0.5
                )
                
                npp_total = df_season['NPP_daily'].sum()
                biomass = npp_total / C_FRAC
                partitioning = HI / ((1 + RS) * (1 - MC))
                yield_gm2 = biomass * partitioning
                yield_tha = yield_gm2 * 0.01
                
                results.append({
                    'PCODE': pcode,
                    'Year': season_year,
                    'Yield_Estimated_t_ha': yield_tha,
                    'NPP_Total_gC': npp_total,
                    'Season_Mode': season_mode,
                    'Start_Date': start_date.date(),
                    'End_Date': end_date.date()
                })
                
        return pd.DataFrame(results)

    def get_max_area_pcode(self, country):
        if self.crop_area_path is None or not os.path.exists(self.crop_area_path):
            return None
        df_area = pd.read_csv(self.crop_area_path)
        df_country = df_area[df_area['country'] == country].dropna(subset=['admin_2'])
        if df_country.empty: return None
        return df_country.groupby('PCODE')['crop_area_ha'].mean().idxmax()

    def run_full_model(self, country, pcode=None):
        df_daily = self.load_and_merge_data()
        df_daily = self.preprocess_fpar(df_daily)
        df_daily = self.calculate_biophysical_variables(df_daily)
        df_calendar = self.load_crop_calendar()
        df_results = self.calculate_yield_estimates(df_daily, df_calendar)
        
        output_path = os.path.join(self.output_dir, "maize_yield_estimates_v3.csv")
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        if pcode is None: pcode = self.get_max_area_pcode(country)
        if pcode: self.plot_example(df_daily, df_results, pcode)
        
        return df_results

    def plot_example(self, df_daily, df_results, pcode):
        import matplotlib.pyplot as plt
        print(f"Generating plots for {pcode}...")
        
        daily_subset = df_daily[df_daily['PCODE'] == pcode]
        if daily_subset.empty: return 
        
        plot_path = os.path.join(self.output_dir, f"{pcode}_timeseries_v3.png")
        # sharex=False to allow different time scales if needed, but True is usually better for alignment.
        # If yield bars (yearly) don't align well, we might need to be careful.
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot 1: FPAR
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_mean'], '.', label='Raw FPAR', alpha=0.3, color='gray')
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_smooth'], '-', label='STSG FPAR (V3)', linewidth=1.5, color='blue')
        axes[0].set_ylabel("FPAR")
        axes[0].set_title(f"FPAR (STSG): {pcode}")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Stress
        # Recalculate stress for plotting if not present (it was calc'd in yield loop but not saved to daily df globally except partially)
        # Actually daily_subset is from df_daily which has Temp_C/VPD_kPa but not Ts/Ws unless we calc them.
        # In run_full_model -> calculate_biophysical_variables adds PAR/VPD/Temp.
        # Ts/Ws are calculated in calculate_yield loop locally.
        # So we recalc here for visualization.
        ts = np.exp( - (daily_subset['Temp_C'] - T_OPT)**2 / (2 * T_SIGMA**2) )
        ws = 1.5 / (1.5 + daily_subset['VPD_kPa'])
        
        axes[1].plot(daily_subset['date'], ts, label='Temp Stress (Ts)', alpha=0.7)
        axes[1].plot(daily_subset['date'], ws, label='Water Stress (Ws)', alpha=0.7)
        axes[1].set_ylabel("Stress Factor (0-1)")
        axes[1].set_title("Biophysical Stress Factors")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Yield
        res_subset = df_results[df_results['PCODE'] == pcode]
        if not res_subset.empty:
            # Use a bar plot width of ~100 days to be visible on multi-year axis
            res_dates = pd.to_datetime(res_subset['Year'].astype(str) + "-06-01") # Mid-year
            
            axes[2].bar(res_dates, res_subset['Yield_Estimated_t_ha'], width=100, label='Yield V3 (t/ha)', color='green', alpha=0.6)
            
            # Annotate mode
            for idx, row in res_subset.iterrows():
                mid_date = pd.to_datetime(f"{int(row['Year'])}-06-01")
                axes[2].text(mid_date, row['Yield_Estimated_t_ha'], f"{row['Yield_Estimated_t_ha']:.2f}\n({row['Season_Mode'][0]})", 
                             ha='center', va='bottom', fontsize=9, color='black')
                             
            axes[2].set_ylabel("Yield (t/ha)")
            axes[2].legend(loc='upper left')
        
        axes[2].set_title(f"Estimated Yield (t/ha) - {pcode}")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
    GADM_DATA_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
    CALENDAR_FILE = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "V3_model")
    COUNTRY = "Angola"
    
    model = MaizeYieldModelV3(
        data_dir=DATA_DIR,
        gadm_data_dir=GADM_DATA_DIR,
        fpar_file=f"{COUNTRY.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv",
        era5_new_file=f"{COUNTRY.replace(' ', '_')}_admin2_new_ERA5_timeseries.csv",
        era5_gadm_file=f"{COUNTRY.replace(' ', '_')}_admin2_ERA5_timeseries_GADM.csv",
        calendar_file=CALENDAR_FILE,
        output_dir=OUTPUT_DIR,
        country=COUNTRY,
        crop_area_file=CROP_AREA_FILE
    )
    
    model.run_full_model(country=COUNTRY)
    
    # Auto-run FAO Verification
    import verify_fao
    print("\nRunning FAO Verification (V3)...")
    verify_fao.verify_fao(country_name=COUNTRY, input_results_dir=OUTPUT_DIR, version='v3')
    
    # Auto-run HSA Verification
    import verify_hsa
    print("\nRunning HarvestStat Africa Verification (V3)...")
    verify_hsa.verify_hsa(country_name=COUNTRY, input_results_dir=OUTPUT_DIR, version='v3')
