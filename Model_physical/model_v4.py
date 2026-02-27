
"""
MODEL V4
DIFFERENCES FROM V3:
1. Diffuse Fertilization Effect: Implements Zheng et al. (2020) logic to boost Light Use Efficiency (epsilon) 
   under cloudy conditions.
   - Calculates Extraterrestrial Radiation (Ra) based on Latitude and DOY.
   - Calculates Clearness Index (Kt) = ssrd / Ra.
   - Estimates Diffuse Fraction (Kd) = 1 - Kt.
   - Calculates Dynamic Epsilon = 2.8 * (1 + 0.5 * Kd).
2. Country Default: Zambia.

GOOD MODEL BUT DOESN'T WORTH IT. THE RESULTS INCREASE SLIGHTLY BUT GIVE MORE POINTS FOR DISCUSSION
"""

import pandas as pd
import numpy as np
import os
import sys
import math
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

# Import V3 logic (re-used for V4)
try:
    from verify_dynamic_calendar_v3 import calculate_dynamic_dates_v3, get_search_window
    from verify_fpar_timeseries import prepare_stsg_data, run_stsg_on_pcode
except ImportError as e:
    print(f"Error importing V3 modules: {e}")
    print("Ensure verify_dynamic_calendar_v3.py and verify_fpar_timeseries.py are in Model_physical/")
    # sys.exit(1) # Don't exit strict, might be testing

def calculate_extraterrestrial_radiation(doy, latitude_deg):
    """
    Calculates Ra (Extraterrestrial Radiation) for a specific day and latitude.
    Based on FAO-56 method.
    Returns: Ra in Joules/m^2/day (to match ERA5 ssrd).
    Vectorized for performance with pandas/numpy arrays.
    """
    # Convert latitude to radians
    lat_rad = np.deg2rad(latitude_deg)
    
    # 1. Inverse relative distance Earth-Sun (dr)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    
    # 2. Solar declination (delta)
    delta = 0.409 * np.sin((2 * np.pi * doy / 365) - 1.39)
    
    # 3. Sunset hour angle (ws)
    # x = -tan(lat) * tan(delta)
    x = -np.tan(lat_rad) * np.tan(delta)
    x = np.clip(x, -1, 1) # Ensure we don't take arccos of >1 or <-1
    ws = np.arccos(x)
    
    # 4. Calculate Ra (MJ/m^2/day)
    # Gsc is solar constant = 0.0820 MJ/m^2/min
    Gsc = 0.0820
    
    Ra_MJ = (24 * 60 / np.pi) * Gsc * dr * (
        (ws * np.sin(lat_rad) * np.sin(delta)) +
        (np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
    )
    
    # 5. Convert MJ to Joules (to match ERA5)
    Ra_Joules = Ra_MJ * 1e6
    
    return Ra_Joules

class MaizeYieldModelV4:
    def __init__(self, data_dir, gadm_data_dir, fpar_file, era5_new_file, era5_gadm_file, calendar_file, output_dir, ndvi_stsg_file, crop_area_file=None):
        self.data_dir = data_dir
        self.gadm_data_dir = gadm_data_dir
        self.fpar_path = os.path.join(data_dir, fpar_file)
        self.era5_new_path = os.path.join(data_dir, era5_new_file)
        self.era5_gadm_path = os.path.join(gadm_data_dir, era5_gadm_file)
        # NDVI STSG file needed for dynamic dates
        self.ndvi_stsg_path = os.path.join("Model_physical", "Results", ndvi_stsg_file) if not os.path.isabs(ndvi_stsg_file) else ndvi_stsg_file
        
        # Latitude Mapping File
        self.lat_mapping_path = os.path.join("Model_physical", "Input", "pcode_lat_lon_mapping.csv")

        self.calendar_path = calendar_file
        self.output_dir = output_dir
        self.crop_area_path = crop_area_file
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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
        # Check columns
        cols_new = ['date', 'PCODE', 'dewpoint_temperature_2m', 'surface_solar_radiation_downwards_sum']
        cols_gadm = ['date', 'PCODE', 'temperature_2m', 'total_precipitation_sum']
        
        # Ensure PCODEs match
        df_daily = pd.merge(
            df_era5_new[cols_new], 
            df_era5_gadm[cols_gadm], 
            on=['date', 'PCODE'], 
            how='inner'
        )
        print(f"  - Merged ERA5 dataframe size: {len(df_daily)}")

        print("Loading FPAR Data...")
        try:
            df_fpar = pd.read_csv(self.fpar_path, parse_dates=['date'])
        except Exception as e:
            print(f"Error loading FPAR Data: {e}")
            sys.exit(1)
            
        print("Merging FPAR with ERA5...")
        df_daily = pd.merge(df_daily, df_fpar[['date', 'PCODE', 'FPAR_mean']], on=['date', 'PCODE'], how='left')
        
        # Load Latitude Mapping
        print("Loading Latitude Data...")
        if os.path.exists(self.lat_mapping_path):
            df_lat = pd.read_csv(self.lat_mapping_path)
            # We only need PCODE and Latitude
            df_daily = pd.merge(df_daily, df_lat[['PCODE', 'Latitude']], on='PCODE', how='left')
            
            # Check for missing latitudes
            missing_lat = df_daily[df_daily['Latitude'].isna()]['PCODE'].unique()
            if len(missing_lat) > 0:
                print(f"Warning: Missing latitude for {len(missing_lat)} PCODEs (e.g., {missing_lat[:3]}). Radiation calc will fail for these.")
            else:
                print("  - Latitude merged successfully.")
        else:
            print(f"Error: Latitude mapping file not found at {self.lat_mapping_path}. Run generate_pcode_lat_lon.py first.")
            sys.exit(1)

        # Load NDVI STSG Data for Dynamic Dates
        print("Loading NDVI STSG Data for Dynamic Dates...")
        if os.path.exists(self.ndvi_stsg_path):
            try:
                df_ndvi = pd.read_csv(self.ndvi_stsg_path, parse_dates=['date'])
                cols_ndvi = ['date', 'PCODE', 'NDVI_STSG']
                if 'NDVI_STSG' not in df_ndvi.columns:
                     if 'NDVI_mean' in df_ndvi.columns:
                         cols_ndvi = ['date', 'PCODE', 'NDVI_mean']
                     else:
                         print("Warning: NDVI STSG file missing NDVI_STSG/NDVI_mean columns.")
                
                df_daily = pd.merge(df_daily, df_ndvi[cols_ndvi], on=['date', 'PCODE'], how='left')
                print(f"  - Merged NDVI STSG data.")
            except Exception as e:
                print(f"Error loading NDVI STSG: {e}")
        else:
            print(f"Warning: NDVI STSG file not found at {self.ndvi_stsg_path}. Dynamic dates might fail.")
            
        return df_daily

    def preprocess_fpar(self, df):
        """
        Applies STSG Smoothing to FPAR. (Same as V3)
        """
        print("Preprocessing FPAR (STSG Smoothing)...")
        # Reuse V3 logic if imported, else fallback to simple interpolation
        if 'prepare_stsg_data' in globals() and 'run_stsg_on_pcode' in globals():
             # ... (Full STSG implementation mostly copied from V3 to avoid dependency issues if import fails)
             # To keep code clean and rely on imports:
             pass 
             # Actually I should implement it or call it. The imports are at the top.
        else:
             print("STSG modules not loaded. Skipping FPAR smoothing (will use valid FPAR_mean or simple interp).")
             # Fallback simple fill
             df['FPAR_smooth'] = df['FPAR_mean'].fillna(0)
             return df

        # -- Full STSG Implementation inline --
        df_valid = df.dropna(subset=['FPAR_mean']).copy()
        
        # Quick exit if empty
        if df_valid.empty:
            df['FPAR_smooth'] = 0
            return df

        print("  - Building reference curves on valid data...")
        try:
            ref_df = prepare_stsg_data(df_valid, col="FPAR_mean")
            
            pcodes = df_valid['PCODE'].unique()
            print(f"  - Processing {len(pcodes)} PCODEs with STSG...")
            
            results_list = []
            count = 0
            
            for pcode in pcodes:
                count += 1
                if count % 20 == 0: print(f"    - {count}/{len(pcodes)}...", end='\r')
                
                try:
                    smoothed = run_stsg_on_pcode(df_valid, pcode, ref_df, col="FPAR_mean")
                    pcode_subset = df_valid[df_valid['PCODE'] == pcode].sort_values('date').copy()
                    
                    if len(smoothed) == len(pcode_subset):
                        pcode_subset['FPAR_STSG_Obs'] = smoothed
                        pcode_subset['FPAR_STSG_Obs'] = pcode_subset['FPAR_STSG_Obs'].clip(0, 1)
                    else:
                        pcode_subset['FPAR_STSG_Obs'] = pcode_subset['FPAR_mean']
                except Exception:
                    pcode_subset = df_valid[df_valid['PCODE'] == pcode].copy()
                    pcode_subset['FPAR_STSG_Obs'] = pcode_subset['FPAR_mean']
                
                results_list.append(pcode_subset[['date', 'PCODE', 'FPAR_STSG_Obs']])
                
            if results_list:
                df_stsg = pd.concat(results_list, ignore_index=True)
                df = pd.merge(df, df_stsg, on=['date', 'PCODE'], how='left')
                
                # Interpolate
                df['FPAR_smooth'] = df.groupby('PCODE')['FPAR_STSG_Obs'].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill').fillna(0)
                )
            else:
                df['FPAR_smooth'] = df['FPAR_mean'].fillna(0)
                
        except Exception as e:
             print(f"STSG Error: {e}")
             df['FPAR_smooth'] = df['FPAR_mean'].fillna(0)
             
        return df

    def calculate_biophysical_variables(self, df):
        """
        Calculates PAR, VPD, and Radiation Efficiency Factors (V4).
        """
        print("Calculating Biophysical Variables (PAR, VPD, Kd, Epsilon)...")
        from pyrealm.core import hygro
        
        # 1. PAR (Direct from ERA5)
        df['PAR_MJ'] = df['surface_solar_radiation_downwards_sum'] * 1.0e-6 * 0.48
        
        # 2. VPD
        temp_c = df['temperature_2m'] - 273.15
        dew_c = df['dewpoint_temperature_2m'] - 273.15
        vp_kpa = hygro.calc_vp_sat(dew_c)
        df['VPD_kPa'] = hygro.convert_vp_to_vpd(vp_kpa, temp_c)
        df['Temp_C'] = temp_c
        
        # -------------------------------------------------------------
        # V4 NEW LOGIC: Diffuse Fertilization Effect
        # -------------------------------------------------------------
        
        # 3. Ra (Extraterrestrial Radiation)
        if 'Latitude' not in df.columns:
            print("Error: Latitude column missing. Cannot calculate Ra.")
            sys.exit(1)
            
        # Ensure DOY
        df['doy'] = df['date'].dt.dayofyear
        
        # Calculate Ra (Joules/m^2/day)
        # We can pass vectors to the function
        df['Ra_Joules'] = calculate_extraterrestrial_radiation(df['doy'].values, df['Latitude'].values)
        
        # Avoid division by zero
        df['Ra_Joules'] = df['Ra_Joules'].replace(0, 1) # Should be fine, Ra is 0 at night results in daily sum 0? No, Ra is daily integral.
        
        # 4. Clearness Index (Kt)
        # Kt = Rs / Ra
        df['Kt'] = df['surface_solar_radiation_downwards_sum'] / df['Ra_Joules']
        df['Kt'] = df['Kt'].clip(0, 0.8) # Cap at 0.8 (Clear sky limit approx)
        df['Kt'] = df['Kt'].fillna(0.5) # Fallback for NaN
        
        # 5. Diffuse Fraction (Kd)
        # Simplified Model: Kd = 1 - Kt
        df['Diffuse_Fraction'] = 1.0 - df['Kt']
        
        # 6. Dynamic Epsilon
        # Zheng et al. 2020 approximation
        alpha = 0.5 
        epsilon_base = 2.8 
        
        df['epsilon_dynamic'] = epsilon_base * (1 + (alpha * df['Diffuse_Fraction']))
        
        return df
    
    def load_crop_calendar(self):
         print("Loading Crop Calendar...")
         try:
             df_cal = pd.read_csv(self.calendar_path)
             if 'FNID' in df_cal.columns:
                 df_cal = df_cal.rename(columns={'FNID': 'PCODE'})
             cols = ['PCODE', 'Maize_1_planting', 'Maize_1_harvest', 'Maize_1_endofseaso']
             return df_cal[cols]
         except Exception as e:
            print(f"Error loading Crop Calendar: {e}")
            sys.exit(1)

    def calculate_yield_estimates(self, df_daily, df_calendar):
        print("Calculating Yield estimates (V4 - Diffuse Enhanced)...")
        
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
            
            f_planting = cal_row.iloc[0]['Maize_1_planting']
            f_harvest = cal_row.iloc[0]['Maize_1_harvest']
            f_end = cal_row.iloc[0]['Maize_1_endofseaso']
            
            if pd.isna(f_planting) or pd.isna(f_harvest): continue
            
            df_pcode = df_daily[df_daily['PCODE'] == pcode].copy()
            df_pcode['Year'] = df_pcode['date'].dt.year
            df_pcode = df_pcode.set_index('date', drop=False)
            
            years = df_pcode['Year'].unique()
            
            for year in years:
                # V3/V4 Dynamic Dates
                try:
                    s_start, s_end = get_search_window(year, f_planting, f_end)
                except Exception:
                    continue
                
                sos, eos, col_used = calculate_dynamic_dates_v3(df_pcode, s_start, s_end)
                season_mode = "Dynamic"
                
                if sos is None or eos is None:
                    season_mode = "Fixed (Fallback)"
                    start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_planting - 1)
                    if f_planting < f_harvest:
                         end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
                    else:
                         end_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
                    season_year = end_date.year
                else:
                    start_date = sos
                    end_date = eos
                    season_year = end_date.year
                
                mask = (df_pcode['date'] >= start_date) & (df_pcode['date'] <= end_date)
                df_season = df_pcode[mask]
                
                if df_season.empty: continue
                
                if end_date > df_pcode['date'].max(): continue
                expected_days = (end_date - start_date).days + 1
                if len(df_season) < expected_days * 0.95: continue
                
                # Stress
                df_season['Ts'] = np.exp( - (df_season['Temp_C'] - T_OPT)**2 / (2 * T_SIGMA**2) )
                df_season['Ws'] = 1.5 / (1.5 + df_season['VPD_kPa'])
                min_stress = np.minimum(df_season['Ts'], df_season['Ws'])
                
                # V4 NPP Calculation: Use epsilon_dynamic
                df_season['NPP_daily'] = (
                    df_season['PAR_MJ'] * 
                    df_season['FPAR_smooth'] * 
                    df_season['epsilon_dynamic'] * # <--- V4 Change
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
        
        output_path = os.path.join(self.output_dir, "maize_yield_estimates_v4.csv")
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
        
        plot_path = os.path.join(self.output_dir, f"{pcode}_timeseries_v4.png")
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True) # Added 4th plot for Epsilon
        
        # Plot 1: FPAR
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_mean'], '.', label='Raw FPAR', alpha=0.3, color='gray')
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_smooth'], '-', label='STSG FPAR', linewidth=1.5, color='blue')
        axes[0].set_ylabel("FPAR")
        axes[0].set_title(f"FPAR (STSG): {pcode}")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Stress
        ts = np.exp( - (daily_subset['Temp_C'] - T_OPT)**2 / (2 * T_SIGMA**2) )
        ws = 1.5 / (1.5 + daily_subset['VPD_kPa'])
        axes[1].plot(daily_subset['date'], ts, label='Temp Stress (Ts)', alpha=0.7)
        axes[1].plot(daily_subset['date'], ws, label='Water Stress (Ws)', alpha=0.7)
        axes[1].set_ylabel("Stress Factor (0-1)")
        axes[1].set_title("Biophysical Stress Factors")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Epsilon Dynamic & Diffuse Fraction
        ax3b = axes[2].twinx()
        l1 = axes[2].plot(daily_subset['date'], daily_subset['epsilon_dynamic'], 'g-', label='Epsilon Dynamic')
        l2 = ax3b.plot(daily_subset['date'], daily_subset['Diffuse_Fraction'], 'r--', label='Diffuse Fraction', alpha=0.5)
        axes[2].set_ylabel("Light Use Efficiency (gC/MJ)")
        ax3b.set_ylabel("Diffuse Fraction (0-1)")
        axes[2].set_title("V4: Dynamic Efficiency vs Diffuse Fraction")
        
        # Legend
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        axes[2].legend(lns, labs, loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Yield
        res_subset = df_results[df_results['PCODE'] == pcode]
        if not res_subset.empty:
            res_dates = pd.to_datetime(res_subset['Year'].astype(str) + "-06-01") 
            axes[3].bar(res_dates, res_subset['Yield_Estimated_t_ha'], width=100, label='Yield V4 (t/ha)', color='darkgreen', alpha=0.6)
            for idx, row in res_subset.iterrows():
                mid_date = pd.to_datetime(f"{int(row['Year'])}-06-01")
                axes[3].text(mid_date, row['Yield_Estimated_t_ha'], f"{row['Yield_Estimated_t_ha']:.2f}", 
                             ha='center', va='bottom', fontsize=9, color='black')
                             
            axes[3].set_ylabel("Yield (t/ha)")
            axes[3].legend(loc='upper left')
        
        axes[3].set_title(f"Estimated Yield (t/ha) - {pcode}")
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
    GADM_DATA_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
    CALENDAR_FILE = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
    CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results")
    COUNTRY = "Zambia" # User requested Zambia
    
    # Needs STSG file name
    # Usually "Zambia_admin2_STSG_smoothed.csv" - Need to check if it exists or needs creating
    # For now assumming standard naming convention or fallback will occur
    STSG_FILE = f"{COUNTRY.replace(' ', '_')}_admin2_STSG_smoothed.csv"
    
    model = MaizeYieldModelV4(
        data_dir=DATA_DIR,
        gadm_data_dir=GADM_DATA_DIR,
        fpar_file=f"{COUNTRY.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv",
        era5_new_file=f"{COUNTRY.replace(' ', '_')}_admin2_new_ERA5_timeseries.csv",
        era5_gadm_file=f"{COUNTRY.replace(' ', '_')}_admin2_ERA5_timeseries_GADM.csv",
        calendar_file=CALENDAR_FILE,
        output_dir=OUTPUT_DIR,
        ndvi_stsg_file=STSG_FILE,
        crop_area_file=CROP_AREA_FILE
    )
    
    model.run_full_model(country=COUNTRY)
    
    # Auto-run FAO Verification
    import verify_fao
    print(f"\nRunning FAO Verification (V4) for {COUNTRY}...")
    verify_fao.verify_fao(country_name=COUNTRY, version='v4')

    # Auto-run HSA Verification
    import verify_hsa
    print("\nRunning HarvestStat Africa Verification (V4)...")
    verify_hsa.verify_hsa(country_name=COUNTRY, version='v4')
