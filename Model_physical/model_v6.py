"""
MODEL V6 – Phased Stress Crop Yield Model (V3.4 Dynamic Calendar)
================================================================
DIFFERENCES FROM V5:
1. Dynamic Crop Calendar V3.4: Uses the synchronized "Strict Fenced" logic.
2. Lookup Optimization: First tries to load pre-calculated results from Global_Calendar_V3_4.csv.
   If country data is missing, it runs the analysis on-the-fly using run_country_calendar_v3_4.
3. Phased NPP Accumulation:
   - Phase 1 (Vegetative, SOS → Silking): NPP weighted by VEGETATIVE_WEIGHT (default 0.5).
   - Phase 2 (Grain Filling, Silking → EOS): NPP weighted by REPRODUCTIVE_WEIGHT (default 1.0).
4. Critical Window Penalties (20-day window centred on Silking):
   - Heat Sterility: If avg Temp > HEAT_THRESHOLD → multiply yield by HEAT_PENALTY.
   - Drought Abortion: If avg Ws < DROUGHT_WS_THRESHOLD → multiply yield by DROUGHT_PENALTY.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# TUNABLE PARAMETERS  –  Change these to experiment
# =============================================================

# --- Stress Factor ---
T_OPT = 25.0          # Optimal temperature (°C)
T_SIGMA = 7.0         # Width of the Gaussian temperature stress curve

# --- Phased NPP Weights ---
VEGETATIVE_WEIGHT = 0.5    # Weight applied to NPP during vegetative phase (SOS → Silking)
REPRODUCTIVE_WEIGHT = 1.0  # Weight applied to NPP during grain-filling phase (Silking → EOS)

# --- Critical Window (around Silking) ---
CRITICAL_WINDOW_DAYS = 10       # Half-window size: total window = 2 × this (days before + after silking)
HEAT_THRESHOLD = 28.0           # Avg temperature (°C) above which heat sterility penalty applies
HEAT_PENALTY = 0.8              # Multiplier applied to final yield when heat sterility is triggered
DROUGHT_WS_THRESHOLD = 0.5      # Avg Ws below which drought abortion penalty applies
DROUGHT_PENALTY = 0.6           # Multiplier applied to final yield when drought abortion is triggered

# --- Biophysical Constants ---
EPSILON_MAX = 2.8       # Maximum Light Use Efficiency (gC/MJ PAR)
HI = 0.35               # Harvest Index
RS = 0.18               # Root-to-Shoot ratio
MC = 0.125              # Moisture Content of grain
C_FRAC = 0.45           # Carbon fraction of dry biomass

# --- Calendar V3.4 Configuration ---
# Looked up from Model_physical/Results/Global_Dynamic_Analysis_V3_4/Global_Calendar_V3_4.csv
# Fallback: run_country_calendar_v3_4 (on-the-fly)
GLOBAL_CALENDAR_RESULTS_DIR = r"Model_physical\Results\Global_Dynamic_Analysis_V3_4"
GLOBAL_CALENDAR_RESULTS_FILE = "Global_Calendar_V3_4.csv"

# =============================================================
# END OF TUNABLE PARAMETERS
# =============================================================


# Add current directory to path to ensure imports work if run from root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Calendar V3.4 logic
try:
    from verify_dynamic_calendar_v3_4 import run_country_calendar_v3_4, preprocess_and_merge, refine_smoothing
    from STSG_smoothing import run_stsg_ndvi, run_stsg_fpar, get_stsg_path
    from visualize_yield_maps import visualize_yield_maps
    import visualize_stress_v5
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure required modules are in Model_physical/")
    sys.exit(1)


class MaizeYieldModelV6:
    def __init__(self, data_dir, gadm_data_dir, fpar_file, era5_new_file,
                 era5_gadm_file, calendar_file, output_dir,
                 country, vi_file, crop_area_file=None):
        """
        Parameters
        ----------
        country : str
            Country name (used to find/generate STSG files).
        vi_file : str
            Path to the VI (NDVI) timeseries CSV used for dynamic calendar detection.
        """
        self.data_dir = data_dir
        self.gadm_data_dir = gadm_data_dir
        self.fpar_path = os.path.join(data_dir, fpar_file)
        self.era5_new_path = os.path.join(data_dir, era5_new_file)
        self.era5_gadm_path = os.path.join(gadm_data_dir, era5_gadm_file)
        self.vi_path = vi_file
        self.country = country
        self.calendar_path = calendar_file
        self.output_dir = output_dir
        self.crop_area_path = crop_area_file

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_and_merge_data(self):
        """Loads ERA5 (two sources), FPAR, and NDVI STSG data and merges them."""
        print("Loading ERA5 Data Source 1 (New – dewpoint + solar radiation)...")
        try:
            df_era5_new = pd.read_csv(self.era5_new_path, parse_dates=['date'])
        except Exception as e:
            print(f"Error loading ERA5 Source 1: {e}"); sys.exit(1)

        print("Loading ERA5 Data Source 2 (GADM – temp + precip)...")
        try:
            df_era5_gadm = pd.read_csv(self.era5_gadm_path, parse_dates=['date'])
        except Exception as e:
            print(f"Error loading ERA5 Source 2: {e}"); sys.exit(1)

        print("Merging ERA5 datasets...")
        cols_new = ['date', 'PCODE', 'dewpoint_temperature_2m',
                    'surface_solar_radiation_downwards_sum']
        cols_gadm = ['date', 'PCODE', 'temperature_2m', 'total_precipitation_sum']
        df_daily = pd.merge(df_era5_new[cols_new], df_era5_gadm[cols_gadm],
                            on=['date', 'PCODE'], how='inner')
        print(f"  - Merged ERA5 dataframe size: {len(df_daily)}")

        # FPAR
        print("Loading FPAR Data...")
        try:
            df_fpar = pd.read_csv(self.fpar_path, parse_dates=['date'])
        except Exception as e:
            print(f"Error loading FPAR Data: {e}"); sys.exit(1)

        print("Merging FPAR with ERA5...")
        df_daily = pd.merge(df_daily, df_fpar[['date', 'PCODE', 'FPAR_mean']],
                            on=['date', 'PCODE'], how='left')

        # NDVI STSG — no longer needed in df_daily for V6 (calendar uses prepare_calendar_data)
        # Kept minimal: just return the merged ERA5+FPAR

        return df_daily

    # ------------------------------------------------------------------
    # FPAR PREPROCESSING (STSG – reused from V3)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # BIOPHYSICAL VARIABLES  (same as V3)
    # ------------------------------------------------------------------
    def calculate_biophysical_variables(self, df):
        """Calculates PAR, VPD, and Temp_C."""
        print("Calculating Biophysical Variables (PAR, VPD)...")
        from pyrealm.core import hygro

        df['PAR_MJ'] = df['surface_solar_radiation_downwards_sum'] * 1.0e-6 * 0.48

        temp_c = df['temperature_2m'] - 273.15
        dew_c  = df['dewpoint_temperature_2m'] - 273.15

        vp_kpa = hygro.calc_vp_sat(dew_c)
        df['VPD_kPa'] = hygro.convert_vp_to_vpd(vp_kpa, temp_c)
        df['Temp_C'] = temp_c

        return df

    # ------------------------------------------------------------------
    # CROP CALENDAR
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # PREPARE NDVI+GDD DATA FOR DYNAMIC CALENDAR V3.2
    # ------------------------------------------------------------------
    def prepare_calendar_data(self):
        """
        Retrieves V3.4 dynamic calendar dates for the target country.
        1. Checks Global_Calendar_V3_4.csv for existing results.
        2. If missing, runs run_country_calendar_v3_4 on-the-fly.
        Returns a dict of {pcode: {year: {'SOS': ..., 'Silking': ..., 'EOS': ...}}}.
        """
        print(f"\n--- Retrieving V3.4 Dynamic Calendar for {self.country} ---")
        
        # The global calendar results are located in Model_physical/Results/Global_Dynamic_Analysis_V3_4/
        # We can derive this relative to the output_dir which is Model_physical/Results/V6_model
        results_path = os.path.join(os.path.dirname(self.output_dir), 
                                    "Global_Dynamic_Analysis_V3_4", "Global_Calendar_V3_4.csv")
        # Normalize path
        results_path = os.path.abspath(results_path)
        
        df_cal_results = None
        
        # Method 1: Global CSV Lookup
        if os.path.exists(results_path):
            print(f"Found global results CSV: {results_path}")
            df_all = pd.read_csv(results_path)
            df_country = df_all[df_all['Country'] == self.country]
            if not df_country.empty:
                print(f"  - Loaded {len(df_country)} records for {self.country} from global results.")
                df_cal_results = df_country
            else:
                print(f"  - No records found in global CSV for {self.country}.")
        else:
            print(f"Global results CSV not found at: {results_path}")

        # Method 2: On-the-fly Generation
        if df_cal_results is None:
            print(f"Running V3.4 analysis on-the-fly for {self.country}...")
            # We use the BASE_DIR logic from the main block if possible, or derive it
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            df_cal_results = run_country_calendar_v3_4(self.country, base_dir=base_dir)
            if df_cal_results is None or df_cal_results.empty:
                print(f"WARNING: No calendar results generated for {self.country}.")
                return {}

        # Convert to daily dates and build dict
        # Ensure dates are timestamps
        for col in ['SOS', 'Silking', 'EOS']:
            if col in df_cal_results.columns:
                df_cal_results[col] = pd.to_datetime(df_cal_results[col])

        # Build nested dict: {pcode: {year: {season_idx: {'SOS': ..., 'Silking': ..., 'EOS': ...}}}}
        cal_dict = {}
        for (pcode, year, season), row in df_cal_results.groupby(['PCODE', 'Year', 'Season']):
            if pcode not in cal_dict: cal_dict[pcode] = {}
            if year not in cal_dict[pcode]: cal_dict[pcode][year] = {}
            
            cal_dict[pcode][year][season] = {
                'SOS': row.iloc[0]['SOS'],
                'Silking': row.iloc[0]['Silking'],
                'EOS': row.iloc[0]['EOS']
            }

        print(f"Calendar data ready for {len(cal_dict)} PCODEs (Multi-season supported).\n")
        return cal_dict

    # ------------------------------------------------------------------
    # YIELD CALCULATION  (V6 – Phased + Critical Window)
    # ------------------------------------------------------------------
    def calculate_yield_estimates(self, df_daily, df_calendar, pcode_cal_dict):
        print("Calculating Yield estimates (V6 – Phased Stress)...")

        results = []
        pcodes = df_daily['PCODE'].unique()
        total = len(pcodes)

        for count, pcode in enumerate(pcodes, 1):
            if count % 10 == 0:
                print(f"  - Processing {count}/{total} PCODEs", end='\r')

            cal_row = df_calendar[df_calendar['PCODE'] == pcode]
            if cal_row.empty:
                continue

            f_planting = cal_row.iloc[0]['Maize_1_planting']
            f_harvest  = cal_row.iloc[0]['Maize_1_harvest']
            f_end      = cal_row.iloc[0]['Maize_1_endofseaso']
            if pd.isna(f_planting) or pd.isna(f_harvest):
                continue

            # Calendar data for this PCODE (NDVI smoothed + GDD)
            # V6: Try to retrieve dynamic dates, but fallback is allowed later
            df_pcode_cal = pcode_cal_dict.get(pcode, None)

            # Daily biophysical data for this PCODE
            df_pcode = df_daily[df_daily['PCODE'] == pcode].copy()
            df_pcode['Year'] = df_pcode['date'].dt.year
            df_pcode = df_pcode.set_index('date', drop=False)

            years = df_pcode['Year'].unique()

            # Season configuration for V3.2 calendar
            seasons_config = [{'index': 1,
                               'planting': int(f_planting),
                               'endofseason': int(f_end)}]

            for year in years:
                # ------------------------------------------------
                # V6 LOGIC: Load pre-calculated V3.4 dates (Dual Seasons supported)
                # ------------------------------------------------
                if pcode not in pcode_cal_dict or year not in pcode_cal_dict[pcode]:
                    # Fallback to fixed calendar (Single season assumed if generic fallback)
                    season_mode = "Fixed (No Entry)"
                    start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_planting - 1)
                    if f_planting < f_harvest:
                        end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
                    else:
                        end_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
                    silking_date = None
                    season_year = end_date.year
                    seasons_to_process = {1: {'SOS': start_date, 'Silking': None, 'EOS': end_date}}
                else:
                    season_mode = "Dynamic V3.4"
                    seasons_to_process = pcode_cal_dict[pcode][year]

                for season_idx, dates in seasons_to_process.items():
                    sos_date = dates['SOS']
                    silking_date = dates['Silking']
                    eos_date = dates['EOS']
                    
                    if pd.isna(sos_date) or pd.isna(eos_date):
                         season_mode_iter = "Fixed (Fallback - Null)"
                         start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_planting - 1)
                         if f_planting < f_harvest:
                             end_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
                         else:
                             end_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=f_harvest - 1)
                         silking_date = None
                         season_year = end_date.year
                    else:
                        start_date = sos_date
                        end_date = eos_date
                        season_mode_iter = season_mode
                        season_year = end_date.year

                    # Filter season slice from the biophysical data
                    mask = (df_pcode['date'] >= start_date) & (df_pcode['date'] <= end_date)
                    df_season = df_pcode[mask].copy()
                    if df_season.empty:
                        continue

                    # Quality checks
                    if end_date > df_pcode['date'].max():
                        continue
                    expected_days = (end_date - start_date).days + 1
                    if len(df_season) < expected_days * 0.95:
                        continue

                    # ------------------------------------------------
                    # Stress factors
                    # ------------------------------------------------
                    df_season['Ts'] = np.exp(-(df_season['Temp_C'] - T_OPT)**2 / (2 * T_SIGMA**2))
                    df_season['Ws'] = 1.5 / (1.5 + df_season['VPD_kPa'])
                    min_stress = np.minimum(df_season['Ts'], df_season['Ws'])

                    # ------------------------------------------------
                    # V6: Phased NPP Accumulation
                    # ------------------------------------------------
                    df_season['NPP_daily'] = (
                        df_season['PAR_MJ'] *
                        df_season['FPAR_smooth'] *
                        EPSILON_MAX *
                        min_stress *
                        0.5  # auto-trophic respiration
                    )

                    if silking_date is not None and start_date <= silking_date <= end_date:
                        # Phase 1: Vegetative (SOS → Silking)
                        mask_veg = df_season['date'] <= silking_date
                        # Phase 2: Reproductive (Silking → EOS)
                        mask_rep = df_season['date'] > silking_date

                        npp_total = (
                            (df_season.loc[mask_veg, 'NPP_daily'] * VEGETATIVE_WEIGHT).sum() +
                            (df_season.loc[mask_rep, 'NPP_daily'] * REPRODUCTIVE_WEIGHT).sum()
                        )
                    else:
                        # No silking date → flat weighting (average of the two weights)
                        avg_weight = (VEGETATIVE_WEIGHT + REPRODUCTIVE_WEIGHT) / 2.0
                        npp_total = (df_season['NPP_daily'] * avg_weight).sum()

                    # ------------------------------------------------
                    # V6: Critical Window Penalty (around Silking)
                    # ------------------------------------------------
                    heat_penalty_applied = 1.0
                    drought_penalty_applied = 1.0

                    if silking_date is not None and start_date <= silking_date <= end_date:
                        cw_start = silking_date - pd.Timedelta(days=CRITICAL_WINDOW_DAYS)
                        cw_end   = silking_date + pd.Timedelta(days=CRITICAL_WINDOW_DAYS)
                        mask_cw = (df_season['date'] >= cw_start) & (df_season['date'] <= cw_end)
                        df_cw = df_season[mask_cw]

                        if not df_cw.empty:
                            avg_temp_cw = df_cw['Temp_C'].mean()
                            avg_ws_cw   = df_cw['Ws'].mean()

                            if avg_temp_cw > HEAT_THRESHOLD:
                                heat_penalty_applied = HEAT_PENALTY
                            if avg_ws_cw < DROUGHT_WS_THRESHOLD:
                                drought_penalty_applied = DROUGHT_PENALTY

                    # ------------------------------------------------
                    # Final yield conversion
                    # ------------------------------------------------
                    biomass = npp_total / C_FRAC
                    partitioning = HI / ((1 + RS) * (1 - MC))
                    yield_gm2 = biomass * partitioning * heat_penalty_applied * drought_penalty_applied
                    yield_tha = yield_gm2 * 0.01

                    results.append({
                        'Country': self.country,
                        'PCODE': pcode,
                        'Year': season_year,
                        'Season': season_idx,
                        'Yield_Estimated_t_ha': yield_tha,
                        'NPP_Total_gC': npp_total,
                        'Season_Mode': season_mode_iter,
                        'Start_Date': start_date.date(),
                        'End_Date': end_date.date(),
                        'Silking_Date': silking_date.date() if silking_date else None,
                        'Heat_Penalty': heat_penalty_applied,
                        'Drought_Penalty': drought_penalty_applied,
                    })

        print()
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # HELPER: max-area PCODE
    # ------------------------------------------------------------------
    def get_max_area_pcode(self, country):
        if self.crop_area_path is None or not os.path.exists(self.crop_area_path):
            return None
        df_area = pd.read_csv(self.crop_area_path)
        df_country = df_area[df_area['country'] == country].dropna(subset=['admin_2'])
        if df_country.empty:
            return None
        return df_country.groupby('PCODE')['crop_area_ha'].mean().idxmax()

    # ------------------------------------------------------------------
    # MAIN RUNNER
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # MAIN RUNNER
    # ------------------------------------------------------------------
    def run_full_model(self, country, pcode=None):
        # 1. Load & merge biophysical data
        df_daily = self.load_and_merge_data()
        df_daily = self.preprocess_fpar(df_daily)
        df_daily = self.calculate_biophysical_variables(df_daily)

        # 2. Load calendar
        df_calendar = self.load_crop_calendar()

        # 3. Prepare NDVI+GDD data for dynamic calendar V3.2
        pcode_cal_dict = self.prepare_calendar_data()

        # 4. Calculate yields
        df_results = self.calculate_yield_estimates(df_daily, df_calendar, pcode_cal_dict)

        # 5. Save (Replace-on-Append logic)
        self.save_and_merge_results(df_results)

        if df_results.empty:
            print(f"WARNING: No yield results generated for {country}. Skipping plotting.")
            return df_results

        # 6. Example plot
        if pcode is None:
            pcode = self.get_max_area_pcode(country)
        if pcode:
            self.plot_example(df_daily, df_results, pcode)

        return df_results

    def save_and_merge_results(self, df_new):
        """
        Saves country results and merges them into the Global V6 CSV.
        Replaces previous records for the same country to avoid duplicates.
        """
        # Save country-specific file
        country_out = os.path.join(self.output_dir, f"maize_yield_estimates_V6_{self.country.replace(' ', '_')}.csv")
        df_new.to_csv(country_out, index=False)
        print(f"\nCountry results saved to: {country_out}")

        # Global Merge (Replace on Append)
        global_out = os.path.join(self.output_dir, "Global_Maize_Yield_V6.csv")
        
        if os.path.exists(global_out):
            df_global = pd.read_csv(global_out)
            # Remove existing rows for this country
            df_global = df_global[df_global['Country'] != self.country]
            # Append new
            df_final = pd.concat([df_global, df_new], ignore_index=True)
            print(f"Updated global results with {self.country} (Replaced previous records).")
        else:
            df_final = df_new
            print(f"Created new global results file with {self.country}.")

        df_final.to_csv(global_out, index=False)
        print(f"Global file updated: {global_out}")

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def plot_example(self, df_daily, df_results, pcode):
        import matplotlib.pyplot as plt
        print(f"Generating plots for {pcode}...")

        daily_subset = df_daily[df_daily['PCODE'] == pcode]
        if daily_subset.empty:
            print(f"  No biophysical data for {pcode}. Skipping plot.")
            return

        if df_results.empty or 'PCODE' not in df_results.columns:
            print(f"  No yield results available for plotting for {pcode}.")
            return

        plot_path = os.path.join(self.output_dir, f"{pcode}_timeseries_V6.png")
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot 1: FPAR
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_mean'], '.',
                     label='Raw FPAR', alpha=0.3, color='gray')
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_smooth'], '-',
                     label='STSG FPAR (V6)', linewidth=1.5, color='blue')
        axes[0].set_ylabel("FPAR")
        axes[0].set_title(f"FPAR (STSG): {pcode}")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Stress
        ts = np.exp(-(daily_subset['Temp_C'] - T_OPT)**2 / (2 * T_SIGMA**2))
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
            # Sort by year and season for better bar mapping
            res_subset = res_subset.sort_values(['Year', 'Season'])
            
            # Simple heuristic for plotting: map Season 1/2 to June/Nov of that year
            def get_plot_date(row):
                if row['Season'] == 1:
                    return pd.to_datetime(f"{int(row['Year'])}-06-01")
                else:
                    return pd.to_datetime(f"{int(row['Year'])}-11-01")
            
            res_dates = res_subset.apply(get_plot_date, axis=1)
            
            axes[2].bar(res_dates, res_subset['Yield_Estimated_t_ha'], width=60,
                        label='Yield V6 (t/ha)', color='green', alpha=0.6)
            
            for i, row in res_subset.iterrows():
                plot_date = get_plot_date(row)
                label_parts = [f"S{row['Season']}: {row['Yield_Estimated_t_ha']:.2f}"]
                if row['Heat_Penalty'] < 1.0:
                    label_parts.append("H!")
                if row['Drought_Penalty'] < 1.0:
                    label_parts.append("D!")
                axes[2].text(plot_date, row['Yield_Estimated_t_ha'],
                             "\n".join(label_parts),
                             ha='center', va='bottom', fontsize=8, color='black')

            axes[2].set_ylabel("Yield (t/ha)")
            axes[2].legend(loc='upper left')

        axes[2].set_title(f"Estimated Yield (t/ha) – {pcode} (Multi-season)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")

def run_model_v6_for_country(country, target_pcode=None):
    """
    Modular entry point to run Model V6 for a specific country.
    """
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
    GADM_DATA_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
    CALENDAR_FILE = os.path.join(BASE_DIR, "GADM", "crop_calendar",
                                 "maize_crop_calendar_extraction.csv")
    CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas",
                                  "africa_crop_areas_glad_filtered.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "V6_model")

    # File names
    VI_FILE = os.path.join(GADM_DATA_DIR,
                           f"{country.replace(' ', '_')}_admin2_VI_timeseries_GADM.csv")

    print(f"\n{'='*60}")
    print(f" RUNNING MODEL V6: {country}")
    print(f"{'='*60}")

    model = MaizeYieldModelV6(
        data_dir=DATA_DIR,
        gadm_data_dir=GADM_DATA_DIR,
        fpar_file=f"{country.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv",
        era5_new_file=f"{country.replace(' ', '_')}_admin2_new_ERA5_timeseries.csv",
        era5_gadm_file=f"{country.replace(' ', '_')}_admin2_ERA5_timeseries_GADM.csv",
        calendar_file=CALENDAR_FILE,
        output_dir=OUTPUT_DIR,
        country=country,
        vi_file=VI_FILE,
        crop_area_file=CROP_AREA_FILE,
    )

    df_results = model.run_full_model(country=country, pcode=target_pcode)

    # -----------------------------------------------------------------
    # Auto-run Verifications
    # -----------------------------------------------------------------
    import verify_fao_dual_season_relative as verify_fao
    print(f"\nRunning FAO Verification (V6) for {country}...")
    verify_fao.verify_fao(country_name=country,
                          input_results_dir=OUTPUT_DIR,
                          version='V6')

    import verify_hsa
    print(f"\nRunning HarvestStat Africa Verification (V6) for {country}...")
    verify_hsa.verify_hsa(country_name=country,
                          input_results_dir=OUTPUT_DIR,
                          version='V6')

    print(f"\nGenerating Yield Anomaly Maps for {country}...")
    visualize_yield_maps(country=country, version='V6')

    print(f"\nRunning Stress Visualization (V6) for {country}...")
    visualize_stress_v5.run_stress_visualization(country, target_pcode=target_pcode)

    return df_results

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Maize Yield Model V6 (Dual Seasons)")
    parser.add_argument("--country", type=str, default="Ethiopia", help="Country name")
    parser.add_argument("--pcode", type=str, default=None, help="Optional target PCODE for plotting")
    
    args = parser.parse_args()
    
    run_model_v6_for_country(args.country, target_pcode=args.pcode)
    print("\nProcessing Complete.")
