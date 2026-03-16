
"""
MODEL V5 – Phased Stress Crop Yield Model
==========================================
DIFFERENCES FROM V3:
1. Dynamic Crop Calendar V3.2: Uses the improved peak-based threshold search with
   GDD-based Silking date (imported from verify_dynamic_calendar_v3_2).
2. Phased NPP Accumulation:
   - Phase 1 (Vegetative, SOS → Silking): NPP weighted by VEGETATIVE_WEIGHT (default 0.5).
   - Phase 2 (Grain Filling, Silking → EOS): NPP weighted by REPRODUCTIVE_WEIGHT (default 1.0).
3. Critical Window Penalties (20-day window centred on Silking):
   - Heat Sterility: If avg Temp > HEAT_THRESHOLD → multiply yield by HEAT_PENALTY.
   - Drought Abortion: If avg Ws < DROUGHT_WS_THRESHOLD → multiply yield by DROUGHT_PENALTY.
4. All outputs saved to results/V5_model/ for clean organisation.
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

# --- Calendar V3.2 Configuration (mirrored from verify_dynamic_calendar_v3_2) ---
# These are re-exported here for easy reference; the actual logic uses the
# constants defined inside verify_dynamic_calendar_v3_2.py.
# If you need to change SOS/EOS thresholds, edit that file directly.

# =============================================================
# END OF TUNABLE PARAMETERS
# =============================================================


# Add current directory to path to ensure imports work if run from root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Calendar V3.2 logic
try:
    from verify_dynamic_calendar_v3_2 import calculate_dates_v3_2, refine_smoothing
    from verify_dynamic_calendar_v5 import preprocess_and_merge
    from STSG_smoothing import run_stsg_ndvi, run_stsg_fpar, get_stsg_path
    from visualize_yield_maps import visualize_yield_maps
    import visualize_stress_v5
except ImportError as e:
    print(f"Error importing V5 modules: {e}")
    print("Ensure required modules are in Model_physical/")
    sys.exit(1)


class MaizeYieldModelV5:
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

        # NDVI STSG — no longer needed in df_daily for V5 (calendar uses prepare_calendar_data)
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
        Load VI + ERA5 GADM data and run preprocess_and_merge (from
        verify_dynamic_calendar_v5) to get daily NDVI + GDD.
        Then apply NDVI smoothing (STSG if available, else iterative SG).
        Returns a dict of {pcode: df_pcode_ready}.
        """
        print("\n--- Preparing NDVI + GDD data for Dynamic Calendar V3.2 ---")

        # Load VI
        print(f"Loading VI Data: {self.vi_path}")
        df_vi = pd.read_csv(self.vi_path, parse_dates=['date'])

        # Load ERA5 GADM (has min/max temp needed for GDD)
        print(f"Loading ERA5 GADM Data: {self.era5_gadm_path}")
        df_era5 = pd.read_csv(self.era5_gadm_path, parse_dates=['date'])

        # Merge and compute GDD using the shared function
        df_cal_data = preprocess_and_merge(df_vi, df_era5)

        # Apply NDVI smoothing
        # Try STSG from Results/STSG/ first, else iterative Savitzky-Golay
        ndvi_stsg_path = get_stsg_path(self.country, "ndvi")

        applied_stsg = False
        if not os.path.exists(ndvi_stsg_path):
            print(f"  NDVI STSG file not found. Generating via STSG_smoothing module...")
            try:
                ndvi_stsg_path = run_stsg_ndvi(self.country)
            except Exception as e:
                print(f"  Error generating NDVI STSG: {e}. Falling back to SG.")
                ndvi_stsg_path = None

        if ndvi_stsg_path and os.path.exists(ndvi_stsg_path):
            print(f"Loading STSG smoothing for NDVI: {ndvi_stsg_path}")
            df_stsg = pd.read_csv(ndvi_stsg_path, parse_dates=['date'])
            df_cal_data = pd.merge(df_cal_data,
                                   df_stsg[['date', 'PCODE', 'NDVI_STSG']],
                                   on=['date', 'PCODE'], how='left')
            if 'NDVI_STSG' in df_cal_data.columns:
                df_cal_data['NDVI_smooth'] = df_cal_data.groupby('PCODE')['NDVI_STSG'].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
                applied_stsg = True

        if not applied_stsg:
            smoothed_parts = []
            for pcode in df_cal_data['PCODE'].unique():
                sub = df_cal_data[df_cal_data['PCODE'] == pcode].copy()
                sub = refine_smoothing(sub)
                smoothed_parts.append(sub)
            df_cal_data = pd.concat(smoothed_parts, ignore_index=True)

        df_cal_data['Year'] = df_cal_data['date'].dt.year

        # Build per-PCODE dict for fast lookup
        pcode_dict = {}
        for pcode, group in df_cal_data.groupby('PCODE'):
            pcode_dict[pcode] = group.sort_values('date').reset_index(drop=True)

        print(f"Calendar data ready for {len(pcode_dict)} PCODEs.\n")
        return pcode_dict

    # ------------------------------------------------------------------
    # YIELD CALCULATION  (V5 – Phased + Critical Window)
    # ------------------------------------------------------------------
    def calculate_yield_estimates(self, df_daily, df_calendar, pcode_cal_dict):
        print("Calculating Yield estimates (V5 – Phased Stress)...")

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
            if pcode not in pcode_cal_dict:
                continue
            df_pcode_cal = pcode_cal_dict[pcode]

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
                # V5 LOGIC: Dynamic dates via V3.2 calendar
                # ------------------------------------------------
                try:
                    season_results = calculate_dates_v3_2(df_pcode_cal, year, seasons_config)
                except Exception:
                    continue

                if 1 not in season_results:
                    continue

                sos_date, silking_date, eos_date = season_results[1]['dates']
                season_mode = "Dynamic"

                # Fallback to fixed calendar if dynamic detection failed
                if sos_date is None or eos_date is None:
                    season_mode = "Fixed (Fallback)"
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
                # V5: Phased NPP Accumulation
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
                # V5: Critical Window Penalty (around Silking)
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
                    'PCODE': pcode,
                    'Year': season_year,
                    'Yield_Estimated_t_ha': yield_tha,
                    'NPP_Total_gC': npp_total,
                    'Season_Mode': season_mode,
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

        # 5. Save
        output_path = os.path.join(self.output_dir, "maize_yield_estimates_v5.csv")
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # 6. Example plot
        if pcode is None:
            pcode = self.get_max_area_pcode(country)
        if pcode:
            self.plot_example(df_daily, df_results, pcode)

        return df_results

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    def plot_example(self, df_daily, df_results, pcode):
        import matplotlib.pyplot as plt
        print(f"Generating plots for {pcode}...")

        daily_subset = df_daily[df_daily['PCODE'] == pcode]
        if daily_subset.empty:
            return

        plot_path = os.path.join(self.output_dir, f"{pcode}_timeseries_v5.png")
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot 1: FPAR
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_mean'], '.',
                     label='Raw FPAR', alpha=0.3, color='gray')
        axes[0].plot(daily_subset['date'], daily_subset['FPAR_smooth'], '-',
                     label='STSG FPAR (V5)', linewidth=1.5, color='blue')
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
            res_dates = pd.to_datetime(res_subset['Year'].astype(str) + "-06-01")
            axes[2].bar(res_dates, res_subset['Yield_Estimated_t_ha'], width=100,
                        label='Yield V5 (t/ha)', color='green', alpha=0.6)
            for _, row in res_subset.iterrows():
                mid_date = pd.to_datetime(f"{int(row['Year'])}-06-01")
                label_parts = [f"{row['Yield_Estimated_t_ha']:.2f}"]
                label_parts.append(f"({row['Season_Mode'][0]})")
                if row['Heat_Penalty'] < 1.0:
                    label_parts.append("H!")
                if row['Drought_Penalty'] < 1.0:
                    label_parts.append("D!")
                axes[2].text(mid_date, row['Yield_Estimated_t_ha'],
                             "\n".join(label_parts),
                             ha='center', va='bottom', fontsize=9, color='black')

            axes[2].set_ylabel("Yield (t/ha)")
            axes[2].legend(loc='upper left')

        axes[2].set_title(f"Estimated Yield (t/ha) – {pcode}")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
    DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
    GADM_DATA_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
    CALENDAR_FILE = os.path.join(BASE_DIR, "GADM", "crop_calendar",
                                 "maize_crop_calendar_extraction.csv")
    CROP_AREA_FILE = os.path.join(BASE_DIR, "GADM", "crop_areas",
                                  "africa_crop_areas_glad_filtered.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "V5_model")
    COUNTRY = "South Africa"

    # File names
    VI_FILE = os.path.join(GADM_DATA_DIR,
                           f"{COUNTRY.replace(' ', '_')}_admin2_VI_timeseries_GADM.csv")

    model = MaizeYieldModelV5(
        data_dir=DATA_DIR,
        gadm_data_dir=GADM_DATA_DIR,
        fpar_file=f"{COUNTRY.replace(' ', '_')}_admin2_FPAR_timeseries_GLAD.csv",
        era5_new_file=f"{COUNTRY.replace(' ', '_')}_admin2_new_ERA5_timeseries.csv",
        era5_gadm_file=f"{COUNTRY.replace(' ', '_')}_admin2_ERA5_timeseries_GADM.csv",
        calendar_file=CALENDAR_FILE,
        output_dir=OUTPUT_DIR,
        country=COUNTRY,
        vi_file=VI_FILE,
        crop_area_file=CROP_AREA_FILE,
    )

    model.run_full_model(country=COUNTRY)

    # -----------------------------------------------------------------
    # Auto-run FAO Verification
    # -----------------------------------------------------------------
    import verify_fao
    print(f"\nRunning FAO Verification (V5) for {COUNTRY}...")
    verify_fao.verify_fao(country_name=COUNTRY,
                          input_results_dir=OUTPUT_DIR,
                          version='v5')

    import verify_fao_relative
    print(f"\nRunning FAO Relative Verification (V5) for {COUNTRY}...")
    verify_fao_relative.verify_fao_relative(country_name=COUNTRY,
                                            input_results_dir=OUTPUT_DIR,
                                            version='v5')



    

    # -----------------------------------------------------------------
    # Auto-run HSA Verification
    # -----------------------------------------------------------------
    import verify_hsa
    print(f"\nRunning HarvestStat Africa Verification (V5) for {COUNTRY}...")
    verify_hsa.verify_hsa(country_name=COUNTRY,
                          input_results_dir=OUTPUT_DIR,
                          version='v5')

    # -----------------------------------------------------------------
    # auto-run visualization
    # -----------------------------------------------------------------
    print(f"\nGenerating Yield Anomaly Maps for {COUNTRY}...")
    visualize_yield_maps(country=COUNTRY, version='v5')

    # -----------------------------------------------------------------
    # Auto-run Stress Visualization
    # -----------------------------------------------------------------
    # Optional: Set a specific PCODE here to visualize a specific location.
    # If None, it will automatically select the PCODE with the maximum crop area.
    TARGET_PCODE = None 
    
    print(f"\nRunning Stress Visualization (V5) for {COUNTRY}...")
    visualize_stress_v5.run_stress_visualization(COUNTRY, target_pcode=TARGET_PCODE)

    print("\nProcessing Complete.")
