"""
MODEL V7 – Optuna-Based Multi-Parameter Optimization
====================================================
Uses Optuna TPE (Tree-structured Parzen Estimator) for intelligent parameter search.
Stores results in a SQLite database (optuna_study.db) for persistence and dashboard analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import optuna
from pyrealm.core import hygro

warnings.filterwarnings("ignore")

# =============================================================
# FIXED CONSTANTS & SEARCH RANGES
# =============================================================
T_SIGMA = 7.0
CRITICAL_WINDOW_DAYS = 15 # Updated by user
HEAT_THRESHOLD = 28.0
DROUGHT_WS_THRESHOLD = 0.5
EPSILON_MAX = 2.8
HI = 0.35
RS = 0.18
MC = 0.125
C_FRAC = 0.45

# --- Paths ---
BASE_DIR = r"c:\Users\MassimoPoretti\Documents\Python Projects\Africa_Maize_Modelling"
DATA_DIR = os.path.join(BASE_DIR, "Model_physical", "Input")
GADM_DATA_DIR = os.path.join(BASE_DIR, "RemoteSensing", "GADM", "extractions")
CALENDAR_PATH = os.path.join(BASE_DIR, "GADM", "crop_calendar", "maize_crop_calendar_extraction.csv")
CROP_AREA_PATH = os.path.join(BASE_DIR, "GADM", "crop_areas", "africa_crop_areas_glad_filtered.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Model_physical", "Results", "V7_model_optimized")

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Imports for specialized logic
try:
    from verify_dynamic_calendar_v3_4 import run_country_calendar_v3_4
    from STSG_smoothing import run_stsg_fpar, get_stsg_path
    import verify_fao_dual_season_optimization as verify_fao
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# =============================================================
# CORE CLASSES
# =============================================================

class SeasonSlice:
    """Memory-cached growing season data for ultra-fast calculation."""
    def __init__(self, pcode, year, season_idx, dates, df_s):
        self.pcode = pcode
        self.year = int(year)
        self.season_idx = season_idx
        self.sos_date = dates['SOS']
        self.silking_date = dates['Silking']
        self.eos_date = dates['EOS']
        
        # NumPy conversion
        self.temp = df_s['Temp_C'].values
        self.vpd = df_s['VPD_kPa'].values
        self.par = df_s['PAR_MJ'].values
        self.fpar = df_s['FPAR_smooth'].values
        self.dates = df_s['date'].values
        
        # Pre-calculated masks (dynamic ones removed to allow optimization)
        if self.silking_date is not None and self.sos_date <= self.silking_date <= self.eos_date:
            self.mask_veg = (self.dates <= np.datetime64(self.silking_date))
            self.mask_rep = (self.dates > np.datetime64(self.silking_date))
        else:
            self.mask_veg = None
            self.mask_rep = None

class MaizeYieldModelV7:
    def __init__(self, country):
        self.country = country
        self.country_clean = country.replace(" ", "_").replace("'","_")
        self.slices = []
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def prepare_data(self):
        """Loads files once and stores optimized 'SeasonSlice' objects in memory."""
        print(f"\n--- [1/2] Preparing Data Caches for {self.country} ---")
        
        # 1. Load Biophysical Data
        fpar_file = os.path.join(DATA_DIR, f"{self.country_clean}_admin2_FPAR_timeseries_GLAD.csv")
        if not os.path.exists(fpar_file):
            print("Data for fPAR admin2 doesn't exist. Taking admin 1 instead")
            fpar_file = os.path.join(DATA_DIR, f"{self.country_clean}_admin1_FPAR_timeseries_GLAD.csv")

        era5_new_file = os.path.join(DATA_DIR, f"{self.country_clean}_admin2_new_ERA5_timeseries.csv")
        if not os.path.exists(era5_new_file):
            print("Data for era5_new admin2 doesn't exist. Taking admin 1 instead")
            era5_new_file = os.path.join(DATA_DIR, f"{self.country_clean}_admin1_new_ERA5_timeseries.csv")

        era5_gadm_file = os.path.join(GADM_DATA_DIR, f"{self.country_clean}_admin2_ERA5_timeseries_GADM.csv")
        if not os.path.exists(era5_gadm_file):
            print("Data for era5 admin2 doesn't exist. Taking admin 1 instead")
            era5_gadm_file = os.path.join(GADM_DATA_DIR, f"{self.country_clean}_admin1_ERA5_timeseries_GADM.csv")
        
        df_era5_new = pd.read_csv(era5_new_file, parse_dates=['date'])
        df_era5_gadm = pd.read_csv(era5_gadm_file, parse_dates=['date'])
        
        df_daily = pd.merge(
            df_era5_new[['date', 'PCODE', 'dewpoint_temperature_2m', 'surface_solar_radiation_downwards_sum']],
            df_era5_gadm[['date', 'PCODE', 'temperature_2m']],
            on=['date', 'PCODE'], how='inner'
        )
        
        df_fpar = pd.read_csv(fpar_file, parse_dates=['date'])
        df_daily = pd.merge(df_daily, df_fpar[['date', 'PCODE', 'FPAR_mean']], on=['date', 'PCODE'], how='left')
        
        # 2. STSG FPAR
        fpar_stsg_path = get_stsg_path(self.country, "fpar")
        if not os.path.exists(fpar_stsg_path):
            print("  Generating FPAR STSG...")
            fpar_stsg_path = run_stsg_fpar(self.country)
        df_fpar_stsg = pd.read_csv(fpar_stsg_path, parse_dates=['date'])
        df_daily = pd.merge(df_daily, df_fpar_stsg[['date', 'PCODE', 'FPAR_STSG']], on=['date', 'PCODE'], how='left')
        df_daily['FPAR_smooth'] = df_daily.groupby('PCODE')['FPAR_STSG'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill').fillna(0)
        )
        
        # 3. Biophysical Vars
        df_daily['PAR_MJ'] = df_daily['surface_solar_radiation_downwards_sum'] * 1.0e-6 * 0.48
        df_daily['Temp_C'] = df_daily['temperature_2m'] - 273.15
        dew_c = df_daily['dewpoint_temperature_2m'] - 273.15
        vp_kpa = hygro.calc_vp_sat(dew_c)
        df_daily['VPD_kPa'] = hygro.convert_vp_to_vpd(vp_kpa, df_daily['Temp_C'])
        
        # 4. Calendar Dict
        pcode_cal_dict = self._get_calendar_dict()
        df_calendar = pd.read_csv(CALENDAR_PATH)
        if 'FNID' in df_calendar.columns: df_calendar = df_calendar.rename(columns={'FNID': 'PCODE'})
        
        # 5. Filter Districts: Must be in both calendar and crop area
        df_cal_pcodes = df_calendar['PCODE'].unique()
        df_area = pd.read_csv(CROP_AREA_PATH)
        df_area_country = df_area[df_area['country'] == self.country]
        crop_area_pcodes = df_area_country['PCODE'].unique()
        valid_pcodes = set(df_cal_pcodes).intersection(set(crop_area_pcodes))
        
        pcodes = [p for p in df_daily['PCODE'].unique() if p in valid_pcodes]
        print(f"  Extracting season slices for {len(pcodes)} districts (Filtered by calendar & crop area)...")
        
        for pcode in pcodes:
            cal_row = df_calendar[df_calendar['PCODE'] == pcode]
            if cal_row.empty: continue
            
            f_planting = cal_row.iloc[0]['Maize_1_planting']
            f_harvest = cal_row.iloc[0]['Maize_1_harvest']
            if pd.isna(f_planting) or pd.isna(f_harvest): continue
            
            df_pcode = df_daily[df_daily['PCODE'] == pcode].copy()
            df_pcode['Year'] = df_pcode['date'].dt.year
            
            for year in df_pcode['Year'].unique():
                if pcode in pcode_cal_dict and year in pcode_cal_dict[pcode]:
                    seasons_to_proc = pcode_cal_dict[pcode][year]
                else:
                    start_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(f_planting) - 1)
                    end_date = start_date + pd.Timedelta(days=(int(f_harvest) - int(f_planting)) % 365)
                    seasons_to_proc = {1: {'SOS': start_date, 'Silking': None, 'EOS': end_date}}
                
                for s_idx, dates in seasons_to_proc.items():
                    if pd.isna(dates['SOS']) or pd.isna(dates['EOS']): continue
                    df_s = df_pcode[(df_pcode['date'] >= dates['SOS']) & (df_pcode['date'] <= dates['EOS'])]
                    if df_s.empty or len(df_s) < ((dates['EOS'] - dates['SOS']).days + 1) * 0.90: continue
                    self.slices.append(SeasonSlice(pcode, year, s_idx, dates, df_s))
        
        print(f"  Prepared {len(self.slices)} memory-resident season slices.")

    def _get_calendar_dict(self):
        cal_path = os.path.join(BASE_DIR, "Model_physical", "Results", "Global_Dynamic_Analysis_V3_4", "Global_Calendar_V3_4.csv")
        if os.path.exists(cal_path):
            df_all = pd.read_csv(cal_path)
            df_country = df_all[df_all['Country'] == self.country.replace(" ", "_")]
        else:
            df_country = run_country_calendar_v3_4(self.country, base_dir=BASE_DIR)
        
        if df_country is None or df_country.empty: return {}
        
        for c in ['SOS', 'Silking', 'EOS']: df_country[c] = pd.to_datetime(df_country[c])
        d = {}
        for (pc, yr, sn), row in df_country.groupby(['PCODE', 'Year', 'Season']):
            if pc not in d: d[pc] = {}
            if yr not in d[pc]: d[pc][yr] = {}
            d[pc][yr][sn] = {'SOS': row.iloc[0]['SOS'], 'Silking': row.iloc[0]['Silking'], 'EOS': row.iloc[0]['EOS']}
        return d

# =============================================================
# OPTIMIZATION ENGINE
# =============================================================

def fast_yield_engine(slices, params_dict, year_align='harvest'):
    """NumPy-based yield engine for ultra-fast iterations.
    
    Parameters
    ----------
    year_align : str
        'harvest' (default) → Year = EOS year.  'plant' → Year = calendar anchor year.
    """
    t_opt = params_dict['T_OPT']
    veg_w = params_dict['VEGETATIVE_WEIGHT']
    rep_w = params_dict['REPRODUCTIVE_WEIGHT']
    heat_p = params_dict['HEAT_PENALTY']
    drought_p = params_dict['DROUGHT_PENALTY']
    cw_days = params_dict.get('CRITICAL_WINDOW_DAYS', 15)
    
    results = []
    for s in slices:
        ts = np.exp(-(s.temp - t_opt)**2 / (2 * T_SIGMA**2))
        f_vpd = 1.5 / (1.5 + s.vpd)
        ws = f_vpd
        
        total_stress = np.minimum(ts, ws)
            
        npp_daily = s.par * s.fpar * EPSILON_MAX * total_stress * 0.5
        
        if s.mask_veg is not None:
            npp_total = (npp_daily[s.mask_veg] * veg_w).sum() + (npp_daily[s.mask_rep] * rep_w).sum()
            
            # Dynamic Critical Window Mask
            cw_start = np.datetime64(s.silking_date - pd.Timedelta(days=cw_days))
            cw_end = np.datetime64(s.silking_date + pd.Timedelta(days=cw_days))
            mask_cw = (s.dates >= cw_start) & (s.dates <= cw_end)
            
            if mask_cw.any():
                hp = heat_p if s.temp[mask_cw].mean() > HEAT_THRESHOLD else 1.0
                dp = drought_p if ws[mask_cw].mean() < DROUGHT_WS_THRESHOLD else 1.0
            else:
                hp, dp = 1.0, 1.0
        else:
            npp_total = npp_daily.sum() * (veg_w + rep_w) / 2.0
            hp, dp = 1.0, 1.0
            
        y_tha = (npp_total / C_FRAC) * (HI / ((1 + RS) * (1 - MC))) * hp * dp * 0.01
        out_year = s.year if year_align == 'plant' else s.eos_date.year
        results.append({
            'PCODE': s.pcode, 
            'Year': out_year, 
            'Season': s.season_idx, 
            'Yield_Estimated_t_ha': y_tha,
            'Start_Date': s.sos_date,
            'End_Date': s.eos_date
        })
        
    return pd.DataFrame(results)

def objective(trial, country, slices, metric='correlation', year_align='harvest'):
    """Optuna objective function."""
    params = {
        'T_OPT': trial.suggest_float('T_OPT', 24.0, 30.0, step=2.0),
        'VEGETATIVE_WEIGHT': trial.suggest_float('VEGETATIVE_WEIGHT', 0.1, 0.6, step=0.1),
        'REPRODUCTIVE_WEIGHT': trial.suggest_float('REPRODUCTIVE_WEIGHT', 0.6, 1, step=0.1),
        'HEAT_PENALTY': trial.suggest_float('HEAT_PENALTY', 0.5, 1.0, step=0.1),
        'DROUGHT_PENALTY': trial.suggest_float('DROUGHT_PENALTY', 0.4, 1.0, step=0.1),
        'CRITICAL_WINDOW_DAYS': trial.suggest_int('CRITICAL_WINDOW_DAYS', 10, 20, step=5)
    }
    
    df_results = fast_yield_engine(slices, params, year_align=year_align)
    if df_results.empty: return -1.0
    
    try:
        _, corr_ker, r2_ker = verify_fao.verify_fao(country, df_model=df_results, save_output=False, version='V7_Optuna')
        val = corr_ker if metric == 'correlation' else r2_ker
        return val if val is not None and not np.isnan(val) else -1.0
    except:
        return -1.0

def run_optimization(country, n_trials=300, objective_metric='correlation', year_align='harvest'):
    model = MaizeYieldModelV7(country)
    model.prepare_data()
    
    print(f"\n--- [2/2] Launching Optuna Optimization for {country} (year_align={year_align}) ---")
    
    # Storage for persistence
    db_path = os.path.join(OUTPUT_DIR, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=f"V7_Optimization_{country}_{objective_metric}",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(lambda trial: objective(trial, country, model.slices, metric=objective_metric, year_align=year_align), n_trials=n_trials)

    print(f"\nOptimization Finished for {country}.")
    print(f"Best {objective_metric}: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
    
    # Final Run & Save
    df_best = fast_yield_engine(model.slices, study.best_params, year_align=year_align)
    df_best['Country'] = country
    df_best['Year'] = df_best['Year'].astype(int)
    
    # Save params
    res_path = os.path.join(OUTPUT_DIR, f"Optimal_Params_Optuna_{country}_{objective_metric}.csv")
    pd.DataFrame([dict(study.best_params, Metric=study.best_value, Objective=objective_metric)]).to_csv(res_path, index=False)
    
    # Full Verification
    verify_fao.verify_fao(country, df_model=df_best, save_output=True, version=f'V7_Final_Optuna_{objective_metric}')
    print(f"Final results saved to {OUTPUT_DIR}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, default="South Africa")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--metric", type=str, default="correlation", choices=["correlation", "r2"])
    parser.add_argument("--year_align", type=str, default="harvest", choices=["harvest", "plant"],
                        help="Year labelling: 'harvest' (default) = EOS year, 'plant' = calendar anchor year")
    args = parser.parse_args()

    predictions = pd.read_csv(r"Model\africa_results\all_africa_maize_yield_predictions.csv")
    countries = predictions["country"].unique().tolist()
    print(countries)

    COUNTRY_LIST = countries
    
    for country in COUNTRY_LIST:
        try:
            run_optimization(country, n_trials=args.trials, objective_metric=args.metric, year_align=args.year_align)
        except Exception as e:
            print(f"Error optimizing for {country}: {e}")
            continue
