**Project Title:** Operational Historical Maize Yield Reconstruction for Kenya (Admin 2 Level)
**Methodology:** Revised EC-LUE (Zheng et al., 2020) implemented via Prince et al. (2001) agronomic conversion.
**Region:** Kenya (Admin 2 / District scale).
**Timeframe:** 2000–Present.

---

### 1. Data Inventory & Mapping
The model relies on three CSV sources. Data is already aggregated at the Admin 2 level.

#### A. Satellite Data (Vegetation)
*   **Path:** `Model_physical\Kenya_admin2_FPAR_timeseries_GLAD.csv`
*   **Key Columns:** `date`, `PCODE` (ID), `FPAR_mean`.
*   **Status:** Pre-processed. `FPAR_mean` is already scaled (0.0 to 1.0). Aggregated using crop mask.
*   **Temporal Resolution:** 8-day (Must be interpolated to reach daily).

#### B. Meteorological Data (Energy & Water)
*   **Source 1:** `Model_physical\Kenya_admin2_new_ERA5_timeseries.csv`
    *   *Variables:* `dewpoint_temperature_2m` (Kelvin), `surface_solar_radiation_downwards_sum` (Joules), `volumetric_soil_water_layer_1` (et al).
*   **Source 2:** `Model_physical\Kenya_admin2_ERA5_timeseries_GADM.csv`
    *   *Variables:* `temperature_2m` (Kelvin), `total_precipitation_sum`.
*   **Action:** These two files must be merged on `date` and `PCODE`.

#### C. Crop Calendar (Phenology)
*   **Path:** `GADM\crop_calendar\maize_crop_calendar_extraction.csv`
*   **Key Columns:** `FNID` (Matches PCODE), `Maize_1_planting` (DOY), `Maize_1_harvest` (DOY).
*   **Logic:** Fixed dates. We assume the sowing and harvesting Day of Year (DOY) is constant for every year.

---

### 2. Pre-Processing Pipeline (Python/Pandas)

**Objective:** Create a unified `daily_dataframe` for each PCODE where every row represents one day with no missing values.

1.  **Merge ERA5:** Join Source 1 and Source 2 on `date` and `PCODE`.
2.  **Merge MODIS:** Join the FPAR data to the ERA5 data.
3.  **Temporal Interpolation (Crucial):**
    *   MODIS FPAR is available only every ~8-16 days. ERA5 is daily.
    *   **Action:** Use `pandas.DataFrame.interpolate(method='linear')` to fill the missing FPAR days between observations.
4.  **Smoothing (Cloud Correction):**
    *   Since cloud masking was not performed at the pixel level, the `FPAR_mean` time series will likely have sudden drops (noise) due to contamination.
    *   **Action:** Apply a **Savitzky-Golay filter** (`scipy.signal.savgol_filter`) to the interpolated FPAR column.
    *   *Parameters:* Window length $\approx$ 31 days, Polyorder $\approx$ 2. This smooths out cloud dips while preserving the seasonal bell curve.

---

### 3. The Biophysical Engine (Calculations)

We calculate physiological variables for every day using the `pyrealm` library where possible to ensure thermodynamic accuracy.

Documentation is here: https://pyrealm.readthedocs.io/en/latest/index.html

#### 3.1 Radiation (PAR)
Convert total solar energy to the visible spectrum available for photosynthesis.
*   Input: `surface_solar_radiation_downwards_sum` ($J/m^2$)
*   Conversion:
    $$PAR_{MJ} = \text{ssrd} \times 10^{-6} \times 0.48$$

#### 3.2 Vapor Pressure Deficit (VPD)
Use `pyrealm` to calculate the "thirst" of the atmosphere.
*   **Input:** `temperature_2m` (Kelvin), `dewpoint_temperature_2m` (Kelvin).
*   **Code Implementation:**
    ```python
    from pyrealm import hygrometric_parameters
    # Convert Kelvin to Celsius for Pyrealm
    temp_c = row['temperature_2m'] - 273.15
    dew_c = row['dewpoint_temperature_2m'] - 273.15
    
    # Calculate VPD (Pa) using rigorous physics
    # Note: We assume standard pressure if surface_pressure is missing
    hygro = hygrometric_parameters.HygroParams(ta=temp_c, dew=dew_c)
    vpd_kpa = hygro.vpd / 1000.0  # Convert Pa to kPa
    ```

---

### 4. The Yield Model (Zheng 2020 Logic)

Iterate through the data for each **Year** and each **PCODE**.

#### 4.1 Define the Season
*   Look up `Maize_1_planting` and `Maize_1_harvest` DOY for the specific PCODE from the Calendar CSV.
*   Slice the daily dataframe to include only days between `Planting` and `Harvest`.

#### 4.2 Calculate Stress Scalars (0.0 to 1.0)
*   **Temperature Stress ($Ts$):** Maize optimum is 30°C.
    $$Ts = \exp\left( - \frac{(T_{air\_C} - 30)^2}{2 \times 5^2} \right)$$
*   **Water Stress ($Ws$):** Use the Zheng (2020) VPD logic.
    *   $VPD_0 \approx 1.5$ kPa (Standard sensitivity for crops).
    $$Ws = \frac{1.5}{1.5 + VPD_{kpa}}$$

#### 4.3 Calculate Daily NPP
$$NPP_{daily} = PAR_{MJ} \times FPAR_{smooth} \times \epsilon_{max} \times \min(Ts, Ws) \times 0.5$$
*   **$\epsilon_{max}$:** Set to **2.8 gC/MJ** (Potential efficiency for C4 Maize).
*   **0.5:** Respiration coefficient (Autotrophic respiration consumes ~50% of GPP).

---

### 5. Final Aggregation (The Prince 2001 Logic)

Sum the `NPP_daily` for the season and convert to Yield.

#### Constants
*   **HI (Harvest Index):** **0.35** (Conservative for Kenya).
*   **RS (Root-to-Shoot):** **0.18**
*   **MC (Moisture Content):** **0.125**
*   **C_frac:** **0.45** (Carbon content of biomass).

#### Final Formula
$$Yield_{g/m^2} = \frac{\sum NPP_{daily}}{0.45} \times \frac{HI}{(1 + RS) \times (1 - MC)}$$
$$Yield_{t/ha} = Yield_{g/m^2} \times 0.01$$

---

### 6. Output Generation
Create a final CSV with the following structure:
*   `PCODE`: District ID
*   `Year`: The crop season year
*   `Yield_Estimated_t_ha`: The model result
*   `NPP_Total_gC`: Intermediate variable (useful for debugging)
*   `Mean_FPAR_Season`: Intermediate variable (useful for debugging)
*   `Mean_VPD_Season`: Intermediate variable (useful for debugging)

This output will allow you to plot the time series (2000–2020) for any district and immediately see the inter-annual variability.

### Implementation notes
* Write comprehensive code with comments and docstrings.
* Add some prints to the terminal to check the progress and see if the results make sense.
* Take one location as example and plot the time series at all the steps (Take the following: KEN.22.5_1)
* This project must written only in /Model_physical folder. Keep results in a subfolder called /Results.
* If you need help you ask. If you don't understand something you ask. If you need documentation you ask.
