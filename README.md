# Sub-National Maize Yield Modeling for Agricultural Risk Management in Africa

This repository contains the source code and data for the research paper, "A Scalable Satellite-Based Framework for Sub-National Yield Modelling to Strengthen Agricultural Risk Management in Africa". The project presents a generalizable machine learning framework that disaggregates national maize yield statistics to the sub-national level using Earth Observation (EO) data.

The primary goal is to create a consistent, robust, and spatially explicit maize yield dataset for Africa to support applications like index-based insurance, catastrophe modeling, and food security assessments.

## Results visualization
![](Plots\output\africa_yield_anomalies_animation.gif)

## Key Features

*   **Scalable Framework:** A transferable model that can be applied across different agro-ecological zones, even in countries without extensive ground-truth yield data.
*   **Data-Driven Methodology:** Utilizes a wide range of publicly available Earth Observation datasets for climate, vegetation, and soil predictors.
*   **Rigorous Curation:** Implements a two-phase data curation protocol to create a high-confidence "Golden Cohort" of 9 countries for model training.
*   **Advanced Modeling:** Employs an Extra-Trees Regressor with an adaptive "analogue" strategy, which tailors the training process to the specific environmental context of the target country.
*   **Open Data & Code:** The generated pan-African yield dataset and the modeling code are openly shared to encourage replication, collaboration, and further research.

## Methodology Overview

The framework is designed to overcome the lack of comparable, sub-national yield data across Africa.

1.  **Data Sources & Preparation:** The model integrates two primary types of data:
    *   **Ground-Truth:** Sub-national maize yield statistics from the HarvestStat Africa database.
    *   **Predictors:** A comprehensive suite of spatio-temporal predictors from EO sources (MODIS, ERA5-Land) and static datasets (GAEZ, HWSD), covering vegetation health, climate stressors, and environmental conditions.

2.  **Data Curation:** A rigorous two-phase protocol is applied to the ground-truth data to ensure structural reliability and agronomic plausibility, resulting in a high-confidence "Golden Cohort" of countries for training.

3.  **Target Variable Transformation:** To ensure transferability, the model predicts a standardized, location-independent metric of relative annual performance. This is achieved by normalizing sub-national yields against a stable, long-term national yield trend derived from FAOSTAT data, effectively decoupling local weather-driven shocks from long-term agronomic progress.

4.  **Yield Estimation & Validation:** An Extremely Randomized Trees (Extra-Trees) model is trained to predict the normalized yield anomalies. The model's performance and transferability are validated using a stringent Leave-One-Country-Out (LOCO) cross-validation strategy.

5.  **Yield Reconstruction:** The model's normalized predictions are converted back into absolute yield values (tonnes/hectare) using the national FAOSTAT trend line as a baseline.

## Repository Structure

```
.
├── FAOSTAT/            # Scripts to download and process FAOSTAT national data
├── GADM/               # Scripts for processing Global Administrative Areas shapefiles
├── GAEZ/               # Scripts for GAEZ potential yield data
├── GEOGLAM/            # Scripts for GEOGLAM crop calendar data
├── HarvestStatAfrica/  # Scripts for the core sub-national ground-truth data
├── HWSD/               # Scripts for Harmonized World Soil Database
├── Model/              # Core modeling and prediction scripts
│   ├── OPTIMIZED_MODEL.py
│   └── PREDICT_UNSEEN_LOCATIONS.py
├── Paper/              # Scientific paper PDF
├── RemoteSensing/      # Scripts for processing remote sensing data (NDVI, ERA5, etc.)
├── .gitignore
├── main.py             # Main script to execute the entire data pipeline
└── README.md
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Africa_Maize_Modelling.git
    cd Africa_Maize_Modelling
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project can be run in different stages:

1.  **Full Pipeline:** To run the entire data acquisition and processing pipeline from scratch, execute the main script. This will download all necessary data and preprocess it.
    ```bash
    python main.py
    ```

2.  **Model Training:** To train the model on the curated "Golden Cohort" and evaluate its performance using Leave-One-Country-Out cross-validation:
    ```bash
    python Model/OPTIMIZED_MODEL.py
    ```

3.  **Prediction for Unseen Locations:** To apply the trained framework and generate yield predictions for all of Africa:
    ```bash
    python Model/PREDICT_UNSEEN_LOCATIONS.py
    ```
    The final predictions will be saved in `Model/africa_results/`.

## Data Availability

This repository serves as the primary storage for the data products generated by this research. The final, pan-African sub-national maize yield dataset is included directly in this repository.

*   **Code:** The code in this repository is provided to ensure full reproducibility of the dataset.
*   **Dataset:** The final dataset is located at [Model/africa_results/all_africa_maize_yield_predictions.csv](Model/africa_results/all_africa_maize_yield_predictions.csv).

Below is a preview of the dataset's structure and contents:

```csv
PCODE,country,year,season_index,pred_yield,has_ground_truth,final_yield,crop_area_ha,national_trend_avg
AGO.2.4_1,Angola,2000,1,0.6982767966030113,False,0.6982767966030113,21862.28,0.8448695652173973
AGO.2.4_1,Angola,2001,1,0.46552029182581445,False,0.46552029182581445,21862.28,0.5258750000000063
AGO.2.4_1,Angola,2002,1,0.4473653672971753,False,0.4473653672971753,21862.28,0.5548745059288578
AGO.2.4_1,Angola,2003,1,0.47213049547333724,False,0.47213049547333724,21862.28,0.5838740118577164
AGO.2.4_1,Angola,2004,1,0.5297595155981496,False,0.5297595155981496,18140.74,0.6128735177865678
AGO.2.4_1,Angola,2005,1,0.5622978757476652,False,0.5622978757476652,18140.74,0.6418730237154193
AGO.2.4_1,Angola,2006,1,0.5946296451543402,False,0.5946296451543402,18140.74,0.6708725296442779
AGO.2.4_1,Angola,2007,1,0.564065491278867,False,0.564065491278867,18140.74,0.6998720355731294
AGO.2.4_1,Angola,2008,1,0.6268055753940258,False,0.6268055753940258,22994.77,0.7288715415019809
...
```

## Citation

If you use this work, please cite the following paper:

```
Poretti, M., Coutu, S., & Wagner, J. (2025). A Scalable Satellite-Based Framework for Sub-National Yield Modelling to Strengthen Agricultural Risk Management in Africa. *Climate Risk Management*.
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
