# Project Overview

This project is a data science pipeline for modeling and predicting maize yield across Africa. It involves several stages:

1.  **Data Acquisition:** Downloading agricultural, climate, and soil data from various sources, including:
    *   **FAOSTAT:** Crop and livestock data.
    *   **GAEZ:** Global Agro-Ecological Zones data, specifically for maize yield potential.
    *   **GEOGLAM:** Global Agricultural Monitoring data.
    *   **HWSD:** Harmonized World Soil Database.
    *   **GADM:** Global Administrative Areas data for shapefiles.
    *   **HarvestStatAfrica:** A custom dataset.

2.  **Data Processing and Feature Engineering:** Extracting, cleaning, and transforming the raw data into a format suitable for machine learning. This includes:
    *   Calculating temporal and spatial features from remote sensing data.
    *   Extracting soil properties.
    *   Processing crop calendar information.
    *   Creating interaction features between different data types.

3.  **Modeling:** Training and evaluating a machine learning model to predict maize yield. The modeling approach is sophisticated, incorporating:
    *   An `ExtraTreesRegressor` model.
    *   A "golden cohort" of countries for robust training.
    *   A `LeaveOneGroupOut` cross-validation strategy by country.
    *   Analogue-based training, which finds similar countries to the target country for more accurate predictions.
    *   PCA for dimensionality reduction.

4.  **Prediction:** Using the trained model to predict maize yield for unseen locations across the entire African continent.

The project is written in Python and utilizes a variety of data science and geospatial libraries, including `pandas`, `numpy`, `scikit-learn`, `rasterio`, `geopandas`, and more.

# Building and Running

The main entry point for the entire pipeline is `main.py`. To run the project, you would typically execute this script.

```bash
python main.py
```

This will trigger the entire data download, processing, and modeling pipeline.

The modeling part of the project can be run independently:

*   **Training the model:**
    ```bash
    python Model/OPTIMIZED_MODEL.py
    ```
*   **Predicting on new data:**
    ```bash
    python Model/PREDICT_UNSEEN_LOCATIONS.py
    ```

# Development Conventions

*   **Modular Structure:** The project is organized into modules based on the data source (e.g., `FAOSTAT`, `GAEZ`, `HWSD`). This promotes code reusability and maintainability.
*   **Clear Naming:** File and function names are descriptive, making it easy to understand their purpose (e.g., `download_faostat.py`, `extract_soil_information.py`).
*   **Configuration:** The modeling scripts use a `BEST_CONFIG` dictionary to manage model parameters and configurations, allowing for easy experimentation.
*   **Virtual Environment:** The project uses a virtual environment (`.venv`) to manage dependencies.
