import pandas as pd
import os
import geopandas as gpd

HSA_DIR = "HarvestStatAfrica/data"

def main():
    os.makedirs(HSA_DIR, exist_ok=True)

    # Download the data from the GitHub repository
    url = "https://raw.githubusercontent.com/HarvestStat/HarvestStat-Africa/refs/heads/main/public/hvstat_africa_data_v1.0.csv"

    df = pd.read_csv(url)

    print(df.head())

    # Save the data to a CSV file
    df.to_csv(os.path.join(HSA_DIR, "hvstat_africa_data_v1.0.csv"), index=False)


    # Shapefile from the GitHub repository
    url = "https://github.com/HarvestStat/HarvestStat-Africa/raw/refs/heads/main/public/hvstat_africa_boundary_v1.0.gpkg"
    gdf = gpd.read_file(url)
    gdf.loc[gdf['ADMIN0'] == "Tanzania", 'ADMIN0'] = "Tanzania, United Republic of"
    print(gdf.head())
    # Save the data to a GeoPackage file
    gdf.to_file(os.path.join(HSA_DIR, "hvstat_africa_boundary_v1.0.gpkg"), driver="GPKG")


    # Crop Calendar from the GitHub repository
    url = "https://raw.githubusercontent.com/HarvestStat/HarvestStat-Africa/refs/heads/main/data/crop_calendar/external_season_calendar.csv"
    crop_calendar = pd.read_csv(url)
    print(crop_calendar.head())
    # Save the data to a CSV file
    crop_calendar_dir = os.path.join("HarvestStatAfrica", "crop_calendar")
    os.makedirs(crop_calendar_dir, exist_ok=True)
    crop_calendar.to_csv(os.path.join(crop_calendar_dir, "external_season_calendar.csv"), index=False)

if __name__ == "__main__":
    main()

