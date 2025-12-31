"""
This script creates the final shapefile for the Africa region by combining the admin 1 and admin 2 shapefiles that are in the final set.

It is useful since there are some countries that are only available at the admin 1 level, but not at the admin 2 level.
"""


import geopandas as gpd
import os
import pandas as pd

list_locations = "GADM/crop_areas/africa_crop_areas_glad_filtered.csv"

list_locations = pd.read_csv(list_locations)

print(list_locations.head())

admin_1_locations = list_locations[list_locations["admin_2"].isna()]["PCODE"].unique()

admin_2_locations = list_locations[list_locations["admin_2"].notna()]["PCODE"].unique()

shapefile_admin1 = gpd.read_file(f"GADM/gadm41_AFR_shp/gadm41_AFR_1_processed.shp")
shapefile_admin2 = gpd.read_file(f"GADM/gadm41_AFR_shp/gadm41_AFR_2_processed.shp")

shapefile_admin1 = shapefile_admin1[shapefile_admin1["FNID"].isin(admin_1_locations)]
shapefile_admin2 = shapefile_admin2[shapefile_admin2["FNID"].isin(admin_2_locations)]

shapefile_final = pd.concat([shapefile_admin1, shapefile_admin2])

shapefile_final.to_file(f"GADM/gadm41_AFR_shp/gadm41_AFR_final.shp")
print(f"Shapefile saved as GADM/gadm41_AFR_shp/gadm41_AFR_final.shp")