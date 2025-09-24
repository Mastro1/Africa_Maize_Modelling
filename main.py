"""
Main script to run all the data extraction and processing pipelines.
"""

import os
import pandas as pd

# FAOSTAT
from FAOSTAT.download_faostat import main as download_faostat

# GLAD
from GLAD.download_GLAD_map_3km import main as download_glad_map_3km

# GAEZ
from GAEZ.download_GAEZ_yield_potential import main as download_gaez
from GAEZ.extract_gaez_values import GAEZYieldExtractor

# GEOGLAM
from GEOGLAM.download_GEOGLAM import main as download_geoglam

# HWSD
from HWSD.download_soil_data import main as download_soil_data
from HWSD.extract_soil_information import extract_soil_properties_all_locations

# GADM
from GADM.DownloadShapefile import main as download_shapefile
from GADM.crop_calendar.extract_crop_calendar import extract_crop_calendar as extract_crop_calendar_GADM
from GADM.CropAreasGADM import main_africa as crop_areas_gadm
from GADM.CropAreasFiltering import main as filter_crop_areas
from GADM.CreateDataGLAM import process_all_africa_glam

# HarvestStatAfrica
from HarvestStatAfrica.crop_calendar.extract_GLAM_calendar import extract_crop_calendar as extract_glam_calendar_HSA
from HarvestStatAfrica.data.download_harvestat_africa_data import main as download_harvestat_africa
from HarvestStatAfrica.CreateDataGLAM import run_glam_for_all_countries
from HarvestStatAfrica.ConcatAllData import concatenate_data as concat_all_data_GLAM

if __name__ == "__main__":

    # General Data
    download_faostat()
    download_glad_map_3km()
    download_gaez()
    download_geoglam()
    download_soil_data()
    download_shapefile()
    download_harvestat_africa()

    # HarvestStatAfrica
    extract_glam_calendar_HSA()

    # GAEZ Yield Potential HSA
    extractor = GAEZYieldExtractor(
    yield_file = "GAEZ/DATA_GAEZ-V5_MAPSET_RES02-YLD_GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif",
    boundaries_path = "HarvestStatAfrica/data/hvstat_africa_boundary_v1.0.gpkg",
    output_dir = "HarvestStatAfrica/yield_potential",
    output_file = "gaez_maize_yield_potential_HSA.csv")
    extractor.run_extraction()

    # HWSD HSA
    extract_soil_properties_all_locations(
        gpkg_path="HarvestStatAfrica/data/hvstat_africa_boundary_v1.0.gpkg",
        admin_level=2,
        raster_path="HWSD/HWSD2.bil",
        mdb_path="HWSD/HWSD2.mdb",
        output_csv="HarvestStatAfrica/HWSD/Africa_Admin2_Soil_Stats.csv",
        topsoil_max_depth=30
    )

    # Add remote sensing data to HSA
    run_glam_for_all_countries(force_redo=False, plot=False)
    concat_all_data_GLAM()

    # GADM
    # Crop areas
    crop_areas_gadm(batch_size=20)

    # Filter crop areas
    filter_crop_areas()

    # Extract crop calendar
    extract_crop_calendar_GADM()

    # GAEZ Yield Potential Admin2
    extractor = GAEZYieldExtractor(
    yield_file = "GAEZ/DATA_GAEZ-V5_MAPSET_RES02-YLD_GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif",
    boundaries_path = "GADM/gadm41_AFR_shp/gadm41_AFR_2_processed.shp",
    output_dir = "GADM/yield_potential",
    output_file = "gaez_maize_yield_potential_admin2.csv")
    extractor.run_extraction()

    # GAEZ Yield Potential Admin1
    print("For prediction admin1")
    extractor = GAEZYieldExtractor(
    yield_file = "GAEZ/DATA_GAEZ-V5_MAPSET_RES02-YLD_GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif",
    boundaries_path = "GADM/gadm41_AFR_shp/gadm41_AFR_1_processed.shp",
    output_dir = "GADM/yield_potential",
    output_file = "gaez_maize_yield_potential_admin1.csv")
    extractor.run_extraction()

    # Concat the two csv files
    df_admin1 = pd.read_csv("GADM/yield_potential/gaez_maize_yield_potential_admin1.csv")
    df_admin2 = pd.read_csv("GADM/yield_potential/gaez_maize_yield_potential_admin2.csv")
    df_merged = pd.concat([df_admin1, df_admin2])
    df_merged.to_csv("GADM/yield_potential/gaez_maize_yield_potential.csv", index=False)

    # HWSD Admin2
    extract_soil_properties_all_locations(
        gpkg_path="GADM/gadm41_AFR_shp/gadm41_AFR_2_processed.shp",
        admin_level=2,
        raster_path="HWSD/HWSD2.bil",
        mdb_path="HWSD/HWSD2.mdb",
        output_csv="GADM/HWSD/Africa_Admin2_Soil_Stats.csv",
        topsoil_max_depth=30
    )

    # HWSD Admin1
    extract_soil_properties_all_locations(
        gpkg_path="GADM/gadm41_AFR_shp/gadm41_AFR_1_processed.shp",
        admin_level=1,
        raster_path="HWSD/HWSD2.bil",
        mdb_path="HWSD/HWSD2.mdb",
        output_csv="GADM/HWSD/Africa_Admin1_Soil_Stats.csv",
        topsoil_max_depth=30
    )

    df_admin1 = pd.read_csv("GADM/HWSD/Africa_Admin1_Soil_Stats.csv")
    df_admin2 = pd.read_csv("GADM/HWSD/Africa_Admin2_Soil_Stats.csv")
    df_merged = pd.concat([df_admin1, df_admin2])
    df_merged.to_csv("GADM/HWSD/Africa_Admin_Soil_Stats.csv", index=False)

    # Create GLAM data
    process_all_africa_glam(force_redo=False)

    