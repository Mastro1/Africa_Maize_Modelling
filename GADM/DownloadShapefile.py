import pygadm
import os
import geopandas as gpd
import pandas as pd


def download_full_shapefile(output_dir="GADM/gadm41_AFR_shp"):
    """
    Downloads GADM shapefiles for all countries in Africa for admin levels 0, 1, and 2.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Admin Level 0 ---
    admin_level_0_file = os.path.join(output_dir, "gadm41_AFR_0.shp")

    # Check if the level 0 file exists. If not, download it.
    if not os.path.exists(admin_level_0_file):
        print("Downloading Admin Level 0 for Africa...")
        try:
            # Use .to_gpd() to get a GeoDataFrame
            gdf_0 = pygadm.Items(name="Africa", content_level=0)
            # Set CRS to WGS84 as per GADM website
            gdf_0 = gdf_0.set_crs("EPSG:4326")
            print(f"Admin Level 0 CRS set to: {gdf_0.crs}")
            gdf_0.to_file(admin_level_0_file)
            print("Admin Level 0 for Africa downloaded successfully.")
        except Exception as e:
            print(f"Failed to download Admin Level 0 for Africa. Error: {e}")
            return  # Exit if we can't get the base file
    else:
        print("Admin Level 0 shapefile already exists. Loading from file.")
        gdf_0 = gpd.read_file(admin_level_0_file)
        # Set CRS to WGS84 as per GADM website
        gdf_0 = gdf_0.set_crs("EPSG:4326")

    # Get the list of unique country names from the 'COUNTRY' column
    countries = gdf_0["NAME_0"].unique()
    print(f"Found {len(countries)} countries to process.")

    # --- Admin Levels 1 and 2 ---
    for level in [1, 2]:
        output_file = os.path.join(output_dir, f"gadm41_AFR_{level}.shp")

        if os.path.exists(output_file):
            print(f"Admin Level {level} shapefile already exists. Skipping.")
            continue

        print(f"\n--- Processing Admin Level {level} ---")
        all_gdfs = []
        failed_countries = []

        for i, country in enumerate(countries):
            print(f"({i + 1}/{len(countries)}) Downloading data for {country} at admin level {level}...")
            try:
                # Use .to_gpd() to get a GeoDataFrame
                country_gdf = pygadm.Items(name=country, content_level=level)
                # Set CRS to WGS84 as per GADM website
                country_gdf = country_gdf.set_crs("EPSG:4326")
                all_gdfs.append(country_gdf)
                print(f"Successfully downloaded data for {country}.")
            except Exception as e:
                failed_countries.append(country)
                print(f"Could not retrieve data for {country} at level {level}. Error: {e}")

        if all_gdfs:
            print(f"\nConcatenating {len(all_gdfs)} GeoDataFrames for admin level {level}...")
            combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=gdf_0.crs)

            print(f"Saving combined shapefile for admin level {level} to {output_file}...")
            combined_gdf.to_file(output_file)
            print(f"Admin Level {level} for Africa saved successfully.")

        if failed_countries:
            print(f"\nCould not retrieve data for the following {len(failed_countries)} countries at level {level}:")
            print(", ".join(failed_countries))

    print("\nDownload process finished.")


def process_shapefiles(input_dir="GADM/gadm41_AFR_shp", tolerance=0.02):
    """
    Simplifies and processes shapefiles in the input directory.
    - Simplifies geometries.
    - For admin level 2, filters out countries with > 150 administrative units.
    - Renames columns to a standard format.
    - Saves them with a '_processed' suffix.
    """
    print("\n--- Processing Shapefiles ---")
    for level in [0, 1, 2]:
        input_file = os.path.join(input_dir, f"gadm41_AFR_{level}.shp")
        output_file = os.path.join(input_dir, f"gadm41_AFR_{level}_processed.shp")

        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}. Skipping processing for level {level}.")
            continue

        print(f"Processing {input_file}...")
        gdf = gpd.read_file(input_file)
        # Set CRS to WGS84 as per GADM website
        gdf = gdf.set_crs("EPSG:4326")
        original_crs = gdf.crs

        # Difference n2: For admin level 2, filter countries
        if level == 2:
            print("Filtering admin level 2 data...")
            # Count admin units per country
            admin_counts = gdf.groupby('NAME_0')['NAME_2'].nunique()
            # Get list of countries to keep
            countries_to_keep = admin_counts[admin_counts <= 350].index.tolist()

            print(f"Keeping {len(countries_to_keep)} out of {len(admin_counts)} countries with <= 350 admin units.")
            
            countries_to_remove = admin_counts[admin_counts > 350].index.tolist()
            print(f"Removing {len(countries_to_remove)} countries: {', '.join(countries_to_remove)}")


            # Filter the GeoDataFrame
            gdf = gdf[gdf['NAME_0'].isin(countries_to_keep)]

        # Difference n1: Simplify geometries
        print(f"Simplifying geometries with tolerance {tolerance}...")
        gdf['geometry'] = gdf['geometry'].simplify(tolerance)

        # Remove empty geometries that might result from simplification
        gdf = gdf[~gdf.geometry.is_empty]

        # Rename columns to match the legacy format
        print("Renaming columns...")
        if level == 0:
            gdf.rename(columns={"NAME_0": "ADMIN0", "GID_0": "FNID"}, inplace=True)
            gdf["ADMIN1"] = None
            gdf["ADMIN2"] = None
        elif level == 1:
            gdf.rename(columns={"NAME_0": "ADMIN0", "NAME_1": "ADMIN1", "GID_1": "FNID"}, inplace=True)
            gdf["ADMIN2"] = None
        elif level == 2:
            gdf.rename(columns={"NAME_0": "ADMIN0", "NAME_1": "ADMIN1", "NAME_2": "ADMIN2", "GID_2": "FNID"}, inplace=True)
            gdf = gdf.dropna(subset=["FNID"])

        # Reorder and select columns to match the old format
        final_cols = ["FNID", "ADMIN0", "ADMIN1", "ADMIN2", "geometry"]
        gdf = gdf[final_cols]

        print(f"Saving processed shapefile to {output_file}...")
        gdf.to_file(output_file)
        print(f"Successfully saved {output_file}")


def main():
    shapefile_dir = os.path.join("GADM", "gadm41_AFR_shp")
    download_full_shapefile(output_dir=shapefile_dir)
    process_shapefiles(input_dir=shapefile_dir)
    print("\nAll processing finished.")

if __name__ == "__main__":
    main()






