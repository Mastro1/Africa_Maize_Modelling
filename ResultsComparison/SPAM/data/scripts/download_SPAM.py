import os
import pandas as pd
import requests
import zipfile

OUTPUT_DIR = "ResultsComparison/SPAM/data"

SPAM_URL = {
    "2000": "https://s3.amazonaws.com/mapspam-data/2000/v3.0.7/geotiff/spam2000v3.0.7_global_yield.geotiff.zip",
    "2005": "https://s3.amazonaws.com/mapspam-data/2005/v3.2/geotiff/spam2005v3r2_global_yield.geotiff.zip",
    "2010": "https://s3.amazonaws.com/mapspam-data/2010/v2.0/geotiff/spam2010v2r0_global_yield.geotiff.zip",
    "2017": "https://s3.amazonaws.com/mapspam-data/2017/ssa/v2.1/geotiff/spam2017v2r1_ssa_yield.geotiff.zip",
    "2020": "As today's date is 2025-12-31, you need to download the SPAM data for 2020 from the Harvard Dataverse. Link: https://dataverse.harvard.edu/file.xhtml?fileId=11596406&version=4.1"
}

def download_SPAM(year):
    url = SPAM_URL[year]

    # Special handling for 2020 - manual download required
    if year == "2020":
        print(f"Year {year}: {url}")
        return

    response = requests.get(url)

    if response.status_code == 200:
        # Create a temporary zip file path
        temp_zip_path = os.path.join(OUTPUT_DIR, f"spam{year}_temp.zip")

        # Save the downloaded content to the temporary zip file
        with open(temp_zip_path, 'wb') as f:
            f.write(response.content)

        # Extract only MAIZ files from the zip file to the output directory
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            # Get list of all files in the zip
            all_files = zip_ref.namelist()

            # Select the appropriate MAIZ file based on year and technology
            maiz_file = None

            if year == "2000":
                # For 2000, select MAIZ.tif (no technology suffix)
                maiz_file = next((f for f in all_files if f.upper().endswith("MAIZ.TIF")), None)
            else:
                # For 2005, 2010, 2017, select MAIZ_A.tif
                maiz_file = next((f for f in all_files if "MAIZ_A.TIF" in f.upper()), None)

            # Extract the selected MAIZ file if found
            if maiz_file:
                # Get just the filename without folder path
                filename = os.path.basename(maiz_file)
                # Extract to output directory with the clean filename
                with open(os.path.join(OUTPUT_DIR, filename), 'wb') as f:
                    f.write(zip_ref.read(maiz_file))
                print(f"Extracted {filename} for year {year}")
            else:
                print(f"No appropriate MAIZ file found for year {year}")

        # Remove the temporary zip file
        os.remove(temp_zip_path)

        print(f"Successfully downloaded and extracted SPAM data for year {year} to {OUTPUT_DIR}")
    else:
        print(f"Failed to download SPAM data for year {year}. Status code: {response.status_code}")


def download_all_SPAM():
    """
    Download and extract SPAM data for all available years.
    """
    print("Starting download of all SPAM datasets...")

    for year in SPAM_URL.keys():
        print(f"\nDownloading SPAM data for year {year}...")
        download_SPAM(year)

    print("\nAll SPAM datasets download completed!")


def rename_maiz_files():
    """
    Rename downloaded MAIZ files to simpler format: maiz_[year].tif
    """
    print("Renaming MAIZ files to simpler format...")

    # Mapping of current filenames to new filenames
    rename_mapping = {
        "spam2000v3r7_yield_MAIZ.tif": "spam_maiz_2000.tif",
        "SPAM2005V3r2_global_Y_TA_MAIZ_A.tif": "spam_maiz_2005.tif",
        "spam2010V2r0_global_Y_MAIZ_A.tif": "spam_maiz_2010.tif",
        "spam2017V2r1_SSA_Y_MAIZ_A.tif": "spam_maiz_2017.tif",
        "spam2020_V2r0_global_Y_MAIZ_A.tif": "spam_maiz_2020.tif"
    }

    renamed_count = 0

    for old_name, new_name in rename_mapping.items():
        old_path = os.path.join(OUTPUT_DIR, old_name)
        new_path = os.path.join(OUTPUT_DIR, new_name)

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            #print(f"Renamed: {old_name} -> {new_name}")
            renamed_count += 1
        else:
            print(f"File not found: {old_name}")

    print(f"\nRenaming completed! {renamed_count} files renamed.")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_all_SPAM()
    rename_maiz_files()