import os
import zipfile
import requests

SOIL_DIR = "HWSD"

def main():
    # Database
    url = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/HWSD/HWSD2_DB.zip"

    response = requests.get(url)
    with open("HWSD2_DB.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("HWSD2_DB.zip", "r") as zip_ref:
        zip_ref.extractall(SOIL_DIR)

    os.remove("HWSD2_DB.zip")

    # Raster
    url = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/HWSD/HWSD2_RASTER.zip"

    response = requests.get(url)
    with open("HWSD2_RASTER.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("HWSD2_RASTER.zip", "r") as zip_ref:
        zip_ref.extractall(SOIL_DIR)

    os.remove("HWSD2_RASTER.zip")

if __name__ == "__main__":
    main()