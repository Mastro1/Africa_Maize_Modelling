import os
import requests

GLAD_DIR = "GLAD"

def main(year=2019):
    url = f"https://glad.geog.umd.edu/Potapov/Global_Crop/Data/Global_cropland_3km_{year}.tif"
    response = requests.get(url)
    with open(os.path.join(GLAD_DIR, f"Global_cropland_3km_{year}.tif"), "wb") as f:
        f.write(response.content)

if __name__ == "__main__":
    main(year=2019)