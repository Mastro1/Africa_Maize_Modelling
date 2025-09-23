import os
import requests

GAEZ_DIR = "GAEZ"

def main():
    url = "https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET/RES02-YLD/GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif"

    response = requests.get(url)
    with open(os.path.join(GAEZ_DIR, "DATA_GAEZ-V5_MAPSET_RES02-YLD_GAEZ-V5.RES02-YLD.HP0120.AGERA5.HIST.MAIZ.HRLM.tif"), "wb") as f:
        f.write(response.content)

if __name__ == "__main__":
    main()