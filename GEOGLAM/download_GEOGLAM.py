import os
import zipfile
import requests

GEOGLAM_DIR = "GEOGLAM"

def main():
    url = "https://zenodo.org/records/10949972/files/GEOGLAM_CM4EW_Calendars_V1.0.zip?download=1"

    response = requests.get(url)
    with open("GEOGLAM_CM4EW_Calendars_V1.0.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("GEOGLAM_CM4EW_Calendars_V1.0.zip", "r") as zip_ref:
        zip_ref.extractall(GEOGLAM_DIR)

    os.remove("GEOGLAM_CM4EW_Calendars_V1.0.zip")

if __name__ == "__main__":
    main()