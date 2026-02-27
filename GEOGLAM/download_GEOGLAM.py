import os
import zipfile
import requests

GEOGLAM_DIR = "GEOGLAM"

def main():
    url = "https://zenodo.org/records/15850408/files/GEOGLAM_CM4EW_Calendars_V1.3.zip?download=1"

    response = requests.get(url)
    with open("GEOGLAM_CM4EW_Calendars_V1.3.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("GEOGLAM_CM4EW_Calendars_V1.3.zip", "r") as zip_ref:
        for member in zip_ref.infolist():
            # Skip directories
            if member.is_dir():
                continue
            
            # Get the path components
            filename = os.path.basename(member.filename)
            if not filename:
                continue
                
            # Construct target path directly in GEOGLAM_DIR
            target_path = os.path.join(GEOGLAM_DIR, filename)
            print(f"Extracting {member.filename} to {target_path}")
            
            # Read the file data and write to target
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

    os.remove("GEOGLAM_CM4EW_Calendars_V1.3.zip")

if __name__ == "__main__":
    main()