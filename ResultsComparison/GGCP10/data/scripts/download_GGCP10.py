# Download GGCP10

import os
import requests
import zipfile
import shutil

# To download the data, you need to go to the link and download the zip file
# In case you downoaded this script unzips the file

# Correct output directory relative to the project root
OUTPUT_DIR = os.path.join("ResultsComparison", "GGCP10", "data")
GGCP10_URL = "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/G1HBNK"


def extract_GGCP10():
    print(f"Starting download from {GGCP10_URL}...")
    
    # Extract specific files
    extracted_count = 0
    with zipfile.ZipFile(r"ResultsComparison\GGCP10\data\dataverse_files.zip", 'r') as zip_ref:
        # List all files in the zip
        all_files = zip_ref.namelist()
        
        # Filter for files inside 'maize_major' folder
        # We look for 'maize_major/' in the path and ensure it's not a directory itself
        maize_files = [f for f in all_files if "Maize" in f]
        
        for file_path in maize_files:
            # We flatten the structure: extract file directly to OUTPUT_DIR
            filename = os.path.basename(file_path)
            if filename:
                target_path = os.path.join(OUTPUT_DIR, filename)
                
                with open(target_path, 'wb') as target_file:
                    with zip_ref.open(file_path) as source_file:
                        shutil.copyfileobj(source_file, target_file)
                
                extracted_count += 1

    print(f"Successfully extracted {extracted_count} files to {OUTPUT_DIR}")
    
    # Clean up
    if os.path.exists(r"ResultsComparison\GGCP10\data\dataverse_files.zip"):
        os.remove(r"ResultsComparison\GGCP10\data\dataverse_files.zip")
        print("Temporary zip file removed.")
    

def open_link():
    import webbrowser
    webbrowser.open(link)

if __name__ == "__main__":
    if not os.path.exists(r"ResultsComparison\GGCP10\data\dataverse_files.zip"):
        open_link()
    else:
        extract_GGCP10()
        
