import os
import requests
import zipfile
import shutil

# Correct output directory relative to the project root
OUTPUT_DIR = os.path.join("ResultsComparison", "GDHY", "data")
GDHY_URL = "https://store.pangaea.de/Publications/IizumiT_2019/gdhy_v1.2_v1.3_20190128.zip"

def download_and_extract_GDHY():
    print(f"Starting download from {GDHY_URL}...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    response = requests.get(GDHY_URL, stream=True)
    
    if response.status_code == 200:
        temp_zip_path = os.path.join(OUTPUT_DIR, "gdhy_temp.zip")
        
        # Download the file
        print("Downloading zip file (this may take a moment)...")
        with open(temp_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Download complete. Extracting 'maize_major' data...")
        
        # Extract specific files
        extracted_count = 0
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            # List all files in the zip
            all_files = zip_ref.namelist()
            
            # Filter for files inside 'maize_major' folder
            # We look for 'maize_major/' in the path and ensure it's not a directory itself
            maize_files = [f for f in all_files if "maize_major/" in f and not f.endswith("/")]
            
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
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
            print("Temporary zip file removed.")
        
    else:
        print(f"Failed to download data. Status code: {response.status_code}")

if __name__ == "__main__":
    download_and_extract_GDHY()
