"""
CropYield5min Data Downloader - SIMPLE VERSION

This script downloads the GlobalCropYield5min dataset from Mendeley.
After download, you MUST manually extract the RAR file:

MANUAL EXTRACTION STEPS:
1. Navigate to: ResultsComparison/cropyield5min/data/
2. Right-click on 'cropyield5min_data.rar'  
3. Choose "Extract here" or "Extract to folder"
4. Keep only files from: GlobalCropYield5min1982_2015_V4/GlobalCropYield5min/Maize/
"""

import os
import requests

OUTPUT_DIR = "ResultsComparison/cropyield5min/data"
url = "https://data.mendeley.com/public-files/datasets/hg8wzgx4yp/files/75ae8d3f-85d1-484f-bd82-5f6c35c2e252/file_downloaded"

def download_file(url, output_dir):
    """
    Simple downloader - manual extraction required!
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use fixed filename
    filename = 'cropyield5min_data.rar'
    output_path = os.path.join(output_dir, filename)

    # Check if file already exists
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        print_extraction_instructions()
        return True

    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    try:
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress
        with open(output_path, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Download progress: {progress:.1f}%", end='\r')

        print("\n✅ Download completed!")
        print_extraction_instructions()
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def print_extraction_instructions():
    """Print clear instructions for manual extraction"""
    print("\n" + "="*70)
    print("🔧 MANUAL EXTRACTION REQUIRED:")
    print("1. Navigate to: ResultsComparison/cropyield5min/data/")
    print("2. Right-click on 'cropyield5min_data.rar'")
    print("3. Choose 'Extract here' or 'Extract to folder'")
    print("4. Keep only files from: GlobalCropYield5min1982_2015_V4/GlobalCropYield5min/Maize/")
    print("="*70)

if __name__ == "__main__":
    print("🌽 CropYield5min Dataset Downloader")
    print("-" * 40)
    
    success = download_file(url, OUTPUT_DIR)
    
    if success:
        print("\n✅ Download process completed!")
        print("📁 File location: ResultsComparison/cropyield5min/data/cropyield5min_data.rar")
        print("⚠️  Remember to extract manually as instructed above!")
    else:
        print("\n❌ Download failed!")