import rasterio
import os

def inspect_tiff(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with rasterio.open(file_path) as src:
        print(f"Inspecting file: {file_path}")
        print("-" * 40)
        print(f"Number of bands: {src.count}")
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"Coordinate Reference System (CRS): {src.crs}")
        print(f"Transform:\n{src.transform}")
        print("-" * 40)
        
        # Checking metadata
        print("Metadata:")
        for key, value in src.meta.items():
            print(f"  {key}: {value}")
        
        print("-" * 40)
        # Checking band descriptions
        for i in range(1, src.count + 1):
            desc = src.descriptions[i-1]
            print(f"Band {i} description: {desc}")
            
            # Read a small sample to see data range
            band_data = src.read(i)
            print(f"  Min: {band_data.min()}")
            print(f"  Max: {band_data.max()}")
            print(f"  Mean: {band_data.mean()}")

if __name__ == "__main__":
    tiff_file = os.path.join("..", "GGCP10_Production_2010_Maize.tif")
    # Absolute path for safety if needed, but relative should work if run from scripts dir
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GGCP10_Production_2010_Maize.tif"))
    inspect_tiff(abs_path)
