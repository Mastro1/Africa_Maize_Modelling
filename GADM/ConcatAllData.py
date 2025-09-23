import pandas as pd
from pathlib import Path

def concatenate_data(input_dir, output_dir, prefix):
    """
    Scans the input directory, groups files by administrative level (0, 1, and 2),
    and concatenates them into single, consolidated CSV files in the output directory.
    """
    print("=== Starting Data Concatenation Process ===")

    # 1. Define input and output directories using pathlib for cross-platform compatibility
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 2. Create the output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory ensured at: {output_dir}")
    except Exception as e:
        print(f"❌ CRITICAL: Could not create output directory. Error: {e}")
        return

    # 3. Check if the input directory exists
    if not input_dir.is_dir():
        print(f"❌ CRITICAL: Input directory not found at {input_dir}. Nothing to process.")
        return

    # 4. Scan and categorize all CSV files in the input directory
    all_files = list(input_dir.glob("*.csv"))
    
    file_groups = {
        "admin0": [f for f in all_files if f"_admin0_merged_data_{prefix}.csv" in f.name],
        "admin1": [f for f in all_files if f"_admin1_remote_sensing_data_{prefix}.csv" in f.name],
        "admin2": [f for f in all_files if f"_admin2_remote_sensing_data_{prefix}.csv" in f.name]
    }

    # 5. Process each group of files
    for level, files in file_groups.items():
        if not files:
            print(f"\n--- No files found for {level}. Skipping. ---")
            continue

        print(f"\n--- Processing {len(files)} files for {level}... ---")

        # Read and concatenate all files for the current level
        try:
            df_list = [pd.read_csv(file) for file in files]
            concatenated_df = pd.concat(df_list, ignore_index=True)
            print(f"   ✓ Read and concatenated {len(files)} files.")
            print(f"   - Total rows in concatenated file: {len(concatenated_df):,}")
            
            # Define the output file name
            if level == "admin0":
                output_name = f"admin0_merged_data_{prefix}_concat.csv"
            else:
                output_name = f"{level}_remote_sensing_data_{prefix}_concat.csv"
            
            output_path = output_dir / output_name

            # Save the concatenated dataframe
            concatenated_df.to_csv(output_path, index=False)
            print(f"   ✓ Successfully saved consolidated file to: {output_path}")

        except Exception as e:
            print(f"   ❌ ERROR: Could not process files for {level}. Error: {e}")

    print("\n=== Data Concatenation Process Finished ===")


if __name__ == "__main__":
    input_dir = Path("RemoteSensing/GADM/preprocessed")
    output_dir = Path("RemoteSensing/GADM/preprocessed_concat")
    concatenate_data(input_dir, output_dir, prefix="GLAM") 