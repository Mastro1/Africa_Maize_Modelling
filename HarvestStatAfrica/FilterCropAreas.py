"""
This script provides functionality to filter crop area data for a single country.
It is designed to work with the output of the 'CropAreas.py' script.
"""

import pandas as pd
from pathlib import Path

CROP_AREA_PERCENTAGE = 95
CROP_INTENSITY_PERCENTAGE = 60

def filter_crop_areas(country_name: str, crop_area_percentage: int = CROP_AREA_PERCENTAGE, crop_intensity_percentage: int = CROP_INTENSITY_PERCENTAGE):
    """
    Filters administrative areas for a given country based on cumulative crop area percentage.

    This function reads the crop area data for a specific country, then filters the
    administrative units (both admin1 and admin2 levels) to retain those that
    cumulatively contribute to a specified percentage of the total crop area for each year.

    Args:
        country_name (str): The name of the country to process.
        crop_area_percentage (int): The cumulative percentage of crop area to retain. Defaults to 95.

    Returns:
        str: The path to the generated filtered CSV file.

    Raises:
        FileNotFoundError: If the input crop area file for the country does not exist.
    """
    input_file = Path(f'HarvestStatAfrica/crop_areas/{country_name}_crop_areas_glad.csv')
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found at: {input_file}. "
            f"Please run the CropAreas script for '{country_name}' first."
        )

    print(f"Reading data for {country_name} from {input_file}...")
    df = pd.read_csv(input_file)

    admin2_df = df[df['admin_2'].notna()].copy()
    admin1_df = df[df['admin_2'].isna()].copy()

    print(f"Processing {country_name}: {len(admin1_df)} admin1 units, {len(admin2_df)} admin2 units.")

    all_filtered_results = []

    for admin_level, data_df in [('admin1', admin1_df), ('admin2', admin2_df)]:
        if data_df.empty:
            print(f"No {admin_level} data to process for {country_name}.")
            continue

        print(f"Filtering {admin_level} data...")
        
        yearly_results = []
        for year, year_data in data_df.groupby('year'):
            if year_data.empty:
                continue

            sorted_year_data = year_data.sort_values('crop_area_ha', ascending=False).copy()
            
            total_area_for_year = sorted_year_data['crop_area_ha'].sum()
            
            if total_area_for_year > 0:
                sorted_year_data['cumulative_area'] = sorted_year_data['crop_area_ha'].cumsum()
                sorted_year_data['cumulative_percentage'] = (sorted_year_data['cumulative_area'] / total_area_for_year) * 100
            else:
                sorted_year_data['cumulative_area'] = 0
                sorted_year_data['cumulative_percentage'] = 0

            filtered_data = sorted_year_data[(sorted_year_data['cumulative_percentage'] <= crop_area_percentage) | (sorted_year_data["crop_percentage"] >= crop_intensity_percentage)]

            if filtered_data.empty and not sorted_year_data.empty:
                filtered_data = sorted_year_data.iloc[[0]]
            
            yearly_results.append(filtered_data)
        
        if yearly_results:
            all_filtered_results.append(pd.concat(yearly_results))

    if not all_filtered_results:
        print(f"No data was left after filtering for {country_name}. No output file will be created.")
        return None

    final_df = pd.concat(all_filtered_results, ignore_index=True)
    
    final_df = final_df.drop(columns=['cumulative_area', 'cumulative_percentage'])

    final_df = final_df.sort_values(['PCODE', 'year']).reset_index(drop=True)
    
    output_dir = input_file.parent
    output_file = output_dir / f'{country_name}_crop_areas_glad_filtered.csv'
    
    final_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully filtered data for {country_name}.")
    print(f"Original number of observations: {len(df)}")
    print(f"Filtered number of observations: {len(final_df)}")
    print(f"Percentage of observations kept: {len(final_df)/len(df)*100:.2f}%")
    print(f"Filtered data saved to: {output_file}")

    return str(output_file)

def main():
    """Main function for command-line execution."""
    print("=== CROP AREA FILTERING SCRIPT ===")
    
    try:
        country = "Zambia"
        print(f"Running filter for country: {country}")
        filter_crop_areas(country)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 