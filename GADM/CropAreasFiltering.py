import pandas as pd
import numpy as np
import os

# Define the percentage of crop area to keep
CROP_AREA_PERCENTAGE = 95
CROP_INTENSITY_PERCENTAGE = 60

def main():
    # Path to the data file
    file_path = os.path.join('GADM', 'crop_areas', 'africa_crop_areas_glad.csv')

    # Read the CSV file
    print("Reading data file...")
    df = pd.read_csv(file_path)

    # Display basic information about the dataset
    print(f"Original dataset shape: {df.shape}")
    print(f"Number of countries: {df['country'].nunique()}")
    print(f"Years in dataset: {sorted(df['year'].unique())}")

    # Separate data into admin1 and admin2 levels
    # Admin2 level: has values in admin_2 column
    admin2_df = df[df['admin_2'].notna() & (df['admin_2'] != '')].copy()
    # Admin1 level: has no values in admin_2 column
    admin1_df = df[df['admin_2'].isna() | (df['admin_2'] == '')].copy()

    print(f"\nAdmin1 level data: {admin1_df.shape[0]} observations")
    print(f"Admin2 level data: {admin2_df.shape[0]} observations")

    # Process each country-year combination at admin2 level
    admin2_results = []
    admin2_kept_rows = 0
    admin2_total_rows = len(admin2_df)
    admin2_stats = []

    print("\nProcessing countries at admin2 level...")

    # Get unique country-year combinations
    country_years_admin2 = admin2_df[['country', 'year']].drop_duplicates()

    for _, row in country_years_admin2.iterrows():
        country = row['country']
        year = row['year']

        # Get data for this country and year
        country_year_data = admin2_df[(admin2_df['country'] == country) & (admin2_df['year'] == year)].copy()

        if len(country_year_data) == 0:
            continue

        # Sort by crop_area_ha in descending order
        country_year_data = country_year_data.sort_values('crop_area_ha', ascending=False)

        # Calculate cumulative sum and percentage
        country_year_data['cumulative_area'] = country_year_data['crop_area_ha'].cumsum()
        total_area = country_year_data['crop_area_ha'].sum()
        country_year_data['cumulative_percentage'] = (country_year_data['cumulative_area'] / total_area) * 100

        # Keep rows up to crop_area_percentage% of total crop area and highly cultivated
        filtered_data = country_year_data[(country_year_data['cumulative_percentage'] <= CROP_AREA_PERCENTAGE) | (country_year_data['crop_percentage'] >= CROP_INTENSITY_PERCENTAGE)]
        #filtered_data = country_year_data[(country_year_data['cumulative_percentage'] <= CROP_AREA_PERCENTAGE)]

        # If filtering removed all rows, keep at least the top row
        if len(filtered_data) == 0 and len(country_year_data) > 0:
            filtered_data = country_year_data.iloc[[0]]

        # Add to results
        admin2_results.append(filtered_data)
        admin2_kept_rows += len(filtered_data)

        # Collect statistics
        admin2_stats.append({
            'country': country,
            'year': year,
            'total_units': len(country_year_data),
            'kept_units': len(filtered_data),
            'dropped_units': len(country_year_data) - len(filtered_data),
            'kept_percentage': len(filtered_data) / len(country_year_data) * 100 if len(country_year_data) > 0 else 0,
            'total_crop_area': total_area,
            'kept_crop_area': filtered_data['crop_area_ha'].sum(),
            'kept_area_percentage': filtered_data['crop_area_ha'].sum() / total_area * 100 if total_area > 0 else 0
        })

    # Combine all filtered admin2 data
    admin2_filtered_df = pd.concat(admin2_results, ignore_index=True) if admin2_results else pd.DataFrame()

    # Process each country-year combination at admin1 level
    admin1_results = []
    admin1_kept_rows = 0
    admin1_total_rows = len(admin1_df)
    admin1_stats = []

    print("\nProcessing countries at admin1 level...")
    # Get unique country-year combinations
    country_years_admin1 = admin1_df[['country', 'year']].drop_duplicates()

    for _, row in country_years_admin1.iterrows():
        country = row['country']
        year = row['year']

        # Get data for this country and year
        country_year_data = admin1_df[(admin1_df['country'] == country) & (admin1_df['year'] == year)].copy()

        if len(country_year_data) == 0:
            continue

        # Sort by crop_area_ha in descending order
        country_year_data = country_year_data.sort_values('crop_area_ha', ascending=False)

        # Calculate cumulative sum and percentage
        country_year_data['cumulative_area'] = country_year_data['crop_area_ha'].cumsum()
        total_area = country_year_data['crop_area_ha'].sum()
        country_year_data['cumulative_percentage'] = (country_year_data['cumulative_area'] / total_area) * 100

        # Keep rows up to crop_area_percentage% of total crop area
        filtered_data = country_year_data[(country_year_data['cumulative_percentage'] <= CROP_AREA_PERCENTAGE) | (country_year_data['crop_percentage'] >= CROP_INTENSITY_PERCENTAGE)]

        # If filtering removed all rows, keep at least the top row
        if len(filtered_data) == 0 and len(country_year_data) > 0:
            filtered_data = country_year_data.iloc[[0]]

        # Add to results
        admin1_results.append(filtered_data)
        admin1_kept_rows += len(filtered_data)

        # Collect statistics
        admin1_stats.append({
            'country': country,
            'year': year,
            'total_units': len(country_year_data),
            'kept_units': len(filtered_data),
            'dropped_units': len(country_year_data) - len(filtered_data),
            'kept_percentage': len(filtered_data) / len(country_year_data) * 100 if len(country_year_data) > 0 else 0,
            'total_crop_area': total_area,
            'kept_crop_area': filtered_data['crop_area_ha'].sum(),
            'kept_area_percentage': filtered_data['crop_area_ha'].sum() / total_area * 100 if total_area > 0 else 0
        })

    # Combine all filtered admin1 data
    admin1_filtered_df = pd.concat(admin1_results, ignore_index=True) if admin1_results else pd.DataFrame()

    # Combine both filtered datasets
    filtered_df = pd.concat([admin1_filtered_df, admin2_filtered_df], ignore_index=True)

    # Save the filtered data
    output_path = os.path.join('GADM', 'crop_areas', 'africa_crop_areas_glad_filtered.csv')
    filtered_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()