"""
Data Availability Checker Module

This module provides functions to check the availability of various types of data
for different countries and administrative levels in the HarvestStat Africa project.
"""

import pandas as pd
import os

# Cache for ground data and admin levels to avoid repeated CSV reads
_GROUND_DATA_CACHE = None
_ADMIN_LEVELS_CACHE = {}


def _load_ground_data():
    """
    Load ground data once and cache it.
    
    Returns:
        pd.DataFrame: Ground data or None if not found
    """
    global _GROUND_DATA_CACHE
    if _GROUND_DATA_CACHE is None:
        try:
            _GROUND_DATA_CACHE = pd.read_csv('HarvestStatAfrica/data/hvstat_africa_data_v1.0.csv')
            _GROUND_DATA_CACHE = _GROUND_DATA_CACHE[_GROUND_DATA_CACHE['product'] == 'Maize']
        except FileNotFoundError:
            print("Warning: Could not load ground data file.")
            _GROUND_DATA_CACHE = pd.DataFrame()
    return _GROUND_DATA_CACHE


def _get_all_admin_levels():
    """
    Get admin levels for all countries at once and cache them.
    
    Returns:
        dict: Dictionary mapping country names to admin levels (1 or 2)
    """
    global _ADMIN_LEVELS_CACHE
    if not _ADMIN_LEVELS_CACHE:
        ground_data = _load_ground_data()
        if not ground_data.empty:
            for _, row in ground_data.iterrows():
                country = row['country']
                admin_level = 1 if row.get('admin_2', '') == 'none' else 2
                _ADMIN_LEVELS_CACHE[country] = admin_level
    return _ADMIN_LEVELS_CACHE


def get_admin_level(country):
    """
    Get the default admin level for a country based on the ground data.
    
    Args:
        country (str): Name of the country
        
    Returns:
        int: Admin level (1 or 2) based on whether admin_2 data exists
    """
    admin_levels = _get_all_admin_levels()
    return admin_levels.get(country, 2)  # Default to admin2 if country not found


def define_type_of_data(file_name):
    """
    Determine the type of data based on the file name.
    
    Args:
        file_name (str): Name of the file to analyze
        
    Returns:
        str: Type of data (VI, NDWI, ERA5, merged_data, crop_areas, remote_sensing, Unknown)
    """
    if "VI" in file_name:
        return "VI"
    elif "NDWI" in file_name:
        return "NDWI"
    elif "ERA5" in file_name:
        return "ERA5"
    elif "merged_data" in file_name:
        return "merged_data"
    elif "crop_areas_glad" in file_name:
        return "crop_areas"
    elif "remote_sensing_data" in file_name:
        return "remote_sensing"
    else:
        return "Unknown"
    
    
def define_admin_level(file_name):
    """
    Determine the administrative level based on the file name.
    
    Args:
        file_name (str): Name of the file to analyze
        
    Returns:
        str: Administrative level (admin0, admin1, admin2, Unknown)
    """
    if "admin0" in file_name:
        return "admin0"
    elif "admin1" in file_name:
        return "admin1"
    elif "admin2" in file_name:
        return "admin2"
    else:
        return "Unknown"
    
    
def define_country(file_name):
    """
    Extract country name from the file name.
    
    Args:
        file_name (str): Name of the file to analyze
        
    Returns:
        str: Country name (handles multi-word countries for EVI/results files)
    """
    # Check if this is a file from EVI/results (VI, NDWI, ERA5 data)
    if any(indicator in file_name for indicator in ["_VI_", "_NDWI_", "_ERA5_"]):
        # For EVI/results files, extract country name up to admin level
        parts = file_name.split("_")
        country_parts = []
        
        for i, part in enumerate(parts):
            if part.startswith("admin"):
                break
            country_parts.append(part)
        
        return "_".join(country_parts).replace("_", " ")
    else:
        # For other files (merged_data, remote_sensing_data, etc.), use original logic
        return file_name.split("_")[0]


def get_country_data_files(country):
    """
    Get list of data files available for a specific country.
    
    Args:
        country (str): Name of the country
        
    Returns:
        list: List of matching data files for the country
    """
    files_to_check = [
        f"{country}_admin0_merged_data.csv", 
        f"{country}_admin1_merged_data.csv", 
        f"{country}_admin2_merged_data.csv", 
        f"{country}_crop_areas_glad.csv", 
        f"{country}_admin0_remote_sensing_data.csv", 
        f"{country}_admin1_remote_sensing_data.csv", 
        f"{country}_admin2_remote_sensing_data.csv"
    ]
    dir_path = f"HarvestStatAfrica/downscaling/data/{country}"
    if os.path.exists(dir_path):
        list_of_files = os.listdir(dir_path)
        matching_files = []
        for file in list_of_files:
            if any(file_to_check in file for file_to_check in files_to_check):
                matching_files.append(file)
        return matching_files
    else:
        return []


def check_remote_sensing_data_availability():
    """
    Check availability of remote sensing data in the EVI results directory.
    
    Returns:
        list: List of dictionaries containing country, type, and admin_level information
    """
    remote_sensing_dir = "HarvestStatAfrica/remote_sensing/extractions"
    if not os.path.exists(remote_sensing_dir):
        return []
        
    list_of_files = os.listdir(remote_sensing_dir)
    data_rows = []
    
    for file in list_of_files:
        country = define_country(file)
        data_type = define_type_of_data(file)
        admin_level = define_admin_level(file)
        if data_type != "Unknown" and admin_level != "Unknown":
            data_rows.append({"country": country, "type": data_type, "admin_level": admin_level})
    
    return data_rows


def check_country_specific_data_availability(countries_list):
    """
    Check availability of country-specific data in downscaling directories.
    
    Args:
        countries_list (list): List of countries to check
        
    Returns:
        list: List of dictionaries containing country, type, and admin_level information
    """
    data_rows = []
    
    for country in countries_list:
        data_files = get_country_data_files(country)
        for file in data_files:
            data_type = define_type_of_data(file)
            admin_level = define_admin_level(file)
            if data_type != "Unknown":
                data_rows.append({"country": country, "type": data_type, "admin_level": admin_level})
    
    return data_rows


def get_data_availability_summary():
    """
    Get a comprehensive summary of data availability across all sources.
    
    Returns:
        pd.DataFrame: DataFrame containing all available data information
    """
    # Load ground data to get list of countries
    ground_data = _load_ground_data()
    list_of_countries = ground_data['country'].unique() if not ground_data.empty else []
    
    # Collect all data availability information
    data_rows = []
    
    # Check remote sensing data
    data_rows.extend(check_remote_sensing_data_availability())
    
    # Check country-specific data
    data_rows.extend(check_country_specific_data_availability(list_of_countries))
    
    # Create DataFrame and sort
    data_available = pd.DataFrame(data_rows)
    if not data_available.empty:
        data_available.sort_values(by=["country", "type", "admin_level"], inplace=True)
        data_available.reset_index(drop=True, inplace=True)
    
    return data_available


def create_comprehensive_data_availability_matrix():
    """
    Create a comprehensive matrix showing data availability by country, data type, and admin level.
    Each data type gets separate columns for admin0, admin1, and admin2.
    
    Returns:
        pd.DataFrame: Comprehensive matrix with countries as rows and detailed columns showing
                     TRUE/FALSE for each data type and admin level combination
    """
    # Get the detailed data availability once
    data_available = get_data_availability_summary()
    
    # Load ground data to get complete list of countries
    ground_data = _load_ground_data()
    all_countries = sorted(ground_data['country'].unique()) if not ground_data.empty else []
    
    # Pre-compute admin levels for all countries
    admin_levels_dict = _get_all_admin_levels()
    
    # Get all possible data types (excluding crop_areas which doesn't have admin levels)
    if not data_available.empty:
        all_data_types = sorted([dt for dt in data_available['type'].unique() if dt != 'crop_areas'])
    else:
        all_data_types = ['VI', 'NDWI', 'ERA5', 'merged_data', 'remote_sensing']
    
    # Define admin levels
    admin_levels = ['admin0', 'admin1', 'admin2']
    
    # Create a lookup dictionary for faster access
    data_lookup = {}
    if not data_available.empty:
        for _, row in data_available.iterrows():
            key = (row['country'], row['type'], row['admin_level'])
            data_lookup[key] = True
    
    # Create comprehensive matrix
    matrix_data = []
    
    for country in all_countries:
        country_row = {'country': country}
        
        # Add required admin level for this country (just the number)
        country_row['required_admin_level'] = admin_levels_dict.get(country, 2)
        
        # Check each data type and admin level combination
        for data_type in all_data_types:
            # Create columns for each admin level
            for admin_level in admin_levels:
                column_name = f"{data_type}_{admin_level}"
                country_row[column_name] = data_lookup.get((country, data_type, admin_level), False)
        
        # Handle crop_areas separately (no admin levels)
        country_row['crop_areas'] = data_lookup.get((country, 'crop_areas', 'Unknown'), False) or \
                                   any(data_lookup.get((country, 'crop_areas', admin), False) for admin in admin_levels)
        
        matrix_data.append(country_row)
    
    # Create DataFrame
    matrix_df = pd.DataFrame(matrix_data)
    
    # Set country as index
    if not matrix_df.empty:
        matrix_df.set_index('country', inplace=True)
    
    return matrix_df


def create_summary_matrix_with_completeness():
    """
    Create a summary matrix showing completeness status for each country and data type.
    Shows whether each country has the required admin levels for each data type.
    
    Returns:
        pd.DataFrame: Summary matrix showing completeness status
    """
    # Get comprehensive matrix
    comprehensive_matrix = create_comprehensive_data_availability_matrix()
    
    if comprehensive_matrix.empty:
        return pd.DataFrame()
    
    # Get all data types (excluding crop_areas and required_admin_level)
    data_types = []
    for col in comprehensive_matrix.columns:
        if col not in ['required_admin_level', 'crop_areas'] and '_admin' in col:
            data_type = col.split('_admin')[0]
            if data_type not in data_types:
                data_types.append(data_type)
    
    # Create summary matrix
    summary_data = []
    
    for country in comprehensive_matrix.index:
        country_row = {'country': country}
        
        # Get required admin level
        required_admin_num = comprehensive_matrix.loc[country, 'required_admin_level']
        required_admin = f'admin{required_admin_num}'
        country_row['required_admin_level'] = required_admin_num
        
        # Check completeness for each data type
        for data_type in sorted(data_types):
            # Check if country has admin0 and required admin level
            has_admin0 = comprehensive_matrix.loc[country, f"{data_type}_admin0"]
            has_required_admin = comprehensive_matrix.loc[country, f"{data_type}_{required_admin}"]
            
            if has_admin0 and has_required_admin:
                status = "COMPLETE"
            elif has_admin0 or has_required_admin:
                available = []
                if has_admin0:
                    available.append("admin0")
                if has_required_admin:
                    available.append(required_admin)
                status = f"PARTIAL ({', '.join(available)})"
            else:
                status = "MISSING"
            
            country_row[data_type] = status
        
        # Handle crop_areas
        country_row['crop_areas'] = "YES" if comprehensive_matrix.loc[country, 'crop_areas'] else "NO"
        
        summary_data.append(country_row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Set country as index
    if not summary_df.empty:
        summary_df.set_index('country', inplace=True)
    
    return summary_df


def get_comprehensive_data_statistics():
    """
    Get comprehensive statistics about data availability including detailed breakdowns.
    
    Returns:
        dict: Comprehensive statistics
    """
    comprehensive_matrix = create_comprehensive_data_availability_matrix()
    
    if comprehensive_matrix.empty:
        return {}
    
    # Get data types
    data_types = []
    for col in comprehensive_matrix.columns:
        if col not in ['required_admin_level', 'crop_areas'] and '_admin' in col:
            data_type = col.split('_admin')[0]
            if data_type not in data_types:
                data_types.append(data_type)
    
    stats = {
        'total_countries': len(comprehensive_matrix),
        'total_data_types': len(data_types) + 1,  # +1 for crop_areas
        'admin_level_breakdown': {},
        'data_type_completeness': {},
        'countries_with_complete_data': [],
        'countries_with_no_data': []
    }
    
    # Calculate admin level breakdown
    for admin_level in ['admin0', 'admin1', 'admin2']:
        admin_cols = [col for col in comprehensive_matrix.columns if col.endswith(f'_{admin_level}')]
        if admin_cols:
            total_possible = len(comprehensive_matrix) * len(admin_cols)
            total_available = comprehensive_matrix[admin_cols].sum().sum()
            stats['admin_level_breakdown'][admin_level] = {
                'available': int(total_available),
                'total_possible': total_possible,
                'percentage': round((total_available / total_possible) * 100, 2) if total_possible > 0 else 0
            }
    
    # Calculate data type completeness
    summary_matrix = create_summary_matrix_with_completeness()
    for data_type in data_types + ['crop_areas']:
        if data_type in summary_matrix.columns:
            if data_type == 'crop_areas':
                complete_count = (summary_matrix[data_type] == 'YES').sum()
            else:
                complete_count = (summary_matrix[data_type] == 'COMPLETE').sum()
            
            stats['data_type_completeness'][data_type] = {
                'complete_countries': int(complete_count),
                'total_countries': len(summary_matrix),
                'completion_rate': round((complete_count / len(summary_matrix)) * 100, 2)
            }
    
    # Find countries with complete/no data
    if not summary_matrix.empty:
        for country in summary_matrix.index:
            row = summary_matrix.loc[country]
            data_cols = [col for col in row.index if col != 'required_admin_level']
            
            # Check if all data types are complete
            all_complete = all(
                (row[col] == 'COMPLETE' if col != 'crop_areas' else row[col] == 'YES') 
                for col in data_cols
            )
            if all_complete:
                stats['countries_with_complete_data'].append(country)
            
            # Check if no data at all
            no_data = all(
                (row[col] == 'MISSING' if col != 'crop_areas' else row[col] == 'NO') 
                for col in data_cols
            )
            if no_data:
                stats['countries_with_no_data'].append(country)
    
    return stats


def print_comprehensive_data_report():
    """
    Print a comprehensive report of data availability with detailed breakdowns.
    """
    print("Comprehensive Data Availability Report")
    print("=" * 70)
    
    # Get comprehensive matrix
    comprehensive_matrix = create_comprehensive_data_availability_matrix()
    
    if comprehensive_matrix.empty:
        print("No data availability information found.")
        return
    
    print("\n1. COMPREHENSIVE MATRIX (TRUE/FALSE for each admin level):")
    print("-" * 60)
    print(comprehensive_matrix.to_string())
    
    # Get summary matrix
    summary_matrix = create_summary_matrix_with_completeness()
    
    print("\n\n2. COMPLETENESS SUMMARY (COMPLETE/PARTIAL/MISSING):")
    print("-" * 60)
    print(summary_matrix.to_string())
    
    # Get statistics
    stats = get_comprehensive_data_statistics()
    
    print(f"\n\n3. STATISTICS:")
    print("-" * 60)
    print(f"Total Countries: {stats['total_countries']}")
    print(f"Total Data Types: {stats['total_data_types']}")
    
    print(f"\nAdmin Level Availability:")
    for admin_level, data in stats['admin_level_breakdown'].items():
        print(f"  {admin_level}: {data['available']}/{data['total_possible']} ({data['percentage']}%)")
    
    print(f"\nData Type Completion Rates:")
    for data_type, data in stats['data_type_completeness'].items():
        print(f"  {data_type}: {data['complete_countries']}/{data['total_countries']} countries ({data['completion_rate']}%)")
    
    if stats['countries_with_complete_data']:
        print(f"\nCountries with Complete Data ({len(stats['countries_with_complete_data'])}):")
        for country in stats['countries_with_complete_data']:
            print(f"  ✓ {country}")
    
    if stats['countries_with_no_data']:
        print(f"\nCountries with No Data ({len(stats['countries_with_no_data'])}):")
        for country in stats['countries_with_no_data']:
            print(f"  ✗ {country}")


# Update the main create_data_availability_matrix to use the comprehensive version
def create_data_availability_matrix():
    """
    Create a pivot matrix showing data availability by country and data type.
    This is now an alias for the comprehensive matrix function.
    
    Returns:
        pd.DataFrame: Comprehensive matrix with detailed admin level breakdown
    """
    return create_comprehensive_data_availability_matrix()


def create_detailed_data_availability_matrix():
    """
    Create a detailed matrix showing exactly which admin levels are available for each country and data type.
    
    Returns:
        pd.DataFrame: Detailed matrix showing admin levels available
    """
    # Get the detailed data availability
    data_available = get_data_availability_summary()
    
    # Load ground data to get complete list of countries
    try:
        ground_data = pd.read_csv("public/hvstat_africa_data_v1.0.csv")
        all_countries = sorted(ground_data['country'].unique())
    except FileNotFoundError:
        print("Warning: Could not load ground data file. Using countries from available data.")
        all_countries = sorted(data_available['country'].unique()) if not data_available.empty else []
    
    # Get all possible data types
    if not data_available.empty:
        all_data_types = sorted(data_available['type'].unique())
    else:
        all_data_types = ['VI', 'NDWI', 'ERA5', 'merged_data', 'crop_areas', 'remote_sensing']
    
    # Create detailed matrix
    matrix_data = []
    
    for country in all_countries:
        country_row = {'country': country}
        country_admin_level = get_admin_level(country)
        country_row['required_admin_level'] = f"admin{country_admin_level}"
        
        # Get available data for this country
        country_data = data_available[data_available['country'] == country] if not data_available.empty else pd.DataFrame()
        
        # Check each data type
        for data_type in all_data_types:
            type_data = country_data[country_data['type'] == data_type] if not country_data.empty else pd.DataFrame()
            available_admin_levels = sorted(type_data['admin_level'].unique()) if not type_data.empty else []
            
            if data_type == 'crop_areas':
                # For crop_areas, show if it exists
                country_row[f"{data_type}"] = "YES" if available_admin_levels else "NO"
            else:
                # For other data types, show which admin levels are available
                admin_status = []
                if 'admin0' in available_admin_levels:
                    admin_status.append('admin0')
                if f'admin{country_admin_level}' in available_admin_levels:
                    admin_status.append(f'admin{country_admin_level}')
                
                if len(admin_status) == 2:
                    country_row[f"{data_type}"] = "COMPLETE"
                elif len(admin_status) == 1:
                    country_row[f"{data_type}"] = f"PARTIAL ({', '.join(admin_status)})"
                else:
                    country_row[f"{data_type}"] = "MISSING"
            
        matrix_data.append(country_row)
    
    # Create DataFrame
    matrix_df = pd.DataFrame(matrix_data)
    
    # Set country as index
    if not matrix_df.empty:
        matrix_df.set_index('country', inplace=True)
    
    return matrix_df


def get_missing_data_summary():
    """
    Get a summary of missing data by country and data type.
    
    Returns:
        dict: Summary containing missing data statistics
    """
    matrix = create_data_availability_matrix()
    
    if matrix.empty:
        return {
            'total_countries': 0,
            'total_data_types': 0,
            'missing_combinations': 0,
            'completion_rate': 0.0,
            'countries_with_no_data': [],
            'data_types_with_most_gaps': [],
            'countries_with_complete_data': []
        }
    
    # Calculate statistics
    total_countries = len(matrix)
    total_data_types = len(matrix.columns)
    total_possible_combinations = total_countries * total_data_types
    
    # Count TRUE values (available data)
    available_combinations = matrix.sum().sum()
    missing_combinations = total_possible_combinations - available_combinations
    completion_rate = (available_combinations / total_possible_combinations) * 100 if total_possible_combinations > 0 else 0
    
    # Countries with no data at all
    countries_with_no_data = matrix[matrix.sum(axis=1) == 0].index.tolist()
    
    # Countries with complete data (all data types available)
    countries_with_complete_data = matrix[matrix.sum(axis=1) == total_data_types].index.tolist()
    
    # Data types with most gaps (least available across countries)
    data_type_availability = matrix.sum().sort_values()
    data_types_with_most_gaps = data_type_availability.head(3).index.tolist()
    
    return {
        'total_countries': total_countries,
        'total_data_types': total_data_types,
        'available_combinations': int(available_combinations),
        'missing_combinations': int(missing_combinations),
        'total_possible_combinations': total_possible_combinations,
        'completion_rate': round(completion_rate, 2),
        'countries_with_no_data': countries_with_no_data,
        'countries_with_complete_data': countries_with_complete_data,
        'data_types_with_most_gaps': data_types_with_most_gaps,
        'data_type_availability_counts': data_type_availability.to_dict()
    }


def print_data_availability_report():
    """
    Print a comprehensive report of data availability including matrix and summary statistics.
    """
    print("Data Availability Report")
    print("=" * 60)
    
    # Get the matrix
    matrix = create_data_availability_matrix()
    
    if matrix.empty:
        print("No data availability information found.")
        return
    
    print("\nData Availability Matrix:")
    print("-" * 40)
    print("(TRUE = Data Available, FALSE = Data Missing)")
    print()
    print(matrix.to_string())
    
    # Get summary statistics
    summary = get_missing_data_summary()
    
    print(f"\n\nSummary Statistics:")
    print("-" * 40)
    print(f"Total Countries: {summary['total_countries']}")
    print(f"Total Data Types: {summary['total_data_types']}")
    print(f"Available Data Combinations: {summary['available_combinations']}/{summary['total_possible_combinations']}")
    print(f"Overall Completion Rate: {summary['completion_rate']}%")
    
    if summary['countries_with_complete_data']:
        print(f"\nCountries with Complete Data ({len(summary['countries_with_complete_data'])}):")
        for country in summary['countries_with_complete_data']:
            print(f"  ✓ {country}")
    
    if summary['countries_with_no_data']:
        print(f"\nCountries with No Data ({len(summary['countries_with_no_data'])}):")
        for country in summary['countries_with_no_data']:
            print(f"  ✗ {country}")
    
    print(f"\nData Type Availability (number of countries with data):")
    for data_type, count in sorted(summary['data_type_availability_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / summary['total_countries']) * 100
        print(f"  {data_type}: {count}/{summary['total_countries']} countries ({percentage:.1f}%)")


def main():
    """
    Main function to run the comprehensive data availability check and print results.
    """
    print("Data Availability Analysis")
    print("=" * 50)
    
    # Get the original detailed data availability
    data_available = get_data_availability_summary()
    print("Raw Data Availability:")
    print("-" * 30)
    print(data_available)
    
    print("\n" + "="*70)
    
    # Get and print comprehensive matrix
    comprehensive_matrix = create_comprehensive_data_availability_matrix()
    print("Comprehensive Data Availability Matrix:")
    print("-" * 50)
    print("(TRUE/FALSE for each data type and admin level)")
    print()
    print(comprehensive_matrix)


if __name__ == "__main__":
    main()
