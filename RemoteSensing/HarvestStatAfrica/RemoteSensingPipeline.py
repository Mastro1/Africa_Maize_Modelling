"""
Optimized Remote Sensing Data Extraction with Task Queue Management

This script leverages the data availability matrix to only extract data that is missing,
and manages Google Earth Engine task queue to maintain optimal throughput.
"""

import sys
import os
import time
import pandas as pd
import ee
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EE_PROJECT = os.getenv('EE_PROJECT')
EE_OPT_URL = os.getenv('EE_OPT_URL')
GD_FOLDER = os.getenv('GD_FOLDER')


# Add the Training directory to Python path so RemoteSensing modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from RemoteSensing.NDVI import extractVI
from RemoteSensing.NDWI import extractNDWI
from RemoteSensing.ERA5 import extractERA5
from HarvestStatAfrica.DataAvailability import create_comprehensive_data_availability_matrix, get_admin_level

# Initialize Earth Engine
try:
    ee.Initialize(project=EE_PROJECT, opt_url=EE_OPT_URL)
    print("✅ Google Earth Engine initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Google Earth Engine: {e}")
    print("Please run 'earthengine authenticate' first")


def get_running_tasks():
    """
    Get the number of currently running tasks in Google Earth Engine.
    
    Returns:
        tuple: (number_of_running_tasks, list_of_running_task_ids)
    """
    try:
        tasks = ee.batch.Task.list()
        running_tasks = [task for task in tasks if task.state in ['RUNNING', 'READY']]
        return len(running_tasks), [task.id for task in running_tasks]
    except Exception as e:
        print(f"❌ Error checking task status: {e}")
        return 0, []


def get_task_status_summary():
    """
    Get a comprehensive summary of all tasks.
    
    Returns:
        dict: Summary of task counts by status
    """
    try:
        tasks = ee.batch.Task.list()
        status_counts = {}
        for task in tasks:
            status = task.state
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
    except Exception as e:
        print(f"❌ Error getting task summary: {e}")
        return {}


def wait_for_available_slot(max_concurrent_tasks=5, check_interval_minutes=2):
    """
    Wait until there's an available slot for new tasks.
    
    Args:
        max_concurrent_tasks (int): Maximum number of concurrent tasks allowed
        check_interval_minutes (int): Minutes to wait between checks
    
    Returns:
        bool: True if slot available, False if error
    """
    while True:
        running_count, running_ids = get_running_tasks()
        
        print(f"📊 Currently running tasks: {running_count}/{max_concurrent_tasks}")
        
        if running_count < max_concurrent_tasks:
            print(f"✅ Slot available! ({max_concurrent_tasks - running_count} free slots)")
            return True
        
        print(f"⏳ Queue full. Waiting {check_interval_minutes} minutes before next check...")
        time.sleep(check_interval_minutes * 60)


def submit_task_with_queue_management(extract_function, country, start_date, end_date, 
                                    shapefile_path, output_prefix, admin_level,
                                    max_concurrent_tasks=5, check_interval_minutes=2):
    """
    Submit a task while respecting queue limits.
    
    Args:
        extract_function: The extraction function to call
        country, start_date, end_date, shapefile_path, output_prefix, admin_level: Task parameters
        max_concurrent_tasks (int): Maximum concurrent tasks
        check_interval_minutes (int): Minutes between queue checks
    
    Returns:
        task: The submitted task or None if failed
    """
    # Wait for available slot
    if not wait_for_available_slot(max_concurrent_tasks, check_interval_minutes):
        return None
    
    # Submit the task
    try:
        task = extract_function(country, start_date, end_date, shapefile_path, output_prefix, admin_level)
        if task:
            print(f"🚀 Task submitted successfully! ID: {task.id}")
        return task
    except Exception as e:
        print(f"❌ Error submitting task: {e}")
        return None


def get_missing_extractions():
    """
    Analyze data availability matrix and determine which extractions are needed.
    
    Returns:
        list: List of dictionaries containing extraction tasks needed
    """
    # Get comprehensive data availability matrix
    matrix = create_comprehensive_data_availability_matrix()
    
    if matrix.empty:
        print("No data availability information found.")
        return []
    
    # Define the data types we can extract
    extractable_types = {
        'VI': 'VI',
        'NDWI': 'NDWI', 
        'ERA5': 'ERA5'
    }
    
    missing_extractions = []
    
    for country in matrix.index:
        required_admin_level = matrix.loc[country, 'required_admin_level']
        
        for data_type, extraction_key in extractable_types.items():
            # Check what admin levels are missing for this data type
            has_admin0 = matrix.loc[country, f'{data_type}_admin0']
            has_required_admin = matrix.loc[country, f'{data_type}_admin{required_admin_level}']
            
            missing_admin_levels = []
            if not has_admin0:
                missing_admin_levels.append(0)
            if not has_required_admin:
                missing_admin_levels.append(required_admin_level)
            
            # Add extraction tasks for missing admin levels
            for admin_level in missing_admin_levels:
                missing_extractions.append({
                    'country': country,
                    'data_type': data_type,
                    'admin_level': admin_level,
                    'extraction_key': extraction_key
                })
    
    return missing_extractions


def print_extraction_summary(missing_extractions):
    """
    Print a summary of what extractions will be performed.
    
    Args:
        missing_extractions (list): List of extraction tasks
    """
    if not missing_extractions:
        print("🎉 All data is already available! No extractions needed.")
        return
    
    print(f"📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total extractions needed: {len(missing_extractions)}")
    
    # Group by country
    countries = {}
    for task in missing_extractions:
        country = task['country']
        if country not in countries:
            countries[country] = []
        countries[country].append(f"{task['data_type']}_admin{task['admin_level']}")
    
    print(f"Countries requiring extractions: {len(countries)}")
    print()
    
    for country, tasks in countries.items():
        print(f"🌍 {country}:")
        for task in sorted(tasks):
            print(f"    ✓ {task}")
        print()
    
    # Group by data type
    data_types = {}
    for task in missing_extractions:
        data_type = task['data_type']
        if data_type not in data_types:
            data_types[data_type] = 0
        data_types[data_type] += 1
    
    print("📈 Extractions by data type:")
    for data_type, count in data_types.items():
        print(f"    {data_type}: {count} extractions")


def run_optimized_extractions_with_queue(dry_run=True, max_concurrent_tasks=5, check_interval_minutes=2):
    """
    Run only the missing data extractions with intelligent queue management.
    
    Args:
        dry_run (bool): If True, just print what would be done without actually running
        max_concurrent_tasks (int): Maximum number of concurrent tasks
        check_interval_minutes (int): Minutes to wait between queue checks
    """
    print("🚀 OPTIMIZED REMOTE SENSING EXTRACTION WITH QUEUE MANAGEMENT")
    print("=" * 80)
    
    # Show current task status
    task_summary = get_task_status_summary()
    print(f"📊 Current GEE Task Status:")
    for status, count in task_summary.items():
        print(f"    {status}: {count}")
    print()
    
    # Get missing extractions
    missing_extractions = get_missing_extractions()
    
    # Print summary
    print_extraction_summary(missing_extractions)
    
    if not missing_extractions:
        return
    
    print(f"\n⚙️ Queue Settings:")
    print(f"    Max concurrent tasks: {max_concurrent_tasks}")
    print(f"    Check interval: {check_interval_minutes} minutes")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No actual extractions will be performed")
        print("Set dry_run=False to run actual extractions")
        return
    
    # Extraction parameters
    extraction_params = {
        'VI': {
            'start_date': "2000-01-01",
            'end_date': "2023-12-31",
            'output_prefix': "VI_timeseries_GLAD",
            'extract_function': extractVI
        },
        'NDWI': {
            'start_date': "2001-02-01", 
            'end_date': "2023-02-01",
            'output_prefix': "NDWI_timeseries_GLAD",
            'extract_function': extractNDWI
        },
        'ERA5': {
            'start_date': "1990-01-01",
            'end_date': "2023-12-31", 
            'output_prefix': "ERA5_timeseries",
            'extract_function': extractERA5
        }
    }
    
    shapefile_path = 'Training\HarvestStatsAfrica\data\hvstat_africa_boundary_v1.0.gpkg'
    
    total_tasks = len(missing_extractions)
    completed_tasks = 0
    submitted_tasks = []
    
    print(f"\n🏃‍♂️ Starting queue-managed extractions...")
    print(f"Total tasks to process: {total_tasks}")
    print("=" * 80)
    
    for i, task_info in enumerate(missing_extractions):
        country = task_info['country']
        data_type = task_info['data_type']
        admin_level = task_info['admin_level']
        
        params = extraction_params[data_type]
        
        print(f"\n📍 TASK {i+1}/{total_tasks}: {country} - {data_type}_admin{admin_level}")
        print("-" * 50)
        
        # Submit task with queue management
        task = submit_task_with_queue_management(
            params['extract_function'],
            country,
            params['start_date'],
            params['end_date'],
            shapefile_path,
            params['output_prefix'],
            admin_level,
            max_concurrent_tasks,
            check_interval_minutes
        )
        
        if task:
            submitted_tasks.append({
                'task_id': task.id,
                'country': country,
                'data_type': data_type,
                'admin_level': admin_level,
                'submitted_at': time.time()
            })
            completed_tasks += 1
            print(f"✅ Progress: {completed_tasks}/{total_tasks} tasks submitted")
        else:
            print(f"❌ Failed to submit task for {country} - {data_type}_admin{admin_level}")
    
    print(f"\n🎉 All {completed_tasks} tasks submitted successfully!")
    print("=" * 80)
    
    # Print summary of submitted tasks
    if submitted_tasks:
        print(f"\n📋 SUBMITTED TASKS SUMMARY:")
        print("-" * 40)
        for task in submitted_tasks:
            print(f"ID: {task['task_id']} | {task['country']} - {task['data_type']}_admin{task['admin_level']}")


def monitor_task_progress(submitted_task_ids, check_interval_minutes=5):
    """
    Monitor progress of submitted tasks.
    
    Args:
        submitted_task_ids (list): List of task IDs to monitor
        check_interval_minutes (int): Minutes between progress checks
    """
    print(f"\n🔍 MONITORING TASK PROGRESS")
    print("=" * 50)
    
    while submitted_task_ids:
        print(f"\n⏰ Checking progress of {len(submitted_task_ids)} tasks...")
        
        try:
            all_tasks = ee.batch.Task.list()
            task_dict = {task.id: task for task in all_tasks}
            
            completed_this_round = []
            
            for task_id in submitted_task_ids:
                if task_id in task_dict:
                    task = task_dict[task_id]
                    status = task.state
                    
                    if status == 'COMPLETED':
                        print(f"✅ Task {task_id}: COMPLETED")
                        completed_this_round.append(task_id)
                    elif status == 'FAILED':
                        print(f"❌ Task {task_id}: FAILED")
                        completed_this_round.append(task_id)
                    elif status in ['RUNNING', 'READY']:
                        print(f"🔄 Task {task_id}: {status}")
                    else:
                        print(f"📊 Task {task_id}: {status}")
                else:
                    print(f"❓ Task {task_id}: Not found")
                    completed_this_round.append(task_id)
            
            # Remove completed tasks from monitoring list
            for task_id in completed_this_round:
                submitted_task_ids.remove(task_id)
            
            if submitted_task_ids:
                print(f"\n⏳ {len(submitted_task_ids)} tasks still running. Checking again in {check_interval_minutes} minutes...")
                time.sleep(check_interval_minutes * 60)
            else:
                print(f"\n🎉 All tasks completed!")
                break
                
        except Exception as e:
            print(f"❌ Error monitoring tasks: {e}")
            time.sleep(check_interval_minutes * 60)


def get_country_extraction_status():
    """
    Get a detailed status of what's missing for each country.
    
    Returns:
        pd.DataFrame: DataFrame showing extraction status by country
    """
    matrix = create_comprehensive_data_availability_matrix()
    
    if matrix.empty:
        return pd.DataFrame()
    
    status_data = []
    
    for country in matrix.index:
        required_admin = matrix.loc[country, 'required_admin_level']
        row = {'country': country, 'required_admin_level': required_admin}
        
        # Check each extractable data type
        for data_type in ['VI', 'NDWI', 'ERA5']:
            has_admin0 = matrix.loc[country, f'{data_type}_admin0']
            has_required = matrix.loc[country, f'{data_type}_admin{required_admin}']
            
            if has_admin0 and has_required:
                status = "COMPLETE"
            elif has_admin0 or has_required:
                missing = []
                if not has_admin0:
                    missing.append("admin0")
                if not has_required:
                    missing.append(f"admin{required_admin}")
                status = f"MISSING: {', '.join(missing)}"
            else:
                status = "MISSING: admin0, admin" + str(required_admin)
            
            row[data_type] = status
        
        status_data.append(row)
    
    return pd.DataFrame(status_data).set_index('country')


def main():
    """
    Main function to run the optimized extraction analysis.
    """
    print("🔍 REMOTE SENSING EXTRACTION OPTIMIZER")
    print("=" * 70)
    
    # Show current status
    status_df = get_country_extraction_status()
    print("\n📊 CURRENT EXTRACTION STATUS:")
    print("-" * 50)
    print(status_df)
    
    # Run dry run analysis
    print("\n" + "=" * 70)
    run_optimized_extractions_with_queue(dry_run=False)
    
    print("\n" + "=" * 70)
    print("💡 TO RUN ACTUAL EXTRACTIONS:")
    print("   from HarvestStatsAfrica.RemoteSensingPipeline import run_optimized_extractions_with_queue")
    print("   run_optimized_extractions_with_queue(dry_run=False)")


if __name__ == "__main__":
    main() 