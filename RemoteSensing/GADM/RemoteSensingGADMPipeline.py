"""
GADM-Based Remote Sensing Data Extraction Pipeline

This script automates the extraction of remote sensing data (VI, NDWI, ERA5)
for all administrative level 1 and 2 regions in Africa, based on the downloaded
GADM shapefiles. It leverages a Google Earth Engine task queue management system
to maintain optimal throughput and avoids re-processing by checking for existing output files.
"""

import sys
import os
import time
import pandas as pd
import geopandas as gpd
import ee
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EE_PROJECT = os.getenv('EE_PROJECT')
EE_OPT_URL = os.getenv('EE_OPT_URL')


# Add the project root to Python path so imports work correctly
# The new script is in ASR/GADM, so we go up two levels to reach the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from RemoteSensing.NDVI import extractVI
from RemoteSensing.NDWI import extractNDWI
from RemoteSensing.ERA5 import extractERA5

# Initialize Earth Engine
try:
    ee.Initialize(project=EE_PROJECT, opt_url=EE_OPT_URL)
    print("✅ Google Earth Engine initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Google Earth Engine: {e}")
    print("Please run 'earthengine authenticate' first")


def get_running_tasks():
    """Get the number of currently running tasks in Google Earth Engine."""
    try:
        tasks = ee.batch.Task.list()
        running_tasks = [task for task in tasks if task.state in ['RUNNING', 'READY']]
        return len(running_tasks), [task.id for task in running_tasks]
    except Exception as e:
        print(f"❌ Error checking task status: {e}")
        return 0, []


def get_task_status_summary():
    """Get a comprehensive summary of all tasks."""
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
    """Wait until there's an available slot for new tasks."""
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
    """Submit a task while respecting queue limits."""
    if not wait_for_available_slot(max_concurrent_tasks, check_interval_minutes):
        return None
    try:
        task = extract_function(country, start_date, end_date, shapefile_path, output_prefix, admin_level)
        if task:
            print(f"🚀 Task submitted successfully! ID: {task.id}")
        return task
    except Exception as e:
        print(f"❌ Error submitting task: {e}")
        return None


def get_gadm_extraction_tasks():
    """
    Determines required extractions based on pre-processed GADM shapefiles.
    It raises an error if the required processed shapefiles are not found.
    """
    print("🔍 Determining extraction tasks from pre-processed GADM shapefiles...")
    
    extraction_params = {
        'VI': {
            'output_prefix': "VI_timeseries_GADM",
        },
        'NDWI': {
            'output_prefix': "NDWI_timeseries_GADM_monthly",
        },
        'ERA5': {
            'output_prefix': "ERA5_timeseries_GADM",
        }
    }
    
    tasks = []
    results_dir = os.path.join("GADM", "remote_sensing")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for level in [2, 1]:
        shapefile_path = os.path.join("GADM", "gadm41_AFR_shp", f"gadm41_AFR_{level}_processed.shp")
        if not os.path.exists(shapefile_path):
            raise FileNotFoundError(f"FATAL: Processed shapefile not found for admin level {level}. Please create it first: {shapefile_path}")

        print(f"Processing admin level {level} from {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        countries = gdf["ADMIN0"].unique()

        for country in countries:
            for data_type, params in extraction_params.items():
                output_filename = f"{country.replace(' ', '_').replace("'", "_")}_admin{level}_{params['output_prefix']}.csv"
                output_path = os.path.join(results_dir, output_filename)

                if os.path.exists(output_path):
                    # print(f"  - Skipping {country} | {data_type} | admin{level} (already exists)")
                    pass
                else:
                    tasks.append({
                        'country': country,
                        'data_type': data_type,
                        'admin_level': level
                    })
    
    if tasks:
        print(f"✅ Found {len(tasks)} new extraction tasks.")
    else:
        print("✅ No new extraction tasks needed.")
    return tasks


def print_extraction_summary(tasks_to_run):
    """Prints a summary of the extraction tasks to be performed."""
    if not tasks_to_run:
        print("\n🎉 All data is already available! No new extractions needed.")
        return

    print(f"\n📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total new extractions needed: {len(tasks_to_run)}")
    
    summary = {}
    for task in tasks_to_run:
        key = (task['country'], task['admin_level'])
        if key not in summary:
            summary[key] = []
        summary[key].append(task['data_type'])
    
    for (country, level), data_types in sorted(summary.items()):
        print(f"  - {country} (Admin {level}): {', '.join(data_types)}")


def run_gadm_extractions(dry_run=True, max_concurrent_tasks=5, check_interval_minutes=2):
    """
    Run data extractions based on GADM shapefiles with queue management.
    """
    print("\n🚀 GADM-BASED REMOTE SENSING EXTRACTION PIPELINE")
    print("=" * 80)
    
    task_summary = get_task_status_summary()
    print("\n📊 Current GEE Task Status:")
    for status, count in task_summary.items():
        print(f"    {status}: {count}")
    
    extraction_tasks = get_gadm_extraction_tasks()
    print_extraction_summary(extraction_tasks)

    if not extraction_tasks:
        return
        
    print(f"\n⚙️ Queue Settings:")
    print(f"    Max concurrent tasks: {max_concurrent_tasks}")
    print(f"    Check interval: {check_interval_minutes} minutes")

    if dry_run:
        print("\n🔍 DRY RUN MODE - No actual extractions will be performed.")
        print("   To run for real, call with dry_run=False.")
        return

    extraction_params = {
        'VI': {
            'start_date': "2000-01-01", 'end_date': "2023-12-31",
            'output_prefix': "VI_timeseries_GADM", 'extract_function': extractVI
        },
        'NDWI': {
            'start_date': "2001-02-01", 'end_date': "2023-02-01",
            'output_prefix': "NDWI_timeseries_GADM", 'extract_function': extractNDWI
        },
        'ERA5': {
            'start_date': "1990-01-01", 'end_date': "2023-12-31", 
            'output_prefix': "ERA5_timeseries_GADM", 'extract_function': extractERA5
        }
    }
    
    gadm_shp_dir = os.path.join("GADM", "gadm41_AFR_shp")
    total_tasks = len(extraction_tasks)
    submitted_tasks_count = 0
    
    print("\n🏃‍♂️ Starting queue-managed extractions...")
    print(f"Total tasks to process: {total_tasks}")
    print("=" * 80)

    for i, task_info in enumerate(extraction_tasks):
        country = task_info['country']
        data_type = task_info['data_type']
        admin_level = task_info['admin_level']
        params = extraction_params[data_type]
        
        # Use the correct, processed shapefile for the given admin level
        shapefile_path = os.path.join(gadm_shp_dir, f"gadm41_AFR_{admin_level}_processed.shp")
        
        print(f"\n📍 TASK {i+1}/{total_tasks}: {country} - {data_type}_admin{admin_level}")
        
        task = submit_task_with_queue_management(
            params['extract_function'], country,
            params['start_date'], params['end_date'],
            shapefile_path, params['output_prefix'], admin_level,
            max_concurrent_tasks, check_interval_minutes
        )
        
        if task:
            submitted_tasks_count += 1
            print(f"✅ Progress: {submitted_tasks_count}/{total_tasks} tasks submitted")
        else:
            print(f"❌ Failed to submit task for {country} - {data_type}_admin{admin_level}")

    print(f"\n🎉 All {submitted_tasks_count} tasks have been submitted!")
    print("Monitor the Google Earth Engine Tasks tab for progress.")

if __name__ == "__main__":
    run_gadm_extractions(dry_run=False) 