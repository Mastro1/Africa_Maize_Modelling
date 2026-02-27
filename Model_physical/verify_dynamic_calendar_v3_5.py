import pandas as pd
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.dates as mdates

# Import shared functions from V5 (Assuming this exists in your env)
from verify_dynamic_calendar_v5 import preprocess_and_merge

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Strict Fenced V4.0)
# ---------------------------------------------------------
USE_STSG_SMOOTHING = True    
SOS_THRESHOLD_PERC = 0.20    
EOS_THRESHOLD_PERC = 0.30   
MIN_AMPLITUDE_SIGNAL = 0.05  

# "Smart Cap" Logic
MAX_SEASON_LENGTH_BUFFER = 45 # Days to add to static length for max allowed dynamic length
SEARCH_WINDOW_HALF_SIZE = 150 # Days to look before/after the anchor (The Floating Window size)

# Smoothing
SG_ITERATIONS = 3            
SG_WINDOW_LEN = 31           
SG_POLY_ORDER = 2            
# ---------------------------------------------------------

def refine_smoothing(df, iters=SG_ITERATIONS, window=SG_WINDOW_LEN, poly=SG_POLY_ORDER, input_col='NDVI_mean'):
    """Applies Savitzky-Golay smoothing iteratively."""
    y = df[input_col].values
    w = window if len(y) > window else (len(y) // 2 * 2 - 1)
    if w < 5: w = 5 if len(y) >= 5 else 3
    for _ in range(iters):
        y = savgol_filter(y, window_length=w, polyorder=poly)
    df['NDVI_smooth'] = y
    return df

def generate_season_anchors(years, seasons_config):
    """
    Generates a master list of ALL season occurrences across all years.
    Returns a list of dicts: [{'season_idx': 1, 'year': 2000, 'anchor_date': ...}, ...]
    """
    anchors = []
    for year in years:
        for season in seasons_config:
            s_idx = season['index']
            # Calculate Static Dates
            p_doy = season['planting']
            e_doy = season['endofseason']
            
            # Planting Date
            p_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(p_doy - 1))
            
            # End Date (Handle wrap-around logic for the static calendar itself)
            if e_doy < p_doy:
                e_date = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(days=int(e_doy - 1))
            else:
                e_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(e_doy - 1))
                
            # The "Anchor" is the middle of the static season
            anchor_date = p_date + (e_date - p_date) / 2
            
            # Static Length (for Smart Cap)
            static_len = (e_date - p_date).days
            
            anchors.append({
                'season_idx': s_idx,
                'year': year, # The "Crop Year" (start year)
                'anchor_date': anchor_date,
                'static_sos': p_date,
                'static_eos': e_date,
                'static_len': static_len
            })
    return anchors

def solve_overlaps_and_fence(anchors):
    """
    Strict Fence Logic:
    If Season 1 of 2000 overlaps with Season 2 of 2000 (or Season 1 of 2001),
    we calculate the midpoint and build a hard wall.
    """
    # Sort chronologically by anchor date
    sorted_anchors = sorted(anchors, key=lambda x: x['anchor_date'])
    
    for i in range(len(sorted_anchors)):
        current = sorted_anchors[i]
        
        # Default Fences (Wide)
        # We start with the anchor and look SEARCH_WINDOW_HALF_SIZE days back/forward
        current['fence_start'] = current['anchor_date'] - pd.Timedelta(days=SEARCH_WINDOW_HALF_SIZE)
        current['fence_end']   = current['anchor_date'] + pd.Timedelta(days=SEARCH_WINDOW_HALF_SIZE)
        
        # 1. Check Backward Overlap (with previous season)
        if i > 0:
            prev = sorted_anchors[i-1]
            # If default fence overlaps with previous static EOS (plus small buffer)
            midpoint = prev['static_eos'] + (current['static_sos'] - prev['static_eos']) / 2
            
            # Hard Fence: Current Start cannot be before the midpoint
            if current['fence_start'] < midpoint:
                current['fence_start'] = midpoint
                
        # 2. Check Forward Overlap (with next season)
        if i < len(sorted_anchors) - 1:
            nxt = sorted_anchors[i+1]
            midpoint = current['static_eos'] + (nxt['static_sos'] - current['static_eos']) / 2
            
            # Hard Fence: Current End cannot be after the midpoint
            if current['fence_end'] > midpoint:
                current['fence_end'] = midpoint
                
    return sorted_anchors

def extract_season_strict(df_pcode, meta):
    """
    The Core "Strict Fenced" Extraction.
    1. Cut the dataframe to the Fence.
    2. Find Max inside Fence.
    3. Search backward/forward, stopping strictly at Fence.
    """
    # 1. Slice Dataframe (The Floating Window)
    subset = df_pcode[(df_pcode['date'] >= meta['fence_start']) & 
                      (df_pcode['date'] <= meta['fence_end'])].copy()
    
    if subset.empty: return None

    # 2. Find Peak (Trapped inside Fence)
    peak_idx = subset['NDVI_smooth'].idxmax()
    peak_date = subset.loc[peak_idx, 'date']
    peak_val = subset.loc[peak_idx, 'NDVI_smooth']
    v_min = subset['NDVI_smooth'].min()
    amp = peak_val - v_min
    
    if amp < MIN_AMPLITUDE_SIGNAL: return None # No signal
    
    # Thresholds
    thresh_sos = v_min + SOS_THRESHOLD_PERC * amp
    thresh_eos = peak_val - EOS_THRESHOLD_PERC * amp # Falling limb logic usually relative to peak or min?
    # Usually EOS is relative to Amp. Let's stick to standard: v_min + 0.X * Amp?
    # Or your V3 logic: v_max - 0.3 * Amp. Let's use yours.
    thresh_eos = peak_val - EOS_THRESHOLD_PERC * amp

    # 3. Search (Trapped)
    # Get location in the FULL dataframe to allow indexing logic, but bounds are strict
    full_idx = df_pcode.index[df_pcode['date'] == peak_date][0]
    peak_pos = df_pcode.index.get_loc(full_idx)
    
    sos_date = meta['fence_start'] # Default to fence
    eos_date = meta['fence_end']   # Default to fence
    
    # Backward Search (SOS)
    # "Stop-Loss": If we hit a valley and go up, stop.
    lowest_seen = peak_val
    for i in range(peak_pos, -1, -1):
        curr_row = df_pcode.iloc[i]
        curr_date = curr_row['date']
        curr_val = curr_row['NDVI_smooth']
        
        # Hit Fence? Stop.
        if curr_date < meta['fence_start']: break
        
        # Stop-Loss (Uphill check) - prevents climbing previous season
        if curr_val < lowest_seen: lowest_seen = curr_val
        if curr_val > lowest_seen + 0.05: # Climbing back up!
            sos_date = df_pcode.iloc[i+1]['date'] # Snap to the bottom
            break

        if curr_val <= thresh_sos:
            sos_date = curr_date
            break

    # Forward Search (EOS)
    for i in range(peak_pos, len(df_pcode)):
        curr_row = df_pcode.iloc[i]
        curr_date = curr_row['date']
        curr_val = curr_row['NDVI_smooth']
        
        # Hit Fence? Stop.
        if curr_date > meta['fence_end']: break
        
        if curr_val <= thresh_eos:
            eos_date = curr_date
            break
            
    # 4. Smart Cap Reality Check
    dynamic_len = (eos_date - sos_date).days
    max_allowed = meta['static_len'] + MAX_SEASON_LENGTH_BUFFER
    
    method = "Dynamic_Strict"
    
    # Fallback if too long
    if dynamic_len > max_allowed:
        sos_date = meta['static_sos']
        eos_date = meta['static_eos']
        method = "Static_Fallback_TooLong"
        
    return {
        'sos': sos_date,
        'eos': eos_date,
        'peak': peak_date,
        'method': method,
        'meta': meta
    }

def plot_season_strict(df_pcode, pcode, year, season_results, output_dir):
    """Plots all seasons associated with a specific Crop Year."""
    # ... (Standard plotting logic, but iterating over the processed results list) ...
    # Simplified for brevity here - standard matplotlib code
    pass

# =========================================================
# MAIN EXECUTION FLOW
# =========================================================

if __name__ == "__main__":
    # ... (Load Args, Dataframes as before) ...
    # Assume df_pcode, seasons_config are loaded and Pre-processing/STSG is done
    
    # 1. Generate the Master Timeline
    unique_years = sorted(df_pcode['Year'].unique())
    all_anchors = generate_season_anchors(unique_years, seasons_config)
    
    # 2. Build the Walls (Resolve Overlaps globally)
    fenced_anchors = solve_overlaps_and_fence(all_anchors)
    
    print(f"Generated {len(fenced_anchors)} distinct seasonal search windows.")
    
    # 3. Iterate by SEASON (Not by Year)
    final_results = []
    
    for meta in fenced_anchors:
        # We only process if the fence is within our data range
        if meta['fence_end'] > df_pcode['date'].max() or meta['fence_start'] < df_pcode['date'].min():
            continue
            
        res = extract_season_strict(df_pcode, meta)
        if res:
            final_results.append(res)
            
            # Print status
            print(f"Season {meta['season_idx']} [{meta['year']}]: {res['method']} "
                  f"({(res['eos'] - res['sos']).days} days) "
                  f"Window: {meta['fence_start'].date()} to {meta['fence_end'].date()}")

    # 4. Plotting (Group back by year for visualization if desired)
    # ...