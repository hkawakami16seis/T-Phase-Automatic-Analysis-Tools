#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06. HDBSCAN Event Estimation (Powell Optimization + Jackknife)
-----------------------------------------------------
Summary:
1. Loads event candidates and initial epicenters (from Step 05).
2. Loads waveform data (SAC) for the relevant time windows.
3. Performs Powell optimization within +/- 2 degrees of the initial coordinate.
4. Estimates errors using Jackknife resampling.
5. Outputs detailed summary and station stats.
"""
import os
import sys
import glob
from datetime import datetime, timedelta
import itertools
import warnings
import gc
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import RegularGridInterpolator
from obspy import read
import obspy

# GPU Library Check
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: CuPy not found. Falling back to CPU.")
    import numpy as cp

warnings.filterwarnings("ignore")

# ==========================================
# 0. Configuration & Paths
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Input Files (From Step 05)
INPUT_CSV = DATA_DIR / "processed" / "id_assigned" / "all_event_detect_with_new_ids.csv"
INITIAL_EPICENTER_CSV = DATA_DIR / "processed" / "id_assigned" / "initial_epicenters.csv"
PRIOR_MAP_PATH = DATA_DIR / "input" / "prior_distribution_masked.npy"

# Waveform Directory (Place .sac files here)
WAVEFORM_DIR = DATA_DIR / "input" / "waveforms"

# Output Directories
OUTPUT_DIR = DATA_DIR / "output" / "estimation_results"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual_events"
DEBUG_JK_DIR = OUTPUT_DIR / "debug_jackknife"

for d in [OUTPUT_DIR, INDIVIDUAL_DIR, DEBUG_JK_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# ==========================================
# 1. PARAMETERS
# ==========================================

# Target Event IDs (Empty list = Process All)
# EVENT_ID_LIST = [1, 2, 3]
EVENT_ID_LIST = []

PRIOR_CONFIG = {
    'MIN_LAT': -60.0, 'MAX_LAT': 70.0,
    'MIN_LON': 100.0, 'MAX_LON': 310.0
}

# Search Configuration
POWELL_SEARCH_RANGE_DEG = 2.0   # Range (+/- degrees) from initial epicenter
SEARCH_RANGE_TIME_SEC = 300.0
VELOCITY = 1.50
MIN_STATIONS = 10               # Minimum stations required for estimation
JACKKNIFE_MIN_STATIONS = 10     # Minimum stations required for Jackknife
SIGMA_THRESHOLD = 2.0           # MAD Filter Threshold

# Waveform Params
WINDOW_SEC = 300
SMOOTH_WINDOW_SEC = 10
BANDPASS = (1.0, 8.0)
TARGET_FS = 1.0
WAVE_LOAD_MARGIN_HOUR = 1

# ==========================================
# 2. Utility Functions
# ==========================================
def haversine_km_cupy_float64(lat1_arr, lon1_arr, lat2_arr, lon2_arr):
    R = 6371.0
    phi1 = cp.radians(lat1_arr)
    phi2 = cp.radians(lat2_arr)
    dphi = cp.radians(lat2_arr - lat1_arr)
    dlon_raw = lon2_arr - lon1_arr
    dlon_raw = (dlon_raw + 180.0) % 360.0 - 180.0
    dlambda = cp.radians(dlon_raw)
    a = cp.sin(dphi/2.0)**2 + cp.cos(phi1)*cp.cos(phi2)*cp.sin(dlambda/2.0)**2
    c = 2.0 * cp.arctan2(cp.sqrt(a), cp.sqrt(1.0-a))
    return R * c

def haversine_km_cpu(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlon_raw = lon2 - lon1
    dlon_raw = (dlon_raw + 180) % 360 - 180
    dlambda = np.radians(dlon_raw)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def load_prior_interpolator(prior_path):
    if not os.path.exists(prior_path):
        print(f'[Warning] Prior file not found: {prior_path}')
        return None
    try:
        prior_grid = np.load(prior_path)
        prior_grid = np.nan_to_num(prior_grid, nan=0.0)
        n_lat, n_lon = prior_grid.shape
        lats = np.linspace(PRIOR_CONFIG['MIN_LAT'], PRIOR_CONFIG['MAX_LAT'], n_lat)
        lons = np.linspace(PRIOR_CONFIG['MIN_LON'], PRIOR_CONFIG['MAX_LON'], n_lon)
        return RegularGridInterpolator((lats, lons), prior_grid, bounds_error=False, fill_value=0.0)
    except Exception as e:
        print(f'[Error] Cannot load prior file: {e}')
        sys.exit(1)

# ==========================================
# 3. Waveform Loading
# ==========================================
def prepare_waveforms_for_window(start_dt, end_dt, station_list):
    """
    Loads SAC files from WAVEFORM_DIR that cover the requested time window.
    """
    req_start = start_dt - timedelta(hours=WAVE_LOAD_MARGIN_HOUR)
    req_end = end_dt + timedelta(hours=WAVE_LOAD_MARGIN_HOUR)
    clean_stations = set(s.split('.')[-1] if '.' in s else s for s in station_list)

    wave_dict = {}

    # Check if directory exists
    if not WAVEFORM_DIR.exists():
        print(f"[Error] Waveform directory not found: {WAVEFORM_DIR}")
        return {}

    # Find all SAC files
    sac_files = glob.glob(str(WAVEFORM_DIR / "*.sac"))

    if not sac_files:
        # print(f"[Warning] No .sac files found in {WAVEFORM_DIR}")
        return {}

    for f in sac_files:
        try:
            # Read header first to check station and time (optimization possible here)
            st = read(f)
            tr = st[0]

            # Station check
            if tr.stats.station.strip() not in clean_stations: continue

            # Time overlap check
            tr_start = tr.stats.starttime.datetime.replace(tzinfo=None)
            tr_end = tr.stats.endtime.datetime.replace(tzinfo=None)

            # Convert request times to naive for comparison
            req_start_naive = req_start.replace(tzinfo=None)
            req_end_naive = req_end.replace(tzinfo=None)

            if (tr_end < req_start_naive) or (tr_start > req_end_naive):
                continue

            # Processing
            tr.trim(obspy.UTCDateTime(req_start), obspy.UTCDateTime(req_end))
            if tr.stats.npts == 0: continue

            tr.filter("bandpass", freqmin=BANDPASS[0], freqmax=BANDPASS[1], corners=4, zerophase=True)
            env = uniform_filter1d(np.abs(tr.data), size=max(1, int(SMOOTH_WINDOW_SEC*tr.stats.sampling_rate)), mode='nearest')
            tr.data = env.astype(np.float64)

            if abs(tr.stats.sampling_rate - TARGET_FS) > 1e-6:
                tr.resample(TARGET_FS)

            wave_dict[tr.stats.station.strip()] = tr
        except Exception:
            pass

    return wave_dict

# ==========================================
# 4. Estimation & Optimization Logic
# ==========================================
def calculate_score_cpu(lat, lon, t0, selected_indices, df_full, wave_dict, interpolator):
    if not (-90 <= lat <= 90): return 0.0

    prior_prob = 1.0
    if interpolator:
        try: prior_prob = float(interpolator([[lat, lon if lon>=0 else lon+360]])[0])
        except: prior_prob = 0.0
    if prior_prob < 1e-9: return 0.0

    subset = df_full.loc[selected_indices]

    # Dynamic MAD Filtering
    lats = subset['st_lat'].values
    lons = subset['st_lon'].values
    obs_times = subset['t_obs_s'].values

    dists = np.array([haversine_km_cpu(lat, lon, sla, slo) for sla, slo in zip(lats, lons)])
    t_preds = t0 + (dists / VELOCITY)
    residuals = obs_times - t_preds

    resid_median = np.median(residuals)
    abs_dev = np.abs(residuals - resid_median)
    mad = np.median(abs_dev)

    sigma_est = max(1.4826 * mad, 0.1)
    limit = SIGMA_THRESHOLD * sigma_est

    valid_mask = abs_dev <= limit

    if np.sum(valid_mask) < MIN_STATIONS:
        return 0.0

    valid_subset = subset # Using all points for waveform stacking, or use valid_subset based on strategy

    traces = []
    for _, row in valid_subset.iterrows():
        sname = row['station_id'].split('.')[-1]
        if sname not in wave_dict: continue
        tr = wave_dict[sname]
        dist = haversine_km_cpu(lat, lon, row['st_lat'], row['st_lon'])
        at = t0 + (dist / VELOCITY)
        center = int((at - tr.stats.starttime.timestamp) * tr.stats.sampling_rate)
        hw = int(WINDOW_SEC * tr.stats.sampling_rate / 2)
        if center-hw < 0 or center+hw+1 >= tr.stats.npts: continue
        seg = tr.data[center-hw : center+hw+1]
        if seg.size > 0:
            l2 = np.linalg.norm(seg)
            if l2 > 1e-9: traces.append(seg/l2)

    if len(traces) < MIN_STATIONS: return 0.0

    max_len = max(len(t) for t in traces)
    stack = np.sum([np.pad(t, (0, max_len-len(t))) for t in traces], axis=0)

    return np.max(stack) * prior_prob

def objective_function_cpu(params, candidates_df, df_full, wave_dict, interpolator):
    return -calculate_score_cpu(params[0], params[1], params[2], candidates_df.index.values, df_full, wave_dict, interpolator)

def estimate_event_from_initial_powell(ev_id, ev_df, wave_dict_cpu, prior_interpolator, init_lat, init_lon, init_t0, verbose=False):
    """
    Performs estimation using Powell optimization starting from an initial coordinate.
    """
    # Range: +/- POWELL_SEARCH_RANGE_DEG
    bounds = [
        (init_lat - POWELL_SEARCH_RANGE_DEG, init_lat + POWELL_SEARCH_RANGE_DEG),
        (init_lon - POWELL_SEARCH_RANGE_DEG, init_lon + POWELL_SEARCH_RANGE_DEG),
        (init_t0 - SEARCH_RANGE_TIME_SEC, init_t0 + SEARCH_RANGE_TIME_SEC)
    ]

    res = minimize(objective_function_cpu, [init_lat, init_lon, init_t0],
                   args=(ev_df, ev_df, wave_dict_cpu, prior_interpolator),
                   method='Powell', bounds=bounds, tol=1e-4)

    flat, flon, ft0 = res.x

    # Post-Process: Classify All Stations
    dists_all = np.array([haversine_km_cpu(flat, flon, r['st_lat'], r['st_lon']) for _, r in ev_df.iterrows()])
    t_preds_all = ft0 + (dists_all / VELOCITY)
    residuals_all = ev_df['t_obs_s'].values - t_preds_all

    # MAD Calculation
    resid_median = np.median(residuals_all)
    mad = np.median(np.abs(residuals_all - resid_median))
    sigma_est = max(1.4826 * mad, 0.1)
    limit = SIGMA_THRESHOLD * sigma_est

    # Create Mask
    final_mask = np.abs(residuals_all - resid_median) <= limit

    # Create a result DataFrame with ALL info
    df_result = ev_df.copy()
    df_result['dist_km'] = dists_all
    df_result['theoretical_t'] = t_preds_all
    df_result['residual'] = residuals_all
    df_result['status'] = np.where(final_mask, 'accepted', 'rejected_mad')

    final_residuals_accepted = residuals_all[final_mask]
    resid_std = np.std(final_residuals_accepted) if len(final_residuals_accepted) > 0 else 999.9

    return {
        'event_id': ev_id, 'lat': flat, 'lon': flon, 't0': ft0,
        'score': -res.fun, 'residual_std': resid_std,
        'all_stations': df_result.to_dict('records')
    }

def perform_jackknife_cpu_serial(init_lat, init_lon, init_t0, accepted_df, wave_dict_cpu, interpolator, event_id, debug_dir):
    """
    Run Jackknife using Powell Optimization around the final result.
    """
    n_stations = len(accepted_df)
    jk_details = []
    jk_results_valid = []

    if n_stations < JACKKNIFE_MIN_STATIONS:
        return np.nan, np.nan, np.nan

    for i in tqdm(range(n_stations), desc=f"  Jackknife Ev{event_id}", leave=False):
        drop_idx = accepted_df.index[i]
        dropped_station_id = accepted_df.loc[drop_idx, 'station_id']
        jk_df = accepted_df.drop(index=drop_idx)

        # Bounds: +/- 5 deg from the specific result
        bounds = [
            (init_lat - POWELL_SEARCH_RANGE_DEG, init_lat + POWELL_SEARCH_RANGE_DEG),
            (init_lon - POWELL_SEARCH_RANGE_DEG, init_lon + POWELL_SEARCH_RANGE_DEG),
            (init_t0 - SEARCH_RANGE_TIME_SEC, init_t0 + SEARCH_RANGE_TIME_SEC)
        ]

        res = minimize(objective_function_cpu, [init_lat, init_lon, init_t0],
                       args=(jk_df, accepted_df, wave_dict_cpu, interpolator),
                       method='Powell', bounds=bounds, tol=1e-4)

        jk_results_valid.append(res.x)
        jk_details.append({
            'iteration': i,
            'dropped_station_id': dropped_station_id,
            'lat': res.x[0], 'lon': res.x[1], 't0': res.x[2]
        })

    if debug_dir:
        pd.DataFrame(jk_details).to_csv(os.path.join(debug_dir, f"jk_detail_event_{event_id}.csv"), index=False)

    jk_results = np.array(jk_results_valid)
    if len(jk_results) < n_stations * 0.5:
        return np.nan, np.nan, np.nan

    scale = len(jk_results) - 1
    std_lat = np.sqrt(scale * np.var(jk_results[:, 0], ddof=0))
    std_lon = np.sqrt(scale * np.var(jk_results[:, 1], ddof=0))
    std_t0 = np.sqrt(scale * np.var(jk_results[:, 2], ddof=0))

    return std_lat, std_lon, std_t0

# ==========================================
# 5. Main Workflow
# ==========================================
def main():
    print("Loading Input Data...")
    if not INPUT_CSV.exists() or not INITIAL_EPICENTER_CSV.exists():
        print(f"[Error] Input files not found. Please run Step 05.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # Rename for internal logic
    if 'slat' in df.columns:
        df = df.rename(columns={'slat': 'st_lat', 'slon': 'st_lon', 'detection_time': 't_obs'})

    df['t_obs'] = pd.to_datetime(df['t_obs'], utc=True, format='mixed')
    df['t_obs_s'] = df['t_obs'].apply(lambda x: x.timestamp())

    # Load Initial Epicenters
    df_init_loc = pd.read_csv(INITIAL_EPICENTER_CSV)
    df_init_loc['est_t0'] = pd.to_datetime(df_init_loc['est_t0'], utc=True, format='mixed')
    df_init_loc['est_t0_s'] = df_init_loc['est_t0'].apply(lambda x: x.timestamp())

    prior = load_prior_interpolator(PRIOR_MAP_PATH)

    if len(EVENT_ID_LIST) == 0:
        unique_ids = sorted([u for u in df['new_event_id'].unique() if u != -1])
    else:
        unique_ids = EVENT_ID_LIST

    print(f"\n[Phase 1] Estimating {len(unique_ids)} events (Powell Only, +/- {POWELL_SEARCH_RANGE_DEG} deg)...")
    print(f"Results will be saved in: {INDIVIDUAL_DIR}")

    for uid in tqdm(unique_ids, desc="Processing Events"):
        sub_df = df[df['new_event_id'] == uid]
        if len(sub_df) < MIN_STATIONS: continue

        # Retrieve Initial Location
        init_row = df_init_loc[df_init_loc['event_id'] == uid]
        if len(init_row) == 0:
            continue

        init_lat = float(init_row.iloc[0]['est_lat'])
        init_lon = float(init_row.iloc[0]['est_lon'])
        init_t0 = init_row.iloc[0]['est_t0_s']

        # Check if initial values are valid
        if pd.isna(init_lat) or pd.isna(init_lon):
            # Fallback to mean of stations if catalog info is missing
            init_lat = sub_df['st_lat'].mean()
            init_lon = sub_df['st_lon'].mean()
            init_t0 = sub_df['t_obs_s'].mean()

        # Load Waveforms
        tm_min, tm_max = sub_df['t_obs_s'].min(), sub_df['t_obs_s'].max()

        # Convert timestamps to datetime objects
        dt_min = pd.to_datetime(tm_min, unit='s', utc=True)
        dt_max = pd.to_datetime(tm_max, unit='s', utc=True)

        wave_dict = prepare_waveforms_for_window(dt_min, dt_max, sub_df['station_id'].unique())

        if not wave_dict:
            # print(f"[Skip] Event {uid}: No waveforms found.")
            continue

        # Execute Estimation (Powell Only)
        res = estimate_event_from_initial_powell(uid, sub_df, wave_dict, prior, init_lat, init_lon, init_t0, verbose=False)

        if res:
            all_stations_df = pd.DataFrame(res['all_stations'])
            accepted_df = all_stations_df[all_stations_df['status'] == 'accepted'].copy()
            rejected_df = all_stations_df[all_stations_df['status'] == 'rejected_mad'].copy()

            n_accepted = len(accepted_df)
            n_rejected = len(rejected_df)

            # Jackknife (CPU Serial, Powell Only)
            if n_accepted >= JACKKNIFE_MIN_STATIONS:
                std_lat, std_lon, std_t0 = perform_jackknife_cpu_serial(
                    res['lat'], res['lon'], res['t0'], accepted_df, wave_dict, prior,
                    event_id=uid, debug_dir=DEBUG_JK_DIR
                )
            else:
                std_lat, std_lon, std_t0 = np.nan, np.nan, np.nan

            summary_data = {
                'event_id': uid,
                'origin_time': pd.to_datetime(res['t0'], unit='s', utc=True).isoformat(),
                'lat': round(res['lat'], 4),
                'lon': round(res['lon'], 4),
                'score': round(res['score'], 4),
                'residual_std': round(res['residual_std'], 4),
                'n_stations_total': len(all_stations_df),
                'n_stations_accepted': n_accepted,
                'n_stations_rejected': n_rejected,
                'std_lat': round(std_lat, 4) if not np.isnan(std_lat) else np.nan,
                'std_lon': round(std_lon, 4) if not np.isnan(std_lon) else np.nan,
                'std_t0': round(std_t0, 4) if not np.isnan(std_t0) else np.nan
            }

            summary_path = os.path.join(INDIVIDUAL_DIR, f"event_{uid}_summary.csv")
            pd.DataFrame([summary_data]).to_csv(summary_path, index=False)

            all_stations_df['t_obs'] = pd.to_datetime(all_stations_df['t_obs_s'], unit='s', utc=True)
            cols_order = [
                'event_id', 'station_id', 'status',
                'st_lat', 'st_lon', 't_obs',
                'residual', 'dist_km', 'theoretical_t'
            ]
            final_cols = [c for c in cols_order if c in all_stations_df.columns]
            stations_path = os.path.join(INDIVIDUAL_DIR, f"event_{uid}_stations.csv")
            all_stations_df[final_cols].to_csv(stations_path, index=False)

        if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()

    print("\nProcessing Complete.")
    print(f"Results saved to: {INDIVIDUAL_DIR}")

if __name__ == "__main__":
    if "__spec__" not in globals(): __spec__ = None
    mp.set_start_method('spawn', force=True)
    main()
