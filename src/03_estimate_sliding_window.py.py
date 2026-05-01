#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03. Real-time Simulation: Sliding Window Event Detection with Internal HDBSCAN
-------------------------------------------------------------------------
Summary:
1. Loads formatted triggers from Step 02.
2. Slices data into time windows.
3. Applies HDBSCAN *within* that window to remove noise.
4. Performs robust grid search and consensus-based T0 estimation.
"""
import os
import sys
import glob
from datetime import datetime, timedelta
import warnings
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

# --- HDBSCAN Import ---
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan
        HDBSCAN = hdbscan.HDBSCAN
    except ImportError:
        print("[Error] HDBSCAN not found. Please install sklearn (v1.3+) or hdbscan.")
        sys.exit(1)

# --- GPU Library check ---
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("WARNING: CuPy not found. Falling back to CPU (Slow).")
    import numpy as cp

warnings.filterwarnings("ignore")

# ==========================================
# 0. Configuration & Paths
# ==========================================
# Identify Repository Root
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Add path to module folder
sys.path.append(str(ROOT_DIR / "modules"))

# ==========================================
# 1. PARAMETERS
# ==========================================

# Input files
INPUT_CSV = DATA_DIR / "input" / "triggers_raw.csv"
PRIOR_MAP_PATH = DATA_DIR / "input" / "prior_distribution_masked.npy"

# Waveform data directory
WAVEFORM_DIR = DATA_DIR / "input" / "waveforms"

# Output directory
OUTPUT_DIR = DATA_DIR / "interim" / "sliding_candidates"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_CSV = OUTPUT_DIR / "events_realtime_final.csv"

# --- Sliding Window Settings ---
WINDOW_MINUTES = 15
SLIDE_STEP_MINUTES = 5

# --- HDBSCAN Settings (Local Filter) ---
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_VELOCITY_KMPS = 1.5

# --- Outlier Rejection & Search Settings ---
MIN_DETECTIONS_IN_WINDOW = 10
T0_CONSENSUS_THRESHOLD_SEC = 120.0
MIN_CONSENSUS_STATIONS = 10
SCORE_THRESHOLD = 0

# --- Search Grid Settings ---
PRIOR_CONFIG = {
    'MIN_LAT': -60.0, 'MAX_LAT': 70.0,
    'MIN_LON': 100.0, 'MAX_LON': 310.0
}
GLOBAL_GRID_STEP_DEG = 0.5
POWELL_SEARCH_RANGE_DEG = 2.0
SEARCH_RANGE_TIME_SEC = 600.0
VELOCITY = 1.50 # km/s (T-phase/Tsunami)

# --- Signal Processing ---
WINDOW_SEC = 300
SMOOTH_WINDOW_SEC = 10
BANDPASS = (1.0, 8.0)
TARGET_FS = 1.0
WAVE_LOAD_MARGIN_HOUR = 1

GPU_GRID_BATCH_SIZE = 1000

# ==========================================
# 2. HDBSCAN Logic (Integrated)
# ==========================================
def run_hdbscan_unified(df, velocity_kmps=1.5, min_cluster_size=5, min_samples=3):
    df = df.copy().reset_index(drop=True)

    if len(df) < min_samples:
        df['event_id'] = -1
        return df

    mean_lat = df['st_lat'].mean()
    mean_lon = df['st_lon'].mean()

    km_per_lat = 111.32
    km_per_lon = 111.32 * np.cos(np.radians(mean_lat))

    x_km = (df['st_lon'] - mean_lon) * km_per_lon
    y_km = (df['st_lat'] - mean_lat) * km_per_lat

    t_start = df['t_obs'].min()
    t_seconds = (df['t_obs'] - t_start).dt.total_seconds()
    z_km = t_seconds * velocity_kmps

    X = np.column_stack([x_km, y_km, z_km])

    try:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
        labels = clusterer.fit_predict(X)
        df['event_id'] = labels
    except Exception as e:
        print(f"HDBSCAN Error: {e}")
        df['event_id'] = -1

    return df

# ==========================================
# 3. Utility Functions
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
    if not os.path.exists(prior_path): return None
    try:
        prior_grid = np.load(prior_path)
        prior_grid = np.nan_to_num(prior_grid, nan=0.0)
        n_lat, n_lon = prior_grid.shape
        lats = np.linspace(PRIOR_CONFIG['MIN_LAT'], PRIOR_CONFIG['MAX_LAT'], n_lat)
        lons = np.linspace(PRIOR_CONFIG['MIN_LON'], PRIOR_CONFIG['MAX_LON'], n_lon)
        return RegularGridInterpolator((lats, lons), prior_grid, bounds_error=False, fill_value=0.0)
    except: return None

# ==========================================
# 4. Waveform Loading & GPU Manager
# ==========================================
def prepare_waveforms_for_window(start_dt, end_dt, station_list):

    clean_stations = set(s.split('.')[-1] if '.' in s else s for s in station_list)
    wave_dict = {}

    # Find all SAC files in WAVEFORM_DIR
    sac_files = glob.glob(str(WAVEFORM_DIR / "*.sac"))

    # Duration (including margin) required for the window being processed
    req_start = start_dt - timedelta(hours=WAVE_LOAD_MARGIN_HOUR)
    req_end = end_dt + timedelta(hours=WAVE_LOAD_MARGIN_HOUR)

    for f in sac_files:
        try:
            st = read(f)
            tr = st[0]

            if tr.stats.station.strip() not in clean_stations: continue

            tr_start = tr.stats.starttime.datetime.replace(tzinfo=None)
            tr_end = tr.stats.endtime.datetime.replace(tzinfo=None)

            if (tr_end < req_start) or (tr_start > req_end):
                continue

            tr.trim(obspy.UTCDateTime(req_start), obspy.UTCDateTime(req_end))
            if tr.stats.npts == 0: continue

            tr.filter("bandpass", freqmin=BANDPASS[0], freqmax=BANDPASS[1], corners=4, zerophase=True)
            env = uniform_filter1d(np.abs(tr.data), size=max(1, int(SMOOTH_WINDOW_SEC*tr.stats.sampling_rate)), mode='nearest')
            tr.data = env.astype(np.float64)
            if abs(tr.stats.sampling_rate - TARGET_FS) > 1e-6: tr.resample(TARGET_FS)

            wave_dict[tr.stats.station.strip()] = tr
        except Exception:
            pass

    return wave_dict

class GpuWaveformManager:
    def __init__(self, wave_dict, station_list):
        self.station_ids = []
        valid_traces = []
        min_start = None
        max_end = None

        for s_full in station_list:
            s_name = s_full.split('.')[-1]
            if s_name in wave_dict:
                tr = wave_dict[s_name]
                t_s = tr.stats.starttime.timestamp
                t_e = tr.stats.endtime.timestamp
                if min_start is None or t_s < min_start: min_start = t_s
                if max_end is None or t_e > max_end: max_end = t_e
                valid_traces.append((s_full, tr))

        if not valid_traces:
            self.valid = False
            return

        self.min_start = min_start
        self.fs = TARGET_FS
        total_samples = int((max_end - min_start) * self.fs) + 100

        n_stat = len(valid_traces)
        wave_matrix_np = np.zeros((n_stat, total_samples), dtype=np.float64)
        self.offsets = np.zeros(n_stat, dtype=np.int32)
        self.start_times = np.zeros(n_stat, dtype=np.float64)

        for i, (s_full, tr) in enumerate(valid_traces):
            self.station_ids.append(s_full)
            t_start = tr.stats.starttime.timestamp
            offset = int((t_start - min_start) * self.fs)
            self.offsets[i] = offset
            self.start_times[i] = t_start
            d = tr.data
            end_idx = min(offset + len(d), total_samples)
            d_len = end_idx - offset
            if d_len > 0:
                wave_matrix_np[i, offset:end_idx] = d[:d_len]

        if HAS_GPU:
            self.wave_matrix = cp.array(wave_matrix_np)
            self.offsets_gpu = cp.array(self.offsets)
            self.start_times_gpu = cp.array(self.start_times)
        else:
            self.wave_matrix = wave_matrix_np
        self.n_stations = n_stat
        self.valid = True

    def get_station_coords(self, df_ev):
        lats, lons, t_obs, valid_indices = [], [], [], []
        station_map = {sid: i for i, sid in enumerate(self.station_ids)}

        for idx, row in df_ev.iterrows():
            sid = row['station_id']
            matched_idx = -1
            if sid in station_map:
                matched_idx = station_map[sid]
            else:
                for mgr_sid, mgr_idx in station_map.items():
                    if sid in mgr_sid:
                        matched_idx = mgr_idx
                        break

            if matched_idx != -1:
                lats.append(row['st_lat'])
                lons.append(row['st_lon'])
                t_obs.append(row['t_obs_s'])
                valid_indices.append(matched_idx)

        if HAS_GPU:
            return (cp.array(lats, dtype=cp.float64), cp.array(lons, dtype=cp.float64),
                    cp.array(t_obs, dtype=cp.float64), cp.array(valid_indices, dtype=cp.int32))
        else:
            return (np.array(lats), np.array(lons), np.array(t_obs), np.array(valid_indices))

# ==========================================
# 5. Search Logic
# ==========================================
def perform_grid_search_gpu_robust(lat_range, lon_range, gpu_wave_mgr, df_ev, interpolator, verbose=False):
    if not gpu_wave_mgr.valid: return None

    st_lats, st_lons, st_tobs, valid_idx_trace = gpu_wave_mgr.get_station_coords(df_ev)
    if len(st_tobs) < MIN_DETECTIONS_IN_WINDOW: return None

    waves_subset = gpu_wave_mgr.wave_matrix[valid_idx_trace]
    offsets_subset = gpu_wave_mgr.offsets_gpu[valid_idx_trace]
    start_times_subset = gpu_wave_mgr.start_times_gpu[valid_idx_trace]

    all_grid_points = []
    for lat in lat_range:
        for lon in lon_range:
            if interpolator:
                prob = float(interpolator([[lat, lon if lon>=0 else lon+360]])[0])
                if prob < 1e-9: continue
            else:
                prob = 1.0
            all_grid_points.append((lat, lon, prob))

    if not all_grid_points: return None
    all_grid_points = np.array(all_grid_points)
    n_total_grid = len(all_grid_points)

    best_score = -1.0
    best_params = None

    hw = int(WINDOW_SEC * gpu_wave_mgr.fs / 2)
    win_len = hw * 2 + 1
    window_offsets = cp.arange(0, win_len, dtype=cp.int32)

    iterator = range(0, n_total_grid, GPU_GRID_BATCH_SIZE)
    if verbose: iterator = tqdm(iterator, leave=False)

    for i in iterator:
        batch_pts = all_grid_points[i : i+GPU_GRID_BATCH_SIZE]

        g_lats = cp.asarray(batch_pts[:, 0], dtype=cp.float64)
        g_lons = cp.asarray(batch_pts[:, 1], dtype=cp.float64)
        g_probs = cp.asarray(batch_pts[:, 2], dtype=cp.float64)

        dists = haversine_km_cupy_float64(g_lats[:, None], g_lons[:, None], st_lats[None, :], st_lons[None, :])
        travel_times = dists / VELOCITY

        t0_candidates = st_tobs[None, :] - travel_times

        t0_est = cp.median(t0_candidates, axis=1)
        residuals = cp.abs(t0_candidates - t0_est[:, None])
        mask1 = residuals < max(T0_CONSENSUS_THRESHOLD_SEC * 2, 10.0)

        t0_candidates_masked = cp.where(mask1, t0_candidates, cp.nan)
        t0_refined = cp.nanmedian(t0_candidates_masked, axis=1)
        t0_final = cp.where(cp.isnan(t0_refined), t0_est, t0_refined)

        final_residuals = cp.abs(t0_candidates - t0_final[:, None])
        inlier_mask = final_residuals <= T0_CONSENSUS_THRESHOLD_SEC
        valid_counts = cp.sum(inlier_mask, axis=1)

        # Stacking
        at_val = t0_final[:, None] + travel_times
        rel_time = at_val - start_times_subset[None, :]
        index_in_trace = cp.floor(rel_time * gpu_wave_mgr.fs).astype(cp.int32)
        center_indices = offsets_subset[None, :] + index_in_trace
        start_indices = center_indices - hw

        max_idx = waves_subset.shape[1] - win_len
        safe_start_indices = cp.clip(start_indices, 0, max_idx)

        gather_idx = safe_start_indices[:, :, None] + window_offsets[None, None, :]
        trace_indices = cp.arange(len(valid_idx_trace))
        gathered_waves = waves_subset[trace_indices[None, :, None], gather_idx]

        segment_norms = cp.linalg.norm(gathered_waves, axis=2, keepdims=True)
        valid_wave_mask = (segment_norms > 1e-9)
        final_mask = valid_wave_mask & inlier_mask[:, :, None]

        segment_norms_safe = cp.where(valid_wave_mask, segment_norms, 1.0)
        normalized_waves = gathered_waves / segment_norms_safe * final_mask

        stacked = cp.sum(normalized_waves, axis=1)
        max_vals = cp.max(stacked, axis=1)

        scores = max_vals * g_probs
        scores = cp.where(valid_counts >= MIN_CONSENSUS_STATIONS, scores, 0.0)

        curr_max_idx = cp.argmax(scores)
        curr_max_score = float(scores[curr_max_idx])

        if curr_max_score > best_score:
            best_score = curr_max_score
            best_params = (
                float(g_lats[curr_max_idx]),
                float(g_lons[curr_max_idx]),
                float(t0_final[curr_max_idx]),
                int(valid_counts[curr_max_idx])
            )

    return best_params, best_score

def objective_function_cpu_robust_strict(params, df_window, wave_dict, interpolator):
    lat, lon, t0 = params
    if not (-90 <= lat <= 90): return 0.0

    prior_prob = 1.0
    if interpolator:
        try: prior_prob = float(interpolator([[lat, lon if lon>=0 else lon+360]])[0])
        except: prior_prob = 0.0
    if prior_prob < 1e-9: return 0.0

    station_best_triggers = {}
    stat_coords = df_window.drop_duplicates('station_id').set_index('station_id')[['st_lat', 'st_lon']]
    stat_dists = {}
    for sid, row in stat_coords.iterrows():
        stat_dists[sid] = haversine_km_cpu(lat, lon, row['st_lat'], row['st_lon'])

    for _, row in df_window.iterrows():
        sid = row['station_id']
        dist = stat_dists.get(sid, 0.0)
        predicted_arrival = t0 + dist / VELOCITY
        resid = abs(row['t_obs_s'] - predicted_arrival)

        if resid > T0_CONSENSUS_THRESHOLD_SEC: continue

        if sid not in station_best_triggers or resid < station_best_triggers[sid]['resid']:
            sname = sid.split('.')[-1]
            if sname not in wave_dict: continue
            tr = wave_dict[sname]
            center = int((predicted_arrival - tr.stats.starttime.timestamp) * tr.stats.sampling_rate)
            hw = int(WINDOW_SEC * tr.stats.sampling_rate / 2)
            if center-hw < 0 or center+hw+1 >= tr.stats.npts: continue
            seg = tr.data[center-hw : center+hw+1]
            if seg.size > 0:
                l2 = np.linalg.norm(seg)
                if l2 > 1e-9:
                    station_best_triggers[sid] = {'resid': resid, 'trace_norm': seg / l2}

    selected_traces = [v['trace_norm'] for v in station_best_triggers.values()]
    if len(selected_traces) < MIN_CONSENSUS_STATIONS: return 0.0

    max_len = max(len(t) for t in selected_traces)
    stack = np.sum([np.pad(t, (0, max_len-len(t))) for t in selected_traces], axis=0)
    return np.max(stack) * prior_prob

def get_final_score_and_count(params, df_window, wave_dict, interpolator):
    score = objective_function_cpu_robust_strict(params, df_window, wave_dict, interpolator)
    lat, lon, t0 = params
    count = 0
    stat_coords = df_window.drop_duplicates('station_id').set_index('station_id')[['st_lat', 'st_lon']]
    for sid, row in stat_coords.iterrows():
        dist = haversine_km_cpu(lat, lon, row['st_lat'], row['st_lon'])
        predicted = t0 + dist/VELOCITY
        triggers = df_window[df_window['station_id']==sid]['t_obs_s'].values
        if np.any(np.abs(triggers - predicted) <= T0_CONSENSUS_THRESHOLD_SEC):
            count += 1
    return score, count

# ==========================================
# 6. Main Sliding Window Loop
# ==========================================
def main_sliding_window():

    print("=== Real-time Simulation: Sliding Window with Internal HDBSCAN ===")

    # 1. Load Data
    print(f"Loading CSV: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print(f"[Error] Input file not found: {INPUT_CSV}")
        print("Please run '02_calc_detection_time.py' first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # Rename columns to match internal logic (Step 02 outputs 'slat', 'slon', 'detection_time')
    if 'slat' in df.columns:
        df = df.rename(columns={'slat': 'st_lat', 'slon': 'st_lon', 'detection_time': 't_obs'})

    # Timestamp conversion
    df['t_obs'] = pd.to_datetime(df['t_obs'], utc=True, format='mixed')
    df['t_obs_s'] = df['t_obs'].apply(lambda x: x.timestamp())

    # --- Auto-detect Analysis Time Range ---
    if df.empty:
        print("[Error] Input CSV is empty.")
        sys.exit(1)

    ANALYSIS_START_TIME = df['t_obs'].min().replace(second=0, microsecond=0)
    ANALYSIS_END_TIME = df['t_obs'].max().replace(second=0, microsecond=0) + timedelta(minutes=1)

    print(f"Analysis Period: {ANALYSIS_START_TIME} to {ANALYSIS_END_TIME}")
    print(f"Total Detections loaded: {len(df)}")

    # Check for SAC files
    if not list(WAVEFORM_DIR.glob("*.sac")):
        print(f"[Warning] No SAC files found in {WAVEFORM_DIR}")
        print(f"Please place your waveform files (.sac) in {WAVEFORM_DIR}")

    prior = load_prior_interpolator(PRIOR_MAP_PATH)

    # Reset Output CSV
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    pd.DataFrame(columns=['window_start', 'window_end', 'est_lat', 'est_lon', 'est_t0', 'score', 'n_unique_stations']).to_csv(OUTPUT_CSV, index=False)

    lats = np.arange(PRIOR_CONFIG['MIN_LAT'], PRIOR_CONFIG['MAX_LAT'], GLOBAL_GRID_STEP_DEG)
    lons = np.arange(PRIOR_CONFIG['MIN_LON'], PRIOR_CONFIG['MAX_LON'], GLOBAL_GRID_STEP_DEG)

    current_time = pd.Timestamp(ANALYSIS_START_TIME).tz_localize('UTC')
    end_time = pd.Timestamp(ANALYSIS_END_TIME).tz_localize('UTC')

    total_steps = int((end_time - current_time).total_seconds() / (SLIDE_STEP_MINUTES*60))
    pbar = tqdm(total=total_steps, desc="Scanning")

    while current_time < end_time:
        window_end = current_time + timedelta(minutes=WINDOW_MINUTES)

        # 2. Window Slice
        mask = (df['t_obs'] >= current_time) & (df['t_obs'] < window_end)
        sub_df_raw = df[mask].copy()

        if len(sub_df_raw) < HDBSCAN_MIN_SAMPLES:
            current_time += timedelta(minutes=SLIDE_STEP_MINUTES)
            pbar.update(1)
            continue

        # 3. Apply HDBSCAN (Local Noise Filter)
        sub_df_clustered = run_hdbscan_unified(
            sub_df_raw,
            velocity_kmps=HDBSCAN_VELOCITY_KMPS,
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES
        )

        sub_df = sub_df_clustered[sub_df_clustered['event_id'] != -1]

        unique_stations = sub_df['station_id'].unique()
        if len(unique_stations) < MIN_CONSENSUS_STATIONS:
            current_time += timedelta(minutes=SLIDE_STEP_MINUTES)
            pbar.update(1)
            continue

        # 4. Waveforms
        wave_dict = prepare_waveforms_for_window(current_time, window_end, unique_stations)

        if not wave_dict:
            current_time += timedelta(minutes=SLIDE_STEP_MINUTES)
            pbar.update(1)
            continue

        wave_mgr = GpuWaveformManager(wave_dict, unique_stations)

        # 5. GPU Grid Search
        if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()
        search_res = perform_grid_search_gpu_robust(lats, lons, wave_mgr, sub_df, prior)

        if search_res:
            best_params, best_score = search_res
            g_lat, g_lon, g_t0, gpu_inliers = best_params

            if best_score > SCORE_THRESHOLD:
                # 6. CPU Optimization
                bounds = [(g_lat-POWELL_SEARCH_RANGE_DEG, g_lat+POWELL_SEARCH_RANGE_DEG),
                          (g_lon-POWELL_SEARCH_RANGE_DEG, g_lon+POWELL_SEARCH_RANGE_DEG),
                          (g_t0-SEARCH_RANGE_TIME_SEC, g_t0+SEARCH_RANGE_TIME_SEC)]

                def func_to_min(p):
                    return -objective_function_cpu_robust_strict(p, sub_df, wave_dict, prior)

                try:
                    res_opt = minimize(func_to_min, [g_lat, g_lon, g_t0],
                                       method='Powell', bounds=bounds, tol=1e-4)

                    final_lat, final_lon, final_t0 = res_opt.x
                    final_score_raw, final_n_stat = get_final_score_and_count(res_opt.x, sub_df, wave_dict, prior)
                except:
                    final_lat, final_lon, final_t0 = g_lat, g_lon, g_t0
                    final_score_raw, final_n_stat = get_final_score_and_count([g_lat, g_lon, g_t0], sub_df, wave_dict, prior)

                if final_score_raw > SCORE_THRESHOLD and final_n_stat >= MIN_CONSENSUS_STATIONS:
                    res_row = {
                        'window_start': current_time.isoformat(),
                        'window_end': window_end.isoformat(),
                        'est_lat': round(final_lat, 4),
                        'est_lon': round(final_lon, 4),
                        'est_t0': pd.to_datetime(final_t0, unit='s', utc=True).isoformat(),
                        'score': round(final_score_raw, 4),
                        'n_unique_stations': int(final_n_stat)
                    }

                    pd.DataFrame([res_row]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                    tqdm.write(f"  Detect: {res_row['est_t0']} | Sc:{final_score_raw:.1f} | St:{final_n_stat}")

        current_time += timedelta(minutes=SLIDE_STEP_MINUTES)
        pbar.update(1)

    pbar.close()
    print(f"Completed. Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    if "__spec__" not in globals(): __spec__ = None
    mp.set_start_method('spawn', force=True)
    main_sliding_window()
