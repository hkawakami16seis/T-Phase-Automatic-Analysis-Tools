#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04. Post-Processing: Unique Trigger Assignment, Event Merging & Scavenging
---------------------------------------------------------------------------------------------
Summary:
1. Loads raw triggers (Step 02) and candidate events (Step 03).
2. Merges duplicate events that are close in time and space.
3. Assigns triggers to events based on residuals (resolves conflicts).
4. Scavenges orphaned triggers to recover weak signals.
5. Outputs a cleaned catalog and individual event files.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil
import sys
from pathlib import Path

# ==========================================
# 0. Configuration & Paths
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Input Paths
INPUT_TRIGGERS_CSV = DATA_DIR / "input" / "triggers_raw.csv"
INPUT_EVENTS_CSV = DATA_DIR / "interim" / "sliding_candidates" / "events_realtime_final.csv"

# Output Paths
OUTPUT_DIR = DATA_DIR / "interim" / "refined_catalog"
OUTPUT_CLEAN_CSV = OUTPUT_DIR / "events_catalog_final.csv"
OUTPUT_DETAILS_DIR = OUTPUT_DIR / "event_details"

# Thresholds
MERGE_TIME_SEC = 30.0
MERGE_DIST_KM = 200.0
MAX_RESIDUAL_SEC = 120.0
MIN_STATIONS_FINAL = 10
VELOCITY = 1.50
SCAVENGE_RESIDUAL_SEC = 60.0

# ==========================================
# Functions
# ==========================================
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def solve_conflicts(events_df, triggers_df):
    print("--- Starting Exclusive Trigger Assignment ---")

    # Check inputs
    if events_df.empty:
        print("Warning: Event list is empty.")
        return pd.DataFrame(), pd.DataFrame()

    events_df['score'] = pd.to_numeric(events_df['score'], errors='coerce')
    events_df['n_unique_stations'] = pd.to_numeric(events_df['n_unique_stations'], errors='coerce')

    # Adjust for output column names in Step03 (window_start, window_end)
    cols_to_keep = ['event_id', 'est_lat', 'est_lon', 'est_t0', 'score', 'n_unique_stations', 'window_start', 'window_end']
    existing_cols = [c for c in cols_to_keep if c in events_df.columns]
    ev_data = events_df[existing_cols].to_dict('records')

    tr_lats = triggers_df['st_lat'].values
    tr_lons = triggers_df['st_lon'].values
    tr_tobs = triggers_df['t_obs_s'].values

    candidates = []
    print("Calculating residuals matrix...")
    for ev in tqdm(ev_data):
        eid = ev['event_id']
        e_lat, e_lon, e_t0 = ev['est_lat'], ev['est_lon'], ev['est_t0']
        q_score = ev['score']

        dists = haversine_np(e_lat, e_lon, tr_lats, tr_lons)
        preds = e_t0 + (dists / VELOCITY)
        residuals = np.abs(tr_tobs - preds)

        match_indices = np.where(residuals <= MAX_RESIDUAL_SEC)[0]

        for idx in match_indices:
            candidates.append((idx, eid, residuals[idx], dists[idx], preds[idx], q_score))

    print(f"Potential associations found: {len(candidates)}")
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()

    cand_df = pd.DataFrame(candidates, columns=['trigger_idx', 'event_id', 'residual', 'dist_km', 'predicted_t', 'score'])

    cand_df = cand_df.sort_values(by=['score', 'residual'], ascending=[False, True])

    assigned_df = cand_df.drop_duplicates(subset=['trigger_idx'], keep='first')

    print("Re-evaluating events...")
    event_counts = assigned_df['event_id'].value_counts()

    valid_events = []
    valid_event_ids = set()

    for ev in ev_data:
        eid = ev['event_id']
        if event_counts.get(eid, 0) >= MIN_STATIONS_FINAL:
            valid_events.append(ev)
            valid_event_ids.add(eid)

    final_assignments = assigned_df[assigned_df['event_id'].isin(valid_event_ids)].copy()

    return pd.DataFrame(valid_events), final_assignments

def scavenge_orphans(valid_events_df, all_triggers_df, assigned_triggers_df):
    """
    Rescan for Orphans that were not assigned to any events.
    If the condition is met, it is added to the event.
    """
    print(f"--- Starting Scavenging (Strict window: +/- {SCAVENGE_RESIDUAL_SEC}s) ---")

    if assigned_triggers_df.empty:
        print("No assignments to start with.")
        return assigned_triggers_df

    assigned_indices = set(assigned_triggers_df['trigger_idx'].unique())
    all_indices = set(all_triggers_df.index)
    orphan_indices = list(all_indices - assigned_indices)

    if not orphan_indices:
        print("No orphan triggers to scavenge.")
        return assigned_triggers_df

    print(f"Scanning {len(orphan_indices)} orphan triggers...")
    orphan_df = all_triggers_df.loc[orphan_indices].copy()
    orphan_lats = orphan_df['st_lat'].values
    orphan_lons = orphan_df['st_lon'].values
    orphan_tobs = orphan_df['t_obs_s'].values
    orphan_orig_indices = orphan_df.index.values

    ev_data = valid_events_df.to_dict('records')
    new_candidates = []

    for ev in tqdm(ev_data, desc="Scavenging"):
        eid = ev['event_id']
        e_lat, e_lon, e_t0 = ev['est_lat'], ev['est_lon'], ev['est_t0']

        dists = haversine_np(e_lat, e_lon, orphan_lats, orphan_lons)
        preds = e_t0 + (dists / VELOCITY)
        residuals = np.abs(orphan_tobs - preds)

        match_indices = np.where(residuals <= SCAVENGE_RESIDUAL_SEC)[0]

        for idx in match_indices:
            new_candidates.append({
                'trigger_idx': orphan_orig_indices[idx],
                'event_id': eid,
                'residual': residuals[idx],
                'dist_km': dists[idx],
                'predicted_t': preds[idx],
                'quality_score': 0
            })

    if not new_candidates:
        print("No orphans matched.")
        return assigned_triggers_df

    scavenged_df = pd.DataFrame(new_candidates)
    scavenged_df = scavenged_df.sort_values(by='residual', ascending=True)
    scavenged_df = scavenged_df.drop_duplicates(subset=['trigger_idx'], keep='first')

    print(f"Rescued {len(scavenged_df)} triggers.")

    cols = ['trigger_idx', 'event_id', 'residual', 'dist_km', 'predicted_t', 'quality_score']
    combined_df = pd.concat([assigned_triggers_df[cols], scavenged_df[cols]], ignore_index=True)

    return combined_df

# ==========================================
# Main Process
# ==========================================
def main():
    print(f"Output Directory: {OUTPUT_DIR}")

    # 0. Directory Setup
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)

    if OUTPUT_DETAILS_DIR.exists():
        shutil.rmtree(OUTPUT_DETAILS_DIR)
    os.makedirs(OUTPUT_DETAILS_DIR)

    # 1. Load Data
    print("Loading Data...")
    if not INPUT_EVENTS_CSV.exists() or not INPUT_TRIGGERS_CSV.exists():
        print(f"[Error] Input files not found.\n{INPUT_EVENTS_CSV}\n{INPUT_TRIGGERS_CSV}")
        print("Please run Steps 02 and 03 first.")
        return

    df_ev = pd.read_csv(INPUT_EVENTS_CSV)
    df_tr = pd.read_csv(INPUT_TRIGGERS_CSV)
    print(f"Loaded Triggers: {len(df_tr)}, Candidate Events: {len(df_ev)}")

    # Column Renaming (Match Step 02 Output to internal logic)
    if 'slat' in df_tr.columns:
        df_tr = df_tr.rename(columns={'slat': 'st_lat', 'slon': 'st_lon', 'detection_time': 't_obs'})

    # Timestamp Conversion
    df_tr['t_obs'] = pd.to_datetime(df_tr['t_obs'], utc=True, format='mixed')
    df_tr['t_obs_s'] = df_tr['t_obs'].apply(lambda x: x.timestamp())

    df_ev['est_t0_dt'] = pd.to_datetime(df_ev['est_t0'])
    df_ev['est_t0'] = df_ev['est_t0_dt'].apply(lambda x: x.timestamp())
    df_ev['event_id'] = range(len(df_ev))

    if len(df_ev) == 0:
        print("No candidate events found. Exiting.")
        return

    # Step 1: Merge Duplicates (Time & Space)
    print("Merging Duplicates...")
    df_ev = df_ev.sort_values('score', ascending=False).reset_index(drop=True)
    keep_indices = []
    dropped_indices = set()
    lats, lons, t0s = df_ev['est_lat'].values, df_ev['est_lon'].values, df_ev['est_t0'].values

    for i in tqdm(range(len(df_ev))):
        if i in dropped_indices: continue
        keep_indices.append(i)
        t_diff = np.abs(t0s[i+1:] - t0s[i])

        candidate_idxs = np.where(t_diff < MERGE_TIME_SEC)[0] + (i + 1)
        for j in candidate_idxs:
            if j in dropped_indices: continue
            dist = haversine_np(lats[i], lons[i], lats[j], lons[j])
            if dist < MERGE_DIST_KM:
                dropped_indices.add(j)

    df_merged = df_ev.iloc[keep_indices].copy()
    print(f"Events after Merging: {len(df_merged)}")

    # Step 2: Exclusive Assignment (Conflict Resolution)
    df_valid, df_assignments = solve_conflicts(df_merged, df_tr)

    if df_valid.empty:
        print("!! Critical: No valid events remaining after assignment.")
        return

    # Step 2.5: Scavenging (Rescue orphaned triggers)
    df_assignments_integrated = scavenge_orphans(df_valid, df_tr, df_assignments)

    print(f"Total Assignments (Integrated): {len(df_assignments_integrated)}")

    # Step 3: Final Catalog
    final_counts = df_assignments_integrated['event_id'].value_counts()
    df_valid['n_unique_stations_final'] = df_valid['event_id'].map(final_counts).fillna(0).astype(int)

    # Save Catalog
    df_final_out = df_valid.copy()
    df_final_out['est_t0'] = pd.to_datetime(df_final_out['est_t0'], unit='s', utc=True).apply(lambda x: x.isoformat())
    cols = ['event_id', 'est_t0', 'est_lat', 'est_lon', 'score', 'n_unique_stations_final', 'quality_score']
    cols = [c for c in cols if c in df_final_out.columns]
    df_final_out[cols].to_csv(OUTPUT_CLEAN_CSV, index=False)
    print(f"Catalog saved: {OUTPUT_CLEAN_CSV}")

    # ----------------------------------------------------
    # Step 4: Save Details (Individual Event Files)
    # ----------------------------------------------------
    print(f"--- Exporting Details to: {OUTPUT_DETAILS_DIR} ---")

    if df_assignments_integrated.empty:
        print("!! Error: No triggers assigned. Skipping details export.")
        return

    # 1. Prepare Trigger Data (Extract from original triggers using indices)
    trigger_indices = df_assignments_integrated['trigger_idx'].values.astype(int)

    try:
        matched_triggers = df_tr.iloc[trigger_indices].copy().reset_index(drop=True)
    except Exception as e:
        print(f"!! Error indexing triggers: {e}")
        return

    if 'event_id' in matched_triggers.columns:
        matched_triggers = matched_triggers.drop(columns=['event_id'])

    # 2. Prepare Meta Data
    assignment_info = df_assignments_integrated.copy().reset_index(drop=True)

    # 3. Concatenate
    full_details = pd.concat([
        matched_triggers,
        assignment_info[['event_id', 'residual', 'dist_km', 'predicted_t']]
    ], axis=1)

    # 4. Format and Save
    full_details['predicted_t_iso'] = pd.to_datetime(full_details['predicted_t'], unit='s', utc=True).apply(lambda x: x.isoformat())

    if pd.api.types.is_datetime64_any_dtype(full_details['t_obs']):
         full_details['t_obs'] = full_details['t_obs'].apply(lambda x: x.isoformat())
    else:
        full_details['t_obs'] = full_details['t_obs'].astype(str)

    cols_to_save = ['station_id', 't_obs', 'residual', 'dist_km', 'predicted_t_iso', 'st_lat', 'st_lon']

    full_details = full_details.dropna(subset=['event_id'])

    saved_count = 0
    # Group by event_id and save separate CSVs
    for eid, group in tqdm(full_details.groupby('event_id'), desc="Saving CSVs"):
        try:
            # Step 05 expects 'event_XXXX.csv'
            fname = OUTPUT_DETAILS_DIR / f"event_{int(eid):04d}.csv"
            group[cols_to_save].to_csv(fname, index=False)
            saved_count += 1
        except Exception as e:
            print(f"Error saving event {eid}: {e}")

    print(f"Process Completed. Saved {saved_count} event detail files.")

if __name__ == "__main__":
    main()
