#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05. Assign IDs and Prepare Data for Precise Estimation
---------------------------------------------------------------------------------------------
Summary:
1. Reads refined event details (Step 04 output).
2. Sorts events chronologically and assigns sequential IDs.
3. Maps initial epicenter estimates from the catalog.
4. Outputs formatted data for Step 06 (Precise Estimation).
5. (Optional) Generates preliminary Time-Lat/Lon plots.
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import tqdm
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.colors as mcolors

# Plotting libraries (Optional)
try:
    import pygmt
    from geopy.distance import geodesic
    HAS_PLOT_LIB = True
except ImportError:
    HAS_PLOT_LIB = False
    print("[Info] PyGMT or Geopy not found. Plotting will be skipped.")

# ==========================================
# 0. Configuration & Paths
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Input Paths (From Step 04)
INPUT_DIR = DATA_DIR / "interim" / "refined_catalog" / "event_details"
CATALOG_PATH = DATA_DIR / "interim" / "refined_catalog" / "events_catalog_final.csv"

# Optional Input (For theoretical arrival comparison in plots)
QUERY_CSV_PATH = DATA_DIR / "input" / "query.csv"

# Output Paths
OUTPUT_DIR = DATA_DIR / "processed" / "id_assigned"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Output Files
OUT_ALL_DETECT_CSV = OUTPUT_DIR / "all_event_detect_with_new_ids.csv"
OUT_MAPPING_CSV = OUTPUT_DIR / "initial_epicenters.csv"
OUT_PLOT_DIR = OUTPUT_DIR / "plots"

# Plotting Parameters
VEL_KMPS = 1.5
MIN_LONGITUDE = 127
MAX_LATITUDE = 50

# ==========================================
# Functions
# ==========================================
def get_dark_colors_by_luminance(min_luminance=0.5):
    all_css4_colors = mcolors.CSS4_COLORS
    all_colors = all_css4_colors.copy()
    all_colors.update(mcolors.TABLEAU_COLORS)
    usable_color_map = {}
    for name, hex_code in all_colors.items():
        rgb_tuple = mcolors.hex2color(hex_code)
        R, G, B = rgb_tuple
        luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
        if luminance < min_luminance: usable_color_map[name] = hex_code
    return usable_color_map

def compute_theoretical_arrival(row, local_event_df, t_velocity=1.5):
    if not HAS_PLOT_LIB: return None
    distance_km = geodesic((row["st_lat"], row["st_lon"]), (local_event_df["latitude"], local_event_df["longitude"])).km
    arrival_sec = distance_km / t_velocity
    return local_event_df["time"] + timedelta(seconds=arrival_sec)

def plot_clustering_results(df_plot, tmin, tmax, output_dir, vel_kmps, event_df):
    if not HAS_PLOT_LIB: return
    if df_plot.empty: return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter Noise
    if 'new_event_id' in df_plot.columns:
        df_events = df_plot[df_plot['new_event_id'] != -1].copy()
        event_ids = sorted(df_events["new_event_id"].unique())
    else:
        df_events = df_plot.copy()
        event_ids = []

    # Color Map
    LUMINANCE_THRESHOLD = 0.85
    usable_color_hex_map = get_dark_colors_by_luminance(min_luminance=LUMINANCE_THRESHOLD)
    usable_colors = list(usable_color_hex_map.values())
    if not usable_colors:
        default_colors_hex = [mcolors.to_hex(f"C{i}") for i in range(10)]
        color_map = {eid: default_colors_hex[i % 10] for i, eid in enumerate(event_ids)}
    else:
        color_map = {eid: usable_colors[i % len(usable_colors)] for i, eid in enumerate(event_ids)}

    # --- Plot: Time vs Latitude ---
    fig_lat = pygmt.Figure()
    fig_lat.basemap(
        region=[tmin, tmax, df_plot["st_lat"].min() - 1, df_plot["st_lat"].max() + 1],
        projection="X15c/8c",
        frame=["xafg+lDate (JST)", "yafg+lLatitude"]
    )

    for eid in event_ids:
        sub = df_events[df_events["new_event_id"] == eid]
        fig_lat.plot(
            x=sub["t_obs"],
            y=sub["st_lat"],
            style="c0.2c",
            pen='0.2p,black',
            fill=color_map.get(eid, "black"),
            transparency=30,
            label=f"Event {eid}"
        )

    # Theoretical Arrivals
    if not event_df.empty:
        for i in range(len(event_df)):
            sub_event_df = event_df.iloc[i]
            theoretical_time_series = df_plot.apply(compute_theoretical_arrival, local_event_df=sub_event_df, t_velocity=vel_kmps, axis=1)
            fig_lat.plot(x=theoretical_time_series, y=df_plot["st_lat"], style="x0.1c", fill="darkgray", transparency=50)

    fig_lat.savefig(os.path.join(output_dir, "time_lat_events_preview.png"), dpi=200)

    # --- Plot: Time vs Longitude ---
    fig_lon = pygmt.Figure()
    fig_lon.basemap(
        region=[tmin, tmax, df_plot["st_lon"].min() - 1, df_plot["st_lon"].max() + 1],
        projection="X15c/8c",
        frame=["xafg+lDate (JST)", "yafg+lLongitude"]
    )

    for eid in event_ids:
        sub = df_events[df_events["new_event_id"] == eid]
        fig_lon.plot(
            x=sub["t_obs"],
            y=sub["st_lon"],
            style="c0.2c",
            pen='0.2p,black',
            fill=color_map.get(eid, "black"),
            transparency=30,
            label=f"Event {eid}"
        )

    if not event_df.empty:
        for i in range(len(event_df)):
            sub_event_df = event_df.iloc[i]
            theoretical_time_series = df_plot.apply(compute_theoretical_arrival, local_event_df=sub_event_df, t_velocity=vel_kmps, axis=1)
            fig_lon.plot(x=theoretical_time_series, y=df_plot["st_lon"], style="x0.1c", fill="darkgray", transparency=50)

    fig_lon.savefig(os.path.join(output_dir, "time_lon_events_preview.png"), dpi=200)
    print(f"  Plots saved to: {output_dir}")

# ==========================================
# Main Process
# ==========================================
def main():
    print(f"--- Step 05: Assign IDs and Prepare Data ---")

    if not INPUT_DIR.exists():
        print(f"[Skip] Input directory not found: {INPUT_DIR}")
        print("Please run Step 04 first.")
        return

    # 1. Load Catalog (for initial epicenter info)
    catalog_data = {}
    if CATALOG_PATH.exists():
        print(f"Loading catalog from: {CATALOG_PATH}")
        try:
            df_cat = pd.read_csv(CATALOG_PATH)
            if 'event_id' in df_cat.columns:
                df_cat['event_id'] = df_cat['event_id'].astype(int)
                catalog_data = df_cat.set_index('event_id').to_dict('index')
        except Exception as e:
            print(f"!! Error loading catalog: {e}")
    else:
        print(f"!! Catalog file not found: {CATALOG_PATH}")

    # 2. Load Event Detail Files
    csv_files = glob.glob(str(INPUT_DIR / "*.csv"))
    if not csv_files:
        print(f"[Skip] No csv files in {INPUT_DIR}.")
        return

    # 3. Sort by Time
    temp_event_list = []
    print(f"Pre-reading {len(csv_files)} files to sort by time...")

    for csv_file in tqdm.tqdm(csv_files, desc="Pre-reading"):
        try:
            df_raw = pd.read_csv(csv_file, parse_dates=["t_obs"])
            df_raw = df_raw.dropna(subset=["st_lat", "st_lon", "t_obs"])

            if df_raw.empty: continue

            min_time = df_raw["t_obs"].min()

            temp_event_list.append({
                "file_path": csv_file,
                "df": df_raw,
                "min_time": min_time,
                "basename": os.path.basename(csv_file)
            })
        except Exception:
            continue

    temp_event_list.sort(key=lambda x: x["min_time"])

    # 4. Assign IDs and Map Info
    all_detection_list = []
    event_mapping_list = []
    current_new_id = 0

    print(f"Processing sorted events...")

    for item in tqdm.tqdm(temp_event_list, desc="Assigning IDs"):
        basename = item["basename"]
        current_new_id += 1

        # Extract Original ID from filename "event_XXXX.csv"
        original_id_str = os.path.splitext(basename)[0].split('_')[-1] # "0001"

        est_lat, est_lon, est_t0 = np.nan, np.nan, None
        original_id = -1

        try:
            if original_id_str.isdigit():
                original_id = int(original_id_str)

            if original_id in catalog_data:
                info = catalog_data[original_id]
                est_lat = info.get('est_lat', np.nan)
                est_lon = info.get('est_lon', np.nan)
                est_t0 = info.get('est_t0', None)
        except:
            pass

        df_raw = item["df"]
        df_raw['new_event_id'] = current_new_id
        all_detection_list.append(df_raw)

        event_mapping_list.append({
            'event_id': current_new_id,
            'original_event_id': original_id,
            'start_time': item["min_time"],
            'est_lat': est_lat,
            'est_lon': est_lon,
            'est_t0': est_t0,
            'filename': basename
        })

    if not all_detection_list:
        print("  No valid data processed.")
        return

    # 5. Save Combined CSV (Input for Step 06)
    all_detect_df = pd.concat(all_detection_list, axis=0, ignore_index=True)

    # Cleaning
    if not pd.api.types.is_datetime64_any_dtype(all_detect_df["t_obs"]):
         all_detect_df["t_obs"] = pd.to_datetime(all_detect_df["t_obs"], format='mixed', errors='coerce')
    if all_detect_df["t_obs"].dt.tz is not None:
        all_detect_df["t_obs"] = all_detect_df["t_obs"].dt.tz_localize(None)

    all_detect_df.to_csv(OUT_ALL_DETECT_CSV, index=False)
    print(f"  Saved All Detections: {OUT_ALL_DETECT_CSV}")

    # 6. Save Mapping (Input for Step 06)
    if event_mapping_list:
        mapping_df = pd.DataFrame(event_mapping_list)
        mapping_df.to_csv(OUT_MAPPING_CSV, index=False)
        print(f"  Saved Initial Epicenters: {OUT_MAPPING_CSV}")

    # 7. (Optional) Plotting
    if HAS_PLOT_LIB:
        # Check for query.csv
        event_df = pd.DataFrame()
        if QUERY_CSV_PATH.exists():
             try:
                event_df = pd.read_csv(QUERY_CSV_PATH, parse_dates=["time"])
                event_df["time"] = event_df["time"].dt.tz_localize(None)
             except: pass

        tmin_plot = all_detect_df["t_obs"].min()
        tmax_plot = all_detect_df["t_obs"].max()

        try:
            plot_clustering_results(
                all_detect_df,
                tmin=tmin_plot,
                tmax=tmax_plot,
                output_dir=OUT_PLOT_DIR,
                vel_kmps=VEL_KMPS,
                event_df=event_df
            )
        except Exception as e:
            print(f"  [Warning] Plotting failed: {e}")

if __name__ == "__main__":
    main()
