#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07. Visualization: Time vs Latitude/Longitude Plot
-----------------------------------------------------
Summary:
1. Loads precise estimation results (from Step 06).
2. Filters for accepted stations.
3. Generates Time-Latitude and Time-Longitude scatter plots.
4. (Optional) Overlays theoretical arrival curves if 'query.csv' exists.
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import tqdm
import datetime
from datetime import timedelta
from pathlib import Path
import matplotlib.colors as mcolors

# Plotting libraries
try:
    import pygmt
    from geopy.distance import geodesic
    HAS_PLOT_LIB = True
except ImportError:
    HAS_PLOT_LIB = False
    print("[Warning] PyGMT or Geopy not found. Plotting cannot proceed.")
    sys.exit(0)

# ==========================================
# 0. Configuration & Paths
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Input Path (From Step 06)
INPUT_DIR = DATA_DIR / "output" / "estimation_results" / "individual_events"

# Optional Input
QUERY_CSV_PATH = DATA_DIR / "input" / "query.csv"

# Output Path
OUTPUT_DIR = DATA_DIR / "output" / "figures" / "timeseries"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Plotting Parameters
VEL_KMPS = 1.5
MIN_LONGITUDE = 127
MAX_LATITUDE = 50

# Time Range for Plotting (Auto-detected if None, or set manually)
PLOT_START_TIME = None # e.g., datetime.datetime(2023, 10, 9, 5, 0)
PLOT_END_TIME = None   # e.g., datetime.datetime(2023, 10, 9, 6, 0)

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
    distance_km = geodesic((row["st_lat"], row["st_lon"]), (local_event_df["latitude"], local_event_df["longitude"])).km
    arrival_sec = distance_km / t_velocity
    return local_event_df["time"] + timedelta(seconds=arrival_sec)

def plot_clustering_results(df_plot, tmin, tmax, output_dir, vel_kmps, event_df, min_longitude, max_latitude):
    if df_plot.empty:
        print("  [Info] Data for plotting is empty.")
        return

    # Filter accepted detections only
    if 'event_id' in df_plot.columns:
        df_events = df_plot[df_plot['event_id'] != -1].copy()
        event_ids = sorted(df_events["event_id"].unique())
    else:
        df_events = df_plot.copy()
        event_ids = []

    # Color Map Setup
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
        projection="X15c/7c",
        frame=["xafg+lDate (UTC)", "yafg+lLatitude"]
    )

    for eid in event_ids:
        sub = df_events[df_events["event_id"] == eid]
        fig_lat.plot(
            x=sub["t_obs"],
            y=sub["st_lat"],
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
            label_txt = 'Theoretical arrival' if i == 0 else None
            fig_lat.plot(x=theoretical_time_series, y=df_plot["st_lat"], style="x0.1c", fill="darkgray", transparency=50, label=label_txt)

    fig_lat.legend(position="JTR+jTL+o1.0c/0c", box=True)
    fig_lat.savefig(os.path.join(output_dir, f"time_lat_events_final.png"), dpi=200)

    # --- Plot: Time vs Longitude ---
    fig_lon = pygmt.Figure()
    fig_lon.basemap(
        region=[tmin, tmax, df_plot["st_lon"].min() - 1, df_plot["st_lon"].max() + 1],
        projection="X15c/7c",
        frame=["xafg+lDate (UTC)", "yafg+lLongitude"]
    )

    for eid in event_ids:
        sub = df_events[df_events["event_id"] == eid]
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
            label_txt = 'Theoretical arrival' if i == 0 else None
            fig_lon.plot(x=theoretical_time_series, y=df_plot["st_lon"], style="x0.1c", fill="darkgray", transparency=50, label=label_txt)

    fig_lon.legend(position="JTR+jTL+o1.0c/0c", box=True)
    fig_lon.savefig(os.path.join(output_dir, f"time_lon_events_final.png"), dpi=200)

# ==========================================
# Main Process
# ==========================================
def main():
    print("--- Step 07: Time-Series Visualization ---")

    if not INPUT_DIR.exists():
        print(f"[Skip] Input directory not found: {INPUT_DIR}")
        print("Please run Step 06 first.")
        return

    # Load Query CSV (Optional)
    event_df = pd.DataFrame()
    if QUERY_CSV_PATH.exists():
        try:
            event_df = pd.read_csv(QUERY_CSV_PATH, parse_dates=["time"])
            # Adjust timezone if necessary (Assuming UTC in CSV, plotting in UTC)
            if event_df["time"].dt.tz is None:
                event_df["time"] = event_df["time"].dt.tz_localize('UTC')
            event_df["time"] = event_df["time"].dt.tz_convert(None) # Make naive for PyGMT
        except Exception as e:
            print(f"[Warning] Failed to load query.csv: {e}")

    # Find Detection Files (from Step 06)
    detection_files = glob.glob(str(INPUT_DIR / "event_*_stations.csv"))
    if not detection_files:
        print(f"[Skip] No event station files found in {INPUT_DIR}.")
        return

    all_detection_list = []
    print(f"Loading {len(detection_files)} event files...")

    for detection_file in tqdm.tqdm(detection_files, desc="Loading"):
        try:
            basename = os.path.basename(detection_file)
            # Filename format: event_[ID]_stations.csv
            # Extract ID: "event_123_stations.csv" -> "123"
            parts = basename.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                event_id = int(parts[1])
            else:
                continue

            df_raw = pd.read_csv(detection_file, parse_dates=["t_obs"])
            df_raw = df_raw.dropna(subset=["st_lat", "st_lon", "t_obs"])

            # Use only accepted stations for final plot
            if 'status' in df_raw.columns:
                df_raw = df_raw[df_raw['status'] == 'accepted']

            df_raw['event_id'] = event_id

            if not df_raw.empty:
                all_detection_list.append(df_raw)
        except Exception:
            pass

    if not all_detection_list:
        print("  No valid data found.")
        return

    # Combine Data
    all_detect_df = pd.concat(all_detection_list, axis=0, ignore_index=True)

    # Timezone Handling (Make naive UTC)
    if not pd.api.types.is_datetime64_any_dtype(all_detect_df["t_obs"]):
         all_detect_df["t_obs"] = pd.to_datetime(all_detect_df["t_obs"], format='mixed', errors='coerce')

    if all_detect_df["t_obs"].dt.tz is not None:
        all_detect_df["t_obs"] = all_detect_df["t_obs"].dt.tz_convert('UTC').dt.tz_localize(None)

    # Determine Plot Range
    tmin_plot = PLOT_START_TIME if PLOT_START_TIME else all_detect_df["t_obs"].min()
    tmax_plot = PLOT_END_TIME if PLOT_END_TIME else all_detect_df["t_obs"].max()

    # Add small margin
    margin = (tmax_plot - tmin_plot) * 0.05
    if margin == timedelta(0): margin = timedelta(minutes=10)
    tmin_plot -= margin
    tmax_plot += margin

    print(f"Plotting Range: {tmin_plot} to {tmax_plot}")

    # Filter Data for Plot
    df_plot = all_detect_df[
        (all_detect_df["t_obs"] >= tmin_plot) &
        (all_detect_df["t_obs"] <= tmax_plot) &
        (all_detect_df["st_lon"] >= MIN_LONGITUDE) &
        (all_detect_df["st_lat"] <= MAX_LATITUDE)
    ].copy()

    if df_plot.empty:
        print("  [Info] No data within the plot range.")
        return

    try:
        plot_clustering_results(
            df_plot,
            tmin=tmin_plot,
            tmax=tmax_plot,
            output_dir=OUTPUT_DIR,
            vel_kmps=VEL_KMPS,
            event_df=event_df,
            min_longitude=MIN_LONGITUDE,
            max_latitude=MAX_LATITUDE
        )
        print(f"  Plots saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"  [Error] Plotting failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import traceback
    main()
