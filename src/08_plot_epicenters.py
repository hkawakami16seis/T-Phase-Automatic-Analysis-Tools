#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08. Visualization: Epicenter Map with Error Ellipse
-----------------------------------------------------
Summary:
1. Loads estimation summary and station details (from Step 06).
2. Calculates error ellipses from Jackknife results (Step 06 debug output).
3. Plots the epicenter, accepted stations, and error ellipses on a map using PyGMT.
4. Includes an inset map showing the broader station distribution.
"""
import os
import sys
import glob
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Plotting
try:
    import pygmt
except ImportError:
    print("[Error] PyGMT not found. Please install pygmt to run this script.")
    sys.exit(0)

# ==========================================
# 0. Configuration & Paths
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Input Path (From Step 06)
ESTIMATION_RESULT_DIR = DATA_DIR / "output" / "estimation_results"

# Output Path
OUTPUT_DIR = DATA_DIR / "output" / "figures" / "maps"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Plotting Parameters ---
LAT_MIN = -60.0
LAT_MAX = 70.0
LON_MIN = 100.0
LON_MAX = 310.0

STATION_LAT_MAX = 60 # Filter out far north stations for clarity if needed
STATION_LON_MIN = 100

# Map Settings
MARGIN_DEG = 3.0  # Local map zoom range (+/- degrees from epicenter)
CONFIDENCE_SIGMA = 1.0 # 1 sigma for ellipse (approx 68%). Use 2.447 for 95%.

# Optional: List of True Epicenters for Comparison (e.g. from USGS)
# Dictionary format: { event_id: [lon, lat], ... }
TRUE_EPICENTERS = {
    # Example: 1: [140.0613, 29.6904]
}

# ==========================================
# Functions
# ==========================================
def norm_lon(lon, wrap=180):
    if wrap == 180:
        return (lon + 180.0) % 360.0 - 180.0
    else:
        return lon % 360.0

def get_jackknife_ellipse_points(csv_path, confidence_level_sigma=1.0):
    """
    Calculates error ellipse parameters from Jackknife scatter CSV.
    """
    if not os.path.exists(csv_path):
        # print(f"Warning: Jackknife file not found: {csv_path}")
        return None, None, None

    try:
        df = pd.read_csv(csv_path)
        N = len(df)
        if N < 2: return None, None, None

        lats = df['lat'].values
        lons = df['lon'].values

        mean_lat = np.mean(lats)
        mean_lon = np.mean(lons)

        # km conversion
        re = 6371.0
        deg2km_lat = 2 * np.pi * re / 360.0
        deg2km_lon = 2 * np.pi * re * np.cos(np.radians(mean_lat)) / 360.0

        y_km = (lats - mean_lat) * deg2km_lat
        x_km = (lons - mean_lon) * deg2km_lon
        data_km = np.stack((x_km, y_km), axis=0)

        # Covariance (Jackknife scaling: (N-1) * Var_sample)
        cov_sample = np.cov(data_km, ddof=0)
        cov_jk = (N - 1) * cov_sample

        vals, vecs = np.linalg.eigh(cov_jk)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Axes length
        major_axis = np.sqrt(vals[0]) * confidence_level_sigma
        minor_axis = np.sqrt(vals[1]) * confidence_level_sigma
        angle_rad = np.arctan2(vecs[1, 0], vecs[0, 0])

        # Generate ellipse points
        theta = np.linspace(0, 2*np.pi, 100)
        x_prime = major_axis * np.cos(theta)
        y_prime = minor_axis * np.sin(theta)

        x_ellipse = x_prime * np.cos(angle_rad) - y_prime * np.sin(angle_rad)
        y_ellipse = x_prime * np.sin(angle_rad) + y_prime * np.cos(angle_rad)

        ellipse_lons = x_ellipse / deg2km_lon + mean_lon
        ellipse_lats = y_ellipse / deg2km_lat + mean_lat

        stats = {
            'mean_lon': mean_lon, 'mean_lat': mean_lat,
            'major_axis': major_axis, 'minor_axis': minor_axis,
            'azimuth': (90 - np.degrees(angle_rad)) % 360
        }

        return ellipse_lons, ellipse_lats, stats

    except Exception as e:
        print(f"Error processing jackknife file {csv_path}: {e}")
        return None, None, None

def main_plotting(internal_event_id, df_source, df_detection, jk_file_path, output_dir, true_epic=None):

    # Filter Stations
    valid_stations = df_detection[(df_detection['st_lat'].notnull()) & (df_detection['st_lon'].notnull())]
    valid_stations = valid_stations[(valid_stations['st_lat'] <= STATION_LAT_MAX) & (valid_stations['st_lon'] >= STATION_LON_MIN)]
    sta_lons = valid_stations['st_lon'].unique()
    sta_lats = valid_stations['st_lat'].unique()

    data = df_source.iloc[0] # Summary row

    # Calculate Ellipse
    el_lons, el_lats, el_stats = get_jackknife_ellipse_points(jk_file_path, confidence_level_sigma=CONFIDENCE_SIGMA)

    # Define Map Regions
    # Global Inset Region
    region_global = [120.0, 160.0, 20.0, 50.0]

    # Local Map Region (Centered on estimated epicenter)
    est_lon = data['lon']
    est_lat = data['lat']

    local_region = [
        est_lon - MARGIN_DEG, est_lon + MARGIN_DEG,
        est_lat - MARGIN_DEG, est_lat + MARGIN_DEG,
    ]

    # Normalize Longitude
    local_region[0] = norm_lon(local_region[0], 180)
    local_region[1] = norm_lon(local_region[1], 180)

    if local_region[0] > local_region[1]:
        local_region[0] -= 360.0 # Handle dateline crossing if needed for PyGMT

    # ----------------------------------------------------
    # Plotting
    # ----------------------------------------------------
    fig = pygmt.Figure()
    map_projection = "M15c"

    # 1. Base Map (Earth Relief)
    # Adjust region for PyGMT grid loading (0-360 preferred sometimes)
    local_region_0_360 = [norm_lon(local_region[0], 360), norm_lon(local_region[1], 360), local_region[2], local_region[3]]
    if local_region_0_360[0] > local_region_0_360[1]:
         local_region_0_360[0] -= 360.0
         local_region_0_360[1] += 360.0

    try:
        grid = pygmt.datasets.load_earth_relief(resolution="15s", region=local_region_0_360)
        fig.grdimage(grid=grid, cmap="oleron", shading="+a-45+nt0.75", projection=map_projection)
    except Exception:
        # Fallback if grid load fails
        fig.basemap(region=local_region, projection=map_projection, frame=True)
        fig.coast(land="gray", water="lightblue")

    fig.basemap(region=local_region, frame=["a", f"+tEvent {internal_event_id}: Epicenter Map"])
    fig.coast(shorelines=True)

    # 2. Plot Stations
    fig.plot(x=sta_lons, y=sta_lats, style="t0.2c", fill="cyan", pen="0.5p,cyan", label="Station")

    # 3. Plot Error Ellipse
    if el_stats:
        long_axis_km = el_stats['major_axis'] * 2
        short_axis_km = el_stats['minor_axis'] * 2
        azimuth_deg = el_stats['azimuth']

        # PyGMT Style: E<azimuth>/<major>/<minor>
        # Note: PyGMT expects axes in degrees if using geographical projection, or km if configured.
        # Here we use the km scaling computed previously but plotting on map requires careful unit handling.
        # Alternatively, plot the polygon points directly which is safer across projections.

        if el_lons is not None:
             fig.plot(x=el_lons, y=el_lats, pen="1.5p,black,-") # Ellipse Outline

    # 4. Plot Estimated Epicenter
    fig.plot(x=est_lon, y=est_lat, style="a0.7c", fill="orange", pen='0.1p,black', label="Est. Epicenter")

    # 5. Plot True Epicenter (if available)
    if true_epic:
        fig.plot(x=true_epic[0], y=true_epic[1], style="a0.7c", fill="red", pen='0.1p,black', label="True Epicenter")

    # 6. Inset Map
    with fig.inset(position="jBR+o0.1c", box=True, region=region_global, projection="M6c"):
        try:
            grid_glob = pygmt.datasets.load_earth_relief(resolution="01m", region=region_global)
            fig.grdimage(grid=grid_glob, cmap="oleron", shading="+a-45+nt0.75")
        except:
            fig.coast(land="gray", water="white")

        fig.coast(shorelines=True)

        # Stations in Inset
        fig.plot(x=sta_lons, y=sta_lats, style="t0.15c", fill="cyan", pen="0.1p,black")

        # Area Rectangle
        rect = [
            [local_region[0], local_region[2]],
            [local_region[0], local_region[3]],
            [local_region[1], local_region[3]],
            [local_region[1], local_region[2]],
            [local_region[0], local_region[2]],
        ]
        fig.plot(data=rect, pen="1p,red")

        # Epicenter in Inset
        fig.plot(x=est_lon, y=est_lat, style="a0.3c", fill="orange", pen="0.1p,black")

    # Save
    out_path = os.path.join(output_dir, f"event_{internal_event_id}_epicenter_map.png")
    fig.savefig(out_path)
    # print(f"Saved map: {out_path}")

# ==========================================
# Main Workflow
# ==========================================
def main():
    print("--- Step 08: Epicenter Map Visualization ---")

    if not ESTIMATION_RESULT_DIR.exists():
        print(f"[Skip] Input directory not found: {ESTIMATION_RESULT_DIR}")
        print("Please run Step 06 first.")
        return

    # Search for Summary Files
    summary_files = sorted(glob.glob(str(ESTIMATION_RESULT_DIR / "individual_events" / "event_*_summary.csv")))

    if not summary_files:
        print("[Skip] No summary files found.")
        return

    print(f"Found {len(summary_files)} events to plot.")

    for sum_file in tqdm(summary_files, desc="Plotting Maps"):
        try:
            # Parse ID
            basename = os.path.basename(sum_file)
            parts = basename.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                event_id = int(parts[1])
            else:
                continue

            # Load Data
            df_sum = pd.read_csv(sum_file)

            det_file = ESTIMATION_RESULT_DIR / "individual_events" / f"event_{event_id}_stations.csv"
            jk_file = ESTIMATION_RESULT_DIR / "debug_jackknife" / f"jk_detail_event_{event_id}.csv"

            if not det_file.exists():
                continue

            df_det = pd.read_csv(det_file)
            # Filter for accepted stations
            if 'status' in df_det.columns:
                df_det = df_det[df_det['status'] == 'accepted']

            # Check for True Epicenter
            true_epic = TRUE_EPICENTERS.get(event_id, None)

            # Plot
            main_plotting(event_id, df_sum, df_det, jk_file, OUTPUT_DIR, true_epic)

        except Exception as e:
            print(f"Error plotting event {sum_file}: {e}")

    print(f"Done. Maps saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
