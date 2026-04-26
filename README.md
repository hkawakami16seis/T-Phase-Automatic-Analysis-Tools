# T-Phase-Automatic-Analysis-Tools

This repository contains the source code for the automated analysis of T-phase signals, as presented in the paper:

> **"Next-Generation Framework for T-phase Analysis: Automated Detection of a Significant Anomalous Event at Sofu Seamount in 2023"**

---

## Overview

This framework provides a pipeline for seismic T-phase detection and epicenter estimation using deep learning (YOLOv5) and robust statistical clustering (HDBSCAN).

It is designed to process large volumes of seismic data to detect submarine volcanic activity or other seafloor acoustic sources.

### Key Features

- **Automated Detection**  
  Utilizes YOLOv5 for high-precision detection of T-phase arrivals in spectrograms.

- **Robust Clustering**  
  Implements a sliding-window HDBSCAN approach to group individual detections into discrete events.

- **Precise Localization**  
  Employs Powell's optimization method for source location estimation.

- **Uncertainty Estimation**  
  Calculates error ellipses using Jackknife resampling to ensure the reliability of epicenters.

- **GPU Acceleration**  
  Supports CuPy for high-speed grid searches and waveform stacking.

---

## Analysis Workflow

The pipeline consists of **8 sequential steps**, each handled by a dedicated script:

| Step | Script | Description |
|------|--------|------------|
| **01** | `01_a_run_yolo.sh` | Runs YOLOv5 inference on spectrogram PNGs to identify T-phase signals |
| **02** | `02_calc_detection_time.py` | Converts YOLO pixel coordinates into timestamps and station coordinates |
| **03** | `03_estimate_sliding_window.py` | Performs sliding-window HDBSCAN clustering to detect events |
| **04** | `04_refine_catalog.py` | Merges duplicate events and resolves trigger conflicts |
| **05** | `05_assign_ids_and_prep.py` | Assigns IDs and prepares data for localization |
| **06** | `06_estimate_precise_jk.py` | Performs Powell optimization and Jackknife error estimation |
| **07** | `07_plot_timeseries.py` | Visualizes event clusters in time-lat/lon space |
| **08** | `08_plot_epicenters.py` | Generates final epicenter maps using PyGMT |

---

## Directory Structure

Ensure your project directory is organized as follows:

```
T-Phase-Automatic-Analysis-Tools/
├── src/                        # Source scripts (01–08)
├── models/                     # YOLOv5 weights (e.g., best.pt)
├── third_party/                # YOLOv5 repository
└── data/
    ├── input/
    │   ├── spectrograms/       # Input PNG files for YOLO
    │   ├── waveforms/          # SAC files for precise estimation
    │   └── prior_distribution_masked.npy
    ├── station_info/           # Station metadata (Hinet_pacific_all.d)
    ├── interim/                # Intermediate results
    ├── processed/              # ID assigned data
    └── output/                 # Final results and figures
```

---

## Installation

### Prerequisites

- **Python 3.8+**

**Core Libraries**
- numpy
- pandas
- scipy
- tqdm
- scikit-learn / hdbscan

**Seismology**
- obspy

**Computer Vision**
- torch
- opencv-python

**Optional (GPU Acceleration)**
- cupy

**Visualization**
- pygmt
- geopy
- matplotlib

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/T-Phase-Automatic-Analysis-Tools.git
cd T-Phase-Automatic-Analysis-Tools
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize YOLOv5

The source code of YOLOv5 is NOT included in this repository. You must clone it manually into the dedicated third_party/ folder.

```bash
mkdir -p third_party
git clone https://github.com/ultralytics/yolov5 third_party/yolov5
---

## Usage Guide

Run the pipeline sequentially from **Step 01 to Step 08**:

### 1. YOLOv5 Detection
`01_a_run_yolo.sh`  
Detects T-phase signals from spectrogram images.

### 2. Detection Time Conversion
`02_calc_detection_time.py`  
Converts bounding boxes into timestamps and station locations.

### 3. Sliding Window Detection
`03_estimate_sliding_window.py`  
Performs sliding-window HDBSCAN for noise filtering and grid search to detect event candidates.

### 4. Catalog Refinement
`04_refine_catalog.py`  
Removes duplicates and resolves conflicts.

### 5. Data Preparation
`05_assign_ids_and_prep.py`  
Prepares structured data for localization.

### 6. Precise Localization & Jackknife
`06_estimate_precise_jk.py`  
Performs final epicenter estimation and uncertainty analysis.

### 7. Time-Series Visualization
`07_plot_timeseries.py`  
Generates time vs location plots.

### 8. Epicenter Mapping
`08_plot_epicenters.py`  
Creates publication-quality maps with error ellipses.

---

## Notes

- Steps **03** and **06** benefit significantly from GPU acceleration.
- Maintain strict step order to ensure pipeline integrity.

---

## Acknowledgement
Station metadata (ID, Latitude, and Longitude) provided in this repository are subsets of the NIED Hi-net station list, used for the reproducibility of this study. We thank NIED for maintaining this critical infrastructure.
