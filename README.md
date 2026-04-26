# T-Phase-Automatic-Analysis-Tools

This repository contains the source code for the automated analysis of T-phase (tertiary wave) signals, as presented in the paper: **"Next-Generation Framework for T-phase Analysis: Automated Detection of a Significant Anomalous Event at Sofu Seamount in 2023"**.


## Overview

This framework provides an end-to-end pipeline for seismic T-phase detection and epicenter estimation using deep learning (YOLOv5) and robust statistical clustering (HDBSCAN). It is designed to process large volumes of hydroacoustic or seismic data to detect submarine volcanic activity or other seafloor acoustic sources.

### Key Features
- **Automated Detection**: Utilizes YOLOv5 for high-precision detection of T-phase arrivals in spectrograms.
- **Robust Clustering**: Implements a sliding-window HDBSCAN approach to group individual detections into discrete events.
- **Precise Localization**: Employs Powell's optimization method for source location estimation.
- **Uncertainty Estimation**: Calculates error ellipses using Jackknife resampling to ensure the reliability of epicenters.
- **GPU Acceleration**: Supports CuPy for high-speed grid searches and waveform stacking.

---

## Analysis Workflow

The pipeline consists of 8 sequential steps. Each script is designed to handle a specific part of the analysis:

| Step | Script | Description |
| :--- | :--- | :--- |
| **01** | `01_a_run_yolo.sh` | Runs YOLOv5 inference on spectrogram PNGs to identify T-phase signals. |
| **02** | `02_calc_detection_time.py` | Converts YOLO pixel coordinates into absolute detection timestamps and station coordinates. |
| **03** | `03_estimate_sliding_window.py` | Performs real-time simulation using sliding windows and HDBSCAN to detect event candidates. |
| **04** | `04_refine_catalog.py` | Merges duplicate events, resolves trigger assignment conflicts, and creates a refined catalog. |
| **05** | `05_assign_ids_and_prep.py` | Assigns chronological IDs and prepares data for precise localization. |
| **06** | `06_estimate_precise_jk.py` | Executes precise Powell optimization and Jackknife resampling for error estimation. |
| **07** | `07_plot_timeseries.py` | Visualizes event clusters in Time-Latitude and Time-Longitude space. |
| **08** | `08_plot_epicenters.py` | Generates final epicenter maps with 1-sigma error ellipses using PyGMT. |

---

## Directory Structure

To ensure the scripts run correctly, organize your project directory as follows:

```text
T-Phase-Automatic-Analysis-Tools/
├── src/                        # Source scripts (01-08)
├── models/                     # YOLOv5 weights (e.g., best.pt)
├── third_party/                # YOLOv5 repository
└── data/
    ├── input/
    │   ├── spectrograms/       # Input PNG files for YOLO
    │   ├── waveforms/          # SAC files for precise estimation
    │   ├── station_info/       # Hinet_pacific_all.d etc.
    │   └── prior_distribution_masked.npy  # Spatial prior map
    ├── interim/                # Intermediate results (CSV/txt)
    ├── processed/              # ID assigned data
    └── output/                 # Final results and figures
```

---    

## Installation

### Prerequisites
The following libraries are required to run the pipeline. For heavy processing (Step 03 & 06), a CUDA-capable GPU is highly recommended.

- **Python 3.8+**
- **Core Processing**: `numpy`, `pandas`, `scipy`, `tqdm`, `scikit-learn` (or `hdbscan`)
- **Seismology**: `obspy` (for SAC file handling and signal processing)
- **Computer Vision**: `torch`, `opencv-python` (required by YOLOv5)
- **GPU Acceleration**: `cupy` (optional, for high-speed waveform stacking)
- **Visualization**: `pygmt`, `geopy`, `matplotlib`

### Setup
1. **Clone the repository**
   ```bash
   git clone [https://github.com/YourUsername/T-Phase-Automatic-Analysis-Tools.git](https://github.com/YourUsername/T-Phase-Automatic-Analysis-Tools.git)
   cd T-Phase-Automatic-Analysis-Tools
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Initialize YOLOv5**
   This project uses YOLOv5 as a submodule or external library. Ensure the third_party/yolov5 directory is populated.
### Usage Guide
The analysis follows a strict sequential order (Step 01 to 08):
1. **Run YOLOv5 Detection** (`01_a_run_yolo.sh`):
Infers T-phase signals from spectrogram PNGs using the trained weights in models/best.pt.
2. **Calculate Detection Times** (`02_calc_detection_time.py`):
Converts YOLO bounding boxes into temporal detections and maps them to station coordinates.
3. **Sliding Window Search** (`03_estimate_sliding_window.py`):
   Simulates real-time detection using HDBSCAN and GPU-accelerated grid search to find event candidates.
4. **Refine Catalog** (`04_refine_catalog.py`):
   Merges spatial-temporal duplicates and resolves conflicts where one trigger might belong to multiple events.
5. **Data Preparation** (`05_assign_ids_and_prep.py`):
   Standardizes event IDs and prepares metadata for high-precision relocation.
6. **Prexise Localization & Jackknife** (`06_estimate_precise_jk.py`):
   Runs Powell optimization for the final epicenter and estimates error ellipses via Jackknife resampling.
7. **Time-Series Plots** (`07_plot_timeseries.py`):
   Generates Time-Lat and Time-Lon plots to visualize event clusters against theoretical arrivals.
8. **Epicenter Maps** (`08_plot_epicenters.py`):
   Creates publication-quality maps with error ellipses using PyGMT.

