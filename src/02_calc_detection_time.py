import os
import sys
import glob
import tqdm
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ==============================================================================
# 02. T-Phase Detection Time Calculator
# ==============================================================================
# Input:  data/interim/yolo_output/exp/labels/*.txt
# Output: data/input/triggers_raw.csv
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

# Input (From Step 01)
RESULTS_BASE_DIR = ROOT_DIR / "data" / "interim" / "yolo_output" / "exp" / "labels"

# Output (For Step 03)
OUTPUT_CSV_PATH = ROOT_DIR / "data" / "input" / "triggers_raw.csv"

# Station Info
STATION_INFO_FILE = ROOT_DIR / "data" / "station_info" / "Hinet_pacific_all.d"
TOTAL_DURATION_SEC = 600

# --- Functions ---

def load_station_info(file_path: Path):
    if not file_path.exists():
        print(f"[ERROR] Station info not found: {file_path}")
        return {}
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None,
                         names=["station_id", "slat", "slon"], engine='python')
        return df.set_index('station_id')[['slat', 'slon']].to_dict(orient='index')
    except Exception as e:
        print(f"[ERROR] Failed to load station info: {e}")
        return {}

def calculate_detection_times(input_dir: Path, output_path: Path,
                              station_dict: dict, duration_sec: int):
    print(f"Processing labels from: {input_dir}")

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found. Run Step 01 first.")
        return

    label_files = sorted(glob.glob(str(input_dir / "*.txt")))
    if not label_files:
        print("[WARNING] No label files found.")
        return

    all_results = []
    for file_path_str in tqdm.tqdm(label_files, desc="Parsing"):
        file_path = Path(file_path_str)
        filename = file_path.stem
        # Filename format: YYYYMMDDhhmm_[STATION]
        if len(filename) < 14 or filename[12] != '_': continue

        timestamp_str = filename[:12]
        station_name = filename[13:]

        try:
            origin_time = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
        except ValueError: continue

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) < 5: continue

                    # YOLO format: class x y w h (conf)
                    x_center = float(tokens[1])
                    conf = float(tokens[5]) if len(tokens) > 5 else 1.0

                    offset = x_center * duration_sec
                    det_time = origin_time + timedelta(seconds=offset)

                    sta = station_dict.get(station_name, {})

                    all_results.append({
                        'station_id': station_name,
                        'detection_time': det_time.isoformat(),
                        'slat': sta.get('slat'),
                        'slon': sta.get('slon'),
                        'confidence': conf,
                        'origin_timestamp': timestamp_str,
                        'file_name': filename
                    })
        except: continue

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.dropna(subset=['slat', 'slon']).sort_values('detection_time')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Saved {len(df)} triggers to {output_path}")
    else:
        print("[WARNING] No valid data extracted.")

if __name__ == "__main__":
    st_data = load_station_info(STATION_INFO_FILE)
    if st_data:
        calculate_detection_times(RESULTS_BASE_DIR, OUTPUT_CSV_PATH, st_data, TOTAL_DURATION_SEC)
