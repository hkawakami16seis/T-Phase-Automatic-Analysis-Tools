#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00. Pre-processing: Batch Generate Spectrograms
-----------------------------------------------------------
Summary:
1. Scans all SAC files in the input directory.
2. Applies standardized signal processing (detrend, filter, taper).
3. Generates the standard multi-panel PNG required for Step 01/02.
   - Top: Waveform
   - Bottom: Spectrogram (Log-scale PSD)
   - Right: Colorbar
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 0. Configuration & Paths
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "data" / "input" / "waveforms"
OUTPUT_DIR = ROOT_DIR / "data" / "input" / "spectrograms"

# --- Signal Processing & Plotting Params ---
# Note: These parameters should match the YOLOv5 training environment
FREQ_MIN = 1.0
FREQ_MAX = 20.0
SPECTROGRAM_WLEN = 5
SPECTROGRAM_PER_LAP = 0.5
DB_VMIN = -180
DB_VMAX = -100

def generate_spectrograms():
    # 1. Setup Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Find all SAC files
    sac_files = sorted(glob.glob(str(INPUT_DIR / "*.sac")))

    if not sac_files:
        print(f"[Warning] No SAC files found in {INPUT_DIR}")
        return

    print(f"Processing {len(sac_files)} files...")

    for sac_path in tqdm(sac_files):
        try:
            # 3. Read Waveform
            st = read(sac_path)
            if len(st) == 0: continue
            tr = st[0]

            # 4. Signal Processing
            tr.detrend(type='demean')
            tr.detrend(type='linear')
            tr.filter(type='bandpass', freqmin=FREQ_MIN, freqmax=FREQ_MAX)
            tr.taper(max_percentage=0.001)

            # 5. Extract Timing for Filename
            # Required format: YYYYMMDDhhmm_[STATION].png
            timestamp = tr.stats.starttime.strftime("%Y%m%d%H%M")
            station = tr.stats.station
            out_fname = OUTPUT_DIR / f"{timestamp}_N.{station}.png"

            # 6. Visualization (Original Layout)
            fig = plt.figure(figsize=(5, 4))
            plt.rcParams['lines.linewidth'] = 0.3

            # Axes definition: [left, bottom, width, height]
            ax1 = fig.add_axes([0.1, 0.8, 0.7, 0.2]) # Waveform
            ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1) # Spectrogram
            ax3 = fig.add_axes([0.82, 0.1, 0.03, 0.6]) # Colorbar

            # Plot Waveform (Top)
            t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
            ax1.plot(t, tr.data, 'k')
            ax1.tick_params(labelbottom=False, labelleft=False, left=False, bottom=True)

            # Plot Spectrogram (Bottom)
            tr.spectrogram(
                samp_rate=tr.stats.sampling_rate,
                wlen=SPECTROGRAM_WLEN,
                per_lap=SPECTROGRAM_PER_LAP,
                show=False, log=True, axes=ax2, dbscale=True
            )
            ax2.set_ylim(0.5, 20)
            ax2.set_ylabel("Frequency [Hz]", fontsize=12)
            ax2.set_xlabel("Time [s]", fontsize=12)

            # Apply Color Limits & Colorbar
            mappable = ax2.collections[0]
            mappable.set_clim(vmin=DB_VMIN, vmax=DB_VMAX)
            cb = plt.colorbar(mappable=mappable, cax=ax3)
            cb.set_label('PSD [dB $m^2s^{-3}$]', fontsize=10)

            # Save
            plt.savefig(out_fname, bbox_inches='tight', dpi=100)
            plt.close(fig)

        except Exception as e:
            print(f"\n[Error] Failed to process {sac_path}: {e}")
            continue

if __name__ == "__main__":
    generate_spectrograms()
    print(f"\nCompleted. Images are saved in: {OUTPUT_DIR}")
