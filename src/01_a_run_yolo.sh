#!/bin/bash

# =========================================================================
#  01. Run YOLOv5s model for T-phase detection
# =========================================================================

# Get the directory where this script is located (src/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Path Configurations ---
PYTHON_SCRIPT="${PROJECT_ROOT}/third_party/yolov5/detect.py"
WEIGHTS_PATH="${PROJECT_ROOT}/models/best.pt"

# Input: Place PNG files here
SOURCE_PATH="${PROJECT_ROOT}/data/input/spectrograms/"

# Output Base
OUTPUT_BASE_DIR="${PROJECT_ROOT}/data/interim/yolo_output"
NAME_DIR="exp"

# Parameters
IMG_SIZE=256
CONF_THRESHOLD=0.5

# ----------------------------------------------------

echo "--- Starting Step 01: YOLO Detection ---"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: yolov5 not found at ${PYTHON_SCRIPT}"
    exit 1
fi

if [ ! -d "${SOURCE_PATH}" ]; then
    echo "WARNING: Input directory not found: ${SOURCE_PATH}"
    mkdir -p "${SOURCE_PATH}"
fi

python "${PYTHON_SCRIPT}" \
    --weights "${WEIGHTS_PATH}" \
    --source "${SOURCE_PATH}" \
    --img ${IMG_SIZE} \
    --conf ${CONF_THRESHOLD} \
    --save-txt \
    --save-conf \
    --nosave \
    --project "${OUTPUT_BASE_DIR}" \
    --name "${NAME_DIR}" \
    --exist-ok

if [ $? -eq 0 ]; then
    echo "--- Step 01 Completed. Next: Run src/02_calc_detection_time.py ---"
else
    echo "--- Step 01 Failed. ---"
fi
