"""
retrain_indian_plates.py
========================
Fine-tune the license-plate detector specifically on Indian vehicle
(car + motorcycle) number plates using a large Roboflow dataset.

Recommended dataset (free, ~7 k images, Indian plates on cars & bikes):
  Workspace : object-detection-xrero
  Project   : indian-license-plate-detection
  Version   : 5
  https://universe.roboflow.com/object-detection-xrero/indian-license-plate-detection

Usage
-----
  # 1. Install dependencies (once)
  pip install roboflow ultralytics

  # 2. Set your Roboflow API key
  export ROBOFLOW_API_KEY="<your_key>"      # or edit ROBOFLOW_API_KEY below

  # 3. Run
  cd <project-root>
  python models/retrain_indian_plates.py

  # 4. After training the new weights will be at:
  #    runs/detect/indian-plate-detector/weights/best.pt
  #
  #    Update MODEL_PATH in models/app.py to point to this file.

GPU note: training on CPU is slow (~8-12 h for 50 epochs on this dataset).
          If a CUDA GPU is available it will be used automatically.
"""

import os
import sys
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration — edit here or set environment variables
# ---------------------------------------------------------------------------
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")

# Roboflow dataset coordinates  (1 650 Indian car/bike plate images)
RF_WORKSPACE  = "nivu"
RF_PROJECT    = "indian-license-plate-knte7"
RF_VERSION    = 1

DATASET_DIR   = "indian_plate_dataset"          # where to download the dataset

# Start from our current best weights (fine-tune) rather than scratch
BASE_MODEL    = "runs/detect/license-plate-retrain/weights/best.pt"
# Fall back to a fresh YOLOv8n if current weights don't exist
if not Path(BASE_MODEL).exists():
    BASE_MODEL = "yolov8n.pt"

EPOCHS        = 80          # more epochs → better convergence on focused data
IMG_SIZE      = 640
BATCH         = 16          # reduce to 8 if you run out of GPU/RAM
WORKERS       = 4
PROJECT       = "runs/detect"
RUN_NAME      = "indian-plate-detector"

# ---------------------------------------------------------------------------
# Step 1 — Download dataset
# ---------------------------------------------------------------------------

def download_dataset() -> str:
    """Download the Roboflow dataset and return the local path."""
    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        sys.exit(
            "\n[ERROR] Roboflow API key not set.\n"
            "  export ROBOFLOW_API_KEY=<your_key>\n"
            "  Get a free key at https://roboflow.com\n"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "\n[ERROR] roboflow package not installed.\n"
            "  pip install roboflow\n"
        )

    print(f"\n[1/3] Downloading dataset '{RF_PROJECT}' v{RF_VERSION} ...")
    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    dataset = project.version(RF_VERSION).download("yolov8", location=DATASET_DIR)
    print(f"      Dataset saved to: {dataset.location}")
    return dataset.location


# ---------------------------------------------------------------------------
# Step 2 — Patch data.yaml so paths are absolute (avoids CWD issues)
# ---------------------------------------------------------------------------

def patch_yaml(dataset_location: str) -> str:
    """Return an absolute-path data.yaml safe to use from any cwd."""
    import yaml

    src_yaml = Path(dataset_location) / "data.yaml"
    if not src_yaml.exists():
        sys.exit(f"[ERROR] data.yaml not found at {src_yaml}")

    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)

    base = Path(dataset_location).resolve()

    def _abs(rel):
        p = Path(rel)
        if p.is_absolute():
            return str(p)
        # Roboflow often writes '../train/images' relative to a subdirectory.
        # Resolve from the yaml's parent directory first.
        candidate = (base / p).resolve()
        if candidate.exists():
            return str(candidate)
        # Strip leading '../' segments and resolve directly inside the dataset dir
        parts = p.parts
        clean_parts = [part for part in parts if part != ".."]
        candidate2 = (base / Path(*clean_parts)).resolve()
        if candidate2.exists():
            return str(candidate2)
        # Last resort: use the resolved candidate even if it doesn't exist yet
        return str(candidate)

    cfg["train"] = _abs(cfg.get("train", "train/images"))
    cfg["val"]   = _abs(cfg.get("val",   "valid/images"))
    if "test" in cfg:
        cfg["test"] = _abs(cfg["test"])

    # Ensure single class 'license_plate'
    cfg["nc"]    = 1
    cfg["names"] = ["license_plate"]

    out_yaml = str(base / "data_abs.yaml")
    with open(out_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"      Patched YAML written to: {out_yaml}")
    print(f"      train : {cfg['train']}")
    print(f"      val   : {cfg['val']}")
    return out_yaml


# ---------------------------------------------------------------------------
# Step 3 — Fine-tune
# ---------------------------------------------------------------------------

def train(data_yaml: str):
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\n[2/3] Starting training on device: {device}")
    print(f"      Base model : {BASE_MODEL}")
    print(f"      Dataset    : {data_yaml}")
    print(f"      Epochs     : {EPOCHS}  |  img: {IMG_SIZE}  |  batch: {BATCH}\n")

    model = YOLO(BASE_MODEL)
    model.train(
        data       = data_yaml,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH,
        workers    = WORKERS,
        device     = device,
        project    = PROJECT,
        name       = RUN_NAME,
        # --- augmentation tuned for Indian street conditions ---
        hsv_h      = 0.015,   # slight hue shift (day/night/shadow)
        hsv_s      = 0.6,     # saturation
        hsv_v      = 0.4,     # brightness variation
        degrees    = 5.0,     # small rotation (tilted plates)
        translate  = 0.1,
        scale      = 0.4,     # zoom: handles far/close plates
        shear      = 2.0,
        perspective= 0.0005,  # mild perspective warp
        flipud     = 0.0,     # plates shouldn't be upside-down
        fliplr     = 0.0,     # horizontal flip breaks plate text
        mosaic     = 1.0,
        mixup      = 0.1,
        # --- keep quality high ---
        patience   = 20,      # early stop if no improvement
        save       = True,
        exist_ok   = True,
        pretrained = True,
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        cos_lr     = True,
        label_smoothing = 0.1,
        conf       = 0.001,
        iou        = 0.6,
        max_det    = 10,
    )

    best = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    print(f"\n[3/3] Training complete!")
    print(f"      Best weights: {best.resolve()}")
    print(
        f"\n  Update MODEL_PATH in models/app.py to:\n"
        f"    MODEL_PATH = \"{best}\"\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset_location = download_dataset()
    data_yaml        = patch_yaml(dataset_location)
    train(data_yaml)
