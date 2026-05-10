# Models Reference

This directory contains all YOLO model weights, training artifacts, and metrics for the ALPR system.

---

## Directory Layout

```
models/
├── indian-plate-detector/    # PRIMARY — Indian license plate detector
├── helmet-detector/          # Helmet / no-helmet classifier
├── license-plate-retrain/    # Fallback plate detector (generic dataset)
├── yolov12n-license-plate/   # UNUSED — oldest, superseded
└── yolov8n.pt                # COCO vehicle detector (stage-1 pipeline)
```

Each trained model folder contains:

| File | Description |
|---|---|
| `weights/best.pt` | Best checkpoint (by mAP50) — used by app |
| `weights/last.pt` | Last epoch checkpoint |
| `results.csv` | Per-epoch training & validation metrics |
| `results.png` | Loss and metric curves plot |
| `confusion_matrix.png` | Raw confusion matrix |
| `confusion_matrix_normalized.png` | Normalised confusion matrix |
| `BoxF1_curve.png` | F1 vs confidence curve |
| `BoxPR_curve.png` | Precision-Recall curve |
| `BoxP_curve.png` | Precision vs confidence |
| `BoxR_curve.png` | Recall vs confidence |
| `labels.jpg` | Label distribution in training set |
| `train_batch*.jpg` | Sample training batches with augmentation |
| `val_batch*_labels.jpg` | Ground-truth labels on val batch |
| `val_batch*_pred.jpg` | Model predictions on val batch |
| `args.yaml` | Full training hyperparameters |

---

## 1. indian-plate-detector ✅ ACTIVE

**Purpose:** Detect license plates on Indian cars and motorcycles.

**Training details:**

| Setting | Value |
|---|---|
| Base model | `license-plate-retrain/weights/best.pt` (fine-tuned, not from scratch) |
| Dataset | Roboflow — `nivu/indian-license-plate-knte7` v1 (1,156 train / 330 val images) |
| Epochs | 80 (early stopped at 70) |
| Image size | 640 |
| Batch | 16 |
| Optimizer | AdamW  lr=0.001 |
| Augmentation | HSV, mosaic, mixup, perspective, scale, small rotation (no flip — preserves plate text) |

**Best epoch metrics (epoch 43):**

| Metric | Value |
|---|---|
| Precision | 0.9909 |
| Recall | 0.9907 |
| **mAP50** | **0.9943** |
| mAP50-95 | 0.8138 |

**Strengths:**
- Trained specifically on Indian street scenes (cars + bikes)
- High recall — very few plates missed
- Fine-tuned from an already-trained plate model, so convergence was fast

**Limitations:**
- Dataset is ~1,600 images; performance on rare plate types (BH series, diplomatic, army) may vary
- mAP50-95 of 0.81 means bounding boxes are slightly loose on very small/distant plates

**Metric files:**
- `confusion_matrix.png` / `confusion_matrix_normalized.png`
- `BoxPR_curve.png` — PR AUC near 1.0
- `results.png` — loss converges smoothly by epoch ~40

---

## 2. helmet-detector ✅ ACTIVE

**Purpose:** Detect whether riders are wearing helmets. Classes: `helmet`, `head` (no helmet).

**Training details:**

| Setting | Value |
|---|---|
| Base model | `yolov8n.pt` (COCO pretrained) |
| Dataset | Roboflow — `joseph-nelson/hard-hat-workers` v1 |
| Epochs | 50 |
| Image size | 640 |
| Batch | 16 |
| Optimizer | Auto (SGD) lr=0.01 |

**Best epoch metrics (epoch 49):**

| Metric | Value |
|---|---|
| Precision | 0.9591 |
| Recall | 0.6073 |
| **mAP50** | **0.6488** |
| mAP50-95 | 0.4348 |

**Strengths:**
- Very high precision — when it says "no helmet", it's almost always correct (few false alarms)
- Works on full frames without needing a cropped rider region

**Limitations:**
- Recall of 0.61 — misses ~40% of bare-head instances, especially when heads are small, occluded, or at angle
- Trained on a general safety-helmet dataset (construction workers), not specifically Indian traffic scenes
- Dataset used `head` class for bare heads which have smaller pixel area than helmets

**Improvement path:** Retrain on an Indian traffic-specific helmet dataset (e.g. riders in traffic footage) to improve recall.

**Metric files:**
- `confusion_matrix_normalized.png` — shows helmet vs head classification breakdown
- `BoxF1_curve.png` — optimal confidence threshold visible here

---

## 3. license-plate-retrain ⚠️ FALLBACK

**Purpose:** Generic license plate detector, used as fallback if `indian-plate-detector` is missing.

**Training details:**

| Setting | Value |
|---|---|
| Base model | `yolov12n-license-plate/weights/best.pt` |
| Dataset | Mixed generic plate dataset (180 images) |
| Epochs | 50 (early stopped at 48) |
| Image size | 640 |
| Batch | 16 |

**Best epoch metrics (epoch 48):**

| Metric | Value |
|---|---|
| Precision | 0.9778 |
| Recall | 0.9800 |
| **mAP50** | **0.9946** |
| mAP50-95 | 0.6714 |

**Note:** mAP50 is high but mAP50-95 is lower than `indian-plate-detector` (0.67 vs 0.81), meaning bounding boxes are less precise. Also the training set was only 180 generic images — not tuned for Indian plates.

---

## 4. yolov12n-license-plate ❌ UNUSED

**Purpose:** Original first-generation plate detector trained from scratch on `yolov12n`.

Superseded by `license-plate-retrain` and then `indian-plate-detector`. Kept for reference only — not loaded by the app.

---

## 5. yolov8n.pt — Vehicle Detector (Stage 1)

**Purpose:** COCO-pretrained YOLOv8n used to locate vehicles (cars, motorcycles, buses, trucks) in the full image before running the plate detector on each vehicle crop.

**Source:** Ultralytics official pretrained weights  
**Classes used:** `car (2)`, `motorcycle (3)`, `bus (5)`, `truck (7)`

This is not fine-tuned — it uses the stock COCO weights. It restricts plate detection to vehicle regions, reducing false positives from shop signs, billboards, etc.

---

## Retraining the Plate Model

```bash
# From project root
export ROBOFLOW_API_KEY="your_key_here"
source venv/bin/activate
python stale/models/retrain_indian_plates.py
```

New weights are saved to `models/indian-plate-detector/weights/best.pt` and the app picks them up automatically on restart.

To retrain the helmet model:

```bash
source venv/bin/activate
python models/helmet-detector/helmet_train.py
```
