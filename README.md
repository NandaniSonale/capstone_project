# Compressed-Domain Object Tracking and Human Activity Recognition (HAR)

A high-performance pipeline for object detection, BAFE (Block-Adaptive Feature Extraction) propagation across P/B-frames, and BiLSTM-based Human Activity Recognition directly within the compressed video domain.

[![GitHub Release](https://img.shields.io/github/v/release/NandaniSonale/capstone_project?label=Visualized%20Videos&color=brightgreen)](https://github.com/NandaniSonale/capstone_project/releases/tag/v1.0.0-artifacts)

---

## 📥 Visualized Output Demonstration Videos

Full visualized `.mp4` video outputs featuring:
- **Bounding Boxes**: Color-coded detection anchors (`DET` on I-frames) and motion propagations (`PROP` on P/B-frames)
- **16×16 Macroblock Microboxes**: Motion grid within Regions of Interest (ROI)
- **Real-Time Motion Vector Arrows**: Visualizing directional magnitude `(dx, dy)`
- **Telemetry HUD Overlays**: Frame type, PTS, and macroblock count

Directly stream or download the rendered demonstration videos from the official GitHub Release:
👉 **[Download Rendered Output Videos (v1.0.0-artifacts)](https://github.com/NandaniSonale/capstone_project/releases/tag/v1.0.0-artifacts)**

---

## 📊 Evaluation & Accuracy Benchmarks

### 1. Compressed-Domain Object Detection & Propagation Accuracy
Evaluated across all **171 videos** and **51,365 frames** from the Walking activity dataset:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **mAP @ IoU 0.50** | **98.48%** | High precision detection and propagation |
| **mAP @ IoU 0.75** | **71.92%** | Strict overlap alignment |
| **mAP @ IoU [0.50:0.95]**| **68.83%** | Comprehensive COCO-style metric |
| **Overall Mean IoU** | **81.59%** | Average across all 51,365 frames |
| **I-Frame Anchor IoU** | **98.55%** | Ground-truth keyframe alignment |
| **BAFE Propagation IoU** | **80.98%** | P and B frame motion propagation |
| **Precision @ 0.50** | **98.48%** | Low false-positive rate |
| **Recall @ 0.50** | **98.73%** | High tracking retention |
| **F1-Score @ 0.50** | **98.60%** | Balanced detection performance |

### 2. BiLSTM Action Recognition Model
Trained on temporal compressed-domain motion features across 5 human activities:
- **Walking F1-Score**: **85.00%** (Precision: 77.27%, Recall: 94.44%)
- **Walking While Using Phone F1-Score**: **81.25%** (Precision: 92.86%, Recall: 72.22%)
- **Standing Still F1-Score**: **68.75%**
- **Macro Precision**: **69.98%**
- **Confusion Matrix**: Saved at [`output/action_confusion_matrix.png`](output/action_confusion_matrix.png)

---

## 📂 Project Architecture

```
capstone_project/
├── compressed_domain_tracker.py      # Core compressed-domain tracker & BAFE propagation engine
├── batch_process.py                  # Batch processor for 171-video dataset with checkpointing
├── render_dataset_videos.py          # Visualized video generator with microboxes & motion arrows
├── train_and_evaluate_action_model.py# BiLSTM activity classifier model
├── evaluate_detector_accuracy.py     # IoU, mAP, precision, and recall evaluator
├── print_accuracy_report.py          # Terminal CLI dashboard displaying all benchmark metrics
├── upload_release_assets.py          # Automation utility for GitHub Releases
├── output/                           # Evaluation reports, confusion matrices, and dataset summaries
│   ├── processing_summary.csv        # Master execution metrics across 171 videos
│   ├── detector_accuracy_report.json # Detailed detection benchmarks
│   ├── action_recognition_metrics.json# Activity classification metrics
│   └── action_confusion_matrix.png   # Action recognition confusion matrix
└── README.md                         # Documentation & Quickstart
```

---

## 🚀 Quickstart & Usage

### 1. View Accuracy Report in Terminal
To view the full formatted accuracy dashboard with detection and classification benchmarks:
```bash
python print_accuracy_report.py
```

### 2. Run Batch Tracking on a Dataset Folder
```bash
python batch_process.py --input "HAR_annotations/Walking" --output "output"
```

### 3. Generate Visualized Output Videos (.mp4)
To render video outputs with bounding boxes, macroblocks, and motion vectors:
```bash
python render_dataset_videos.py --output "output"
```

### 4. Evaluate Object Detector & Tracker Accuracy
```bash
python evaluate_detector_accuracy.py --annotations "HAR_annotations/Walking" --output "output"
```

### 5. Train & Evaluate Action Recognition Model
```bash
python train_and_evaluate_action_model.py --output "output"
```

---

## 👥 Authors & Collaborators
- **Nandani Sonale**
- **Keshav**
