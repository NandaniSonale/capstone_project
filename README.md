# Compressed-Domain Object Tracking & P/B-Frame BAFE Motion Vector Propagation

[![GitHub Release](https://img.shields.io/github/v/release/NandaniSonale/capstone_project?label=Visualized%20Videos&color=brightgreen)](https://github.com/NandaniSonale/capstone_project/releases/tag/v1.0.0-artifacts)

A high-performance pipeline for compressed-domain human detection, **Box-Aligned Feature Extraction (BAFE) propagation across P/B-frames**, and macroblock motion vector extraction directly within the H.264 video codec domain without full RGB pixel decoding or optical flow computation.

---

## 📥 Visualized Output Demonstration Videos

Full visualized `.mp4` video outputs featuring:
- **Bounding Boxes**: Color-coded detection anchors (`DET` on I-frames) and motion propagations (`PROP` on P/B-frames).
- **16×16 Macroblock Microboxes**: Motion grid within Regions of Interest (ROI).
- **Real-Time Motion Vector Arrows**: Visualizing directional magnitude `(dx, dy)`.
- **Telemetry HUD Overlays**: Frame type (I / P / B), PTS timestamp, and active macroblock count.

Stream or download rendered demonstration videos directly from GitHub Releases:
👉 **[Download Rendered Output Videos (v1.0.0-artifacts)](https://github.com/NandaniSonale/capstone_project/releases/tag/v1.0.0-artifacts)**

---

## 🎯 Core Project Concept & Architecture

```
                  +-------------------------------------------------+
                  |       Compressed H.264 Video Stream (.mp4)      |
                  +-------------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        [ I-FRAME KEYFRAMES ]                         [ P / B FRAMES ]
        Extract 4x4 DCT Sub-Bands                     Extract Motion Vectors (dx, dy)
        (Low, Mid, High Frequencies)                  Extract DCT Energy (sum|coeffs|)
                    |                                             |
                    v                                             |
        SSD Object Detector (300x300)                             |
        Initial Human BBox [cx, cy, w, h]                         |
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                             +---------------------------+
                             |   BAFE Motion Propagation |
                             |   (Box-Aligned Median MV) |
                             +---------------------------+
                                           |
                                           v
                             +---------------------------+
                             |   Spatial ROI MB Filter   |
                             |   Keep MBs inside Box     |
                             +---------------------------+
                                           |
                                           v
                             +---------------------------+
                             | Temporal ROI Motion Data  |
                             | (roi_motion_data.json)    |
                             +---------------------------+
```

### Key Principles
1. **Zero Pixel Decoding**: Operations remain 100% inside H.264 codec transform domain & motion vectors. No RGB decoding or OpenCV optical flow calculation.
2. **Spatial Bounding Box Filter**: Human bounding box acts as a spatial ROI filter:
   $$\text{Action Features} = \text{Motion Vectors (dx, dy)} + \text{DCT Energy inside ROI across frames}$$
3. **BAFE Propagation**: Updates bounding box position on P and B frames using median motion vector displacement.

---

## 🧠 Algorithmic Foundations

### Algorithm 1: Sub-Band Frequency Feature Map Generation
*Source File:* [`feature_map.py`](file:///c:/Users/newuser/capstone_project/feature_map.py)
Converts raw I-frame DCT coefficients from the H.264 bitstream into 3-channel spatial frequency maps without decoding pixels:
1. Divide $16 \times 16$ macroblock into 16 sub-blocks of $4 \times 4$ DCT coefficients.
2. Group coefficients into 3 frequency bands:
   - **Low Frequency:** $E_{\text{low}} = \sum |c_0, c_1, c_4|^2$
   - **Mid Frequency:** $E_{\text{mid}} = \sum |c_2, c_3, c_5, c_6, c_8|^2$
   - **High Frequency:** $E_{\text{high}} = \sum |c_7, c_9, c_{10}, c_{11}, c_{12}, c_{13}, c_{14}, c_{15}|^2$
3. Populate 3D spatial feature map tensor: $[ \log(1 + E_{\text{low}}), \log(1 + E_{\text{mid}}), \log(1 + E_{\text{high}}) ]$.

### Algorithm 2: Box-Aligned Feature Extraction (BAFE) & Propagation
*Source File:* [`bafe_propagation.py`](file:///c:/Users/newuser/capstone_project/bafe_propagation.py)
Propagates human bounding boxes across P and B frames using macroblock motion vectors:
1. Sample motion vectors $(dx, dy)$ for macroblocks inside and adjacent to the active bounding box.
2. Calculate median displacement $(\text{median\_dx}, \text{median\_dy})$.
3. Convert H.264 quarter-pixel MVs to normalized grid space:
   $$\Delta \text{grid\_x} = -\frac{\text{median\_dx}}{4.0} \times \frac{\text{grid\_size}}{\text{frame\_width}}$$
   $$\Delta \text{grid\_y} = -\frac{\text{median\_dy}}{4.0} \times \frac{\text{grid\_size}}{\text{frame\_height}}$$
4. Update bounding box center: $\text{new\_cx} = \text{cx} + \Delta \text{grid\_x}$, $\text{new\_cy} = \text{cy} + \Delta \text{grid\_y}$.

### Algorithm 3: P/B Frame Compressed-Domain Motion Data Format
*Source File:* [`compressed_domain_tracker.py`](file:///c:/Users/newuser/capstone_project/compressed_domain_tracker.py)
Extracted P/B frame macroblock motion data serialized per frame (`roi_motion_data.json`):
```json
{
  "frame_512": [
    {"mb_x": 12, "mb_y": 8, "dx": 2, "dy": 1, "dct_energy": 14.85},
    {"mb_x": 13, "mb_y": 8, "dx": 3, "dy": 1, "dct_energy": 19.20}
  ],
  "frame_1024": [
    {"mb_x": 12, "mb_y": 8, "dx": 2, "dy": 2, "dct_energy": 16.10}
  ]
}
```

---

## 📁 Repository Structure

```
capstone_project/
├── bafe_propagation.py               # BAFE propagation & macroblock grid displacement engine
├── compressed_domain_tracker.py      # Core compressed-domain tracker & H.264 extraction orchestrator
├── feature_map.py                    # Sub-band DCT frequency map generator for I-frames
├── batch_process.py                  # Dataset batch runner across video folders with checkpointing
├── batch_process_folder.py           # Single-folder dataset tracking pipeline
├── render_dataset_videos.py          # Annotated video renderer (bounding boxes, microboxes, MV arrows)
├── debug_extraction.py               # Data extraction verification & troubleshooting script
├── upload_release_assets.py          # Automation utility for GitHub Release video uploads
├── FFmpeg/                           # Custom FFmpeg source tree with h264_coeff_extract.c hook
├── output/                           # Results directory
│   └── processing_summary.csv        # Master dataset execution logs across all videos
└── README.md                         # Project Documentation & Quickstart
```

---

## 🚀 Quickstart & Command Guide

### 1. Single Video Tracking & P/B Propagation
Run compressed-domain tracker on a single video file:
```bash
python compressed_domain_tracker.py --video "Human Activity Recognition - Video Dataset/Walking/Walking (23).mp4" --model "best_model .h5"
```

### 2. Batch Tracking on Dataset Folder
Run BAFE propagation across all dataset video folders:
```bash
python batch_process.py --input "Human Activity Recognition - Video Dataset/Walking" --output "output"
```

### 3. Render Visualized Annotated Videos (.mp4)
Generate output videos with bounding boxes, macroblocks, and motion vector arrows:
```bash
python render_dataset_videos.py --output "output"
```

### 4. Verify P-Frame Extraction & Motion Data
```bash
python debug_extraction.py
```

---

## 👥 Authors & Collaborators
- **Nandani Sonale**
