# 🎯 TASK BRIEF: Compressed-Domain P-Frame ROI Motion Extraction

## Executive Summary

Extract **compressed-domain motion information** from P-frames using human bounding boxes detected on I-frames. This is the core action-recognition feature extraction pipeline.

**Core Principle:** Spatial ROI filter + Temporal motion vectors + DCT energy = Action signal

---

## GOAL

Use the human bounding box **already detected on the I-frame** and extract **ONLY compressed-domain motion information** from every **P-frame** inside that bbox.

### ✅ DO:
- Extract motion vectors (dx, dy) from macroblocks inside ROI
- Extract DCT energy from the same macroblocks
- Filter macroblocks by: **if macroblock center inside bbox → keep it**
- Process every P-frame in sequence
- Stay completely inside **FFmpeg compressed-domain pipeline**

### ❌ DO NOT:
- Decode RGB frames
- Compute optical flow
- Use CNN feature extraction
- Use OpenCV trackers
- Average whole-frame motion
- Process background macroblocks outside bbox
- Leave compressed domain

---

## What To Extract From Each P-Frame

### 1. Motion Vectors (PRIMARY ACTION SIGNAL)

For every macroblock **inside the ROI bbox**:

**Extract per-macroblock:**
- `dx` - horizontal motion displacement
- `dy` - vertical motion displacement

**These represent:**
- Motion direction
- Motion magnitude  
- Temporal movement patterns
- Primary action-recognition feature

### 2. Residual / DCT Information

For the **same macroblocks inside ROI**:

**Extract:**
- DCT Energy (preferred implementation):
  ```
  DCT Energy = Σ |coefficients|²
  ```

**This represents:**
- Motion intensity
- Appearance change magnitude
- Macroblock deformation

---

## Required Per-Macroblock Data Structure

For **every ROI macroblock**, extract:

```json
{
  "mb_x": 128,
  "mb_y": 96,
  "dx": 2,
  "dy": 1,
  "dct_energy": 18.2
}
```

This is sufficient for version 1.0.

---

## Filtering Logic

### ROI Macroblock Selection Condition

```python
if macroblock_center_inside_bbox:
    keep_macroblock()
else:
    ignore_macroblock()
```

**Key point:** Only process macroblocks whose center point falls within the bounding box region.

---

## Expected Output Format

For every human action sequence, the temporal ROI motion data should look like:

```json
{
  "frame_1": [
    {"mb_x": 32, "mb_y": 24, "dx": 2, "dy": 1, "dct_energy": 12.1},
    {"mb_x": 33, "mb_y": 24, "dx": 3, "dy": 1, "dct_energy": 18.0},
    {"mb_x": 32, "mb_y": 25, "dx": 2, "dy": 2, "dct_energy": 15.4}
  ],
  
  "frame_2": [
    {"mb_x": 32, "mb_y": 24, "dx": 2, "dy": 2, "dct_energy": 15.4},
    {"mb_x": 33, "mb_y": 24, "dx": 4, "dy": 1, "dct_energy": 21.0},
    {"mb_x": 32, "mb_y": 25, "dx": 1, "dy": 3, "dct_energy": 18.7}
  ]
}
```

This **temporal compressed-domain ROI motion sequence** becomes the **direct input for the action recognition model**.

---

## Core Project Idea

```
Bounding Box = Spatial Filter
├── Filters out background
├── Focuses on human region
└── Enables efficient feature extraction

Action Signal = Motion Vectors + DCT Residuals
├── Motion direction (dx, dy)
├── Motion intensity (dct_energy)
└── Temporal changes across P-frames
```

**The entire extraction pipeline exists to produce this single feature:**
```
Temporal Compressed-Domain ROI Motion = f(Motion Vectors, DCT Energy, inside_bbox_frames)
```

---

## Implementation Checklist

- [ ] I-frame human detection ✓ (Already implemented)
- [ ] Bounding box propagation to P-frames ✓ (Already implemented)
- [ ] P-frame motion vector extraction ✓ (Already implemented)
- [ ] P-frame DCT energy computation ✓ (Already implemented)  
- [ ] Macroblock ROI filtering (need to verify)
- [ ] Temporal sequence collection (need to verify)
- [ ] Output JSON generation (need to verify)
- [ ] Validation with test videos
- [ ] Performance optimization if needed

---

## Data Flow Pipeline

```
Video File
    ↓
FFmpeg Custom Decoder (H.264)
    ├─→ I-Frame: Extract DCT coefficients → CNN Detector → Bounding Boxes
    │
    └─→ P-Frame: Extract Motion Vectors + DCT Energy
                    ↓
                Filter by ROI (bbox)
                    ↓
                Collect temporal sequence
                    ↓
            Output: roi_motion_data.json
                    ↓
            Action Recognition Model Input
```

---

## Success Criteria

1. ✅ For every P-frame, identify macroblocks inside human bbox
2. ✅ Extract motion vectors (dx, dy) for those macroblocks
3. ✅ Compute DCT energy for those macroblocks  
4. ✅ Store in temporal structure with frame_N keys
5. ✅ Generate valid JSON output
6. ✅ Validate output with actual video test cases
7. ✅ No RGB decoding at any point
8. ✅ Stay 100% in compressed domain

---

## Files Involved

- `compressed_domain_tracker.py` - Main orchestrator
  - I-frame detection
  - P-frame motion filtering
  - Output generation
  
- `FFmpeg/FFmpeg/libavcodec/` - Custom C code
  - P-frame motion vector extraction
  - DCT coefficient extraction
  
- `tracking_outputs/roi_motion_data.json` - Output file

---

## Notes

- Macroblock size: 16×16 pixels
- Motion vectors: Typically in quarter-pixel units
- DCT domain: Quantized integer coefficients
- Frame references: I (0), P (1-3), B (2)
