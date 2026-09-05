# 🎯 TASK BRIEF: Compressed-Domain P-Frame ROI Motion Extraction

## 1. Goal

Use the human bounding box already detected on the I-frame and extract **ONLY compressed-domain motion information** from every P-frame inside that bbox.

* **NO RGB decoding.**
* **NO optical flow.**
* **NO OpenCV tracking.**
* **Stay completely inside FFmpeg compressed-domain/DCT pipeline.**

---

## 2. What To Extract From Each P-Frame

For every propagated bbox region, extract **ONLY**:

### A. Motion Vectors (MOST IMPORTANT)
From macroblocks inside the bbox.

For every macroblock inside ROI:
* `dx` — horizontal displacement
* `dy` — vertical displacement

**These represent:**
* Motion direction
* Motion magnitude
* Temporal movement

*This is the primary action-recognition signal.*

### B. Residual / DCT Information
Inside the **SAME** bbox macroblocks:
* DCT coefficients **OR** DCT energy

**Preferred initial implementation:**
$$\text{DCT Energy} = \sum |\text{coefficients}|$$
for each macroblock.

**This represents:**
* Motion intensity
* Appearance change
* Deformation

---

## 3. Important Filtering Logic

Process **ONLY** macroblocks inside the bounding box.

**Condition:**
```python
if macroblock_center_inside_bbox:
    keep_it()
else:
    ignore_it()
```

**Ignore all background macroblocks.**

---

## 4. Required Per-Macroblock Data Structure

For every ROI macroblock:

```json
{
  "mb_x": 128,
  "mb_y": 96,
  "dx": 2,
  "dy": 1,
  "dct_energy": 18.2
}
```

---

## 5. What NOT To Do

**DO NOT:**
* Decode RGB frames
* Compute optical flow
* Use CNN feature extraction
* Use OpenCV trackers
* Average whole-frame motion
* Process outside bbox
* Leave compressed domain

**Stay fully in:**
* Motion vectors
* Residual DCT domain **ONLY**

---

## 6. Final Expected Output

For every human sequence, produce temporal ROI motion data across P-frames (`tracking_outputs/roi_motion_data.json`):

```json
{
  "frame_1": [
    {"mb_x": 32, "mb_y": 24, "dx": 2, "dy": 1, "dct_energy": 12.1},
    {"mb_x": 33, "mb_y": 24, "dx": 3, "dy": 1, "dct_energy": 18.0}
  ],
  "frame_2": [
    {"mb_x": 32, "mb_y": 24, "dx": 2, "dy": 2, "dct_energy": 15.4},
    {"mb_x": 33, "mb_y": 24, "dx": 4, "dy": 1, "dct_energy": 21.0}
  ]
}
```

This temporal compressed-domain ROI motion sequence becomes the **direct input for the action recognition model**.

---

## 7. Core Project Idea

```
Bounding Box = Spatial Filter
```

$$\text{Action Information} = \text{Motion Vectors} + \text{DCT Residual Changes inside Human ROI across time}$$
