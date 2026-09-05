# 📢 TEAM TASK SPECIFICATION: COMPRESSED-DOMAIN P-FRAME ROI EXTRACTION

---

## 🎯 GOAL

Use the human bounding box already detected on the I-frame and extract **ONLY compressed-domain motion information** from every P-frame inside that bbox.

* **NO RGB decoding.**
* **NO optical flow.**
* **NO OpenCV tracking.**
* **Stay completely inside FFmpeg compressed-domain/DCT pipeline.**

---

## 📊 WHAT THEY NEED TO EXTRACT FROM EACH P-FRAME

For every propagated bbox region, extract **ONLY**:

### 1. Motion Vectors (MOST IMPORTANT)
From macroblocks inside the bbox.

For every macroblock inside ROI, extract:
* `dx` — horizontal displacement
* `dy` — vertical displacement

**These represent:**
* motion direction
* motion magnitude
* temporal movement

*This is the primary action-recognition signal.*

---

### 2. Residual / DCT Information
Inside the **SAME** bbox macroblocks, extract:
* DCT coefficients **OR** DCT energy

**Preferred initial implementation:**
$$\text{DCT Energy} = \sum |\text{coefficients}|$$
for each macroblock.

**This represents:**
* motion intensity
* appearance change
* deformation

---

## ⚠️ IMPORTANT: ROI FILTER CONDITION

They must **ONLY** process macroblocks inside the bounding box.

```python
if macroblock_center_inside_bbox:
    keep_it()
else:
    ignore_it()
```

**Ignore all background macroblocks.**

---

## 📋 REQUIRED PER-MACROBLOCK DATA

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

*That is enough for version 1.*

---

## ❌ WHAT NOT TO DO

**DO NOT:**
* ❌ decode RGB frames
* ❌ compute optical flow
* ❌ use CNN feature extraction
* ❌ use OpenCV trackers
* ❌ average whole-frame motion
* ❌ process outside bbox
* ❌ leave compressed domain

**Stay fully in:**
* Motion vectors
* Residual DCT domain **ONLY**

---

## 🏁 FINAL OUTPUT EXPECTED

For every human sequence: **Temporal ROI motion data across P-frames**.

### Example Output (`tracking_outputs/roi_motion_data.json`):

```json
{
  "frame_512": [
    {
      "mb_x": 32,
      "mb_y": 24,
      "dx": 2,
      "dy": 1,
      "dct_energy": 12.1
    },
    {
      "mb_x": 33,
      "mb_y": 24,
      "dx": 3,
      "dy": 1,
      "dct_energy": 18.0
    }
  ],
  "frame_1024": [
    {
      "mb_x": 32,
      "mb_y": 24,
      "dx": 2,
      "dy": 2,
      "dct_energy": 15.4
    },
    {
      "mb_x": 33,
      "mb_y": 24,
      "dx": 4,
      "dy": 1,
      "dct_energy": 21.0
    }
  ]
}
```

This temporal compressed-domain ROI motion sequence becomes the **direct input for the action recognition model**.

---

## 💡 CORE PROJECT IDEA

```
Bounding Box = Spatial Filter
```

Actual action information comes from:
$$\text{Action Feature} = \text{Motion Vectors} + \text{DCT Residual Changes inside Human ROI across time}$$

That is the entire purpose of this extraction stage.

---

## 🚀 HOW TO EXECUTE PIPELINE

```bash
# PowerShell / Bash:
python compressed_domain_tracker.py --video "Human Activity Recognition - Video Dataset/Clapping/Clapping (1).mp4" --model "best_model .h5"
```
