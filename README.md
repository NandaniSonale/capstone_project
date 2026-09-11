# Unified Compressed-Domain Framework for Object Detection and Human Action Recognition

## 📌 Overview

This project presents a unified framework for **object detection, bounding-box tracking, and human action recognition using H.264 compressed-domain information**.

Instead of fully decoding every video frame into RGB and performing conventional image processing, the framework directly utilizes information available from the H.264 compressed representation, including:

- DCT/frequency-domain coefficients
- Macroblock-level information
- Motion vectors
- DCT energy
- I/P/B frame structure
- ROI-based compressed-domain motion information

A trained **SSD-based object detection model** is used on I-frames to obtain the initial object bounding box. For subsequent frames, the system uses **BAFE (Box-Aligned Feature Extraction)** to estimate object motion from relevant macroblocks and propagate the **bounding box** across frames.

The resulting tracking information is visualized and can be used as temporal information for the Human Action Recognition (HAR) stage.

---

## 🎯 Project Objectives

The major objectives of the project are:

1. Process H.264 video directly using compressed-domain information.
2. Extract DCT/frequency-domain features from I-frames.
3. Extract motion vectors and DCT energy from compressed video information.
4. Detect objects using a trained SSD-based detector.
5. Propagate the detected **bounding box** across P/B frames.
6. Use BAFE to obtain motion information from ROI-aligned macroblocks.
7. Generate frame-level tracking and motion information.
8. Visualize bounding boxes, motion vectors, and ROI macroblocks.
9. Evaluate detection and bounding-box tracking performance.
10. Use the extracted temporal information as the foundation for Human Action Recognition.

---

# 🏗️ System Architecture

```text
                    H.264 Compressed Video
                              │
                              ▼
                    ┌───────────────────┐
                    │  Batch Processing │
                    └─────────┬─────────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │ H.264 Compressed-Domain│
                  │       Analysis          │
                  └────────────┬────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
              I-Frames      P-Frames      B-Frames
                 │             │             │
                 ▼             ▼             ▼
          DCT/Frequency     Motion Vectors  Propagated
             Features       + DCT Energy     Bounding Box
                 │             │             │
                 ▼             │             │
          SSD Detection        │             │
                 │             │             │
                 ▼             │             │
          Initial Bounding     │             │
               Box             │             │
                 │             │             │
                 └─────────────┼─────────────┘
                               ▼
                     BAFE-Based Processing
                               │
                               ▼
                     ROI + Macroblocks
                               │
                               ▼
                       Motion dx / dy
                               │
                               ▼
                     Median Motion Estimate
                               │
                               ▼
                  Bounding-Box Propagation
                               │
                               ▼
                     Tracking Results
                               │
                               ▼
                  Visualization / Evaluation
                               │
                               ▼
                Temporal Features for HAR
🔹 1. H.264 Compressed-Domain Processing

The framework works with the compressed representation of H.264 video.

An H.264 video consists of different types of frames:

I-frames – independently coded frames
P-frames – predicted using reference frames
B-frames – bidirectionally predicted frames

The project uses the information available in these compressed frames instead of relying entirely on RGB-frame processing.

The main compressed-domain information used by the system includes:

H.264 Bitstream
     │
     ├── DCT / Frequency Information
     │
     ├── Macroblock Information
     │
     ├── Motion Vectors
     │
     └── DCT Energy

This allows the system to obtain useful spatial and temporal information directly from the compressed representation.

🔹 2. I-Frame Processing

I-frames provide the starting point for object detection.

The processing pipeline is:

I-Frame
   │
   ▼
DCT Coefficient Extraction
   │
   ▼
Frequency-Domain Feature Generation
   │
   ▼
SSD Object Detection Model
   │
   ▼
Initial Bounding Box

The initial bounding box obtained from the detector becomes the reference bounding box for subsequent tracking.

DCT Frequency Groups

The extracted DCT coefficients are grouped into frequency bands:

Low Frequency:
[0, 1, 4]

Mid Frequency:
[2, 3, 5, 6, 8]

High Frequency:
[7, 9, 10, 11, 12, 13, 14, 15]

For the frequency-map generation, energy is calculated using:

Energy = log1p(sum(coefficients²))

These frequency-domain features provide information from the compressed representation of the I-frame.

🔹 3. SSD Object Detection

A trained SSD-based object detection model is used to detect the target object on I-frames.

The detector provides:

Bounding-box coordinates
Detection confidence
Object location

The detection stage initializes the tracking process.

I-Frame
   │
   ▼
DCT / Frequency Features
   │
   ▼
Trained SSD Detector
   │
   ▼
Detected Object
   │
   ▼
Initial Bounding Box

The detection confidence comes from the SSD model prediction and is not calculated from motion vectors or BAFE.

🔹 4. P-Frame Processing

P-frames contain compressed-domain motion information.

The project extracts information such as:

PTS
Macroblock X
Macroblock Y
dx
dy
DCT Energy

Each motion record represents compressed-domain information associated with a macroblock.

Example:

PTS = 512
mb_x = 25
mb_y = 47
dx = -6
dy = -24
dct_energy = 44.0

Where:

PTS = Presentation Time Stamp
mb_x, mb_y = macroblock grid coordinates
dx, dy = H.264 motion-vector components
dct_energy = DCT energy associated with the macroblock
🔹 5. Macroblocks

H.264 processing uses 16 × 16 pixel macroblocks.

For a video resolution of 960 × 540:

Macroblocks horizontally:

ceil(960 / 16) = 60

Macroblocks vertically:

ceil(540 / 16) = 34

Therefore, approximately:

60 × 34 = 2040 macroblocks

are present in the frame.

The tracking system does not store all macroblocks in the ROI JSON. It selects the macroblocks relevant to the propagated bounding box.

🔹 6. Motion Vectors

Motion vectors describe the displacement associated with H.264 macroblocks.

The extracted values include:

dx
dy

The project uses H.264 quarter-pixel motion-vector representation.

The conversion is approximately:

pixel displacement = MV / 4

Because H.264 motion vectors point toward the reference block, the forward object displacement is handled as:

forward_dx = -dx / 4
forward_dy = -dy / 4

For example:

dx = -6
dy = -24

gives approximately:

forward_dx = +1.5 pixels
forward_dy = +6 pixels

These motion values are used by BAFE to estimate how the bounding box should move.

🔹 7. DCT Energy

DCT energy provides additional compressed-domain frequency information.

For P-frame macroblocks, the extracted motion record contains:

mb_x
mb_y
dx
dy
dct_energy

DCT energy is stored in the ROI motion data as:

"dct_energy"

The current BAFE bounding-box displacement is primarily determined using the aggregated motion-vector information (dx, dy).

The DCT energy is retained as an additional compressed-domain feature.

🔹 8. Bounding-Box Propagation

The framework does not propagate the object itself.

It propagates the bounding box associated with the detected object.

The process is:

SSD Detection
      │
      ▼
Initial Bounding Box
      │
      ▼
BAFE Motion Analysis
      │
      ▼
Estimate Bounding-Box Displacement
      │
      ▼
Move Bounding-Box Center
      │
      ▼
Propagated Bounding Box

The bounding-box width, height, and confidence are preserved while the center position is updated according to the estimated motion.

🔹 9. BAFE – Box-Aligned Feature Extraction

BAFE (Box-Aligned Feature Extraction) is the main method used for bounding-box propagation.

BAFE aligns compressed-domain macroblock features with the current bounding box.

The process is:

Previous Bounding Box
          │
          ▼
Expand ROI by 20%
          │
          ▼
Create Box-Aligned ROI
          │
          ▼
Divide ROI into 3 × 3 Grid
          │
          ▼
Find Relevant 16 × 16 Macroblocks
          │
          ▼
Collect Motion Vectors
          │
          ▼
Aggregate Motion Using Median
          │
          ▼
Calculate Bounding-Box Displacement
          │
          ▼
Update Bounding-Box Center
BAFE Parameters

The current implementation uses:

Grid size              = 3 × 3
Neighborhood scale     = 20%
Macroblock size        = 16 × 16
Motion aggregation     = Median
🔹 10. ROI Generation

The previous bounding box is expanded by approximately 20% to create a neighborhood around the object.

This provides additional macroblocks around the bounding box that may contain useful motion information.

             Expanded ROI
      ┌───────────────────────┐
      │                       │
      │    ┌─────────────┐    │
      │    │ Bounding Box│    │
      │    │             │    │
      │    └─────────────┘    │
      │                       │
      └───────────────────────┘

The expanded ROI is divided into a:

3 × 3 grid

Macroblocks are assigned to the corresponding grid cells.

🔹 11. Motion Aggregation

Motion vectors from relevant macroblocks may contain noise or outliers.

Therefore, BAFE uses the median motion rather than simply using one macroblock.

For example:

Macroblock motions:

dx: -5, -6, -6, -7, -40
dy: -23, -24, -24, -25, 50

The extreme values can be outliers.

The median provides a more robust estimate:

Median dx ≈ -6
Median dy ≈ -24

The resulting motion is then converted into bounding-box displacement.

🔹 12. Bounding-Box Update

The estimated displacement is applied to the center of the bounding box.

Conceptually:

New Center X = Old Center X + displacement X

New Center Y = Old Center Y + displacement Y

The current implementation keeps:

Width       → unchanged
Height      → unchanged
Confidence  → unchanged

Only the center position is updated based on the estimated motion.

🔹 13. B-Frame Processing

B-frames are also handled as part of the bounding-box tracking pipeline.

When an active bounding box is available, the propagated state is used for tracking.

The B-frame processing uses the available compressed-domain motion information and propagated state to continue bounding-box tracking.

The primary direct compressed-domain motion extraction in the current implementation is performed for P-frame data.

🔹 14. ROI Macroblock Filtering

After bounding-box propagation, the system selects the macroblocks whose centers fall inside the propagated bounding box.

The selected information includes:

mb_x
mb_y
dx
dy
dct_energy

This produces ROI-specific compressed-domain motion information.

The information is stored in:

roi_motion_data.json

Example structure:

{
    "frame_512": [
        {
            "mb_x": 25,
            "mb_y": 47,
            "dx": -6,
            "dy": -24,
            "dct_energy": 44.0
        }
    ]
}

The frame identifier in this structure is based on the PTS, so frame_512 should be interpreted as a frame associated with PTS 512, not necessarily frame number 512.

🔹 15. Tracking Source

The tracking results identify whether the bounding box came from detection or propagation.

Two important sources are:

DET
PROP
DET

DET indicates that the bounding box was obtained from the object detector.

This is primarily associated with I-frame detection.

PROP

PROP indicates that the bounding box is being tracked using propagation.

The general process is:

I-Frame
   │
   └── DET → Initial Bounding Box
              │
              ▼
         P-Frame
              │
              └── PROP → Updated Bounding Box
                              │
                              ▼
                         Next Frame
🔹 16. Detection vs Bounding-Box Tracking

The project separates detection from bounding-box tracking.

Detection
SSD Model
   ↓
Find Object
   ↓
Generate Bounding Box
Bounding-Box Tracking
Existing Bounding Box
   ↓
BAFE
   ↓
Motion Vectors
   ↓
Bounding-Box Displacement
   ↓
Updated Bounding Box

Therefore, the detector initializes the bounding box, while the compressed-domain motion information is used to track and propagate that bounding box across subsequent frames.

🔹 17. Visualization

The project provides visualization of the tracking results.

The visualization can display:

Bounding boxes
DET / PROP source
Macroblock regions
Motion-vector arrows
ROI information
Frame telemetry

The visualization pipeline reads the generated tracking CSV files and ROI motion information.

Tracking CSV
      +
ROI Motion JSON
      │
      ▼
Visualization Renderer
      │
      ▼
Annotated Video

The visualization stage is primarily responsible for displaying the results. It does not perform the main BAFE algorithm.

🔹 18. Output Files

The project generates several tracking and analysis outputs.

Tracking Outputs
tracking/
motion/
propagation/
Main Result Files
tracking_results.csv
p_frames_tracking.csv
b_frames_tracking.csv
roi_motion_data.json
motion_summary.json
propagation_summary.json
video_summary.json
Frequency-Domain Output

Frequency maps are generated as:

.npy

files.

These contain the extracted frequency-domain information from the processed I-frames.

🔹 19. Project Processing Flow

The complete processing pipeline can be summarized as:

H.264 Video
     │
     ▼
Compressed-Domain Extraction
     │
     ├───────────────┐
     │               │
     ▼               ▼
I-Frame          P/B Frames
     │               │
     ▼               ▼
DCT Features     Motion Information
     │               │
     ▼               │
SSD Detection       │
     │               │
     ▼               │
Initial BBox         │
     │               │
     └───────┬───────┘
             ▼
           BAFE
             │
             ▼
       ROI Expansion
             │
             ▼
         3 × 3 Grid
             │
             ▼
     16 × 16 Macroblocks
             │
             ▼
       Motion dx / dy
             │
             ▼
      Median Aggregation
             │
             ▼
   Bounding-Box Displacement
             │
             ▼
    Propagated Bounding Box
             │
             ▼
      Tracking Results
             │
             ▼
       Visualization
             │
             ▼
       Evaluation / HAR
📊 20. Dataset

The current project processing includes the Walking dataset.

Current processed dataset statistics:

Metric	Value
Videos	171
Frames	51,365
Dataset	Walking

The dataset is used for object detection and bounding-box tracking evaluation.

📈 21. Detection and Tracking Results

The current evaluation reports the following results:

Metric	Result
mAP @ 0.50	98.48%
mAP @ 0.75	71.92%
mAP @ 0.50:0.95	68.83%
Mean IoU	81.59%
I-Frame Anchor IoU	98.55%
BAFE Propagation IoU	80.98%
Precision	98.48%
Recall	98.73%
F1 Score	98.60%

These metrics evaluate the detection and bounding-box tracking performance of the implemented pipeline.

🔬 22. Role of Each Component
Component	Purpose
H.264 Bitstream	Provides compressed-domain information
I-Frame	Provides the initial detection frame
DCT Coefficients	Provide frequency-domain information
SSD Detector	Generates the initial object bounding box
Macroblocks	Provide spatial units for compressed-domain analysis
Motion Vectors	Provide object/region motion information
DCT Energy	Provides additional frequency-domain information
BAFE	Extracts ROI-aligned motion features
Median Motion	Provides robust motion estimation
Bounding-Box Propagation	Updates the bounding-box position
ROI Filtering	Selects macroblocks inside the propagated bounding box
Tracking CSV	Stores frame-level tracking results
ROI JSON	Stores ROI macroblock motion information
Visualization	Displays bounding boxes and motion information
HAR	Uses temporal information for future action-recognition processing
