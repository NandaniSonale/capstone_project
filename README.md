Unified Compressed-Domain Framework for Object Detection and Human Action Recognition
Overview

This project implements a compressed-domain object detection and tracking pipeline for H.264 video. The system uses information available in the compressed representation instead of performing object detection independently on every decoded frame.

The implemented pipeline processes I-frames, P-frames, and B-frames and uses:

I-frame DCT/frequency information
I-frame object detection
P-frame motion vectors
Macroblock-level motion information
DCT energy
Box-Aligned Feature Extraction (BAFE)
Bounding-box propagation
ROI macroblock extraction
Object tracking
Tracking visualization
Detection and tracking evaluation
1. H.264 Compressed-Domain Processing

The project extracts information from the H.264 compressed representation.

The implemented pipeline works with:

I-frames
P-frames
B-frames
Macroblocks
Motion vectors
DCT/frequency information
DCT energy
PTS information

The extracted information is used for object detection and temporal tracking.

2. I-Frame Processing

I-frames are used for object detection and DCT/frequency feature extraction.

I-Frame
   │
   ├── DCT Coefficients
   │        ↓
   │   Frequency Features
   │
   └── YOLO
        ↓
   Object Detection
        ↓
   Initial BBox

YOLO provides the initial bounding box and confidence for the detected object.

This bounding box is then used as the reference for propagation in subsequent frames.

3. DCT / Frequency Feature Extraction

DCT coefficients are extracted from the I-frame compressed information.

The coefficients are grouped into:

Low Frequency
Mid Frequency
High Frequency

The energy for the frequency groups is calculated as:

Energy = log(1 + Σ coefficient²)

The resulting frequency information is stored in .npy files.

4. P-Frame Motion Information

P-frame compressed-domain information is extracted at the macroblock level.

The extracted information contains:

PTS
mb_x
mb_y
dx
dy
DCT Energy

Example:

{
    "mb_x": 25,
    "mb_y": 47,
    "dx": -6,
    "dy": -24,
    "dct_energy": 44.0
}

Here:

mb_x, mb_y represent macroblock coordinates.
dx, dy represent motion-vector components.
dct_energy represents the associated DCT energy.
PTS identifies the temporal position of the frame.
5. B-Frame Processing

B-frames are included in the temporal tracking pipeline.

The propagated object state is maintained through B-frames using the available propagation and temporal information.

Previous BBox
     │
     ▼
B-Frame
     │
     ▼
Temporal Propagation
     │
     ▼
Updated Object State

The primary direct compressed-domain motion extraction in the current implementation is from the P-frame motion records. B-frame processing is handled through the propagation/tracking pipeline.

6. BAFE
Box-Aligned Feature Extraction

The project implements BAFE (Box-Aligned Feature Extraction) for bounding-box propagation.

BAFE uses the previous bounding box to select relevant compressed-domain macroblocks and their motion information.

Previous BBox
     │
     ▼
20% Neighborhood
     │
     ▼
3 × 3 Grid
     │
     ▼
16 × 16 Macroblocks
     │
     ▼
Extract dx / dy
     │
     ▼
Median Motion
     │
     ▼
BBox Displacement
     │
     ▼
Updated BBox
20% Neighborhood

The previous bounding box is expanded by 20% to include nearby macroblocks around the object.

3 × 3 Grid

The expanded ROI is divided into a 3 × 3 grid to preserve the spatial distribution of the motion information.

16 × 16 Macroblocks

The relevant H.264 macroblocks are identified inside the ROI.

Median Motion

The motion vectors from the selected macroblocks are aggregated using the median to reduce the effect of noisy or outlier motion vectors.

Bounding-Box Update

The resulting motion is used to update the center of the bounding box.

The current implementation retains:

BBox Width       → unchanged
BBox Height      → unchanged
Confidence       → retained
BBox Center      → updated using motion
7. Object Propagation

The implemented propagation process is:

I-Frame
   │
   ▼
YOLO Detection
   │
   ▼
Initial Bounding Box
   │
   ▼
P/B Frame
   │
   ▼
BAFE
   │
   ▼
Motion-Based Displacement
   │
   ▼
Updated Bounding Box

This allows the object to be tracked across subsequent frames without applying YOLO independently to every frame.

8. ROI Macroblock Extraction

After propagation, macroblocks associated with the propagated object region are selected.

The ROI motion information is stored in:

roi_motion_data.json

The stored information includes:

mb_x
mb_y
dx
dy
dct_energy

This provides the macroblock-level compressed-domain information around the tracked object.

9. Object Tracking Outputs

The tracking pipeline generates:

tracking_results.csv
p_frames_tracking.csv
b_frames_tracking.csv

The tracking results contain information such as:

PTS
Frame type
Detection/propagation source
Confidence
Bounding-box coordinates

Additional generated information includes:

roi_motion_data.json
motion_summary.json
propagation_summary.json
video_summary.json
10. Visualization

The tracking results are visualized using the generated tracking and ROI information.

The visualization includes:

Bounding boxes
DET / PROP labels
16 × 16 macroblock microboxes
Motion-vector arrows
Frame information
Tracking telemetry

This provides a visual representation of the compressed-domain object tracking process.

11. Dataset Processing

The pipeline supports batch processing of multiple videos.

The main processing scripts are:

batch_process.py
batch_process_folder.py

The implemented pipeline has been processed on the Walking dataset.

Videos processed : 171
Frames processed : 51,365
12. Object Detection and Tracking Evaluation

The implemented object detection and tracking pipeline has been evaluated using the following metrics:

Metric	Result
mAP @ 0.50	98.48%
mAP @ 0.75	71.92%
mAP @ 0.50:0.95	68.83%
Mean IoU	81.59%
I-frame Anchor IoU	98.55%
BAFE Propagation IoU	80.98%
Precision	98.48%
Recall	98.73%
F1 Score	98.60%

These results represent the implemented object detection and compressed-domain tracking pipeline.

13. Action Recognition / HAR

The project title includes Human Action Recognition, but the HAR component is currently under development.

The completed tracking pipeline provides temporal object and motion information that will be used for the action-recognition stage.

H.264 Video
     │
     ▼
Compressed-Domain Processing
     │
     ▼
Object Detection
     │
     ▼
P/B Temporal Tracking
     │
     ▼
Motion / Temporal Information
     │
     ▼
Human Action Recognition
        🚧

HAR is therefore not included as a completed result in the current implementation.

Project Status
Component	Status
H.264 compressed-domain processing	✅ Completed
I-frame processing	✅ Completed
P-frame processing	✅ Completed
B-frame processing	✅ Completed
DCT/frequency extraction	✅ Completed
I-frame object detection	✅ Completed
P-frame motion-vector extraction	✅ Completed
Macroblock processing	✅ Completed
DCT energy extraction/storage	✅ Completed
BAFE	✅ Completed
ROI macroblock extraction	✅ Completed
Bounding-box propagation	✅ Completed
Object tracking	✅ Completed
Tracking visualization	✅ Completed
Batch processing	✅ Completed
Object detection evaluation	✅ Completed
Tracking/propagation evaluation	✅ Completed
Human Action Recognition	🚧 In Progress
HAR evaluation	🚧 In Progress
