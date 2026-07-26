╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║       📋 TEAM TASK COMMUNICATION: COMPRESSED-DOMAIN P-FRAME ROI EXTRACTION     ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝


👋 TEAM MESSAGE
════════════════════════════════════════════════════════════════════════════════

The compressed-domain P-frame ROI motion extraction pipeline is FULLY IMPLEMENTED
and ready for testing and validation.

This document communicates:
✓ What the task is
✓ What's been completed  
✓ How to run it
✓ What output to expect


🎯 TASK SUMMARY
════════════════════════════════════════════════════════════════════════════════

GOAL:
  Extract compressed-domain motion information from every P-frame inside the
  human bounding box detected on the I-frame.

  NO RGB decoding. NO optical flow. NO OpenCV tracking.
  ONLY: Motion vectors + DCT energy from H.264 bitstream.

KEY INSIGHT:
  Bounding Box = Spatial Filter
  Motion Vectors + DCT Energy = Temporal Action Signal
  
  → Temporal Compressed-Domain ROI Motion = Direct Input for Action Recognition


✅ WHAT'S ALREADY IMPLEMENTED
════════════════════════════════════════════════════════════════════════════════

1. ✓ P-FRAME MOTION VECTOR EXTRACTION
   File: FFmpeg/FFmpeg/libavcodec/h264_coeff_extract.c
   
   Extracts for every P-frame macroblock:
   • dx, dy (motion vectors from codec)
   • dct_energy (sum of absolute DCT coefficients)
   
   Output format: [pts | mb_x | mb_y | dx | dy | dct_energy]
   Binary file: tracking_outputs/extraction_cache/video_data_P.bin

2. ✓ I-FRAME DETECTION (SPATIAL ANCHOR)
   File: compressed_domain_tracker.py (lines 144-166)
   
   • Load pre-trained SSD model
   • Extract frequency maps from I-frame DCT coefficients
   • Detect human bounding boxes: [confidence, cx, cy, w, h]
   • Result: Human region coordinates for temporal filtering

3. ✓ ROI FILTERING (SPATIAL SELECTION)
   File: compressed_domain_tracker.py (lines 170-198)
   
   For every P-frame:
   • Convert MB grid coordinates to detection space
   • Check if MB center is inside bbox
   • Keep only macroblocks within human region
   • Discard all background data

4. ✓ TEMPORAL AGGREGATION (TIME-SERIES COLLECTION)
   File: compressed_domain_tracker.py (lines 197-198)
   
   Collect filtered MBs across consecutive P-frames:
   {
     "frame_512":  [...motion data...],
     "frame_1024": [...motion data...],
     "frame_1536": [...motion data...]
   }

5. ✓ JSON OUTPUT GENERATION
   File: compressed_domain_tracker.py (lines 234-238)
   Output: tracking_outputs/roi_motion_data.json
   
   Ready for action recognition model input


📊 WHAT TO EXTRACT (DETAILED SPECIFICATION)
════════════════════════════════════════════════════════════════════════════════

For EVERY macroblock INSIDE the human bounding box:

Required Per-Macroblock Fields:
┌─────────────────────────────────────────────────────────────┐
│ {                                                           │
│   "mb_x": 32,           ← Macroblock X coordinate          │
│   "mb_y": 24,           ← Macroblock Y coordinate          │
│   "dx": 2,              ← Horizontal motion (PRIMARY)       │
│   "dy": 1,              ← Vertical motion (PRIMARY)         │
│   "dct_energy": 12.1    ← Residual magnitude (SECONDARY)   │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘

Filter Condition:
┌─────────────────────────────────────────────────────────────┐
│ if macroblock_center_inside_bounding_box:                  │
│     extract_and_keep_macroblock()                          │
│ else:                                                       │
│     ignore_macroblock()                                    │
└─────────────────────────────────────────────────────────────┘

Temporal Organization:
┌─────────────────────────────────────────────────────────────┐
│ {                                                           │
│   "frame_512": [                                           │
│     {"mb_x":32, "mb_y":24, "dx":2,  "dy":1, ...},         │
│     {"mb_x":33, "mb_y":24, "dx":3,  "dy":1, ...},         │
│     {"mb_x":32, "mb_y":25, "dx":1,  "dy":2, ...}          │
│   ],                                                       │
│   "frame_1024": [                                          │
│     {"mb_x":32, "mb_y":24, "dx":2,  "dy":2, ...},         │
│     {"mb_x":33, "mb_y":24, "dx":4,  "dy":1, ...}          │
│   ]                                                        │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘

This temporal sequence becomes the DIRECT INPUT for action recognition.


❌ WHAT NOT TO DO
════════════════════════════════════════════════════════════════════════════════

✗ Decode RGB frames          ← Use bitstream directly
✗ Compute optical flow       ← Use codec motion vectors
✗ Use CNN feature extraction ← Use DCT coefficients
✗ Use OpenCV trackers        ← Use detector bounding box
✗ Average whole-frame motion ← Use ROI-filtered MBs
✗ Process background area    ← Filter by bbox
✗ Leave compressed domain    ← Stay 100% in codec domain

Everything must stay inside FFmpeg's H.264 compressed-domain pipeline.


🚀 HOW TO RUN THE EXTRACTION
════════════════════════════════════════════════════════════════════════════════

1. SIMPLE COMMAND:
   ─────────────────
   
   BASH/Linux:
   python3 compressed_domain_tracker.py \
       --video "path/to/video.mp4" \
       --model "best_model .h5"
   
   PowerShell (Windows):
   python3 compressed_domain_tracker.py `
       --video "path/to/video.mp4" `
       --model "best_model .h5"
   
   Single line (all platforms):
   python3 compressed_domain_tracker.py --video "path/to/video.mp4" --model "best_model .h5"

2. EXAMPLE WITH DATASET VIDEO:
   ──────────────────────────────
   
   PowerShell (Windows):
   python3 compressed_domain_tracker.py -video "Human Activity Recognition - Video Dataset/Clapping/Clapping (1).mp4" -model "best_model .h5"
   
   Or (single line):
   python3 compressed_domain_tracker.py --video "Human Activity Recognition - Video Dataset/Clapping/Clapping (1).mp4" --model "best_model .h5"

3. WHAT HAPPENS INTERNALLY:
   ─────────────────────────
   a) FFmpeg loads video in custom extraction mode
   b) Extracts I-frame DCT coefficients
   c) Builds frequency feature maps
   d) Runs SSD detector → Gets bounding boxes
   e) Extracts P-frame motion vectors + energy
   f) Filters macroblocks by ROI
   g) Collects temporal sequence
   h) Outputs roi_motion_data.json

4. EXPECTED CONSOLE OUTPUT:
   ────────────────────────
   [INIT] Video: ...
   [INIT] Model: best_model .h5
   [STEP 1] Starting Full Video Extraction Pass...
   [STEP 2] Processing frames and saving results to: tracking_outputs/tracking_results.csv
   [PTS 0] I-Frame: Detected 1 boxes.
   [PTS 512] P-Frame: Extracted 42 ROI macroblocks.
   [PTS 1024] P-Frame: Extracted 39 ROI macroblocks.
   ...
   Task complete. Tracking results saved in tracking_outputs/tracking_results.csv
   ROI Motion Data saved in tracking_outputs/roi_motion_data.json


📁 OUTPUT FILES
════════════════════════════════════════════════════════════════════════════════

MAIN OUTPUT (USE THIS):
┌───────────────────────────────────────────────────────────┐
│ tracking_outputs/roi_motion_data.json                    │
│                                                           │
│ JSON file with temporal ROI motion data                  │
│ Format: {"frame_PTS": [macroblock records]}              │
│ Ready for action recognition model input                 │
└───────────────────────────────────────────────────────────┘

DEBUG OUTPUTS:
├─ tracking_outputs/tracking_results.csv
│  └─ Frame-by-frame detection results
│
└─ tracking_outputs/extraction_cache/
   ├─ video_data_P.bin (P-frame motion vectors)
   ├─ video_data_I.bin (I-frame DCT coefficients)
   └─ pts_*.npy (frequency maps)


✅ VERIFICATION & VALIDATION
════════════════════════════════════════════════════════════════════════════════

RUN VERIFICATION SCRIPT:
┌─────────────────────────────────────────────────────────┐
│ python3 debug_extraction.py                             │
│                                                         │
│ This will check:                                        │
│ • P-frame binary has motion vector data ✓              │
│ • I-frame detections were successful ✓                │
│ • ROI filtering produced output ✓                      │
│ • Motion vectors have realistic values ✓               │
│ • DCT energy is positive ✓                             │
└─────────────────────────────────────────────────────────┘

MANUAL INSPECTION:
┌─────────────────────────────────────────────────────────┐
│ # View the output (formatted)                           │
│ cat tracking_outputs/roi_motion_data.json | \           │
│   python3 -m json.tool | head -50                      │
│                                                         │
│ # Check file exists                                     │
│ ls -lh tracking_outputs/roi_motion_data.json           │
│                                                         │
│ # View detection results                               │
│ head tracking_outputs/tracking_results.csv              │
└─────────────────────────────────────────────────────────┘

SUCCESS CRITERIA:
┌─────────────────────────────────────────────────────────┐
│ ✓ roi_motion_data.json is NOT empty ({})               │
│ ✓ Contains multiple frame entries                      │
│ ✓ Each frame has macroblock records                    │
│ ✓ dx, dy values are non-zero (motion detected)         │
│ ✓ dct_energy values are positive                       │
│ ✓ Temporal ordering preserved (frame_512 < frame_1024)│
│ ✓ No RGB decoding in process                          │
│ ✓ No optical flow computation                         │
└─────────────────────────────────────────────────────────┘


🔍 UNDERSTAND THE DATA
════════════════════════════════════════════════════════════════════════════════

What is dx, dy?
  These are MOTION VECTORS from the H.264 codec.
  
  dx, dy = How much the 16×16 macroblock moved from previous frame
  
  Example: dx=2, dy=1
  → Moved ~0.5 pixels right (2 quarter-pixels)
  → Moved ~0.25 pixels down (1 quarter-pixel)
  
  Why it matters for action recognition:
  • Motion patterns reveal actions
  • Clapping = hands moving up/down → high dy values
  • Walking = body moving horizontally → high dx values
  • Standing = minimal motion → dx≈0, dy≈0

What is dct_energy?
  Sum of absolute DCT coefficients for the macroblock.
  
  dct_energy = Σ |DCT_coefficient[i]|
  
  Example: dct_energy = 12.1
  → Macroblock has 12.1 units of transform activity
  → High energy = complex texture or sudden change
  → Low energy = simple, uniform region
  
  Why it matters for action recognition:
  • Appearance changes reveal actions
  • Deformation (limb movement) increases energy
  • Occlusion/disocclusion detected
  • Clothing texture variation captured

Why both dx, dy AND dct_energy?
  Motion vectors (dx, dy) capture MOVEMENT
  DCT energy captures APPEARANCE CHANGE
  
  Together = Complete action signal
  
  Example: Clapping motion
  • Hands move up → high dy
  • Hands come together → appearance change → energy spike
  • Pattern repeats → temporal sequence captures cycle


🎓 IMPLEMENTATION ARCHITECTURE
════════════════════════════════════════════════════════════════════════════════

Data Flow:

  H.264 Video File
       ↓
  FFmpeg Custom Decoder
       ├─→ I-FRAME:  Extract DCT → Build Frequency Maps → SSD Detector
       │             Output: Bounding boxes [cx, cy, w, h]
       │
       └─→ P-FRAME:  Extract Motion Vectors (dx, dy)
                     Extract DCT Energy
                     Output: {mb_x, mb_y, dx, dy, energy}
       ↓
  ROI Filtering:
       Filter macroblocks by bbox
       Keep only MBs where: (cx-w/2 ≤ mb_gx ≤ cx+w/2) AND (...)
       ↓
  Temporal Aggregation:
       Collect across consecutive P-frames
       Organize by frame timestamp
       ↓
  JSON Output:
       {"frame_512": [...], "frame_1024": [...], ...}
       ↓
  Action Recognition Model Input


📚 KEY DOCUMENTATION
════════════════════════════════════════════════════════════════════════════════

Read these files for more details:

1. TASK_BRIEF.md
   └─ Detailed task specification and requirements

2. IMPLEMENTATION_SUMMARY.md
   └─ Technical architecture and data structures

3. QUICKSTART.py (run it!)
   └─ Step-by-step execution guide with examples

4. debug_extraction.py (run it!)
   └─ Verification and troubleshooting script

5. Project_Algorithms.md
   └─ Original project algorithm descriptions

6. Source code:
   ├─ compressed_domain_tracker.py (main orchestrator)
   ├─ feature_map.py (frequency map builder)
   ├─ FFmpeg/FFmpeg/libavcodec/h264_coeff_extract.c (extraction logic)
   └─ FFmpeg/FFmpeg/libavcodec/h264_coeff_extract.h (interface)


🔧 TROUBLESHOOTING QUICK REFERENCE
════════════════════════════════════════════════════════════════════════════════

Problem: roi_motion_data.json is empty {}
Solution 1: Check I-frame detections
  grep "I-Frame: Detected" (look for count > 0)
Solution 2: Check P-frame binary
  ls -lh tracking_outputs/extraction_cache/video_data_P.bin (should exist)
Solution 3: Run debug script
  python3 debug_extraction.py (check all stages)

Problem: No I-frame detections
Cause: Model not detecting humans
Solutions:
  • Check model file: ls -la best_model\ .h5
  • Lower confidence threshold: --conf_threshold 0.1
  • Verify model trained on similar data

Problem: FFmpeg not found
Solution:
  cd FFmpeg/FFmpeg
  ./configure && make -j$(nproc)
  ls -la ffmpeg.exe (verify binary created)

Problem: TensorFlow import error
Solution:
  pip install tensorflow numpy opencv-python av


📋 NEXT STEPS FOR THE TEAM
════════════════════════════════════════════════════════════════════════════════

1. RUN THE EXTRACTION
   
   PowerShell:
   python3 compressed_domain_tracker.py --video "dataset/video.mp4" --model "best_model .h5"
   
   Bash:
   python3 compressed_domain_tracker.py \
       --video "dataset/video.mp4" \
       --model "best_model .h5"

2. VERIFY THE OUTPUT
   python3 debug_extraction.py

3. INSPECT THE DATA
   cat tracking_outputs/roi_motion_data.json | python3 -m json.tool

4. VALIDATE WITH TEST VIDEOS
   Test with multiple action types:
   • Clapping (vertical hand motion)
   • Walking (horizontal body motion)
   • Sitting (minimal motion)

5. FEED TO ACTION RECOGNITION MODEL
   Load roi_motion_data.json and use as input features

6. MEASURE PERFORMANCE
   Compare action classification accuracy with this approach


🎯 SUCCESS DEFINITION
════════════════════════════════════════════════════════════════════════════════

The extraction pipeline is successful when:

✓ roi_motion_data.json contains temporal motion sequences
✓ Motion vectors show expected patterns (clapping → vertical, etc.)
✓ DCT energy correlates with motion intensity
✓ No RGB decoding occurred (stayed in compressed domain)
✓ Action recognition model achieves target accuracy
✓ Processing is efficient (no pixel reconstruction overhead)


════════════════════════════════════════════════════════════════════════════════

                           You're ready to extract!
                    Start with: python3 compressed_domain_tracker.py

════════════════════════════════════════════════════════════════════════════════
