#!/usr/bin/env python3
"""
QUICK START GUIDE: Compressed-Domain P-Frame ROI Motion Extraction

This guide walks through running the complete extraction pipeline and
validating the output for action recognition model input.
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                        TASK COMMUNICATION BRIEF                            ║
║                                                                            ║
║  Goal: Extract compressed-domain motion information from P-frames         ║
║        using human bounding boxes detected on I-frames                    ║
╚════════════════════════════════════════════════════════════════════════════╝

CORE CONCEPT
════════════

The pipeline works in 3 stages:

1. I-FRAME DETECTION
   • Extract DCT coefficients from I-frames (no decoding!)
   • Build spatial frequency maps (3-channel)
   • Run SSD detector → Detect human bounding boxes
   Output: Bounding box coordinates [cx, cy, w, h]

2. P-FRAME MOTION EXTRACTION  
   • Extract motion vectors (dx, dy) for each macroblock
   • Compute DCT energy for each macroblock
   • Format: pts | mb_x | mb_y | dx | dy | dct_energy
   Output: Binary file with all macroblocks

3. ROI FILTERING & AGGREGATION
   • For each P-frame: Filter macroblocks inside bbox
   • Collect temporal sequence across frames
   • Organize: {"frame_512": [...], "frame_1024": [...], ...}
   Output: JSON file ready for action recognition model


═══════════════════════════════════════════════════════════════════════════════
WHAT THE TEAM NEEDS TO EXTRACT
═══════════════════════════════════════════════════════════════════════════════

For EVERY macroblock INSIDE the human bounding box:

Required fields:
  ✓ mb_x, mb_y       → Macroblock position
  ✓ dx, dy           → Motion vectors (PRIMARY FEATURE)
  ✓ dct_energy       → Residual magnitude (SECONDARY FEATURE)

Filter condition:
  if (mb_center.x inside bbox AND mb_center.y inside bbox):
      extract_macroblock()

Temporal organization:
  {
    "frame_512": [
      {"mb_x": 32, "mb_y": 24, "dx": 2, "dy": 1, "dct_energy": 12.1},
      {"mb_x": 33, "mb_y": 24, "dx": 3, "dy": 1, "dct_energy": 18.0},
      ...
    ],
    "frame_1024": [...]
  }


═══════════════════════════════════════════════════════════════════════════════
WHAT NOT TO DO
═══════════════════════════════════════════════════════════════════════════════

❌ NO RGB decoding           - Stay in compressed domain
❌ NO optical flow           - Use motion vectors from codec
❌ NO OpenCV tracking        - Use bounding box from detector
❌ NO CNN features           - Use DCT coefficients directly
❌ NO whole-frame averaging  - Process only inside bbox
❌ NO background processing  - Filter spatially by bbox


═══════════════════════════════════════════════════════════════════════════════
STEP 1: VERIFY INSTALLATION
═══════════════════════════════════════════════════════════════════════════════

Required files:
  ✓ FFmpeg/FFmpeg/ffmpeg.exe         (custom compiled with extraction)
  ✓ best_model .h5                   (SSD detector model)
  ✓ compressed_domain_tracker.py     (main orchestrator)
  ✓ feature_map.py                   (frequency map builder)

Check installation:
""")

print("  python3 -c \"import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')\"")
print("  python3 -c \"import av; print(f'PyAV: {av.__version__}')\"")
print("  ls -la FFmpeg/FFmpeg/ffmpeg.exe")
print("  ls -la best_model\\ .h5")

print("""

═══════════════════════════════════════════════════════════════════════════════
STEP 2: PREPARE VIDEO
═══════════════════════════════════════════════════════════════════════════════

Video format: MP4 with H.264 codec (compressed domain)
Expected input: Human Activity Recognition video dataset

Example videos in dataset:
  • Human Activity Recognition - Video Dataset/Clapping/Clapping (1).mp4
  • Human Activity Recognition - Video Dataset/Walking/Walking (1).mp4
  • etc.

Video characteristics:
  • Encoded with H.264 (MPEG-4 AVC)
  • Contains I-frames (key frames) and P-frames (predicted frames)
  • No need to decode - extraction happens at bitstream level


═══════════════════════════════════════════════════════════════════════════════
STEP 3: RUN EXTRACTION PIPELINE
═══════════════════════════════════════════════════════════════════════════════

Command:
""")

print("""  python3 compressed_domain_tracker.py \\
      --video "path/to/video.mp4" \\
      --model "best_model .h5"
""")

print("""
Example:
""")

print("""  python3 compressed_domain_tracker.py \\
      --video "Human Activity Recognition - Video Dataset/Clapping/Clapping (1).mp4" \\
      --model "best_model .h5"
""")

print("""
What happens:
  1. FFmpeg loads video in custom mode
  2. Extracts binary P-frame data to: tracking_outputs/extraction_cache/video_data_P.bin
  3. Extracts binary I-frame data to: tracking_outputs/extraction_cache/video_data_I.bin
  4. Builds frequency maps: tracking_outputs/extraction_cache/pts_*.npy
  5. Runs SSD detector on each I-frame
  6. Filters P-frame data by detected bounding boxes
  7. Collects temporal sequence
  8. Outputs: tracking_outputs/roi_motion_data.json

Expected output:
  [INIT] Video: ...
  [INIT] Model: best_model .h5
  [STEP 1] Starting Full Video Extraction Pass...
  [STEP 2] Processing frames and saving results...
  [PTS 0] I-Frame: Detected X boxes.
  [PTS 512] P-Frame: Extracted Y ROI macroblocks.
  ...
  Task complete. Tracking results saved...
  ROI Motion Data saved in tracking_outputs/roi_motion_data.json


═══════════════════════════════════════════════════════════════════════════════
STEP 4: VERIFY OUTPUT
═══════════════════════════════════════════════════════════════════════════════

Debug script:
""")

print("  python3 debug_extraction.py")

print("""
This will show:
  • P-frame motion vector statistics
  • I-frame detection summary
  • ROI data collection status
  • Any issues in the pipeline

Manual inspection:
""")

print("""  # View ROI motion data (pretty-printed)
  cat tracking_outputs/roi_motion_data.json | python3 -m json.tool

  # Check file sizes
  ls -lh tracking_outputs/extraction_cache/

  # Check CSV results
  head tracking_outputs/tracking_results.csv
""")

print("""
Expected roi_motion_data.json structure:
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
      },
      ...
    ],
    "frame_1024": [...]
  }

Success criteria:
  ✓ roi_motion_data.json is NOT empty {}
  ✓ Contains multiple frame entries (frame_512, frame_1024, etc.)
  ✓ Each frame has multiple macroblock entries
  ✓ dx, dy values are non-zero (motion detected)
  ✓ dct_energy values are positive floats


═══════════════════════════════════════════════════════════════════════════════
STEP 5: FEED TO ACTION RECOGNITION MODEL
═══════════════════════════════════════════════════════════════════════════════

The output JSON is ready for your action recognition model:
""")

print("""  import json

  # Load ROI motion data
  with open('tracking_outputs/roi_motion_data.json', 'r') as f:
      roi_data = json.load(f)

  # Each frame's macroblocks become input features
  for frame_id, macroblocks in roi_data.items():
      # macroblocks = [
      #   {"mb_x": 32, "mb_y": 24, "dx": 2, "dy": 1, "dct_energy": 12.1},
      #   ...
      # ]
      
      # Extract temporal motion features
      motion_features = [
          [mb['dx'], mb['dy'], mb['dct_energy']]
          for mb in macroblocks
      ]
      
      # Feed to model for action classification
      # prediction = model.predict(motion_features)
""")

print("""
═══════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

Problem: roi_motion_data.json is empty {}

Solution 1: Check I-frame detections
""")
print("  grep 'DET\\|DETECTED' tracking_results.csv")
print("""
  If 0 detections:
    • Model may not be loading
    • Confidence threshold too high (try --conf_threshold 0.1)
    • Model not trained on this dataset

Solution 2: Check P-frame binary exists
""")
print("  ls -lh tracking_outputs/extraction_cache/video_data_P.bin")
print("""
  If file not found:
    • FFmpeg extraction failed
    • Check FFmpeg custom build
    • Verify H264_COEFF_EXTRACT_FILE environment variable

Solution 3: Check data flow
""")
print("  python3 debug_extraction.py")
print("""

Problem: FFmpeg says "command not found"

Solution: Compile custom FFmpeg
""")
print("  cd FFmpeg/FFmpeg")
print("  ./configure --disable-doc --disable-programs")
print("  make -j$(nproc)")
print("  ls -la ffmpeg.exe")

print("""

═══════════════════════════════════════════════════════════════════════════════
KEY POINTS TO REMEMBER
═══════════════════════════════════════════════════════════════════════════════

1. MOTION VECTORS ARE THE PRIMARY FEATURE
   • dx, dy directly from H.264 bitstream
   • No decoding, no optical flow needed
   • Represents temporal pixel movement

2. DCT ENERGY IS THE SECONDARY FEATURE
   • Sum of absolute DCT coefficients
   • Indicates macroblock complexity/change
   • Complement motion with appearance change

3. BOUNDING BOX FILTERS SPATIAL REGION
   • Only process macroblocks inside bbox
   • Ignore background completely
   • Efficient feature extraction

4. TEMPORAL SEQUENCE MATTERS
   • Collect motion across multiple P-frames
   • Frame ordering preserved
   • Time-series input for action recognition

5. EVERYTHING STAYS COMPRESSED
   • No RGB pixel reconstruction
   • No optical flow computation
   • No OpenCV processing
   • Direct from H.264 codec domain


═══════════════════════════════════════════════════════════════════════════════
FILES YOU'LL INTERACT WITH
═══════════════════════════════════════════════════════════════════════════════

INPUT:
  • video file (H.264/MP4)
  • best_model .h5 (SSD detector)

PROCESS:
  • compressed_domain_tracker.py (main script)
  • feature_map.py (frequency map builder)
  • FFmpeg/FFmpeg/ffmpeg.exe (custom decoder)

OUTPUT:
  • tracking_outputs/roi_motion_data.json (main output)
  • tracking_outputs/tracking_results.csv (debug info)
  • tracking_outputs/extraction_cache/ (temporary)

VERIFY:
  • debug_extraction.py (validation script)


═══════════════════════════════════════════════════════════════════════════════
REFERENCE: DATA STRUCTURES
═══════════════════════════════════════════════════════════════════════════════

Macroblock (16×16 pixels in video)
  mb_x, mb_y: Position in video (index)
  dx, dy: Quarter-pixel displacement
  dct_energy: Sum of |DCT_coefficients|

Example:
  {"mb_x": 32, "mb_y": 24, "dx": 2, "dy": 1, "dct_energy": 12.1}
  → Macroblock at grid position (32, 24)
  → Moved 0.5 pixels right, 0.25 pixels down
  → Has 12.1 units of transform coefficient energy

Bounding Box (from detector)
  cx, cy: Center in grid coordinates [0-7] for 7×7 SSD grid
  w, h: Width/height in same grid units
  confidence: Detection confidence [0-1]

Example:
  {"cx": 3.5, "cy": 3.5, "w": 2.0, "h": 3.0, "confidence": 0.95}
  → Human centered at grid position (3.5, 3.5)
  → Spans roughly 2 grid cells horizontally, 3 vertically

Filtering Example:
  MB (32, 24) in 64×48 MB grid (1024×768 video)
  → Grid coordinate: (32.5 × 7/64, 24.5 × 7/48) = (3.55, 3.57)
  
  Bbox: center (3.5, 3.5), size (2.0, 3.0)
  → Bbox area: [2.5-4.5] × [2.0-5.0]
  
  Is MB inside bbox?
  → 2.5 ≤ 3.55 ≤ 4.5 ✓
  → 2.0 ≤ 3.57 ≤ 5.0 ✓
  → YES → Keep this macroblock


════════════════════════════════════════════════════════════════════════════════

                    Ready to extract compressed-domain motion?
                          Run: python3 compressed_domain_tracker.py

════════════════════════════════════════════════════════════════════════════════
""")
