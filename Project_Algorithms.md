# Project Algorithms: Compressed-Domain Object Tracking Video Analytics

This document outlines the core algorithms implemented in the software pipeline described in Chapter 8. The pseudo-code here is formulated directly from the local project repository code, ensuring exact alignment with the methodology and bypassing generic pseudo-code templates to maintain academic originality.

### Algorithm 1: Automated Dataset Ground-Truth Preparation
**(Maps to Section 8.2 & `dataset_prep.py`)**
This defines the generation pipeline for ground-truth dataset labels using YOLO. It processes the compressed H.264 video files directly to map tracking coordinates to the matching frequency maps later.

```text
Algorithm 1: Dataset Ground Truth Annotation Pipeline
Input: root_directory containing compressed H.264 video streams
Output: Normalized coordinate annotation text files (.txt)

1: Initialize YOLOv8 object tracking model
2: for each compressed_video_file in root_directory do
3:     cap = Open VideoCapture(compressed_video_file)
4:     W, H = ExtractVideoResolution(cap)
5:     Create equivalent output_directory structure
6:     Open annotation_file for writing
7:     frame_index = 0
8:     
9:     while cap has frames do
10:        frame = cap.read()
11:        results = YOLO.track(frame)
12:        
13:        for each object_box in results do
14:            class_id, confidence = ExtractClassAndConf(object_box)
15:            if class_id == HUMAN_CLASS and confidence > 0.5 then
16:                [x1, y1, x2, y2] = object_box.coordinates
17:                track_id = object_box.id
18:                
19:                // Normalize coordinates relative to video dimensions
20:                x1_norm, y1_norm = x1 / W, y1 / H
21:                x2_norm, y2_norm = x2 / W, y2 / H
22:                
23:                Write (frame_index, track_id, HUMAN_CLASS, x1_norm, y1_norm, x2_norm, y2_norm) 
24:                      to annotation_file
25:            end if
26:        end for
27:        frame_index = frame_index + 1
28:    end while
29:    cap.release()
30: end for
```

---

### Algorithm 2: H.264 Coefficient Extraction & Frequency Map Generation
**(Maps to Section 8.2 / 8.3 & `feature_map.py`)**  
Describes how compressed-domain coefficient residuals are parsed without falling back to full RGB decoding, approximated into spatial frequency channels, and converted into log-normalized RGB-equivalent feature tensors.

```text
Algorithm 2: Sub-Band Frequency Feature Map Generation
Input: compressed_coefficients.bin extracted from the H.264 bitstream via custom FFmpeg struct
Output: 3D Spatial Feature Map Tensor (Height x Width x 3)

1: Initialize HDR_SIZE and COEFF_BYTES based on predetermined struct sizes
2: Calculate num_macroblocks = file_size / RECORD_SIZE
3: Determine map spatial resolution max_mb_x, max_mb_y across target frames
4: 
5: Set grid_rows = (max_mb_y + 1) * 4
6: Set grid_cols = (max_mb_x + 1) * 4
7: Initialize float32 feature_tensor of size (grid_rows, grid_cols, 3) initialized to 0
8: 
9: Define 4x4 spatial frequency band indices:
10:    low_indices  = {0, 1, 4}
11:    mid_indices  = {2, 3, 5, 6, 8}
12:    high_indices = {7, 9, 10, 11, 12, 13, 14, 15}
13: 
14: for each chunk in compressed_coefficients.bin do
15:    Extract header (frame_num, mb_x, mb_y, mb_type)
16:    if frame_num == target_frame then
17:        Extract and cast 256 Luma Quantized transform coefficients to float
18:        
19:        for block_idx = 0 to 15 do
20:            block_coeffs = subset of Luma coefficients for block_idx
21:            
22:            energy_low = Sum of square amounts of block_coeffs[low_indices]
23:            energy_mid = Sum of square amounts of block_coeffs[mid_indices]
24:            energy_high = Sum of square amounts of block_coeffs[high_indices]
25:            
26:            // Translate Z-scan macroblock index into linear pixel grid
27:            Calculate global_row and global_col from mb_y, mb_x 
28:            
29:            // Populate feature channels with log-scaled spatial approximations
30:            feature_tensor[global_row, global_col, 0] = log(1 + energy_low)
31:            feature_tensor[global_row, global_col, 1] = log(1 + energy_mid)
32:            feature_tensor[global_row, global_col, 2] = log(1 + energy_high)
33:        end for
34:    end if
35: end for
36: Return feature_tensor
```

---

### Algorithm 3: High-Efficiency Object Tracking and Propagation
**(Maps to Section 8.3 / 8.5 & `compressed_domain_tracker.py` + `object_detection_basic.py`)**  
Defines the inference loop. Detects exclusively on compressed I-frames using the previously derived frequency map tensors directly on the SSD network, propagating localizations for inter-frame components without decoding them.

```text
Algorithm 3: Compressed-Domain Temporal Object Tracking Framework
Input: Compressed Video Bitstream V, Custom FFmpeg Executable E, Trained SSD300 Model M
Output: csv_results containing temporal bounding box traces

1: Run Executable E directly over Bitstream V to extract frequency maps (.npy) for all I-Frames
2: active_boxes = Empty List
3: Open csv_results for writing
4: 
5: for each parsed frame in Bitstream V do
6:     pts_timestamp = frame.pts
7:     frame_type = determine_h264_type(frame) 
8:     
9:     if frame_type == 'I_FRAME' then
10:        Load corresponding frequency_map from Cache via pts_timestamp
11:        Normalize frequency_map by subtracting mean and dividing by standard deviation
12:        Resize frequency_map to M input resolution (300 x 300)
13:        
14:        predictions = M.predict(frequency_map)
15:        active_boxes = Empty List
16:        
17:        for each grid_cell mapping in predictions do
18:            if grid_cell.confidence >= CONFIDENCE_THRESHOLD then
19:                cx, cy, w, h = grid_cell.relative_bounding_box
20:                Convert cx, cy to absolute map scale based on grid position
21:                Append [confidence, cx, cy, w, h] to active_boxes
22:            end if
23:        end for
24:        Record detection_source = "DETECTION_" + frame_type
25:        
26:    else if frame_type == 'P_FRAME' then
27:        if length(active_boxes) > 0 then
28:            // ROI-Based Motion Extraction Stage
29:            // For every macroblock center (mb_x, mb_y) inside active_boxes:
30:            //   Extract dx, dy (Motion Vectors)
31:            //   Extract Σ |coefficients| (DCT Energy)
32:            // Append to temporal ROI sequence
33:            Record detection_source = "PROPAGATION_ROI_EXTRACT"
34:        else
35:            Record detection_source = "NONE"
36:        end if
37:    end if
38:    
39:    if length(active_boxes) > 0 then
40:        for each box in active_boxes do
41:            Append (pts_timestamp, frame_type, detection_source, box) to csv_results
42:        end for
43:    else
44:        Append empty metric entries to csv_results
45:    end if
46: end for
47: Export temporal ROI sequence to JSON for action recognition input
```
