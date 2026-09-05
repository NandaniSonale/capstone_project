import os
import cv2
import av
import pandas as pd
import json
import numpy as np
import argparse

def generate_video(
    video_path=r"Human Activity Recognition - Video Dataset\Walking\Walking (23).mp4",
    csv_path=r"tracking_outputs\tracking_results.csv",
    json_path=r"tracking_outputs\roi_motion_data.json",
    output_mp4_paths=[
        r"visuals\tracking_visualization.mp4"
    ],
    grid_size=7
):
    print(f"--- Compressed-Domain Video Output Generator (Single Primary Target Edition) ---")
    print(f"Reading Video: {video_path}")
    print(f"Reading CSV: {csv_path}")
    print(f"Reading JSON: {json_path}")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV results not found: {csv_path}")
        return

    # Load CSV tracking data
    df = pd.read_csv(csv_path)
    boxes_by_pts = df.groupby('pts')

    # Load JSON motion vector & DCT energy data
    roi_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            roi_data = json.load(f)
        print(f"Loaded {len(roi_data)} frame ROI motion records.")

    # Open PyAV video container
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    width = stream.width
    height = stream.height
    fps = float(stream.average_rate) if stream.average_rate else 30.0

    print(f"Video resolution: {width}x{height} @ {fps:.2f} FPS")

    # Dynamic scaling for UI elements based on resolution width
    scale_factor = max(1.0, width / 1280.0)
    banner_height = int(120 * (height / 1080.0))
    box_thickness = max(4, int(4 * scale_factor))
    text_scale_title = 0.85 * scale_factor
    text_scale_sub = 0.62 * scale_factor
    text_thickness = max(2, int(2 * scale_factor))

    # Ensure output directory exists
    for p in output_mp4_paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # Initialize Video Writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = []
    for p in output_mp4_paths:
        w = cv2.VideoWriter(p, fourcc, fps, (width, height))
        if w.isOpened():
            writers.append((p, w))
            print(f"Created video writer target: {p}")
        else:
            print(f"[WARNING] Could not open VideoWriter for {p}")

    if not writers:
        print("[ERROR] No VideoWriters could be initialized.")
        return

    frame_count = 0
    print("Processing video frames and rendering high-visibility overlays...")

    pict_map = {'1': 'I', '2': 'P', '3': 'B', 1: 'I', 2: 'P', 3: 'B', 'I': 'I', 'P': 'P', 'B': 'B'}
    i_frame_hold_counter = 0

    for frame in container.decode(video=0):
        pts = frame.pts
        raw_type = frame.pict_type.name if hasattr(frame.pict_type, 'name') else str(frame.pict_type)
        f_type = pict_map.get(raw_type, 'I' if frame_count == 0 else 'P')
        
        # Highlight I-Frame Green DET box for 6 frames (~0.2s) so both I-frames (PTS 0 and PTS 80896) stand out clearly
        if f_type == 'I':
            i_frame_hold_counter = 6
        elif i_frame_hold_counter > 0:
            i_frame_hold_counter -= 1

        is_i_frame_active = (f_type == 'I' or i_frame_hold_counter > 0)

        img = frame.to_ndarray(format='bgr24')
        h_img, w_img = img.shape[:2]

        active_mbs_count = 0
        frame_key = f"frame_{pts}"
        if frame_key in roi_data:
            active_mbs_count = len(roi_data[frame_key])

        # -------------------------------------------------------------
        # 1. DRAW HIGH-CONTRAST TOP HUD HEADER BANNER
        # -------------------------------------------------------------
        cv2.rectangle(img, (0, 0), (w_img, banner_height), (15, 18, 26), -1)
        cv2.line(img, (0, banner_height), (w_img, banner_height), (0, 200, 255), max(2, int(3 * scale_factor)))

        # Line 1: Primary Frame Status
        type_str = "I-FRAME" if is_i_frame_active else "P-FRAME"
        type_color = (0, 255, 0) if is_i_frame_active else (255, 200, 0)
        hud_line1 = f"FRAME: {frame_count:04d} | PTS: {pts} | TYPE: {type_str}"
        cv2.putText(img, hud_line1, (int(20 * scale_factor), int(42 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale_title, (255, 255, 255), text_thickness, cv2.LINE_AA)

        # Line 2: ROI & Legend (Clean high contrast rendering)
        hud_line2 = f"ROI Macroblocks: {active_mbs_count} | [I-Frame: Green Box (DCT)] | [P-Frame: Light Blue Box] | [MVs: Red Arrows]"
        cv2.putText(img, hud_line2, (int(20 * scale_factor), int(90 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale_sub, (0, 225, 255), max(1, text_thickness), cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. DRAW ROI MOTION VECTORS & DCT ENERGY (P-FRAMES)
        # -------------------------------------------------------------
        if frame_key in roi_data:
            mbs = roi_data[frame_key]

            for mb in mbs:
                mb_x = mb['mb_x']
                mb_y = mb['mb_y']
                dx = mb['dx']
                dy = mb['dy']
                energy = mb.get('dct_energy', 0.0)

                # Center of 16x16 macroblock in pixel space
                center_x = int(mb_x * 16 + 8)
                center_y = int(mb_y * 16 + 8)

                # A. DCT Energy Grid Heatmap (Normalized across 30k..90k energy range)
                if energy > 5000:
                    norm_e = min(1.0, max(0.0, (energy - 30000.0) / 60000.0))
                    # Heatmap color: Green (Low Energy) -> Yellow -> Bright Red (High Energy)
                    e_color = (0, int(255 * (1.0 - norm_e)), int(255 * norm_e))
                    top_left = (mb_x * 16, mb_y * 16)
                    bot_right = ((mb_x + 1) * 16, (mb_y + 1) * 16)
                    cv2.rectangle(img, top_left, bot_right, e_color, 1)

                # B. Motion Vector Arrow Overlay (Scaled for 16x16 macroblock grid)
                if dx != 0 or dy != 0:
                    mv_scale = 0.8  # Scale motion displacement vector for visual clarity
                    end_x = int(center_x - dx * mv_scale)
                    end_y = int(center_y - dy * mv_scale)

                    mag = np.sqrt(dx*dx + dy*dy)
                    # Color: Bright Red for larger motion, Bright Yellow/Orange for subtle motion
                    mv_color = (0, 0, 255) if mag > 20.0 else (0, 215, 255)
                    line_thick = max(1, int(1.5 * scale_factor))

                    cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y),
                                    mv_color, line_thick, tipLength=0.35)
                else:
                    # Subtle marker dot for stationary macroblock
                    dot_radius = max(2, int(2 * scale_factor))
                    cv2.circle(img, (center_x, center_y), dot_radius, (160, 160, 160), -1)

        # -------------------------------------------------------------
        # 3. DRAW BOUNDING BOXES (I-Frame DET vs P-Frame PROP)
        # -------------------------------------------------------------
        if pts in boxes_by_pts.groups:
            frame_boxes = boxes_by_pts.get_group(pts)

            for _, row in frame_boxes.iterrows():
                if row['confidence'] <= 0:
                    continue

                cx_rel = row['cx'] / grid_size
                cy_rel = (row['cy'] / grid_size) - 0.02
                w_rel = row['w'] / grid_size
                h_rel = row['h'] / grid_size

                # Full Human Body Coverage bounds (head to toe enclosure):
                h_rel = max(0.64, h_rel * 1.35)
                w_rel = max(0.24, w_rel * 1.35)

                xmin = int((cx_rel - w_rel/2) * w_img)
                ymin = int((cy_rel - h_rel/2) * h_img)
                xmax = int((cx_rel + w_rel/2) * w_img)
                ymax = int((cy_rel + h_rel/2) * h_img)

                # Clamp bounding box inside frame
                xmin, ymin = max(0, xmin), max(banner_height + 5, ymin)
                xmax, ymax = min(w_img - 1, xmax), min(h_img - 1, ymax)

                is_det_box = (row['source'] == 'DET' or is_i_frame_active)
                box_color = (0, 255, 0) if is_det_box else (255, 200, 0)  # Pure Green (0,255,0) = I-Frame (DCT), Light Blue (255,200,0) = P-Frame (PROP)

                # Draw main ROI Bounding Box with thick lines
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
                
                # Draw Corner Brackets for visual clarity
                bracket_len = int(min(xmax - xmin, ymax - ymin) * 0.18)
                bracket_thick = box_thickness + 2
                if bracket_len > 5:
                    # Top-Left
                    cv2.line(img, (xmin, ymin), (xmin + bracket_len, ymin), box_color, bracket_thick)
                    cv2.line(img, (xmin, ymin), (xmin, ymin + bracket_len), box_color, bracket_thick)
                    # Top-Right
                    cv2.line(img, (xmax, ymin), (xmax - bracket_len, ymin), box_color, bracket_thick)
                    cv2.line(img, (xmax, ymin), (xmax, ymin + bracket_len), box_color, bracket_thick)
                    # Bottom-Left
                    cv2.line(img, (xmin, ymax), (xmin + bracket_len, ymax), box_color, bracket_thick)
                    cv2.line(img, (xmin, ymax), (xmin, ymax - bracket_len), box_color, bracket_thick)
                    # Bottom-Right
                    cv2.line(img, (xmax, ymax), (xmax - bracket_len, ymax), box_color, bracket_thick)
                    cv2.line(img, (xmax, ymax), (xmax, ymax - bracket_len), box_color, bracket_thick)

                # Draw Label Badge Above Box
                lbl_text = f"{'I-Frame (DCT)' if is_det_box else 'P-Frame (Propagation)'} | Conf: {row['confidence']:.2f}"
                lbl_scale = 0.7 * scale_factor
                lbl_thick = max(2, int(2 * scale_factor))
                (w_lbl, h_lbl), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, lbl_thick)

                badge_y_min = max(banner_height + 5, ymin - h_lbl - 12)
                cv2.rectangle(img, (xmin, badge_y_min), (xmin + w_lbl + 16, badge_y_min + h_lbl + 10), (15, 15, 15), -1)
                cv2.rectangle(img, (xmin, badge_y_min), (xmin + w_lbl + 16, badge_y_min + h_lbl + 10), box_color, 2)
                cv2.putText(img, lbl_text, (xmin + 8, badge_y_min + h_lbl + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, box_color, lbl_thick, cv2.LINE_AA)

        # Write to output video targets
        for _, writer in writers:
            writer.write(img)

        frame_count += 1

    # Release video writers
    for path, writer in writers:
        writer.release()
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Successfully generated video: {path} ({size_mb:.2f} MB)")

    print("High-visibility video output generation complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '-v', default=r"C:\Users\newuser\capstone_project\Human Activity Recognition - Video Dataset\Clapping\Clapping (1).mp4")
    parser.add_argument('--csv', '-c', default=r"tracking_outputs\tracking_results.csv")
    parser.add_argument('--json', '-j', default=r"tracking_outputs\roi_motion_data.json")
    args = parser.parse_args()

    generate_video(args.video, args.csv, args.json)
