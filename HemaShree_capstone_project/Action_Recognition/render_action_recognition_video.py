"""
render_action_recognition_video.py
===================================
Performs Compressed-Domain Action Recognition inference on a video and
renders a fully annotated, production-quality visualization MP4 output video:
- Shows the bounding box tracked across I, P, and B frames (BAFE propagation)
- Displays HUD with the predicted action, confidence, and class breakdown
- Shows real-time frame type (I-frame DET, P-frame PROP, B-frame HOLD) and motion energy
- Produces a web-compatible H.264 MP4 output with +faststart
"""

import os
import sys
import json
import cv2
import av
import numpy as np
import pandas as pd
import subprocess

# Robust root directory detection
curr = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(curr, "action_recognition_model")):
    ROOT_DIR = curr
elif os.path.exists(os.path.join(os.path.dirname(curr), "action_recognition_model")):
    ROOT_DIR = os.path.dirname(curr)
elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(curr)), "action_recognition_model")):
    ROOT_DIR = os.path.dirname(os.path.dirname(curr))
else:
    ROOT_DIR = r"C:\Users\newuser\capstone_project"

ACTION_MODEL_DIR = os.path.join(ROOT_DIR, "action_recognition_model")
sys.path.insert(0, ACTION_MODEL_DIR)

from evaluate import predict_action, DEFAULT_CLASSES


def render_action_video(
    video_path,
    tracking_csv,
    roi_json_path,
    weights_path,
    output_mp4,
    grid_size=7.0,
    artifact_dir=None
):
    print(f"\n{'='*70}")
    print(f"COMPRESSED-DOMAIN ACTION RECOGNITION & VIDEO RENDERING")
    print(f"{'='*70}")
    print(f"Video input      : {video_path}")
    print(f"Tracking CSV     : {tracking_csv}")
    print(f"ROI Motion JSON  : {roi_json_path}")
    print(f"Model Weights    : {weights_path}")
    print(f"Output MP4       : {output_mp4}")
    print(f"{'='*70}\n")

    # 1. Run Action Recognition inference
    print("[1/4] Running Action Recognition inference on compressed-domain features...")
    pred_result = predict_action(roi_json_path, weights_path, model_type="tcnn")
    if not pred_result:
        raise RuntimeError("Action recognition inference returned None!")

    action_label = pred_result["predicted_action"]
    confidence = pred_result["confidence"]
    probs = pred_result["class_probabilities"]
    print(f"  -> Identified Action: {action_label.upper()} ({confidence*100:.1f}%)")

    # 2. Load tracking results
    print("[2/4] Loading bounding box timeline...")
    df = pd.read_csv(tracking_csv)
    # Map PTS -> list of box records
    boxes_by_pts = {}
    for _, row in df.iterrows():
        pts = int(row['pts'])
        if pts not in boxes_by_pts:
            boxes_by_pts[pts] = []
        boxes_by_pts[pts].append(row)

    # Load motion energies
    with open(roi_json_path, 'r', encoding='utf-8') as f:
        roi_motion = json.load(f)

    # 3. Open video container
    print("[3/4] Decoding video stream and rendering overlay...")
    container = av.open(video_path)
    stream = container.streams.video[0]
    w = stream.width
    h = stream.height
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    total_frames = int(stream.frames) if stream.frames else len(df)

    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    temp_raw_mp4 = output_mp4.replace(".mp4", "_temp.mp4")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_raw_mp4, fourcc, fps, (w, h))

    saved_artifacts = []
    frame_idx = 0

    for frame in container.decode(video=0):
        pts = frame.pts if frame.pts is not None else frame_idx
        img = frame.to_ndarray(format='bgr24')

        # Determine frame type from tracking CSV
        f_type = 'P'
        if pts in boxes_by_pts and len(boxes_by_pts[pts]) > 0:
            csv_f_type = str(boxes_by_pts[pts][0].get('frame_type', '')).strip().upper()
            if csv_f_type in ('I', 'P', 'B'):
                f_type = csv_f_type
            else:
                pt_val = getattr(frame.pict_type, 'value', frame.pict_type)
                f_type = {1: 'I', 2: 'P', 3: 'B'}.get(pt_val, 'P')
        else:
            pt_val = getattr(frame.pict_type, 'value', frame.pict_type)
            f_type = {1: 'I', 2: 'P', 3: 'B'}.get(pt_val, 'P')

        # 1. Draw Bounding Boxes
        if pts in boxes_by_pts:
            for row in boxes_by_pts[pts]:
                cx_rel = float(row['cx']) / grid_size
                cy_rel = float(row['cy']) / grid_size
                w_rel = float(row['w']) / grid_size
                h_rel = float(row['h']) / grid_size

                xmin = max(0, int((cx_rel - w_rel / 2.0) * w))
                ymin = max(0, int((cy_rel - h_rel / 2.0) * h))
                xmax = min(w - 1, int((cx_rel + w_rel / 2.0) * w))
                ymax = min(h - 1, int((cy_rel + h_rel / 2.0) * h))

                source = str(row['source']).upper()
                row_ftype = str(row.get('frame_type', f_type)).strip().upper()

                # Detection (I) = Bright Green, P-Frame Propagation = Bright Cyan, B-Frame = Amber Orange
                if source == 'DET' or row_ftype == 'I':
                    box_color = (46, 204, 113)   # Green
                    tag = f"I-DET ({float(row['confidence']):.2f})"
                elif row_ftype == 'B':
                    box_color = (0, 165, 255)     # Bright Orange / Amber
                    tag = "B-FRAME (HOLD)"
                else:
                    box_color = (255, 225, 0)     # Bright Cyan / Blue-Green
                    tag = "P-FRAME (BAFE)"

                # Draw bounding box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, 3)

                # Draw corner brackets for high-tech look
                corner_len = min(25, (xmax - xmin) // 4, (ymax - ymin) // 4)
                if corner_len > 0:
                    cv2.line(img, (xmin, ymin), (xmin + corner_len, ymin), (255, 255, 255), 4)
                    cv2.line(img, (xmin, ymin), (xmin, ymin + corner_len), (255, 255, 255), 4)
                    cv2.line(img, (xmax, ymin), (xmax - corner_len, ymin), (255, 255, 255), 4)
                    cv2.line(img, (xmax, ymin), (xmax, ymin + corner_len), (255, 255, 255), 4)
                    cv2.line(img, (xmin, ymax), (xmin + corner_len, ymax), (255, 255, 255), 4)
                    cv2.line(img, (xmin, ymax), (xmin, ymax - corner_len), (255, 255, 255), 4)
                    cv2.line(img, (xmax, ymax), (xmax - corner_len, ymax), (255, 255, 255), 4)
                    cv2.line(img, (xmax, ymax), (xmax, ymax - corner_len), (255, 255, 255), 4)

                # Tag above box
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                tag_y = max(ymin - 8, th + 10)
                cv2.rectangle(img, (xmin, tag_y - th - 6), (xmin + tw + 10, tag_y + 4), box_color, -1)
                cv2.putText(img, tag, (xmin + 5, tag_y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Draw HUD Header
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (w - 20, 150), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.rectangle(img, (20, 20), (w - 20, 150), (50, 60, 70), 2)

        # Title
        cv2.putText(img, "COMPRESSED-DOMAIN ACTION RECOGNITION", (40, 52),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (180, 200, 220), 2, cv2.LINE_AA)

        # Predicted Action banner
        action_text = f"ACTION: {action_label.upper()}"
        conf_text = f"{confidence * 100:.1f}% CONFIDENCE"
        cv2.putText(img, action_text, (40, 95),
                    cv2.FONT_HERSHEY_DUPLEX, 1.15, (0, 255, 128), 3, cv2.LINE_AA)
        cv2.putText(img, conf_text, (40, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 230, 210), 1, cv2.LINE_AA)

        # Right side info block
        f_badge_color = (46, 204, 113) if f_type == 'I' else ((0, 165, 255) if f_type == 'B' else (255, 225, 0))
        cv2.putText(img, f"Frame: {frame_idx + 1}/{total_frames} (PTS {pts})", (w - 380, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(img, "Frame Type: ", (w - 380, 92),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(img, (w - 250, 72), (w - 180, 102), f_badge_color, -1)
        cv2.putText(img, f_type, (w - 225, 95),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Motion Energy stats
        mbs = roi_motion.get(f"frame_{pts}", [])
        m_count = len(mbs)
        cv2.putText(img, f"ROI Macroblocks: {m_count}", (w - 380, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 210, 240), 1, cv2.LINE_AA)

        # Bottom Bar: Model architecture info
        cv2.rectangle(img, (20, h - 50), (w - 20, h - 15), (20, 25, 30), -1)
        cv2.putText(img, "Model: Temporal 1D-CNN  |  Features: BAFE Motion Vectors (dx, dy) + DCT Residual Energy  |  Zero Pixel Decoding",
                    (35, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 180, 190), 1, cv2.LINE_AA)

        writer.write(img)

        # Save sample keyframes for artifacts (I, B, and P frames)
        if artifact_dir:
            if f_type == 'I' and 'frame_I' not in [os.path.basename(p) for p in saved_artifacts]:
                p_i = os.path.join(artifact_dir, "action_vis_frame_I.jpg")
                cv2.imwrite(p_i, img)
                saved_artifacts.append(p_i)
            elif f_type == 'B' and len([p for p in saved_artifacts if 'frame_B' in p]) < 2:
                p_b = os.path.join(artifact_dir, f"action_vis_frame_B_{frame_idx}.jpg")
                cv2.imwrite(p_b, img)
                saved_artifacts.append(p_b)
            elif f_type == 'P' and 'frame_P' not in [os.path.basename(p) for p in saved_artifacts]:
                p_p = os.path.join(artifact_dir, "action_vis_frame_P.jpg")
                cv2.imwrite(p_p, img)
                saved_artifacts.append(p_p)

        frame_idx += 1

    writer.release()
    container.close()

    # 4. Remux with ffmpeg to ensure standard H.264 + faststart web playback
    print("[4/4] Remuxing to standard H.264 MP4 with +faststart...")
    ffmpeg_exe = os.path.join(ROOT_DIR, "FFmpeg", "FFmpeg", "ffmpeg.exe")
    if not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe, "-y", "-i", temp_raw_mp4,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_mp4
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(output_mp4):
        try:
            os.remove(temp_raw_mp4)
        except Exception:
            pass
        print(f"[OK] Video rendering successfully completed: {output_mp4}")
    else:
        if os.path.exists(temp_raw_mp4):
            if os.path.exists(output_mp4):
                os.remove(output_mp4)
            os.rename(temp_raw_mp4, output_mp4)
        print(f"[OK] Video saved: {output_mp4}")

    return {
        "output_video": output_mp4,
        "action": action_label,
        "confidence": confidence,
        "probabilities": probs,
        "total_frames": frame_idx,
        "artifacts": saved_artifacts
    }


if __name__ == '__main__':
    video_file = os.path.join(ROOT_DIR, r"Human Activity Recognition - Video Dataset\Walking\Walking (1).mp4")
    csv_file = os.path.join(ROOT_DIR, r"output\Walking (1)\tracking\tracking_results.csv")
    roi_file = os.path.join(ROOT_DIR, r"output\Walking (1)\motion\roi_motion_data.json")
    weights_file = os.path.join(ROOT_DIR, r"action_recognition_model\action_recognition_tcnn.pt")
    out_video = os.path.join(ROOT_DIR, r"output\Walking (1)\Walking_1_Action_Recognition_Output.mp4")
    art_dir = r"C:\Users\newuser\.gemini\antigravity-ide\brain\bfbf25a3-dabe-4415-b83f-d4921d81ca0d"

    render_action_video(video_file, csv_file, roi_file, weights_file, out_video, artifact_dir=art_dir)
