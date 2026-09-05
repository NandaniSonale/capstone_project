#!/usr/bin/env python3
"""
Render Visualized Videos for Compressed-Domain Tracking Dataset
Generates .mp4 videos showing:
  1. Human Bounding Box (DET vs PROP)
  2. Microboxes (16x16 Macroblock Grid inside ROI)
  3. Motion Vector Directional Arrows (dx, dy)
  4. Real-time Telemetry HUD (Frame, PTS, Frame Type, Microbox Count)
"""

import os
import re
import sys
import time
import argparse
import json
import cv2
import numpy as np
import pandas as pd


def natural_sort_key(s):
    """Sort strings containing numbers naturally: 'Walking (2)' before 'Walking (10)'."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def find_raw_video_source(video_name, search_dirs):
    """Search for matching raw video (.mp4, .avi) if available."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    for sdir in search_dirs:
        if not sdir or not os.path.exists(sdir):
            continue
        for root, _, files in os.walk(sdir):
            for f in files:
                if f.lower().endswith(video_extensions):
                    base = os.path.splitext(f)[0]
                    if base.lower() == video_name.lower():
                        return os.path.join(root, f)
    return None


def render_single_video(
    video_dir,
    raw_video_path=None,
    output_video_path=None,
    width=540,
    height=960,
    fps=25,
    force=False
):
    """
    Renders one video with bounding boxes, microboxes, motion vectors, and HUD.
    Returns dict with summary statistics.
    """
    video_name = os.path.basename(os.path.normpath(video_dir))
    if output_video_path is None:
        output_video_path = os.path.join(video_dir, f"{video_name}_visualized.mp4")

    if os.path.exists(output_video_path) and not force:
        size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        if size_mb > 0.05:
            return {
                "video_name": video_name,
                "status": "SKIPPED",
                "output_video": output_video_path,
                "file_size_mb": round(size_mb, 2),
                "duration_s": 0.0,
                "frames_rendered": 0
            }

    csv_path = os.path.join(video_dir, "tracking", "tracking_results.csv")
    roi_path = os.path.join(video_dir, "motion", "roi_motion_data.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Tracking CSV is empty: {csv_path}")

    motion_data = {}
    if os.path.exists(roi_path):
        try:
            with open(roi_path, 'r', encoding='utf-8') as f:
                motion_data = json.load(f)
        except Exception:
            motion_data = {}

    # Open raw video container if available
    raw_cap = None
    if raw_video_path and os.path.exists(raw_video_path):
        raw_cap = cv2.VideoCapture(raw_video_path)
        if not raw_cap.isOpened():
            raw_cap = None

    t0 = time.time()
    total_frames = len(df)
    scale_x = width / 1080.0
    scale_y = height / 1920.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_video_path}")

    grid_w = max(int(16 * scale_x), 4)

    for idx in range(total_frames):
        row = df.iloc[idx]
        pts = int(row['pts'])
        f_type = str(row['frame_type'])
        source = str(row['source'])
        conf = float(row['confidence'])
        mbs = motion_data.get(f"frame_{pts}", [])

        # 1. Base Frame
        if raw_cap is not None:
            ret, frame = raw_cap.read()
            if ret and frame is not None:
                canvas = cv2.resize(frame, (width, height))
            else:
                canvas = np.full((height, width, 3), (16, 20, 26), dtype=np.uint8)
        else:
            # Compressed domain grid canvas
            canvas = np.full((height, width, 3), (16, 20, 26), dtype=np.uint8)
            # Subtle macroblock grid lines
            for gx in range(0, width, grid_w * 4):
                cv2.line(canvas, (gx, 0), (gx, height), (24, 30, 38), 1)
            for gy in range(0, height, grid_w * 4):
                cv2.line(canvas, (0, gy), (width, gy), (24, 30, 38), 1)

        # 2. Coordinates calculation (grid [0, 7] -> pixel)
        cx_rel = float(row['cx']) / 7.0
        cy_rel = float(row['cy']) / 7.0
        w_rel = float(row['w']) / 7.0
        h_rel = float(row['h']) / 7.0

        xmin = int((cx_rel - w_rel / 2.0) * width)
        ymin = int((cy_rel - h_rel / 2.0) * height)
        xmax = int((cx_rel + w_rel / 2.0) * width)
        ymax = int((cy_rel + h_rel / 2.0) * height)

        # 3. Soft ROI glow
        if source != 'NONE' and (xmax > xmin and ymax > ymin):
            overlay = canvas.copy()
            glow_color = (20, 50, 35) if source == 'DET' else (35, 45, 20)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), glow_color, -1)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # 4. Microboxes (16x16 macroblocks inside ROI) & Motion Vector Quivers
        for mb in mbs:
            mx = int(mb['mb_x'] * 16 * scale_x)
            my = int(mb['mb_y'] * 16 * scale_y)
            mw = max(int(16 * scale_x), 2)
            mh = max(int(16 * scale_y), 2)

            # Microbox outline
            cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (50, 165, 95), 1)

            # Motion vector arrow
            cx = mx + mw // 2
            cy = my + mh // 2
            # dx, dy are quarter-pixel units in H.264
            dx = int((mb['dx'] / 4.0) * scale_x * 2.5)
            dy = int((mb['dy'] / 4.0) * scale_y * 2.5)

            if abs(dx) > 0 or abs(dy) > 0:
                cv2.arrowedLine(canvas, (cx, cy), (cx + dx, cy + dy), (0, 240, 255), 1, tipLength=0.35)
            else:
                cv2.circle(canvas, (cx, cy), 1, (0, 200, 220), -1)

        # 5. Bounding Box & Accents
        if source != 'NONE' and (xmax > xmin and ymax > ymin):
            # DET = Bright Green, PROP = Cyan / Gold
            box_color = (0, 255, 128) if source == 'DET' else (255, 200, 0)
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), box_color, 2)

            # Corner brackets
            corner_len = min(12, max(4, (xmax - xmin) // 6))
            for pt1, pt2 in [
                ((xmin, ymin), (xmin + corner_len, ymin)),
                ((xmin, ymin), (xmin, ymin + corner_len)),
                ((xmax, ymin), (xmax - corner_len, ymin)),
                ((xmax, ymin), (xmax, ymin + corner_len)),
                ((xmin, ymax), (xmin + corner_len, ymax)),
                ((xmin, ymax), (xmin, ymax - corner_len)),
                ((xmax, ymax), (xmax - corner_len, ymax)),
                ((xmax, ymax), (xmax, ymax - corner_len))
            ]:
                cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

            # Box Badge Tag
            tag_y = max(ymin - 24, 76)
            tag_w = 175
            cv2.rectangle(canvas, (xmin, tag_y), (xmin + tag_w, tag_y + 22), box_color, -1)
            tag_text = f"{f_type}-FRAME {source} ({conf:.2f})"
            cv2.putText(canvas, tag_text, (xmin + 6, tag_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

        # 6. Top Telemetry HUD
        cv2.rectangle(canvas, (0, 0), (width, 70), (10, 14, 18), -1)
        cv2.line(canvas, (0, 70), (width, 70), (45, 55, 68), 1)
        cv2.putText(canvas, f"{video_name.upper()} | COMPRESSED DOMAIN ROI", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        hud_info = f"Frame: {idx + 1}/{total_frames} (PTS {pts}) | {f_type} ({source}) | Microboxes: {len(mbs)}"
        cv2.putText(canvas, hud_info, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1)

        # 7. Bottom Legend Footer
        cv2.rectangle(canvas, (0, height - 32), (width, height), (10, 14, 18), -1)
        cv2.line(canvas, (0, height - 32), (width, height - 32), (45, 55, 68), 1)
        cv2.putText(canvas, "Green BBox = DET | Amber BBox = PROP | Cyan Arrow = MV | Microbox = 16x16 MB",
                    (12, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (170, 190, 200), 1)

        writer.write(canvas)

    writer.release()
    if raw_cap is not None:
        raw_cap.release()

    duration = time.time() - t0
    size_mb = os.path.getsize(output_video_path) / (1024 * 1024)

    return {
        "video_name": video_name,
        "status": "SUCCESS",
        "output_video": output_video_path,
        "file_size_mb": round(size_mb, 2),
        "duration_s": round(duration, 2),
        "frames_rendered": total_frames
    }


def main():
    parser = argparse.ArgumentParser(description="Render visual videos with bounding boxes and microboxes.")
    parser.add_argument("--output", "-o", default="output", help="Directory containing per-video output folders")
    parser.add_argument("--raw-videos", "-r", default=r"C:\Users\keshav\OneDrive\Desktop\nandani", help="Directory with raw video files if any")
    parser.add_argument("--width", type=int, default=540, help="Output video width (default: 540)")
    parser.add_argument("--height", type=int, default=960, help="Output video height (default: 960)")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS (default: 25)")
    parser.add_argument("--force", action="store_true", help="Force re-rendering even if video already exists")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit max videos to render")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    if not os.path.exists(output_dir):
        print(f"[ERROR] Output directory not found: {output_dir}")
        sys.exit(1)

    # Find all video subdirectories
    subdirs = []
    for item in os.listdir(output_dir):
        full_path = os.path.join(output_dir, item)
        if os.path.isdir(full_path):
            tracking_csv = os.path.join(full_path, "tracking", "tracking_results.csv")
            if os.path.exists(tracking_csv):
                subdirs.append(full_path)

    subdirs.sort(key=natural_sort_key)
    if args.max_videos:
        subdirs = subdirs[:args.max_videos]

    total_videos = len(subdirs)
    print(f"\n{'=' * 60}")
    print(f"BATCH VIDEO RENDERER: COMPRESSED DOMAIN VISUALIZATION")
    print(f"{'=' * 60}")
    print(f"Output Directory : {output_dir}")
    print(f"Total Videos     : {total_videos}")
    print(f"Resolution       : {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Force Overwrite  : {args.force}")
    print(f"{'=' * 60}\n")

    search_dirs = [args.raw_videos, output_dir, os.getcwd()]

    success_count = 0
    skipped_count = 0
    failed_count = 0
    total_frames_rendered = 0
    t_start = time.time()
    results = []

    for i, v_dir in enumerate(subdirs, 1):
        v_name = os.path.basename(v_dir)
        print(f"[{i}/{total_videos}] Rendering: {v_name}...", end="", flush=True)

        raw_source = find_raw_video_source(v_name, search_dirs)

        try:
            res = render_single_video(
                video_dir=v_dir,
                raw_video_path=raw_source,
                width=args.width,
                height=args.height,
                fps=args.fps,
                force=args.force
            )
            results.append(res)
            if res["status"] == "SKIPPED":
                skipped_count += 1
                print(f" -> [SKIPPED] ({res['file_size_mb']} MB)")
            else:
                success_count += 1
                total_frames_rendered += res["frames_rendered"]
                print(f" -> [SUCCESS] {res['frames_rendered']} frames in {res['duration_s']}s ({res['file_size_mb']} MB)")
        except Exception as exc:
            failed_count += 1
            print(f" -> [FAILED] {exc}")
            results.append({
                "video_name": v_name,
                "status": "FAILED",
                "error": str(exc),
                "duration_s": 0.0
            })

    total_duration = time.time() - t_start

    # Update master processing_summary.csv if it exists
    summary_csv = os.path.join(output_dir, "processing_summary.csv")
    if os.path.exists(summary_csv):
        try:
            summary_df = pd.read_csv(summary_csv)
            video_to_path = {
                r["video_name"]: r.get("output_video", "") for r in results if r.get("status") in ("SUCCESS", "SKIPPED")
            }
            summary_df["visualized_video"] = summary_df["video_name"].map(video_to_path).fillna("")
            summary_df.to_csv(summary_csv, index=False)
            print(f"\n[INFO] Updated master summary CSV with visualized video paths: {summary_csv}")
        except Exception as exc:
            print(f"\n[WARN] Failed to update master summary CSV: {exc}")

    print(f"\n{'=' * 60}")
    print(f"VIDEO RENDERING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total Videos         : {total_videos}")
    print(f"Newly Rendered       : {success_count}")
    print(f"Already Completed    : {skipped_count}")
    print(f"Failed               : {failed_count}")
    print(f"Total Frames         : {total_frames_rendered:,}")
    print(f"Total Elapsed Time   : {total_duration:.2f}s ({total_duration / 60:.2f} mins)")
    if success_count > 0:
        print(f"Average Time/Video   : {total_duration / success_count:.2f}s")
    print(f"Output Directory     : {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
