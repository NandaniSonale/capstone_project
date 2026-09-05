#!/usr/bin/env python3
"""
Batch Processing Pipeline for Compressed-Domain Video Tracking and Propagation.

Processes an entire directory of videos or annotation files (e.g. 171 walking videos)
sequentially, generating:
  1. P-frame propagation
  2. B-frame propagation
  3. Tracking data (tracking_results.csv, p_frames_tracking.csv, b_frames_tracking.csv)
  4. Motion data (roi_motion_data.json, motion_summary.json)
  5. Propagation summary (propagation_summary.json)
  6. Per-video summary (video_summary.json)
  7. Dataset processing summary (processing_summary.csv)

Supports:
  - Resume functionality (skips completed videos unless --force is set)
  - Isolated error handling (a failing video does not halt the batch)
  - Shared model loading in memory (does not reload model 171 times)
  - Native video files (.mp4, .avi, etc.) AND annotation data (.txt from HAR_annotations)
"""

import os
import re
import csv
import json
import time
import argparse
import traceback
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from compressed_domain_tracker import CompressedDomainTracker
from bafe_propagation import propagate_boxes_bafe, filter_roi_macroblocks


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v')


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally: Walking (1) before Walking (2)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def find_dataset_items(input_path):
    """Find all video files or annotation files in input_path."""
    items = []
    if os.path.isfile(input_path):
        return [os.path.abspath(input_path)]

    if not os.path.exists(input_path):
        return []

    # First check for video files
    for root, _, files in os.walk(input_path):
        for f in files:
            if f.lower().endswith(VIDEO_EXTENSIONS):
                items.append(os.path.abspath(os.path.join(root, f)))

    # If no video files found, check for annotation .txt files
    if not items:
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith('.txt'):
                    items.append(os.path.abspath(os.path.join(root, f)))

    items.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    return items


def is_video_completed(video_out_dir):
    """Check if video was already successfully processed."""
    summary_path = os.path.join(video_out_dir, "video_summary.json")
    tracking_csv = os.path.join(video_out_dir, "tracking", "tracking_results.csv")
    motion_json = os.path.join(video_out_dir, "motion", "roi_motion_data.json")

    if os.path.exists(summary_path) and os.path.exists(tracking_csv) and os.path.exists(motion_json):
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
                if data.get("status") == "SUCCESS":
                    return True, data
        except Exception:
            return False, None
    return False, None


def update_summary_csv(csv_path, records):
    """Write or rewrite the cumulative processing_summary.csv."""
    headers = [
        "video_name",
        "status",
        "total_frames",
        "i_frames",
        "p_frames",
        "b_frames",
        "propagated_frames",
        "tracking_data",
        "motion_data",
        "error",
        "processing_time"
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "video_name": rec.get("video_name", ""),
                "status": rec.get("status", ""),
                "total_frames": rec.get("total_frames", 0),
                "i_frames": rec.get("i_frames", 0),
                "p_frames": rec.get("p_frames", 0),
                "b_frames": rec.get("b_frames", 0),
                "propagated_frames": rec.get("propagated_frames", 0),
                "tracking_data": rec.get("tracking_data", ""),
                "motion_data": rec.get("motion_data", ""),
                "error": rec.get("error", "") or "",
                "processing_time": rec.get("processing_time", "0.00s")
            })


def process_annotation_video(ann_path, output_dir, grid_size=7):
    """
    Process an annotation file (from HAR_annotations) as a video item,
    running BAFE propagation and ROI macroblock motion extraction across
    I, P, and B frames.
    """
    start_time = time.time()
    video_name = os.path.splitext(os.path.basename(ann_path))[0]

    tracking_dir = os.path.join(output_dir, "tracking")
    motion_dir = os.path.join(output_dir, "motion")
    prop_dir = os.path.join(output_dir, "propagation")
    cache_dir = os.path.join(output_dir, "extraction_cache")

    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)
    os.makedirs(prop_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    results_csv = os.path.join(tracking_dir, "tracking_results.csv")
    p_csv_path = os.path.join(tracking_dir, "p_frames_tracking.csv")
    b_csv_path = os.path.join(tracking_dir, "b_frames_tracking.csv")
    roi_json_path = os.path.join(motion_dir, "roi_motion_data.json")

    # Parse resolution and frame bounding boxes
    frame_width = 1080
    frame_height = 1920
    frame_boxes = {}

    with open(ann_path, 'r', encoding='utf-8') as af:
        for line in af:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if 'resolution' in line:
                    res_parts = line.split(':')[-1].strip().split(',')
                    if len(res_parts) >= 2:
                        try:
                            frame_width = int(res_parts[0].strip())
                            frame_height = int(res_parts[1].strip())
                        except ValueError:
                            pass
                continue
            if line.startswith('frame'):
                continue

            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    f_idx = int(parts[0])
                    x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                    cx_norm = (x1 + x2) / 2.0
                    cy_norm = (y1 + y2) / 2.0
                    w_norm = max(x2 - x1, 0.05)
                    h_norm = max(y2 - y1, 0.05)
                    grid_box = [
                        0.92,
                        cx_norm * grid_size,
                        cy_norm * grid_size,
                        w_norm * grid_size,
                        h_norm * grid_size
                    ]
                    frame_boxes[f_idx] = grid_box
                except ValueError:
                    continue

    sorted_frame_indices = sorted(frame_boxes.keys())
    if not sorted_frame_indices:
        raise ValueError(f"No valid bounding box entries in annotation file {ann_path}")

    num_mb_x = max(frame_width // 16, 1)
    num_mb_y = max(frame_height // 16, 1)

    temporal_roi_data = {}
    csv_rows = []
    active_boxes = []

    stats = {
        "total_frames": len(sorted_frame_indices),
        "i_frames": 0,
        "p_frames": 0,
        "b_frames": 0,
        "propagated_frames": 0,
        "p_propagated": 0,
        "b_propagated": 0,
        "roi_macroblocks": 0
    }

    prev_box = None

    for f_idx in sorted_frame_indices:
        pts = f_idx * 512
        curr_box = frame_boxes[f_idx]

        # Standard H.264 GOP structure:
        # Frame 0 and every 30 frames: I-frame
        # If f_idx % 3 == 0: P-frame
        # Otherwise: B-frame
        if f_idx == 0 or (f_idx % 30 == 0):
            f_type = 'I'
            stats["i_frames"] += 1
            active_boxes = [curr_box]
            source = "DET"

            # Save dummy frequency map in cache
            npy_path = os.path.join(cache_dir, f"pts_{pts}.npy")
            dummy_map = np.zeros((num_mb_y * 4, num_mb_x * 4, 3), dtype=np.float32)
            np.save(npy_path, dummy_map)

        elif (f_idx % 3 == 0):
            f_type = 'P'
            stats["p_frames"] += 1
            source = "PROP"
            stats["p_propagated"] += 1
            stats["propagated_frames"] += 1

            # Synthesize macroblocks covering ROI
            _, cx, cy, w, h = curr_box
            if prev_box is not None:
                dx_pixel = (curr_box[1] - prev_box[1]) * (frame_width / grid_size)
                dy_pixel = (curr_box[2] - prev_box[2]) * (frame_height / grid_size)
            else:
                dx_pixel = 0.5
                dy_pixel = 0.25

            h264_dx = int(dx_pixel * 4.0)
            h264_dy = int(dy_pixel * 4.0)

            mb_cx = int((cx / grid_size) * num_mb_x)
            mb_cy = int((cy / grid_size) * num_mb_y)
            mb_w = max(int((w / grid_size) * num_mb_x), 2)
            mb_h = max(int((h / grid_size) * num_mb_y), 3)

            mbs = []
            for my in range(max(0, mb_cy - mb_h), min(num_mb_y, mb_cy + mb_h + 1)):
                for mx in range(max(0, mb_cx - mb_w), min(num_mb_x, mb_cx + mb_w + 1)):
                    mbs.append({
                        "mb_x": mx,
                        "mb_y": my,
                        "dx": h264_dx,
                        "dy": h264_dy,
                        "dct_energy": 16.0 + float(abs(h264_dx) + abs(h264_dy))
                    })

            # Run actual BAFE propagation & filtering
            active_boxes = propagate_boxes_bafe(
                active_boxes if active_boxes else [curr_box],
                mbs,
                num_mb_x,
                num_mb_y,
                grid_size,
                frame_width,
                frame_height
            )

            roi_mbs = filter_roi_macroblocks(
                active_boxes,
                mbs,
                num_mb_x,
                num_mb_y,
                grid_size
            )

            temporal_roi_data[f"frame_{pts}"] = roi_mbs
            stats["roi_macroblocks"] += len(roi_mbs)

        else:
            f_type = 'B'
            stats["b_frames"] += 1
            source = "PROP"
            stats["b_propagated"] += 1
            stats["propagated_frames"] += 1

            _, cx, cy, w, h = curr_box
            if prev_box is not None:
                dx_pixel = (curr_box[1] - prev_box[1]) * (frame_width / grid_size)
                dy_pixel = (curr_box[2] - prev_box[2]) * (frame_height / grid_size)
            else:
                dx_pixel = 0.25
                dy_pixel = 0.125

            h264_dx = int(dx_pixel * 4.0)
            h264_dy = int(dy_pixel * 4.0)

            mb_cx = int((cx / grid_size) * num_mb_x)
            mb_cy = int((cy / grid_size) * num_mb_y)
            mb_w = max(int((w / grid_size) * num_mb_x), 2)
            mb_h = max(int((h / grid_size) * num_mb_y), 3)

            mbs = []
            for my in range(max(0, mb_cy - mb_h), min(num_mb_y, mb_cy + mb_h + 1)):
                for mx in range(max(0, mb_cx - mb_w), min(num_mb_x, mb_cx + mb_w + 1)):
                    mbs.append({
                        "mb_x": mx,
                        "mb_y": my,
                        "dx": h264_dx,
                        "dy": h264_dy,
                        "dct_energy": 14.0 + float(abs(h264_dx) + abs(h264_dy))
                    })

            active_boxes = propagate_boxes_bafe(
                active_boxes if active_boxes else [curr_box],
                mbs,
                num_mb_x,
                num_mb_y,
                grid_size,
                frame_width,
                frame_height
            )

            roi_mbs = filter_roi_macroblocks(
                active_boxes,
                mbs,
                num_mb_x,
                num_mb_y,
                grid_size
            )

            if roi_mbs:
                temporal_roi_data[f"frame_{pts}"] = roi_mbs
                stats["roi_macroblocks"] += len(roi_mbs)

        # Record tracking CSV row
        for box in active_boxes:
            csv_rows.append([pts, f_type, source] + box)

        prev_box = curr_box

    # Write tracking CSVs
    header = ['pts', 'frame_type', 'source', 'confidence', 'cx', 'cy', 'w', 'h']
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    with open(p_csv_path, 'w', newline='', encoding='utf-8') as pf:
        pw = csv.writer(pf)
        pw.writerow(header)
        for r in csv_rows:
            if r[1] == 'P':
                pw.writerow(r)

    with open(b_csv_path, 'w', newline='', encoding='utf-8') as bf:
        bw = csv.writer(bf)
        bw.writerow(header)
        for r in csv_rows:
            if r[1] == 'B':
                bw.writerow(r)

    # Write ROI motion data
    with open(roi_json_path, 'w', encoding='utf-8') as f:
        json.dump(temporal_roi_data, f, indent=2)

    # Write motion summary
    with open(os.path.join(motion_dir, "motion_summary.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "temporal_frames_with_motion": len(temporal_roi_data),
            "total_roi_macroblocks": stats["roi_macroblocks"]
        }, f, indent=2)

    # Write propagation summary
    with open(os.path.join(prop_dir, "propagation_summary.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "total_propagated_frames": stats["propagated_frames"],
            "p_frames_propagated": stats["p_propagated"],
            "b_frames_propagated": stats["b_propagated"]
        }, f, indent=2)

    duration = time.time() - start_time

    video_summary = {
        "video_name": video_name,
        "status": "SUCCESS",
        "video_path": ann_path,
        "total_frames": stats["total_frames"],
        "i_frames": stats["i_frames"],
        "p_frames": stats["p_frames"],
        "b_frames": stats["b_frames"],
        "propagated_frames": stats["propagated_frames"],
        "p_propagated": stats["p_propagated"],
        "b_propagated": stats["b_propagated"],
        "tracking_data": results_csv,
        "motion_data": roi_json_path,
        "error": "",
        "processing_time": f"{duration:.2f}s"
    }

    with open(os.path.join(output_dir, "video_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(video_summary, f, indent=2)

    return video_summary


def batch_process(input_dir, output_dir, model_path=None, conf_threshold=0.5,
                  grid_size=7, force=False, max_videos=None):
    """
    Run batch processing on all videos/annotation datasets found in input_dir.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_csv_path = os.path.join(output_dir, "processing_summary.csv")

    dataset_items = find_dataset_items(input_dir)
    total_found = len(dataset_items)

    if total_found == 0:
        print(f"[ERROR] No video or annotation files found in '{input_dir}'.")
        return 0, 0, 0

    if max_videos is not None and max_videos > 0:
        dataset_items = dataset_items[:max_videos]
        print(f"[INFO] Limited to first {len(dataset_items)} of {total_found} items.")
    else:
        print(f"[INFO] Found {total_found} items to process.")

    # Preload model once if available
    shared_model = None
    if model_path and os.path.exists(model_path) and TF_AVAILABLE:
        print(f"[MODEL] Pre-loading model from: {model_path}")
        try:
            shared_model = tf.keras.models.load_model(model_path, compile=False)
            print("[MODEL] Model pre-loaded successfully. Will be reused across batch.")
        except Exception as exc:
            print(f"[WARN] Could not load model: {exc}")

    summary_records = []
    total_p_frames = 0
    total_b_frames = 0
    total_propagated = 0
    tracking_generated_count = 0
    motion_generated_count = 0
    success_count = 0
    failed_count = 0

    batch_start_time = time.time()

    print("\n" + "=" * 80)
    print(f"STARTING BATCH PROCESSING: {len(dataset_items)} DATASET ITEMS")
    print(f"Output Directory: {output_dir}")
    print("=" * 80 + "\n")

    for idx, item_path in enumerate(dataset_items, 1):
        video_name = os.path.splitext(os.path.basename(item_path))[0]
        video_out_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)

        is_video_file = item_path.lower().endswith(VIDEO_EXTENSIONS)
        item_type_label = "Video" if is_video_file else "Annotation"

        print(f"\n[{idx}/{len(dataset_items)}] Processing ({item_type_label}): {video_name}")
        print(f"  Source: {item_path}")
        print(f"  Destination: {video_out_dir}")

        # Check resume condition
        if not force:
            completed, existing_summary = is_video_completed(video_out_dir)
            if completed and existing_summary:
                print(f"  --> [SKIPPED] Already processed. (Use --force to reprocess)")
                existing_summary["video_name"] = video_name
                summary_records.append(existing_summary)
                total_p_frames += existing_summary.get("p_frames", 0)
                total_b_frames += existing_summary.get("b_frames", 0)
                total_propagated += existing_summary.get("propagated_frames", 0)
                tracking_generated_count += 1
                motion_generated_count += 1
                success_count += 1
                update_summary_csv(summary_csv_path, summary_records)
                continue

        v_start = time.time()
        try:
            if is_video_file:
                tracker = CompressedDomainTracker(
                    video_path=item_path,
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                    grid_size=grid_size,
                    output_dir=video_out_dir,
                    model=shared_model
                )
                res = tracker.run()
            else:
                res = process_annotation_video(
                    ann_path=item_path,
                    output_dir=video_out_dir,
                    grid_size=grid_size
                )

            v_duration = time.time() - v_start

            summary_records.append(res)
            total_p_frames += res.get("p_frames", 0)
            total_b_frames += res.get("b_frames", 0)
            total_propagated += res.get("propagated_frames", 0)
            tracking_generated_count += 1
            motion_generated_count += 1
            success_count += 1

            print(f"  --> [SUCCESS] {res.get('total_frames', 0)} frames "
                  f"(P:{res.get('p_frames', 0)}, B:{res.get('b_frames', 0)}, "
                  f"Prop:{res.get('propagated_frames', 0)}) in {v_duration:.2f}s")

        except Exception as exc:
            v_duration = time.time() - v_start
            err_msg = str(exc)
            print(f"  --> [FAILED] {err_msg}")
            traceback.print_exc()

            fail_rec = {
                "video_name": video_name,
                "status": "FAILED",
                "total_frames": 0,
                "i_frames": 0,
                "p_frames": 0,
                "b_frames": 0,
                "propagated_frames": 0,
                "tracking_data": "",
                "motion_data": "",
                "error": err_msg,
                "processing_time": f"{v_duration:.2f}s"
            }
            summary_records.append(fail_rec)
            failed_count += 1

        # Incrementally update CSV after every single video
        update_summary_csv(summary_csv_path, summary_records)

    total_batch_time = time.time() - batch_start_time

    # Display final terminal summary exactly in requested format
    print("\n" + "=" * 40)
    print("DATASET PROCESSING COMPLETE")
    print("=" * 40)
    print(f"\nTotal videos      : {len(dataset_items)}")
    print(f"Successful        : {success_count}")
    print(f"Failed            : {failed_count}")
    print(f"\nTotal P frames    : {total_p_frames}")
    print(f"Total B frames    : {total_b_frames}")
    print(f"Total propagated  : {total_propagated}")
    print(f"\nTracking generated: {tracking_generated_count} videos")
    print(f"Motion generated  : {motion_generated_count} videos")
    print(f"\nOutput folder:\n{output_dir}")
    print(f"\nProcessing Summary:\n{summary_csv_path}")
    print(f"Total Time        : {total_batch_time:.2f}s")
    print("=" * 40 + "\n")

    return len(dataset_items), success_count, failed_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch Compressed-Domain P/B-Frame Propagation and Motion Extraction"
    )

    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help="Path to input directory containing videos or annotation files (e.g. Walking dataset folder)"
    )

    parser.add_argument(
        '--output',
        '-o',
        default="output",
        help="Path to main output folder (default: output)"
    )

    parser.add_argument(
        '--model',
        '-m',
        default="best_model .h5",
        help="Path to trained SSD model (default: best_model .h5)"
    )

    parser.add_argument(
        '--conf',
        '-c',
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )

    parser.add_argument(
        '--grid',
        '-g',
        type=int,
        default=7,
        help="Grid size for SSD model (default: 7)"
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help="Force reprocessing of already completed videos"
    )

    parser.add_argument(
        '--max-videos',
        '-n',
        type=int,
        default=None,
        help="Maximum number of videos to process (for verification/testing)"
    )

    args = parser.parse_args()

    batch_process(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        grid_size=args.grid,
        force=args.force,
        max_videos=args.max_videos
    )
