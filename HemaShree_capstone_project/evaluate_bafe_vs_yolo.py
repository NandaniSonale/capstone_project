"""
=============================================================================
evaluate_bafe_vs_yolo.py - BAFE Propagation vs YOLO Detection Benchmark
=============================================================================
Evaluates compressed-domain Box-Aligned Feature Extraction (BAFE) bounding box
propagation against full-pixel YOLOv8 person detections across multiple videos.

Metrics Evaluated:
  1. Intersection over Union (IoU): Overlap between BAFE box and YOLO box.
  2. Center Location Error (CLE): Euclidean pixel distance between box centers.
  3. Success Rate (IoU >= 0.50): Percentage of frames tracked successfully.
  4. Precision Rate (CLE <= 50 px): Percentage of frames within tolerance.
  5. P-Frame vs B-Frame Propagation Analysis: Evaluating forward vs bi-directional
     compressed-domain motion vectors.
=============================================================================
"""

import os
import sys
import json
import argparse
import time
import cv2
import av
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


def compute_iou(box1, box2):
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0


def compute_center_error(box1, box2):
    """Computes Euclidean distance between centers of two boxes [x1, y1, x2, y2]."""
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0

    return float(np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2))


def evaluate_single_video(video_path, csv_path, yolo_model, grid_size=7.0, max_frames=60, save_visuals_dir=None):
    """
    Evaluates BAFE propagation against YOLO detection for a single video.
    """
    if not os.path.exists(video_path) or not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.empty or 'pts' not in df.columns:
        return None

    container = av.open(video_path)
    stream = container.streams.video[0]
    w_img, h_img = stream.width, stream.height

    frame_records = []
    prev_actor_box = None
    processed_count = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for frame in container.decode(video=0):
        pts = frame.pts
        sub = df[df['pts'] == pts]
        if sub.empty:
            continue

        r = sub.iloc[0]
        # Convert BAFE 7x7 grid coordinates to full-frame pixel coordinates
        cx_rel = r['cx'] / grid_size
        cy_rel = r['cy'] / grid_size
        w_rel = r['w'] / grid_size
        h_rel = r['h'] / grid_size

        bafe_box = [
            float((cx_rel - w_rel / 2.0) * w_img),
            float((cy_rel - h_rel / 2.0) * h_img),
            float((cx_rel + w_rel / 2.0) * w_img),
            float((cy_rel + h_rel / 2.0) * h_img)
        ]

        # Decode RGB frame for YOLO detection
        img_bgr = frame.to_ndarray(format='bgr24')

        # Run YOLOv8 for person detection (class 0 = person)
        results = yolo_model(img_bgr, classes=[0], verbose=False)
        detected_boxes = results[0].boxes.xyxy.cpu().numpy()

        f_type = str(r.get('frame_type', 'P'))
        source = str(r.get('source', 'PROP'))

        if len(detected_boxes) > 0:
            # Find the best matching person detection
            if prev_actor_box is not None:
                # Match closest to previous track position
                ious = [compute_iou(prev_actor_box, b) for b in detected_boxes]
                best_idx = int(np.argmax(ious))
            else:
                # Match highest overlap with initial BAFE box
                ious = [compute_iou(bafe_box, b) for b in detected_boxes]
                best_idx = int(np.argmax(ious))

            yolo_box = detected_boxes[best_idx].tolist()
            prev_actor_box = yolo_box

            iou = compute_iou(bafe_box, yolo_box)
            cle = compute_center_error(bafe_box, yolo_box)

            rec = {
                'pts': int(pts),
                'frame_type': f_type,
                'source': source,
                'bafe_box': bafe_box,
                'yolo_box': yolo_box,
                'iou': iou,
                'cle': cle,
                'success': bool(iou >= 0.50),
                'precise': bool(cle <= 50.0)
            }
            frame_records.append(rec)

            # Save annotated visual sample for key frames (e.g., first 5 frames, then every 10 frames)
            if save_visuals_dir and (processed_count < 3 or processed_count % 10 == 0):
                vis_img = img_bgr.copy()
                # Draw YOLO detection in GREEN
                yx1, yy1, yx2, yy2 = [int(v) for v in yolo_box]
                cv2.rectangle(vis_img, (yx1, yy1), (yx2, yy2), (0, 255, 0), 2)
                cv2.putText(vis_img, "YOLO Reference", (yx1, max(yy1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                # Draw BAFE propagation in CYAN
                bx1, by1, bx2, by2 = [int(v) for v in bafe_box]
                cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
                cv2.putText(vis_img, f"BAFE {f_type} ({source})", (bx1, min(by2 + 20, h_img - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                # Overlay metric HUD badge
                hud_text = f"Frame {pts} | Type: {f_type} ({source}) | IoU: {iou:.3f} | CLE: {cle:.1f}px"
                cv2.rectangle(vis_img, (15, 15), (550, 55), (0, 0, 0), -1)
                cv2.putText(vis_img, hud_text, (25, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

                vis_path = os.path.join(save_visuals_dir, f"{video_name}_frame_{pts:06d}.jpg")
                cv2.imwrite(vis_path, vis_img)

        processed_count += 1
        if processed_count >= max_frames:
            break

    if not frame_records:
        return None

    # Calculate overall and stratified metrics
    ious = [f['iou'] for f in frame_records]
    cles = [f['cle'] for f in frame_records]

    p_ious = [f['iou'] for f in frame_records if f['frame_type'] == 'P']
    b_ious = [f['iou'] for f in frame_records if f['frame_type'] == 'B']
    prop_ious = [f['iou'] for f in frame_records if f['source'] == 'PROP']

    summary = {
        'video_name': video_name,
        'total_evaluated_frames': len(frame_records),
        'mean_iou': float(np.mean(ious)),
        'std_iou': float(np.std(ious)),
        'median_iou': float(np.median(ious)),
        'prop_mean_iou': float(np.mean(prop_ious)) if prop_ious else 0.0,
        'p_frame_mean_iou': float(np.mean(p_ious)) if p_ious else 0.0,
        'b_frame_mean_iou': float(np.mean(b_ious)) if b_ious else 0.0,
        'mean_cle_px': float(np.mean(cles)),
        'success_rate_iou_05': float(np.mean([1 if x >= 0.50 else 0 for x in ious]) * 100.0),
        'precision_rate_cle_50': float(np.mean([1 if x <= 50.0 else 0 for x in cles]) * 100.0),
        'min_iou': float(np.min(ious)),
        'max_iou': float(np.max(ious))
    }

    return summary, frame_records


def run_benchmark(num_videos=10, output_dir="evaluation_results", max_frames_per_video=50):
    """
    Runs BAFE vs YOLO comparison across a diverse set of multiple videos.
    """
    os.makedirs(output_dir, exist_ok=True)
    visuals_dir = os.path.join(output_dir, "visual_comparisons")
    os.makedirs(visuals_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("      BAFE PROPAGATION vs. YOLOv8 OBJECT DETECTOR BENCHMARK")
    print("=" * 75)
    print(f"  Target Videos to Test : {num_videos}")
    print(f"  Frames per Video      : Up to {max_frames_per_video} frames")
    print(f"  Results Destination   : {output_dir}")
    print(f"  Visual Annotations    : {visuals_dir}\n")

    print("[INIT] Loading YOLOv8n detector weights for ground truth comparison...")
    yolo_model = YOLO('yolov8n')

    # Candidate videos across Walking and Walking While Reading Book
    candidate_list = [
        ('Human Activity Recognition - Video Dataset/Walking/Walking (1).mp4', 'output/Walking (1)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking/Walking (2).mp4', 'output/Walking (2)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking/Walking (3).mp4', 'output/Walking (3)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking/Walking (4).mp4', 'output/Walking (4)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking/Walking (5).mp4', 'output/Walking (5)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking/Walking (10).mp4', 'output/Walking (10)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking/Walking (23).mp4', 'output/Walking (23)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking While Reading Book/Walking While Reading Book (1).mp4', 'output/Walking While Reading Book (1)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking While Reading Book/Walking While Reading Book (2).mp4', 'output/Walking While Reading Book (2)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking While Reading Book/Walking While Reading Book (3).mp4', 'output/Walking While Reading Book (3)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking While Reading Book/Walking While Reading Book (4).mp4', 'output/Walking While Reading Book (4)/tracking/tracking_results.csv'),
        ('Human Activity Recognition - Video Dataset/Walking While Reading Book/Walking While Reading Book (10).mp4', 'output/Walking While Reading Book (10)/tracking/tracking_results.csv'),
    ]

    selected_candidates = candidate_list[:num_videos]

    all_summaries = []
    all_frames_data = {}

    start_time = time.time()

    for idx, (v_path, c_path) in enumerate(selected_candidates, 1):
        v_name = os.path.splitext(os.path.basename(v_path))[0]
        print(f"[{idx:02d}/{len(selected_candidates):02d}] Evaluating '{v_name}'...", flush=True)

        res = evaluate_single_video(
            v_path, c_path, yolo_model,
            max_frames=max_frames_per_video,
            save_visuals_dir=visuals_dir
        )

        if res is not None:
            summary, frames = res
            all_summaries.append(summary)
            all_frames_data[v_name] = frames

            print(f"     -> Frames: {summary['total_evaluated_frames']:2d} | "
                  f"Mean IoU: {summary['mean_iou']:.3f} | "
                  f"P-IoU: {summary['p_frame_mean_iou']:.3f} | "
                  f"B-IoU: {summary['b_frame_mean_iou']:.3f} | "
                  f"Success (IoU>=0.5): {summary['success_rate_iou_05']:.1f}% | "
                  f"Mean CLE: {summary['mean_cle_px']:.1f}px")
        else:
            print(f"     [WARN] Could not evaluate {v_name}")

    elapsed = time.time() - start_time
    print(f"\nCompleted evaluation in {elapsed:.1f}s.\n")

    if not all_summaries:
        print("[ERROR] No videos were successfully evaluated.")
        return

    # Convert summaries to DataFrame
    df_summary = pd.DataFrame(all_summaries)
    csv_out_path = os.path.join(output_dir, "bafe_vs_yolo_summary.csv")
    df_summary.to_csv(csv_out_path, index=False)
    print(f"[REPORT] Summary CSV saved: {csv_out_path}")

    json_out_path = os.path.join(output_dir, "bafe_vs_yolo_detailed.json")
    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump({'summaries': all_summaries, 'frames': all_frames_data}, f, indent=2)
    print(f"[REPORT] Detailed JSON saved: {json_out_path}")

    # Generate Comprehensive Plots
    generate_comparison_plots(df_summary, all_frames_data, output_dir)

    # Print Final Macro Metrics Table
    print_macro_report(df_summary)


def generate_comparison_plots(df_summary, all_frames_data, output_dir):
    """
    Generates high-resolution comparative figures.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Multi-Video IoU and CLE Metrics Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names_short = [n.replace("Walking While Reading Book", "W-Book") for n in df_summary['video_name']]
    x = np.arange(len(names_short))
    width = 0.35

    # Plot 1: Mean IoU (P vs B frames)
    axes[0].bar(x - width/2, df_summary['p_frame_mean_iou'], width, label='P-Frames (Forward MV)', color='#1f77b4', alpha=0.9)
    axes[0].bar(x + width/2, df_summary['b_frame_mean_iou'], width, label='B-Frames (Bi-directional MV)', color='#2ca02c', alpha=0.9)
    axes[0].axhline(0.50, color='red', linestyle='--', linewidth=1.5, label='PASCAL VOC Threshold (0.50)')
    axes[0].set_ylabel('Intersection over Union (IoU)', fontsize=12)
    axes[0].set_title('BAFE vs YOLO: Mean IoU per Video (P vs B Frames)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names_short, rotation=35, ha='right', fontsize=9)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc='lower right', frameon=True)

    # Plot 2: Tracking Success Rate and Precision Rate
    axes[1].bar(x - width/2, df_summary['success_rate_iou_05'], width, label='Success Rate (IoU >= 0.50)', color='#ff7f0e', alpha=0.9)
    axes[1].bar(x + width/2, df_summary['precision_rate_cle_50'], width, label='Precision Rate (CLE <= 50px)', color='#9467bd', alpha=0.9)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_title('BAFE Propagation Tracking Accuracy (%)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names_short, rotation=35, ha='right', fontsize=9)
    axes[1].set_ylim(0.0, 105.0)
    axes[1].legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bafe_vs_yolo_comparison_barchart.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Bar chart saved: {plot_path}")

    # Figure 2: Temporal IoU Trajectories over consecutive frames
    plt.figure(figsize=(12, 6))
    for v_name, frames in list(all_frames_data.items())[:6]:
        v_short = v_name.replace("Walking While Reading Book", "W-Book")
        pts_list = [f['pts'] for f in frames]
        iou_list = [f['iou'] for f in frames]
        plt.plot(pts_list, iou_list, marker='o', markersize=3, label=v_short, linewidth=1.8, alpha=0.85)

    plt.axhline(0.50, color='red', linestyle='--', linewidth=1.8, label='Success Threshold (IoU = 0.50)')
    plt.xlabel('Video Presentation Timestamp (PTS)', fontsize=12)
    plt.ylabel('Intersection over Union (IoU)', fontsize=12)
    plt.title('BAFE Propagation Stability over Consecutive Frames vs YOLOv8 Reference', fontsize=13, fontweight='bold')
    plt.ylim(0.0, 1.05)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    trajectory_path = os.path.join(output_dir, "bafe_vs_yolo_temporal_trajectories.png")
    plt.savefig(trajectory_path, dpi=300)
    plt.close()
    print(f"[PLOT] Temporal trajectories saved: {trajectory_path}")


def print_macro_report(df_summary):
    """
    Prints a formatted summary table of benchmark results.
    """
    avg_iou = df_summary['mean_iou'].mean()
    avg_p_iou = df_summary['p_frame_mean_iou'].mean()
    avg_b_iou = df_summary['b_frame_mean_iou'].mean()
    avg_cle = df_summary['mean_cle_px'].mean()
    avg_success = df_summary['success_rate_iou_05'].mean()
    avg_precision = df_summary['precision_rate_cle_50'].mean()
    total_frames = df_summary['total_evaluated_frames'].sum()

    print("\n" + "=" * 75)
    print("         MACRO BENCHMARK SUMMARY (BAFE PROPAGATION vs. YOLOv8)")
    print("=" * 75)
    print(f"{'Video Name':<32} | {'Frames':^7} | {'Mean IoU':^10} | {'P-IoU':^8} | {'B-IoU':^8} | {'Success%':^9}")
    print("-" * 75)

    for _, row in df_summary.iterrows():
        v_short = row['video_name']
        print(f"{v_short:<32} | {row['total_evaluated_frames']:^7d} | {row['mean_iou']:^10.3f} | "
              f"{row['p_frame_mean_iou']:^8.3f} | {row['b_frame_mean_iou']:^8.3f} | {row['success_rate_iou_05']:^8.1f}%")

    print("=" * 75)
    print(f"{'OVERALL AVERAGE':<32} | {total_frames:^7d} | {avg_iou:^10.3f} | "
          f"{avg_p_iou:^8.3f} | {avg_b_iou:^8.3f} | {avg_success:^8.1f}%")
    print("=" * 75)
    print(f"  Overall Mean IoU               : {avg_iou:.4f} (Benchmark: >= 0.50 is standard)")
    print(f"  P-Frame Mean IoU (Forward)     : {avg_p_iou:.4f}")
    print(f"  B-Frame Mean IoU (Bi-dir)      : {avg_b_iou:.4f}")
    print(f"  Overall Center Location Error  : {avg_cle:.2f} pixels")
    print(f"  Tracking Success Rate (IoU>=0.5): {avg_success:.2f}%")
    print(f"  Precision Rate (CLE <= 50px)   : {avg_precision:.2f}%")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate BAFE Propagation vs YOLOv8 Object Detections.")
    parser.add_argument('--num_videos', '-n', type=int, default=10, help="Number of videos to evaluate")
    parser.add_argument('--max_frames', '-f', type=int, default=50, help="Max frames to evaluate per video")
    parser.add_argument('--output_dir', '-o', type=str, default="evaluation_results", help="Output directory")

    args = parser.parse_args()

    run_benchmark(
        num_videos=args.num_videos,
        output_dir=args.output_dir,
        max_frames_per_video=args.max_frames
    )
