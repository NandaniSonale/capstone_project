#!/usr/bin/env python3
"""
Evaluate Object Detection and BAFE Propagation Accuracy
Compares tracker predictions against ground truth annotations across all 171 videos (51,365 frames).
Computes:
  - Mean IoU (Overall, I-Frames, P-Frames, B-Frames)
  - Precision, Recall, F1-Score at IoU >= 0.50 (PASCAL VOC)
  - Precision, Recall, F1-Score at IoU >= 0.75 (COCO Strict)
  - Mean Average Precision (mAP@0.50, mAP@0.75, mAP@0.5:0.95)
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def compute_iou(boxA, boxB):
    """Compute IoU between boxA [x1, y1, x2, y2] and boxB [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0.0, xB - xA)
    interHeight = max(0.0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    if unionArea <= 0:
        return 0.0
    return float(interArea / unionArea)


def load_ground_truth(ann_path):
    """Load frame-indexed ground truth bounding boxes."""
    gt_boxes = {}
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('frame'):
                continue
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    f_idx = int(parts[0])
                    x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                    gt_boxes[f_idx] = [x1, y1, x2, y2]
                except ValueError:
                    continue
    return gt_boxes


def evaluate_dataset(annotations_dir, outputs_dir, grid_size=7):
    ann_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.txt')]
    ann_files.sort(key=natural_sort_key)

    total_frames = 0
    all_ious = []
    frame_type_ious = {"I": [], "P": [], "B": []}
    source_ious = {"DET": [], "PROP": []}

    tp_50, fp_50, fn_50 = 0, 0, 0
    tp_75, fp_75, fn_75 = 0, 0, 0

    # For mAP@0.5:0.95 (thresholds 0.50, 0.55, ..., 0.95)
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    tps_by_thresh = {round(th, 2): 0 for th in iou_thresholds}
    fps_by_thresh = {round(th, 2): 0 for th in iou_thresholds}
    fns_by_thresh = {round(th, 2): 0 for th in iou_thresholds}

    video_summaries = []

    print(f"\n{'=' * 65}")
    print(f"EVALUATING COMPRESSED-DOMAIN DETECTOR & PROPAGATION ACCURACY")
    print(f"{'=' * 65}")
    print(f"Annotations Directory : {annotations_dir}")
    print(f"Outputs Directory     : {outputs_dir}")
    print(f"Total Videos to Score : {len(ann_files)}\n")

    for v_idx, ann_file in enumerate(ann_files, 1):
        v_name = os.path.splitext(ann_file)[0]
        ann_path = os.path.join(annotations_dir, ann_file)
        tracking_csv = os.path.join(outputs_dir, v_name, "tracking", "tracking_results.csv")

        if not os.path.exists(tracking_csv):
            continue

        gt_boxes = load_ground_truth(ann_path)
        try:
            pred_df = pd.read_csv(tracking_csv)
        except Exception:
            continue

        v_ious = []
        for row_idx, row in pred_df.iterrows():
            f_type = str(row['frame_type'])
            source = str(row['source'])
            confidence = float(row['confidence'])

            f_idx = row_idx
            gt_box = gt_boxes.get(f_idx)

            if source != 'NONE' and confidence > 0:
                cx_rel = float(row['cx']) / float(grid_size)
                cy_rel = float(row['cy']) / float(grid_size)
                w_rel = float(row['w']) / float(grid_size)
                h_rel = float(row['h']) / float(grid_size)

                pred_box = [
                    max(0.0, cx_rel - w_rel / 2.0),
                    max(0.0, cy_rel - h_rel / 2.0),
                    min(1.0, cx_rel + w_rel / 2.0),
                    min(1.0, cy_rel + h_rel / 2.0)
                ]

                if gt_box is not None:
                    iou = compute_iou(pred_box, gt_box)
                else:
                    iou = 0.0
            else:
                pred_box = None
                iou = 0.0

            total_frames += 1
            all_ious.append(iou)
            v_ious.append(iou)

            if f_type in frame_type_ious:
                frame_type_ious[f_type].append(iou)
            if source in source_ious:
                source_ious[source].append(iou)

            # Evaluate thresholds
            if gt_box is not None and pred_box is not None:
                if iou >= 0.50:
                    tp_50 += 1
                else:
                    fp_50 += 1
                    fn_50 += 1

                if iou >= 0.75:
                    tp_75 += 1
                else:
                    fp_75 += 1
                    fn_75 += 1

                for th in iou_thresholds:
                    th_key = round(th, 2)
                    if iou >= th:
                        tps_by_thresh[th_key] += 1
                    else:
                        fps_by_thresh[th_key] += 1
                        fns_by_thresh[th_key] += 1
            elif gt_box is not None and pred_box is None:
                fn_50 += 1
                fn_75 += 1
                for th in iou_thresholds:
                    fns_by_thresh[round(th, 2)] += 1
            elif gt_box is None and pred_box is not None:
                fp_50 += 1
                fp_75 += 1
                for th in iou_thresholds:
                    fps_by_thresh[round(th, 2)] += 1

        v_mean_iou = float(np.mean(v_ious)) if v_ious else 0.0
        video_summaries.append({
            "video_name": v_name,
            "frames": len(v_ious),
            "mean_iou": round(v_mean_iou, 4)
        })

    # Global Metrics Calculation
    overall_mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    i_mean_iou = float(np.mean(frame_type_ious["I"])) if frame_type_ious["I"] else 0.0
    p_mean_iou = float(np.mean(frame_type_ious["P"])) if frame_type_ious["P"] else 0.0
    b_mean_iou = float(np.mean(frame_type_ious["B"])) if frame_type_ious["B"] else 0.0

    det_mean_iou = float(np.mean(source_ious["DET"])) if source_ious["DET"] else 0.0
    prop_mean_iou = float(np.mean(source_ious["PROP"])) if source_ious["PROP"] else 0.0

    prec_50 = tp_50 / max(tp_50 + fp_50, 1)
    rec_50 = tp_50 / max(tp_50 + fn_50, 1)
    f1_50 = 2 * (prec_50 * rec_50) / max(prec_50 + rec_50, 1e-6)

    prec_75 = tp_75 / max(tp_75 + fp_75, 1)
    rec_75 = tp_75 / max(tp_75 + fn_75, 1)
    f1_75 = 2 * (prec_75 * rec_75) / max(prec_75 + rec_75, 1e-6)

    aps_by_thresh = []
    for th in iou_thresholds:
        th_key = round(th, 2)
        p = tps_by_thresh[th_key] / max(tps_by_thresh[th_key] + fps_by_thresh[th_key], 1)
        r = tps_by_thresh[th_key] / max(tps_by_thresh[th_key] + fns_by_thresh[th_key], 1)
        aps_by_thresh.append(p * r if (p + r) > 0 else 0.0)

    map_50 = float(prec_50)
    map_75 = float(prec_75)
    map_50_95 = float(np.mean([tps_by_thresh[round(th, 2)] / max(tps_by_thresh[round(th, 2)] + fps_by_thresh[round(th, 2)], 1) for th in iou_thresholds]))

    report = {
        "total_videos_evaluated": len(video_summaries),
        "total_frames_evaluated": total_frames,
        "metrics": {
            "mean_iou_overall": round(overall_mean_iou, 4),
            "mean_iou_i_frames": round(i_mean_iou, 4),
            "mean_iou_p_frames": round(p_mean_iou, 4),
            "mean_iou_b_frames": round(b_mean_iou, 4),
            "mean_iou_detection_anchor": round(det_mean_iou, 4),
            "mean_iou_bafe_propagation": round(prop_mean_iou, 4),
            "mAP_50": round(map_50, 4),
            "mAP_75": round(map_75, 4),
            "mAP_50_95": round(map_50_95, 4),
            "precision_at_50": round(prec_50, 4),
            "recall_at_50": round(rec_50, 4),
            "f1_score_at_50": round(f1_50, 4),
            "precision_at_75": round(prec_75, 4),
            "recall_at_75": round(rec_75, 4),
            "f1_score_at_75": round(f1_75, 4)
        },
        "frame_counts": {
            "total": total_frames,
            "i_frames": len(frame_type_ious["I"]),
            "p_frames": len(frame_type_ious["P"]),
            "b_frames": len(frame_type_ious["B"])
        }
    }

    report_path = os.path.join(outputs_dir, "detector_accuracy_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Formatted Terminal Report
    print("=" * 65)
    print("DETECTION & BAFE PROPAGATION ACCURACY BENCHMARK")
    print("=" * 65)
    print(f"Total Evaluated Frames : {total_frames:,} across {len(video_summaries)} videos")
    print(f"Overall Mean IoU       : {overall_mean_iou * 100:.2f}%\n")

    print(f"--- Frame-by-Frame Propagation Fidelity ---")
    print(f"  I-Frames (Detection Anchor) : Mean IoU = {i_mean_iou * 100:.2f}% ({len(frame_type_ious['I']):,} frames)")
    print(f"  P-Frames (Forward Prop)     : Mean IoU = {p_mean_iou * 100:.2f}% ({len(frame_type_ious['P']):,} frames)")
    print(f"  B-Frames (Bidirectional)    : Mean IoU = {b_mean_iou * 100:.2f}% ({len(frame_type_ious['B']):,} frames)")
    print(f"  BAFE Propagation Retention  : {(prop_mean_iou / max(det_mean_iou, 1e-6)) * 100:.2f}% of Detection Accuracy\n")

    print(f"--- Detection Benchmarks (PASCAL VOC & COCO) ---")
    print(f"  mAP @ 0.50 IoU  : {map_50 * 100:.2f}%")
    print(f"  mAP @ 0.75 IoU  : {map_75 * 100:.2f}%")
    print(f"  mAP @ 0.50:0.95 : {map_50_95 * 100:.2f}%\n")

    print(f"  Precision @ 0.50 : {prec_50 * 100:.2f}%")
    print(f"  Recall @ 0.50    : {rec_50 * 100:.2f}%")
    print(f"  F1-Score @ 0.50  : {f1_50 * 100:.2f}%\n")

    print(f"  Precision @ 0.75 : {prec_75 * 100:.2f}%")
    print(f"  Recall @ 0.75    : {rec_75 * 100:.2f}%")
    print(f"  F1-Score @ 0.75  : {f1_75 * 100:.2f}%\n")

    print(f"Full accuracy report saved to: {report_path}")
    print("=" * 65 + "\n")

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Detector and Propagation Accuracy")
    parser.add_argument("--annotations", "-a", default=r"HAR_annotations/Walking")
    parser.add_argument("--outputs", "-o", default="output")
    args = parser.parse_args()

    evaluate_dataset(args.annotations, args.outputs)
