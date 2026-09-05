#!/usr/bin/env python3
"""
Print Complete Accuracy Benchmark Report to Terminal
Displays:
  1. Compressed-Domain Object Detector & BAFE Propagation Metrics (51,365 frames)
  2. Action Recognition Deep Learning Model (Bi-LSTM) Evaluation & Per-Class Accuracy
  3. Confusion Matrix & Summary
"""

import os
import sys
import json

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def print_report():
    output_dir = "output"
    det_path = os.path.join(output_dir, "detector_accuracy_report.json")
    act_path = os.path.join(output_dir, "action_recognition_metrics.json")

    print("\n" + "=" * 74)
    print("      COMPRESSED-DOMAIN MACHINE LEARNING ACCURACY BENCHMARK REPORT      ")
    print("=" * 74 + "\n")

    # 1. OBJECT DETECTOR & BAFE TRACKER BENCHMARK
    if os.path.exists(det_path):
        with open(det_path, 'r', encoding='utf-8') as f:
            det = json.load(f)

        m = det.get("metrics", {})
        fc = det.get("frame_counts", {})
        print("+-" + "-" * 70 + "-+")
        print("|  1. COMPRESSED-DOMAIN OBJECT DETECTOR & BAFE PROPAGATION ACCURACY      |")
        print("+-" + "-" * 70 + "-+")
        print(f"|  * Total Videos Evaluated   : {det.get('total_videos_evaluated', 171):>6d} videos                             |")
        print(f"|  * Total Frames Evaluated   : {det.get('total_frames_evaluated', 51365):>6,d} frames                             |")
        print(f"|  * Overall Mean IoU         : {m.get('mean_iou_overall', 0)*100:>6.2f}%                                    |")
        print("|                                                                        |")
        print("|  [FRAME-TYPE ACCURACY & PROPAGATION FIDELITY]                          |")
        print(f"|    - I-Frames (Detection Anchor) : {m.get('mean_iou_i_frames', 0)*100:>6.2f}% Mean IoU ({fc.get('i_frames', 0):>6,d} frames)    |")
        print(f"|    - P-Frames (Forward Prop)     : {m.get('mean_iou_p_frames', 0)*100:>6.2f}% Mean IoU ({fc.get('p_frames', 0):>6,d} frames)    |")
        print(f"|    - B-Frames (Bidirectional)    : {m.get('mean_iou_b_frames', 0)*100:>6.2f}% Mean IoU ({fc.get('b_frames', 0):>6,d} frames)    |")
        retention = (m.get('mean_iou_bafe_propagation', 0) / max(m.get('mean_iou_detection_anchor', 1), 1e-6)) * 100
        print(f"|    - BAFE Propagation Retention  : {retention:>6.2f}% of Detection Accuracy       |")
        print("|                                                                        |")
        print("|  [DETECTION ACCURACY BENCHMARKS (PASCAL VOC & COCO)]                   |")
        p50 = m.get('precision_at_50', 0) * 100
        r50 = m.get('recall_at_50', 0) * 100
        print(f"|    - mAP @ 0.50 IoU  : {m.get('mAP_50', 0)*100:>6.2f}%  (Precision: {p50:.2f}%, Recall: {r50:.2f}%)   |")
        print(f"|    - F1-Score @ 0.50 : {m.get('f1_score_at_50', 0)*100:>6.2f}%                                            |")
        print(f"|    - mAP @ 0.75 IoU  : {m.get('mAP_75', 0)*100:>6.2f}%  (Strict COCO threshold)                 |")
        print(f"|    - mAP @ 0.50:0.95 : {m.get('mAP_50_95', 0)*100:>6.2f}%  (COCO Average Precision)               |")
        print("+-" + "-" * 70 + "-+\n")

    # 2. ACTION RECOGNITION DEEP LEARNING MODEL (BI-LSTM)
    if os.path.exists(act_path):
        with open(act_path, 'r', encoding='utf-8') as f:
            act = json.load(f)

        tm = act.get("test_metrics", {})
        pcm = act.get("per_class_metrics", {})
        classes = act.get("classes", [])
        cm = act.get("confusion_matrix", [])

        print("+-" + "-" * 70 + "-+")
        print("|  2. ACTION RECOGNITION DEEP LEARNING MODEL (Bi-LSTM)                   |")
        print("+-" + "-" * 70 + "-+")
        print(f"|  * Architecture     : {act.get('model_architecture', 'Bi-LSTM'):<48s} |")
        print(f"|  * Input Features   : 60 Timesteps x 8 Motion Features (dx, dy, energy, area)  |")
        print(f"|  * Number of Classes: {act.get('num_classes', 5):<48d} |")
        print("|                                                                        |")
        print("|  [GLOBAL TEST SET PERFORMANCE]                                         |")
        print(f"|    - Overall Test Accuracy : {tm.get('test_accuracy', 0)*100:>6.2f}%  (Baseline Random Guess: 20.00%) |")
        print(f"|    - Test Loss             : {tm.get('test_loss', 0):>6.4f}                                     |")
        print(f"|    - Macro Precision       : {tm.get('macro_precision', 0)*100:>6.2f}%                                     |")
        print(f"|    - Macro Recall          : {tm.get('macro_recall', 0)*100:>6.2f}%                                     |")
        print(f"|    - Macro F1-Score        : {tm.get('macro_f1_score', 0)*100:>6.2f}%                                     |")
        print("|                                                                        |")
        print("|  [PER-CLASS ACCURACY BREAKDOWN]                                        |")
        print("|  " + f"{'Class Name':<28s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<10s}" + "   |")
        print("|  " + "-" * 66 + "   |")
        for c in classes:
            stats = pcm.get(c, {})
            p = stats.get("precision", 0) * 100
            r = stats.get("recall", 0) * 100
            f1 = stats.get("f1_score", 0) * 100
            print(f"|  {c:<28s} {p:>8.2f}%   {r:>8.2f}%   {f1:>8.2f}%" + "    |")
        print("|                                                                        |")
        print("|  [CONFUSION MATRIX]                                                    |")
        header_cols = " ".join([f"{c[:7]:>8s}" for c in classes])
        print(f"|  {'Act \\ Pred':<16s} {header_cols}   |")
        for i, row in enumerate(cm):
            row_str = " ".join([f"{val:>8d}" for val in row])
            print(f"|  {classes[i][:15]:<16s} {row_str}   |")
        print("+-" + "-" * 70 + "-+\n")

    print("+-" + "-" * 70 + "-+")
    print("|  SAVED BENCHMARK ARTIFACTS & MODEL WEIGHTS                             |")
    print("+-" + "-" * 70 + "-+")
    print("|  * Detector Accuracy Report: output/detector_accuracy_report.json       |")
    print("|  * Action Recognition Model: output/action_recognition_bilstm.pt        |")
    print("|  * Action Model Metrics    : output/action_recognition_metrics.json     |")
    print("|  * Confusion Matrix Plot   : output/action_confusion_matrix.png         |")
    print("|  * Complete Walkthrough    : walkthrough.md                             |")
    print("+-" + "-" * 70 + "-+\n")


if __name__ == '__main__':
    print_report()
