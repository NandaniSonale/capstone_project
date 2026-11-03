"""Run complete-dataset or merged long-duration HAR/YOLO evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from har_yolo_benchmark import (
    ACTION_CLASSES,
    HAR_CLASSES,
    VIDEO_EXTENSIONS,
    HARClassifier,
    ProcessMemoryMonitor,
    action_features,
    add_detection_frame,
    classification_summary,
    detection_summary,
    hardware_metadata,
    model_profile,
    natural_key,
    read_annotation,
    write_json,
    yolo_profile,
)


ANNOTATION_PROVENANCE_WARNING = (
    "The repository's annotation-generation script uses YOLOv8n to create class-0 "
    "person boxes at confidence >0.5. Unless independent manual ground truth is "
    "confirmed, agreement metrics against these files are diagnostic only."
)


def discover_complete_videos(
    dataset_root: Path,
    annotations_root: Path,
    max_videos_per_class: int | None,
) -> list[dict[str, Any]]:
    videos = []
    for label in HAR_CLASSES:
        class_dir = dataset_root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dataset class directory missing: {class_dir}")
        class_videos = sorted(
            (path for path in class_dir.iterdir()
             if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS),
            key=lambda path: natural_key(path.name),
        )
        if max_videos_per_class is not None:
            class_videos = class_videos[:max_videos_per_class]
        for path in class_videos:
            annotation = annotations_root / label / f"{path.stem}.txt"
            videos.append(
                {
                    "video_id": f"{label}/{path.stem}",
                    "true_action": label,
                    "video_path": path,
                    "annotation_path": annotation if annotation.is_file() else None,
                    "segment_start": 0,
                    "segment_end_exclusive": None,
                }
            )
    return videos


def load_long_segments(manifest_path: Path, merged_video: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Segment manifest not found: {manifest_path}")
    if not merged_video.is_file():
        raise FileNotFoundError(f"Merged video not found: {merged_video}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        segments = list(csv.DictReader(stream))
    if not segments:
        raise ValueError(f"No segments found in {manifest_path}")
    loaded = []
    expected_start = 0
    for row in segments:
        start = int(row["start_frame"])
        end = int(row["end_frame_exclusive"])
        label = row["label"]
        if label not in HAR_CLASSES:
            raise ValueError(f"Unknown action class {label!r} in {manifest_path}")
        if start != expected_start or end <= start:
            raise ValueError(f"Non-contiguous or invalid frame boundaries: {row}")
        loaded.append(
            {
                "video_id": row["source_video"],
                "true_action": label,
                "video_path": merged_video,
                "annotation_path": None,
                "segment_start": start,
                "segment_end_exclusive": end,
                "source_video": row["source_video"],
                "source_annotation": row["source_annotation"],
            }
        )
        expected_start = end
    return loaded


def build_annotation_map(rows: list[dict[str, Any]]) -> dict[int, list[np.ndarray]]:
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if row["class"] != 0:
            continue
        grouped[row["frame"]].append(np.asarray([row["x1"], row["y1"], row["x2"], row["y2"]]))
    return grouped


def action_prediction(
    item: dict[str, Any],
    rows: list[dict[str, Any]],
    classifier: HARClassifier,
) -> dict[str, Any]:
    prep_start = time.perf_counter()
    segment_rows = [
        row for row in rows
        if item["segment_start"] <= row["frame"] < item["segment_end_exclusive"]
    ]
    features = action_features(segment_rows)
    feature_time_ms = (time.perf_counter() - prep_start) * 1000.0
    prediction, confidence, tensor_prep_ms, inference_ms, postprocess_ms = classifier.predict(features)
    video_read_ms = item.get("annotation_read_ms", 0.0)
    return {
        "video_id": item["video_id"],
        "true_action": item["true_action"],
        "predicted_action": prediction,
        "confidence": confidence,
        "annotation_read_ms": video_read_ms,
        "feature_preprocessing_ms": feature_time_ms,
        "tensor_preprocessing_ms": tensor_prep_ms,
        "inference_ms": inference_ms,
        "postprocessing_ms": postprocess_ms,
        "total_har_pipeline_ms": video_read_ms + feature_time_ms + tensor_prep_ms + inference_ms + postprocess_ms,
        "annotation_rows": len(segment_rows),
        "annotation_source": str(item.get("annotation_path") or item.get("source_annotation") or ""),
    }


def _ground_truth_for_frame(
    annotation_map: dict[int, list[np.ndarray]],
    global_frame: int,
    width: int,
    height: int,
) -> list[np.ndarray]:
    return [
        np.asarray([box[0] * width, box[1] * height, box[2] * width, box[3] * height], dtype=np.float32)
        for box in annotation_map.get(global_frame, [])
    ]


def run_yolo_over_video(
    items: list[dict[str, Any]],
    annotation_map: dict[int, list[np.ndarray]],
    yolo: YOLO,
    imgsz: int,
    confidence: float,
    iou: float,
    device: torch.device,
    max_frames_per_video: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    detection_state = {
        "confidence_threshold": confidence,
        "ground_truth_count": 0,
        "prediction_count": 0,
        "confidence_prediction_count": 0,
        "confidence_true_positives": 0,
        "confidence_false_positives": 0,
        "ranked_predictions": [],
    }
    warmup_capture = cv2.VideoCapture(str(items[0]["video_path"]))
    if not warmup_capture.isOpened():
        raise OSError(f"Could not open video for YOLO warm-up: {items[0]['video_path']}")
    warmup_ok, warmup_frame = warmup_capture.read()
    warmup_capture.release()
    if not warmup_ok:
        raise ValueError(f"No frame available for YOLO warm-up: {items[0]['video_path']}")
    warmup_start = time.perf_counter()
    yolo.predict(
        source=warmup_frame,
        imgsz=imgsz,
        conf=confidence,
        iou=iou,
        device=str(device),
        verbose=False,
    )
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
    initialized_predictor = True

    for item in items:
        video_path: Path = item["video_path"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Could not open video: {video_path}")
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = time.perf_counter()
        decode_ms = 0.0
        preprocess_ms = 0.0
        inference_ms = 0.0
        postprocess_ms = 0.0
        yolo_call_ms = 0.0
        frames_processed = 0
        frame_index = 0
        segment_start = int(item["segment_start"])
        segment_end = item["segment_end_exclusive"]

        while True:
            if max_frames_per_video is not None and frames_processed >= max_frames_per_video:
                break
            if segment_end is not None and frame_index >= int(segment_end):
                break
            decode_start = time.perf_counter()
            ok, frame = capture.read()
            decode_ms += (time.perf_counter() - decode_start) * 1000.0
            if not ok:
                break

            if frame_index >= segment_start:
                call_start = time.perf_counter()
                result = yolo.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=confidence,
                    iou=iou,
                    device=str(device),
                    verbose=False,
                )[0]
                call_ms = (time.perf_counter() - call_start) * 1000.0
                yolo_call_ms += call_ms
                timings = result.speed or {}
                preprocess_ms += float(timings.get("preprocess", 0.0))
                inference_ms += float(timings.get("inference", 0.0))
                postprocess_ms += float(timings.get("postprocess", 0.0))

                predictions: list[tuple[np.ndarray, float]] = []
                if result.boxes is not None and len(result.boxes):
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                    scores = result.boxes.conf.detach().cpu().numpy()
                    predictions = [
                        (box.astype(np.float32), float(score))
                        for box, class_id, score in zip(boxes, classes, scores)
                        if class_id == 0
                    ]
                global_frame = frame_index
                item_annotation_map = item.get("annotation_map", annotation_map)
                gt = _ground_truth_for_frame(
                    item_annotation_map, global_frame, frame.shape[1], frame.shape[0]
                )
                add_detection_frame(detection_state, predictions, gt)
                frames_processed += 1
            frame_index += 1

        capture.release()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        actual_duration = elapsed_ms / 1000.0
        rows.append(
            {
                "video_id": item["video_id"],
                "true_action": item["true_action"],
                "video_path": str(video_path),
                "frames_processed": frames_processed,
                "declared_video_frames": declared_frames,
                "segment_start_frame": segment_start,
                "segment_end_frame_exclusive": segment_end,
                "decode_ms": decode_ms,
                "yolo_wrapper_ms": yolo_call_ms,
                "yolo_preprocessing_ms": preprocess_ms,
                "yolo_inference_ms": inference_ms,
                "yolo_postprocessing_ms": postprocess_ms,
                "yolo_other_call_ms": max(0.0, yolo_call_ms - preprocess_ms - inference_ms - postprocess_ms),
                "total_wall_ms": elapsed_ms,
                "fps_wall": frames_processed / actual_duration if actual_duration else 0.0,
                "yolo_inference_fps_internal": (
                    1000.0 * frames_processed / inference_ms if inference_ms else 0.0
                ),
                "annotation_path": str(item.get("annotation_path") or item.get("source_annotation") or ""),
            }
        )
        if len(items) > 1:
            print(f"YOLO processed {item['video_id']} ({frames_processed} frames)")

    detection_state["warmup_ms"] = warmup_ms
    return rows, detection_state


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    output_dir: Path,
    yolo_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    cls: dict[str, Any],
    det: dict[str, Any],
    model_data: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    yolo_fps = [row["fps_wall"] for row in yolo_rows if row["fps_wall"] > 0]
    har_latency = [row["total_har_pipeline_ms"] for row in action_rows if row["total_har_pipeline_ms"] > 0]
    axes[0].boxplot(yolo_fps or [0], tick_labels=["YOLOv8n"])
    axes[0].set_ylabel("Frames per second (whole-video wall time)")
    axes[0].set_title("YOLO throughput")
    axes[1].boxplot(har_latency or [0], tick_labels=["HAR classifier"])
    axes[1].set_ylabel("Milliseconds per video classification")
    axes[1].set_title("HAR pipeline latency")
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_comparison.png", dpi=160)
    plt.close(fig)

    matrix = np.asarray(cls["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_xticks(range(len(cls["class_order"])), labels=cls["class_order"], rotation=45, ha="right")
    axis.set_yticks(range(len(cls["class_order"])), labels=cls["class_order"])
    axis.set_xlabel("Predicted HAR class")
    axis.set_ylabel("Dataset class")
    axis.set_title("HAR five-output checkpoint against all seven dataset classes")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            axis.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "har_confusion_matrix.png", dpi=160)
    plt.close(fig)

    metrics = [("Precision", det.get("precision")), ("Recall", det.get("recall")),
               ("F1", det.get("f1")), ("AP50", det.get("ap50")),
               ("mAP50:95", det.get("map50_95"))]
    fig, axis = plt.subplots(figsize=(8, 4))
    names = [name for name, value in metrics if value is not None]
    values = [value for _name, value in metrics if value is not None]
    axis.bar(names, values)
    axis.set_ylim(0, 1)
    axis.set_title("YOLO annotation agreement / detection metrics")
    fig.tight_layout()
    fig.savefig(output_dir / "detection_metrics.png", dpi=160)
    plt.close(fig)

    model_names = ["HAR BiLSTM", "YOLOv8n"]
    profile_values = [
        (
            model_data[name]["size_mb"],
            model_data[name]["profile"]["parameters"] / 1e6,
            (model_data[name]["profile"]["macs"] or 0) / 1e9,
        )
        for name in model_names
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, position, title, ylabel in (
        (axes[0], 0, "Weights on disk", "MB"),
        (axes[1], 1, "Parameter count", "Millions"),
        (axes[2], 2, "Operations per input", "GMAC"),
    ):
        axis.bar(model_names, [values[position] for values in profile_values])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "model_resources.png", dpi=160)
    plt.close(fig)


def markdown_report(results: dict[str, Any]) -> str:
    detection = results["detection_metrics"]
    classification = results["har_classification"]
    models = results["models"]
    valid_text = (
        "independent ground truth was explicitly confirmed"
        if results["annotations_treated_as_ground_truth"]
        else "pseudo-label agreement only; not valid independent ground-truth metrics"
    )
    lines = [
        "# HAR and YOLOv8 Evaluation",
        "",
        f"- Experiment: `{results['experiment']}`",
        f"- Run timestamp (UTC): `{results['timestamp_utc']}`",
        f"- Videos/segments evaluated: {results['videos_evaluated']}",
        f"- Frames passed through YOLO: {results['frames_evaluated']}",
        f"- Annotation status: {valid_text}",
        f"- Detection metric interpretation: {results['annotation_metric_status']}",
        "",
        "## Hardware and execution",
        "",
        f"- CPU: {results['hardware']['cpu']}",
        f"- Logical CPUs: {results['hardware']['logical_cpu_count']}",
        f"- RAM: {results['hardware']['ram_total_bytes'] / 1e9:.2f} GB",
        f"- GPU: {results['hardware']['gpu'] or 'None detected'}",
        f"- Runtime: Python {results['hardware']['python']}, PyTorch {results['hardware']['torch']}, "
        f"OpenCV {results['hardware']['opencv']}",
        f"- Peak resident memory sampled during run: {results['peak_rss_mb']:.1f} MB",
        f"- Peak allocated GPU memory: "
        f"{results['gpu_peak_allocated_mb'] if results['gpu_peak_allocated_mb'] is not None else 'N/A (CPU run)'} MB",
        "",
        "## Model and performance comparison",
        "",
        "| Model | Weights | Size (MB) | Parameters | MACs / input | FLOPs / input |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, info in models.items():
        profile_info = info["profile"]
        lines.append(
            f"| {name} | `{info['weights']}` | {info['size_mb']:.2f} | "
            f"{profile_info['parameters']:,} | {profile_info['macs'] if profile_info['macs'] is not None else 'N/A'} | "
            f"{profile_info['flops'] if profile_info['flops'] is not None else 'N/A'} |"
        )
    lines.extend(
        [
            "",
            "| Measurement | Result |",
            "|---|---:|",
            f"| YOLO wall-time FPS (median video) | {results['yolo_median_fps']:.3f} |",
            f"| HAR sequence classifications per second (median full pipeline) | "
            f"{results['har_sequences_per_second']:.3f} |",
            f"| YOLO wrapper latency (ms/frame) | {results['yolo_wrapper_ms_per_frame']:.3f} |",
            f"| YOLO preprocess / infer / postprocess (ms/frame) | "
            f"{results['yolo_preprocess_ms_per_frame']:.3f} / "
            f"{results['yolo_inference_ms_per_frame']:.3f} / "
            f"{results['yolo_postprocess_ms_per_frame']:.3f} |",
            f"| HAR preprocessing / inference / postprocessing (median ms/video) | "
            f"{results['har_median_preprocessing_ms']:.3f} / "
            f"{results['har_median_inference_ms']:.3f} / "
            f"{results['har_median_postprocessing_ms']:.3f} |",
            "",
            "## Detection metrics",
            "",
            f"- Precision / recall / F1 at confidence {detection['confidence_threshold']:.2f}, IoU 0.50: "
            f"{detection['precision']:.4f} / {detection['recall']:.4f} / {detection['f1']:.4f}",
            f"- AP50 / mAP50:95: {detection['ap50']} / {detection['map50_95']}",
            f"- Ground-truth boxes: {detection['ground_truth_boxes']:,}; predictions at confidence threshold: "
            f"{detection['predictions']:,}",
            f"- **Validity:** {results['annotation_metric_status']}.",
            "",
            "## HAR action classification",
            "",
            f"- Accuracy: {classification['accuracy']}",
            f"- Macro precision / recall / F1 over all seven dataset labels: "
            f"{classification['macro_precision']} / {classification['macro_recall']} / {classification['macro_f1']}",
            f"- This checkpoint has outputs for only: {', '.join(ACTION_CLASSES)}.",
            "",
            "| Dataset class | Support | Precision | Recall | F1 | Model output supported |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for label, item in classification["per_class"].items():
        lines.append(
            f"| {label} | {item['support']} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['supported_by_model']} |"
        )
    lines.extend(
        [
            "",
            "## Method and limitations",
            "",
            "YOLOv8n is evaluated as a COCO person detector at the configured image size, confidence, and NMS IoU. "
            "The HAR checkpoint is a five-class BiLSTM over 60x8 features. Feature generation reproduces the "
            "repository's annotation-derived feature conversion and uses the saved training normalization cache. "
            "HAR inference latency is therefore not a raw-video end-to-end comparison with YOLO.",
            "",
            ANNOTATION_PROVENANCE_WARNING,
            "",
            "The top-level `best_model .h5` has a convolutional image-model configuration and is not the selected HAR checkpoint. "
            "The action checkpoint does not emit `Meet and Split` or `Walking While Reading Book`; metrics include "
            "all seven dataset classes and those unsupported-class predictions cannot be learned from this checkpoint. "
            "The repository cache/metrics do not provide a held-out video-ID manifest, so overlap with training videos "
            "cannot be excluded. These measurements are not a clean independent action-recognition test.",
            "",
            "Preprocessing, neural-network inference, and postprocessing timings are measured separately where the "
            "framework exposes them. YOLO throughput includes video decoding and per-frame wrapper overhead; the "
            "HAR time describes parsing/processing supplied annotation files and one sequence classification.",
            "",
            "See `video_results.csv`, `action_predictions.csv`, `comparison_table.csv`, `results.json`, and the PNG "
            "figures for reproducible per-video outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)

    if args.experiment == "complete":
        items = discover_complete_videos(
            args.dataset_root, args.annotations_root, args.max_videos_per_class
        )
        if not items:
            raise ValueError("No dataset videos found")
        rows_by_item: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            if item["annotation_path"]:
                start = time.perf_counter()
                rows_by_item[item["video_id"]] = read_annotation(item["annotation_path"])
                item["annotation_read_ms"] = (time.perf_counter() - start) * 1000.0
            else:
                rows_by_item[item["video_id"]] = []
                item["annotation_read_ms"] = 0.0
            item["segment_end_exclusive"] = (
                max((row["frame"] for row in rows_by_item[item["video_id"]]), default=-1) + 1
            )
        annotation_path_for_all = None
    else:
        items = load_long_segments(args.manifest, args.merged_video)
        annotation_path_for_all = args.merged_annotations
        if not annotation_path_for_all.is_file():
            raise FileNotFoundError(f"Merged annotations missing: {annotation_path_for_all}")
        annotation_load_start = time.perf_counter()
        shared_rows = read_annotation(annotation_path_for_all)
        shared_annotation_read_ms = (time.perf_counter() - annotation_load_start) * 1000.0
        rows_by_item = {item["video_id"]: shared_rows for item in items}
        per_segment_read_ms = shared_annotation_read_ms / len(items)
        for item in items:
            item["annotation_read_ms"] = per_segment_read_ms

    har = HARClassifier(args.har_weights, args.normalization_cache, device_name)
    yolo = YOLO(str(args.yolo_weights))
    yolo.model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    har_profile = model_profile(har.model, (1, 60, 8), device)
    yolo_model_profile = yolo_profile(yolo, args.imgsz, device)
    yolo_model_profile["profile_input"] = f"1x3x{args.imgsz}x{args.imgsz}"

    action_rows = []
    annotation_map: dict[int, list[np.ndarray]] = defaultdict(list)
    for item in items:
        annotation_rows = rows_by_item[item["video_id"]]
        if args.experiment == "complete":
            item["annotation_map"] = build_annotation_map(annotation_rows)
        if annotation_rows:
            action_rows.append(action_prediction(item, annotation_rows, har))
        else:
            action_rows.append(
                {
                    "video_id": item["video_id"],
                    "true_action": item["true_action"],
                    "predicted_action": None,
                    "confidence": None,
                    "annotation_read_ms": item.get("annotation_read_ms", 0.0),
                    "feature_preprocessing_ms": None,
                    "tensor_preprocessing_ms": None,
                    "inference_ms": None,
                    "postprocessing_ms": None,
                    "total_har_pipeline_ms": None,
                    "annotation_rows": 0,
                    "annotation_source": str(item.get("annotation_path") or annotation_path_for_all or ""),
                }
            )
    if args.experiment == "long":
        annotation_map.update(build_annotation_map(shared_rows))

    if args.experiment == "long":
        yolo_items = [
            {
                "video_id": "merged_video",
                "true_action": "multiple",
                "video_path": args.merged_video,
                "annotation_path": None,
                "source_annotation": str(args.merged_annotations),
                "segment_start": 0,
                "segment_end_exclusive": None,
            }
        ]
    else:
        yolo_items = items

    with ProcessMemoryMonitor() as memory_monitor:
        yolo_rows, detection_state = run_yolo_over_video(
            yolo_items,
            annotation_map,
            yolo,
            args.imgsz,
            args.confidence,
            args.iou,
            device,
            args.max_frames_per_video,
        )
    peak_rss_mb = memory_monitor.peak_rss / (1024 * 1024)
    det = detection_summary(detection_state)
    cls = classification_summary(action_rows)
    total_frames = sum(row["frames_processed"] for row in yolo_rows)
    total_yolo_wall = sum(row["total_wall_ms"] for row in yolo_rows)
    total_yolo_call = sum(row["yolo_wrapper_ms"] for row in yolo_rows)
    total_inference = sum(row["yolo_inference_ms"] for row in yolo_rows)
    total_preprocess = sum(row["yolo_preprocessing_ms"] for row in yolo_rows)
    total_postprocess = sum(row["yolo_postprocessing_ms"] for row in yolo_rows)
    timed_actions = [row for row in action_rows if row["total_har_pipeline_ms"] is not None]
    median_value = lambda key: statistics.median(row[key] for row in timed_actions) if timed_actions else 0.0
    yolo_weight_size = args.yolo_weights.stat().st_size
    har_weight_size = har.weights_path.stat().st_size
    annotated_count = sum(bool(rows_by_item[item["video_id"]]) for item in items)
    annotation_status = (
        "independent ground truth explicitly confirmed by the operator"
        if args.annotations_are_ground_truth
        else "annotation agreement only; source labels are YOLOv8-generated pseudo-labels"
    )
    det["is_independent_ground_truth"] = args.annotations_are_ground_truth
    det["metric_status"] = annotation_status
    det["annotation_provenance_note"] = ANNOTATION_PROVENANCE_WARNING
    video_counts_by_class = {
        label: sum(item["true_action"] == label for item in items)
        for label in HAR_CLASSES
    }
    results = {
        "experiment": args.experiment,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "videos_evaluated": len(items),
        "videos_with_annotations": annotated_count,
        "videos_by_class": video_counts_by_class,
        "frames_evaluated": total_frames,
        "frames_per_video_limit": args.max_frames_per_video,
        "videos_per_class_limit": args.max_videos_per_class,
        "annotations_treated_as_ground_truth": args.annotations_are_ground_truth,
        "annotation_metric_status": annotation_status,
        "annotation_provenance_note": ANNOTATION_PROVENANCE_WARNING,
        "detection_metrics": det,
        "har_classification": cls,
        "hardware": hardware_metadata(),
        "peak_rss_mb": peak_rss_mb,
        "gpu_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda" else None
        ),
        "runtime": {
            "yolo_warmup_ms": detection_state["warmup_ms"],
            "yolo_wall_fps": total_frames * 1000.0 / total_yolo_wall if total_yolo_wall else 0.0,
            "yolo_wall_ms_per_frame": total_yolo_wall / total_frames if total_frames else 0.0,
            "yolo_wrapper_ms_per_frame": total_yolo_call / total_frames if total_frames else 0.0,
            "yolo_preprocess_ms_per_frame": total_preprocess / total_frames if total_frames else 0.0,
            "yolo_inference_ms_per_frame": total_inference / total_frames if total_frames else 0.0,
            "yolo_postprocess_ms_per_frame": total_postprocess / total_frames if total_frames else 0.0,
            "yolo_median_fps_per_video": statistics.median(
                row["fps_wall"] for row in yolo_rows if row["fps_wall"] > 0
            ) if any(row["fps_wall"] > 0 for row in yolo_rows) else 0.0,
            "har_median_preprocessing_ms": median_value("annotation_read_ms") + median_value("feature_preprocessing_ms") + median_value("tensor_preprocessing_ms"),
            "har_median_inference_ms": median_value("inference_ms"),
            "har_median_postprocessing_ms": median_value("postprocessing_ms"),
            "har_median_total_ms_per_video": median_value("total_har_pipeline_ms"),
            "har_sequences_per_second": (
                1000.0 / median_value("total_har_pipeline_ms")
                if median_value("total_har_pipeline_ms") > 0 else 0.0
            ),
        },
        "models": {
            "HAR BiLSTM": {
                "weights": str(args.har_weights.resolve()),
                "size_bytes": har_weight_size,
                "size_mb": har_weight_size / (1024 * 1024),
                "profile": har_profile,
                "input": "1x60x8 annotation-derived sequence",
                "classes": har.classes,
            },
            "YOLOv8n": {
                "weights": str(args.yolo_weights.resolve()),
                "size_bytes": yolo_weight_size,
                "size_mb": yolo_weight_size / (1024 * 1024),
                "profile": yolo_model_profile,
                "input": f"1x3x{args.imgsz}x{args.imgsz} decoded RGB/BGR video frame",
                "classes": ["person (COCO class 0)"],
            },
        },
        "experiment_parameters": {
            "device": device_name,
            "imgsz": args.imgsz,
            "yolo_confidence": args.confidence,
            "yolo_iou": args.iou,
            "normalization_cache": str(args.normalization_cache.resolve()),
            "annotation_read_time_shared_across_long_segments": args.experiment == "long",
        },
    }
    results["yolo_median_fps"] = results["runtime"]["yolo_median_fps_per_video"]
    results["yolo_wrapper_ms_per_frame"] = results["runtime"]["yolo_wrapper_ms_per_frame"]
    results["yolo_preprocess_ms_per_frame"] = results["runtime"]["yolo_preprocess_ms_per_frame"]
    results["yolo_inference_ms_per_frame"] = results["runtime"]["yolo_inference_ms_per_frame"]
    results["yolo_postprocess_ms_per_frame"] = results["runtime"]["yolo_postprocess_ms_per_frame"]
    results["har_median_preprocessing_ms"] = results["runtime"]["har_median_preprocessing_ms"]
    results["har_median_inference_ms"] = results["runtime"]["har_median_inference_ms"]
    results["har_median_postprocessing_ms"] = results["runtime"]["har_median_postprocessing_ms"]
    results["har_sequences_per_second"] = results["runtime"]["har_sequences_per_second"]

    comparison = [
        {
            "model": model_name,
            "weight_size_mb": info["size_mb"],
            "parameters": info["profile"]["parameters"],
            "macs": info["profile"]["macs"],
            "flops": info["profile"]["flops"],
            "peak_process_rss_mb": peak_rss_mb,
            "fps_whole_video": results["runtime"]["yolo_wall_fps"] if model_name == "YOLOv8n" else None,
            "har_sequences_per_second": (
                results["runtime"]["har_sequences_per_second"] if model_name == "HAR BiLSTM" else None
            ),
            "median_inference_ms": (
                results["runtime"]["yolo_inference_ms_per_frame"] if model_name == "YOLOv8n"
                else results["runtime"]["har_median_inference_ms"]
            ),
            "preprocessing_ms": (
                results["runtime"]["yolo_preprocess_ms_per_frame"] if model_name == "YOLOv8n"
                else results["runtime"]["har_median_preprocessing_ms"]
            ),
            "postprocessing_ms": (
                results["runtime"]["yolo_postprocess_ms_per_frame"] if model_name == "YOLOv8n"
                else results["runtime"]["har_median_postprocessing_ms"]
            ),
            "detection_precision": det["precision"] if model_name == "YOLOv8n" else None,
            "detection_recall": det["recall"] if model_name == "YOLOv8n" else None,
            "detection_f1": det["f1"] if model_name == "YOLOv8n" else None,
            "detection_ap50": det["ap50"] if model_name == "YOLOv8n" else None,
            "detection_map50_95": det["map50_95"] if model_name == "YOLOv8n" else None,
            "action_accuracy": cls["accuracy"] if model_name == "HAR BiLSTM" else None,
            "action_macro_precision": cls["macro_precision"] if model_name == "HAR BiLSTM" else None,
            "action_macro_recall": cls["macro_recall"] if model_name == "HAR BiLSTM" else None,
            "action_macro_f1": cls["macro_f1"] if model_name == "HAR BiLSTM" else None,
            "detection_metric_status": annotation_status,
        }
        for model_name, info in results["models"].items()
    ]

    write_json(output_dir / "results.json", results)
    write_json(output_dir / "detection_metrics.json", det)
    write_json(output_dir / "har_classification_metrics.json", cls)
    write_csv(output_dir / "video_results.csv", yolo_rows)
    write_csv(output_dir / "action_predictions.csv", action_rows)
    write_csv(output_dir / "comparison_table.csv", comparison)
    write_plots(output_dir, yolo_rows, action_rows, cls, det, results["models"])
    (output_dir / "research_report.md").write_text(markdown_report(results), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    complete = subparsers.add_parser("complete", help="Evaluate every dataset video.")
    complete.add_argument("--dataset-root", type=Path, default=Path("Human Activity Recognition - Video Dataset"))
    complete.add_argument("--annotations-root", type=Path, default=Path("HAR_annotations"))
    complete.add_argument("--output-dir", type=Path, default=Path("evaluation_results\\complete_dataset"))
    complete.add_argument("--max-videos-per-class", type=int, default=None, help="Smoke-test limit; omit to evaluate all videos.")
    complete.add_argument("--max-frames-per-video", type=int, default=None, help="Smoke-test limit; omit to evaluate all frames.")

    long = subparsers.add_parser("long", help="Evaluate the 70-video merged long-duration stream.")
    long.add_argument("--manifest", type=Path, default=Path("evaluation_results\\long_duration\\segments.csv"))
    long.add_argument("--merged-video", type=Path, default=Path("evaluation_results\\long_duration\\long_duration_70_videos.mp4"))
    long.add_argument("--merged-annotations", type=Path, default=Path("evaluation_results\\long_duration\\merged_annotations.csv"))
    long.add_argument("--output-dir", type=Path, default=Path("evaluation_results\\long_duration\\evaluation"))
    long.add_argument("--max-videos-per-class", type=int, default=None)
    long.add_argument("--max-frames-per-video", type=int, default=None, help="Smoke-test limit; omit to evaluate the full merged stream.")

    for command in (complete, long):
        command.add_argument("--har-weights", type=Path, default=Path("action_recognition_model\\action_recognition_bilstm.pt"))
        command.add_argument("--normalization-cache", type=Path, default=Path("action_recognition_model\\action_dataset_cache.npz"))
        command.add_argument("--yolo-weights", type=Path, default=Path("yolov8n.pt"))
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        command.add_argument("--imgsz", type=int, default=640)
        command.add_argument("--confidence", type=float, default=0.25)
        command.add_argument("--iou", type=float, default=0.70)
        command.add_argument(
            "--annotations-are-ground-truth",
            action="store_true",
            help="Declare the supplied person boxes independently verified ground truth.",
        )

    args = parser.parse_args()
    if args.imgsz < 32 or not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        parser.error("--imgsz must be >=32 and confidence/IoU must be in [0,1]")
    if args.max_videos_per_class is not None and args.max_videos_per_class < 1:
        parser.error("--max-videos-per-class must be positive")
    if args.max_frames_per_video is not None and args.max_frames_per_video < 1:
        parser.error("--max-frames-per-video must be positive")
    result = run(args)
    print(json.dumps({
        "experiment": result["experiment"],
        "videos_evaluated": result["videos_evaluated"],
        "frames_evaluated": result["frames_evaluated"],
        "yolo_wall_fps": result["runtime"]["yolo_wall_fps"],
        "har_action_accuracy": result["har_classification"]["accuracy"],
        "annotation_metric_status": result["annotation_metric_status"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
