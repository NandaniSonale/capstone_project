"""Shared, reproducible evaluation helpers for the HAR/YOLO experiments."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from thop import profile
from ultralytics import YOLO


ACTION_CLASSES = [
    "Walking",
    "Clapping",
    "Sitting",
    "Standing Still",
    "Walking While Using Phone",
]
HAR_CLASSES = [
    "Clapping",
    "Meet and Split",
    "Sitting",
    "Standing Still",
    "Walking",
    "Walking While Reading Book",
    "Walking While Using Phone",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", value)]


def read_annotation(path: str | Path) -> list[dict[str, Any]]:
    """Read the repository's normalized per-frame person-box CSV format."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        lines = (line for line in stream if not line.lstrip().startswith("#"))
        reader = csv.DictReader(lines)
        required = {"frame", "x1_norm", "y1_norm", "x2_norm", "y2_norm"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unsupported annotation schema in {path}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                box = {
                    "frame": int(row["frame"]),
                    "class": int(row.get("class") or 0),
                    "track_id": int(row.get("track_id") or -1),
                    "x1": float(row["x1_norm"]),
                    "y1": float(row["y1_norm"]),
                    "x2": float(row["x2_norm"]),
                    "y2": float(row["y2_norm"]),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid annotation at {path}:{line_number}: {error}") from error
            if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
                raise ValueError(f"Invalid box coordinates at {path}:{line_number}")
            rows.append(box)
    return rows


def action_features(rows: list[dict[str, Any]], max_timesteps: int = 60) -> np.ndarray:
    """Reproduce the annotation-based feature conversion used by the existing trainer."""
    rows = sorted(rows, key=lambda row: row["frame"])
    if not rows:
        return np.zeros((max_timesteps, 8), dtype=np.float32)

    sequence = []
    for index, current in enumerate(rows):
        previous = rows[index - 1] if index else current
        cx = (current["x1"] + current["x2"]) / 2.0
        cy = (current["y1"] + current["y2"]) / 2.0
        prev_cx = (previous["x1"] + previous["x2"]) / 2.0
        prev_cy = (previous["y1"] + previous["y2"]) / 2.0
        width = max(current["x2"] - current["x1"], 0.02)
        height = max(current["y2"] - current["y1"], 0.02)
        dx = (cx - prev_cx) * 100.0
        dy = (cy - prev_cy) * 100.0
        magnitude = math.sqrt(dx * dx + dy * dy)
        density = max(width * 1080.0 / 16.0, 1.0) * max(height * 1920.0 / 16.0, 1.0)
        energy = math.log1p(15.0 + abs(dx) * 4.0 + abs(dy) * 4.0)
        sequence.append(
            [dx, dy, abs(dx) * 0.3, abs(dy) * 0.3, magnitude, energy, energy * 0.25, density / 100.0]
        )

    values = np.asarray(sequence, dtype=np.float32)
    if len(values) > max_timesteps:
        indexes = np.linspace(0, len(values) - 1, max_timesteps).astype(int)
        values = values[indexes]
    elif len(values) < max_timesteps:
        values = np.vstack((values, np.zeros((max_timesteps - len(values), 8), dtype=np.float32)))
    return values


class HARClassifier:
    def __init__(self, weights: Path, cache: Path, device: str = "auto"):
        from action_recognition_model.models import BiLSTMActionClassifier

        if not weights.is_file():
            raise FileNotFoundError(f"HAR weights not found: {weights}")
        if not cache.is_file():
            raise FileNotFoundError(f"Training normalization cache not found: {cache}")
        cache_data = np.load(cache, allow_pickle=True)
        self.classes = [str(value) for value in cache_data["class_names"].tolist()]
        if self.classes != ACTION_CLASSES:
            raise ValueError(
                f"Checkpoint class order {self.classes!r} does not match the recorded "
                f"metrics class order {ACTION_CLASSES!r}"
            )
        self.mean = cache_data["mean"].astype(np.float32)
        self.std = cache_data["std"].astype(np.float32)
        if self.mean.shape != (1, 1, 8) or self.std.shape != (1, 1, 8):
            raise ValueError(f"Unexpected feature normalization shapes: {self.mean.shape}, {self.std.shape}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but this PyTorch installation has no CUDA device")
        self.device = torch.device(device)
        self.model = BiLSTMActionClassifier(num_classes=len(self.classes))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.weights_path = weights

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> tuple[str, float, float, float, float]:
        prep_start = time.perf_counter()
        normalized = (features[np.newaxis, ...] - self.mean) / self.std
        tensor = torch.from_numpy(normalized.astype(np.float32, copy=False)).to(self.device)
        preprocessing_ms = (time.perf_counter() - prep_start) * 1000.0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        infer_start = time.perf_counter()
        logits = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - infer_start) * 1000.0

        post_start = time.perf_counter()
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
        result = self.classes[int(index)], float(confidence), preprocessing_ms, (time.perf_counter() - post_start) * 1000.0
        return result[0], result[1], result[2], inference_ms, result[3]


class ProcessMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.1):
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "ProcessMemoryMonitor":
        self.peak_rss = self.process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()


def model_profile(model: torch.nn.Module, input_shape: tuple[int, ...], device: torch.device) -> dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    try:
        sample = torch.zeros(input_shape, device=device)
        macs, profiled_parameters = profile(model, inputs=(sample,), verbose=False)
        return {
            "parameters": parameters,
            "macs": int(macs),
            "flops": int(macs * 2),
            "flops_convention": "2 x THOP MACs; operator coverage depends on THOP",
            "thop_parameters": int(profiled_parameters),
            "profile_error": None,
        }
    except (RuntimeError, ValueError, TypeError, NotImplementedError) as error:
        return {
            "parameters": parameters,
            "macs": None,
            "flops": None,
            "flops_convention": "unavailable",
            "thop_parameters": None,
            "profile_error": f"{type(error).__name__}: {error}",
        }


def yolo_profile(model: YOLO, imgsz: int, device: torch.device) -> dict[str, Any]:
    network = model.model
    info = model_profile(network, (1, 3, imgsz, imgsz), device)
    info["profile_input"] = f"1x3x{imgsz}x{imgsz}"
    return info


def hardware_metadata() -> dict[str, Any]:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        gpu_name = None
        gpu_memory = None
    try:
        cpu = platform.processor() or platform.uname().processor
    except (OSError, RuntimeError):
        cpu = "unknown"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "gpu": gpu_name,
        "gpu_memory_bytes": gpu_memory,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "opencv": cv2.__version__,
    }


def iou_xyxy(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def add_detection_frame(
    state: dict[str, Any],
    predictions: list[tuple[np.ndarray, float]],
    ground_truth: list[np.ndarray],
    thresholds: np.ndarray = IOU_THRESHOLDS,
) -> None:
    """Accumulate confidence-ranked matches for AP and thresholded P/R/F1."""
    state["ground_truth_count"] += len(ground_truth)
    ordered = sorted(predictions, key=lambda entry: entry[1], reverse=True)
    per_threshold_matches: list[list[bool]] = []
    for threshold in thresholds:
        matched: set[int] = set()
        frame_matches = []
        for box, _confidence in ordered:
            choices = [
                (iou_xyxy(box, truth), index)
                for index, truth in enumerate(ground_truth)
                if index not in matched
            ]
            best_iou, best_index = max(choices, default=(0.0, -1))
            is_match = best_iou >= threshold
            if is_match:
                matched.add(best_index)
            frame_matches.append(is_match)
        per_threshold_matches.append(frame_matches)

    for index, (box, confidence) in enumerate(ordered):
        match_flags = tuple(flags[index] for flags in per_threshold_matches)
        state["ranked_predictions"].append((confidence, match_flags))
        state["prediction_count"] += 1
        if confidence >= state["confidence_threshold"]:
            state["confidence_prediction_count"] += 1
            if match_flags[0]:
                state["confidence_true_positives"] += 1
            else:
                state["confidence_false_positives"] += 1


def _average_precision(true_positives: np.ndarray, total_ground_truth: int) -> float | None:
    if total_ground_truth == 0:
        return None
    if len(true_positives) == 0:
        return 0.0
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(1 - true_positives)
    recall = cumulative_tp / total_ground_truth
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    sampled = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        valid = precision[recall >= recall_level]
        sampled.append(float(valid.max()) if len(valid) else 0.0)
    return float(np.mean(sampled))


def detection_summary(state: dict[str, Any]) -> dict[str, Any]:
    total_gt = state["ground_truth_count"]
    tp = state["confidence_true_positives"]
    fp = state["confidence_false_positives"]
    fn = max(0, total_gt - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    ranked = sorted(state["ranked_predictions"], key=lambda item: item[0], reverse=True)
    aps = []
    for threshold_index in range(len(IOU_THRESHOLDS)):
        matches = np.asarray([flags[threshold_index] for _, flags in ranked], dtype=np.float64)
        aps.append(_average_precision(matches, total_gt))
    available_aps = [value for value in aps if value is not None]
    return {
        "confidence_threshold": state["confidence_threshold"],
        "iou_threshold": 0.5,
        "ground_truth_boxes": total_gt,
        "predictions": state["confidence_prediction_count"],
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": aps[0],
        "map50_95": float(np.mean(available_aps)) if available_aps else None,
        "ap_by_iou": {f"{threshold:.2f}": value for threshold, value in zip(IOU_THRESHOLDS, aps)},
    }


def classification_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = HAR_CLASSES
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row in records:
        if row.get("predicted_action") in index and row["true_action"] in index:
            matrix[index[row["true_action"]], index[row["predicted_action"]]] += 1

    per_class = {}
    for class_index, label in enumerate(labels):
        tp = int(matrix[class_index, class_index])
        fp = int(matrix[:, class_index].sum() - tp)
        fn = int(matrix[class_index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": int(matrix[class_index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "supported_by_model": label in ACTION_CLASSES,
        }

    total = int(matrix.sum())
    return {
        "samples": total,
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "macro_precision": float(np.mean([row["precision"] for row in per_class.values()])) if total else None,
        "macro_recall": float(np.mean([row["recall"] for row in per_class.values()])) if total else None,
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])) if total else None,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_order": labels,
    }


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, contents: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2, default=json_default), encoding="utf-8")
