#!/usr/bin/env python3
"""Predict actions and render annotated demo videos for all 7 HAR classes."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "action_recognition_model_7class"
OUTPUT_DIR = ROOT / "output" / "action_recognition_7class"
DATASET_DIR = ROOT / "Human Activity Recognition - Video Dataset"
ANNOTATIONS_DIR = ROOT / "HAR_annotations"
sys.path.insert(0, str(ROOT / "action_recognition_model"))

from train import ALL_ACTION_CLASSES, extract_features_from_annotation_file, natural_sort_key
from train_seven_class_action_model import summarize_sequences


def load_boxes(annotation_path):
    boxes = {}
    with annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line in annotation_file:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("frame,"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                frame_idx = int(parts[0])
                x1, y1, x2, y2 = (float(value) for value in parts[3:7])
            except ValueError:
                continue
            boxes.setdefault(frame_idx, []).append((x1, y1, x2, y2))
    if not boxes:
        raise ValueError(f"No valid bounding boxes found in {annotation_path}")
    return boxes


def summarize_sequences(sequences):
    return np.concatenate(
        [
            sequences.mean(axis=1),
            sequences.std(axis=1),
            np.quantile(sequences, [0.1, 0.5, 0.9], axis=1)
            .transpose(1, 2, 0)
            .reshape(len(sequences), -1),
        ],
        axis=1,
    )


def annotate_video(video_path, annotation_path, output_path, ground_truth, prediction, confidence, probabilities):
    boxes = load_boxes(annotation_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
    if raw_path.exists():
        raw_path.unlink()
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {raw_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            for x1, y1, x2, y2 in boxes.get(frame_idx, []):
                left = max(0, min(width - 1, round(x1 * width)))
                top = max(0, min(height - 1, round(y1 * height)))
                right = max(0, min(width - 1, round(x2 * width)))
                bottom = max(0, min(height - 1, round(y2 * height)))
                cv2.rectangle(frame, (left, top), (right, bottom), (50, 220, 100), 3)

            overlay_height = min(170, height)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (18, 24, 32), -1)
            cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
            cv2.putText(frame, "7-CLASS ACTION RECOGNITION", (24, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (220, 230, 240), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Ground truth: {ground_truth}", (24, 78),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Prediction: {prediction} ({confidence:.1%})", (24, 120),
                        cv2.FONT_HERSHEY_DUPLEX, 0.82, (50, 240, 150), 2, cv2.LINE_AA)

            top_classes = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
            summary = "  |  ".join(f"{name}: {probability:.0%}" for name, probability in top_classes)
            font_scale = 0.65
            while font_scale > 0.35:
                text_width = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
                if text_width <= width - 48:
                    break
                font_scale *= 0.9
            cv2.putText(frame, summary, (24, height - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames were decoded from {video_path}")

    ffmpeg = ROOT / "FFmpeg" / "FFmpeg" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"Bundled ffmpeg executable not found: {ffmpeg}")
    command = [
        str(ffmpeg), "-y", "-i", str(raw_path), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
        raw_path.unlink()
    elif "Unknown encoder 'libx264'" in result.stderr and raw_path.is_file():
        raw_path.replace(output_path)
        print(f"ffmpeg has no libx264 encoder; keeping MP4V output: {output_path}")
    else:
        raise RuntimeError(
            f"ffmpeg failed for {output_path}: {result.stderr[-2000:]}"
        )
    return frame_idx, expected_frames


def main():
    model_path = MODEL_DIR / "action_recognition_rf.joblib"
    metrics_path = MODEL_DIR / "action_recognition_rf_metrics.json"
    for path in (model_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required trained-model artifact not found: {path}")

    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    classes = metrics["classes"]
    if len(classes) != 7:
        raise ValueError(f"Expected a 7-class checkpoint; metrics list {len(classes)} classes")

    checkpoint = joblib.load(model_path)
    if checkpoint["classes"] != classes:
        raise ValueError("Classifier and metrics class order differ")
    model = checkpoint["model"]
    mean = checkpoint["normalization_mean"]
    std = checkpoint["normalization_std"]

    results = []
    for class_name in ALL_ACTION_CLASSES:
        class_video_dir = DATASET_DIR / class_name
        class_annotation_dir = ANNOTATIONS_DIR / class_name
        videos = sorted(class_video_dir.glob("*.mp4"), key=natural_sort_key)
        if len(videos) < 121:
            raise ValueError(f"{class_name} has fewer than 121 videos; no held-out sample available")

        video_path = videos[120]
        annotation_path = class_annotation_dir / f"{video_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation missing for representative video: {annotation_path}")
        if metrics["held_out_representatives"].get(class_name) != annotation_path.name:
            raise ValueError(f"{annotation_path} was not reserved as a held-out training example")

        features = extract_features_from_annotation_file(str(annotation_path))
        normalized = (features[None, ...] - mean) / std
        probabilities_array = model.predict_proba(summarize_sequences(normalized))[0]
        probabilities = {
            class_name: float(probability)
            for class_name, probability in zip(classes, probabilities_array)
        }
        predicted_class = classes[int(np.argmax(probabilities_array))]
        confidence = float(np.max(probabilities_array))

        output_path = OUTPUT_DIR / "videos" / class_name / f"{video_path.stem}_action_recognition.mp4"
        rendered_frames, expected_frames = annotate_video(
            video_path, annotation_path, output_path, class_name,
            predicted_class, confidence, probabilities
        )
        results.append({
            "ground_truth": class_name,
            "prediction": predicted_class,
            "confidence": confidence,
            "video": str(video_path.relative_to(ROOT)),
            "annotation": str(annotation_path.relative_to(ROOT)),
            "output_video": str(output_path.relative_to(ROOT)),
            "rendered_frames": rendered_frames,
            "source_frames": expected_frames,
            "class_probabilities": probabilities,
        })
        print(f"{class_name}: predicted {predicted_class} ({confidence:.1%}) -> {output_path}")

    representative_accuracy = sum(
        result["ground_truth"] == result["prediction"] for result in results
    ) / len(results)
    summary_path = OUTPUT_DIR / "video_predictions.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump({
            "feature_source": "YOLO annotation-derived bounding-box motion features",
            "representative_selection": "121st naturally sorted video per class; excluded from training annotations",
            "representative_accuracy": representative_accuracy,
            "results": results,
        }, summary_file, indent=2)
    print(f"Held-out representative accuracy: {representative_accuracy:.1%} ({sum(result['ground_truth'] == result['prediction'] for result in results)}/{len(results)})")
    print(f"Saved prediction summary: {summary_path}")


if __name__ == "__main__":
    main()
