"""Create a reproducible 70-clip long-duration video and aligned manifests."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import cv2

from har_yolo_benchmark import (
    HAR_CLASSES,
    VIDEO_EXTENSIONS,
    natural_key,
    read_annotation,
    write_json,
)


def discover_videos(dataset_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for label in HAR_CLASSES:
        class_dir = dataset_root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dataset class directory missing: {class_dir}")
        result[label] = sorted(
            (path for path in class_dir.iterdir()
             if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS),
            key=lambda path: natural_key(path.name),
        )
    return result


def select_videos(
    dataset_root: Path,
    annotation_root: Path,
    samples_per_class: int,
    seed: int,
) -> list[tuple[str, Path, Path]]:
    videos_by_class = discover_videos(dataset_root)
    randomizer = random.Random(seed)
    selected = []
    for label in HAR_CLASSES:
        videos = videos_by_class[label]
        if len(videos) < samples_per_class:
            raise ValueError(
                f"Class {label!r} has {len(videos)} videos; "
                f"{samples_per_class} are required"
            )
        class_selection = sorted(
            randomizer.sample(videos, samples_per_class),
            key=lambda path: natural_key(path.name),
        )
        for video_path in class_selection:
            annotation_path = annotation_root / label / f"{video_path.stem}.txt"
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Matching annotation missing: {annotation_path}")
            selected.append((label, video_path, annotation_path))
    return selected


def merge_videos(
    selected: list[tuple[str, Path, Path]],
    output_video: Path,
    manifest_csv: Path,
    manifest_json: Path,
    merged_annotations: Path,
    target_fps: float | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
) -> list[dict[str, Any]]:
    if not selected:
        raise ValueError("No videos selected")

    first = cv2.VideoCapture(str(selected[0][1]))
    if not first.isOpened():
        raise OSError(f"Cannot open video: {selected[0][1]}")
    first_width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    first_height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_fps = float(first.get(cv2.CAP_PROP_FPS))
    first.release()
    width = target_width or first_width
    height = target_height or first_height
    fps = target_fps or first_fps
    if width < 2 or height < 2 or fps <= 0:
        raise ValueError(f"Invalid output video dimensions or FPS: {width}x{height}@{fps}")
    width -= width % 2
    height -= height % 2

    output_video.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_annotations.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Could not create merged video: {output_video}")

    segments: list[dict[str, Any]] = []
    annotation_output_rows: list[dict[str, Any]] = []
    current_global_frame = 0
    try:
        for segment_index, (label, video_path, annotation_path) in enumerate(selected):
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise OSError(f"Cannot open source video: {video_path}")
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if source_fps <= 0 or reported_frames <= 0:
                capture.release()
                raise ValueError(f"Invalid video stream metadata: {video_path}")

            source_annotations = read_annotation(annotation_path)
            annotations_by_frame: dict[int, list[dict[str, Any]]] = {}
            for row in source_annotations:
                annotations_by_frame.setdefault(row["frame"], []).append(row)

            start_frame = current_global_frame
            output_count = max(1, round(reported_frames * fps / source_fps))
            current_source_frame = -1
            frame = None
            written = 0
            for relative_output_frame in range(output_count):
                source_frame_index = min(
                    reported_frames - 1,
                    round(relative_output_frame * source_fps / fps),
                )
                while current_source_frame < source_frame_index:
                    ok, decoded = capture.read()
                    if not ok:
                        break
                    current_source_frame += 1
                    frame = decoded
                if frame is None or current_source_frame < source_frame_index:
                    break

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                writer.write(frame)

                global_frame = current_global_frame
                current_global_frame += 1
                written += 1
                for row in annotations_by_frame.get(source_frame_index, []):
                    annotation_output_rows.append(
                        {
                            "frame": global_frame,
                            "track_id": row["track_id"],
                            "class": row["class"],
                            "x1_norm": row["x1"],
                            "y1_norm": row["y1"],
                            "x2_norm": row["x2"],
                            "y2_norm": row["y2"],
                        }
                    )

            capture.release()
            if written == 0:
                raise ValueError(f"No frames decoded from {video_path}")
            segments.append(
                {
                    "segment_index": segment_index,
                    "label": label,
                    "source_video": str(video_path.resolve()),
                    "source_annotation": str(annotation_path.resolve()),
                    "source_fps": source_fps,
                    "target_fps": fps,
                    "source_frame_count_reported": reported_frames,
                    "output_frame_count": written,
                    "start_frame": start_frame,
                    "end_frame_exclusive": current_global_frame,
                    "width": width,
                    "height": height,
                }
            )
            print(f"[{segment_index + 1}/{len(selected)}] {label}: {video_path.name} ({written} frames)")
    finally:
        writer.release()

    fieldnames = [
        "frame", "track_id", "class", "x1_norm", "y1_norm", "x2_norm", "y2_norm"
    ]
    with merged_annotations.open("w", newline="", encoding="utf-8") as stream:
        output = csv.DictWriter(stream, fieldnames=fieldnames)
        output.writeheader()
        output.writerows(annotation_output_rows)

    with manifest_csv.open("w", newline="", encoding="utf-8") as stream:
        columns = list(segments[0])
        output = csv.DictWriter(stream, fieldnames=columns)
        output.writeheader()
        output.writerows(segments)

    write_json(
        manifest_json,
        {
            "video": str(output_video.resolve()),
            "annotations": str(merged_annotations.resolve()),
            "target_fps": fps,
            "width": width,
            "height": height,
            "total_frames": current_global_frame,
            "segments": segments,
        },
    )
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select 10 videos per HAR class and merge into one long-duration video."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("Human Activity Recognition - Video Dataset"))
    parser.add_argument("--annotations-root", type=Path, default=Path("HAR_annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results\\long_duration"))
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=None, help="Output FPS; defaults to the first selected video's FPS.")
    parser.add_argument("--width", type=int, default=None, help="Output width; defaults to the first selected video's width.")
    parser.add_argument("--height", type=int, default=None, help="Output height; defaults to the first selected video's height.")
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive")

    selected = select_videos(
        args.dataset_root, args.annotations_root, args.samples_per_class, args.seed
    )
    output_dir = args.output_dir
    merged_video = output_dir / f"long_duration_{len(selected)}_videos.mp4"
    segments = merge_videos(
        selected,
        merged_video,
        output_dir / "segments.csv",
        output_dir / "segments.json",
        output_dir / "merged_annotations.csv",
        args.fps,
        args.width,
        args.height,
    )
    print(f"Merged {len(segments)} source clips into {merged_video}")
    print(f"Frame-boundary manifest: {output_dir / 'segments.csv'}")
    print(f"Aligned annotations: {output_dir / 'merged_annotations.csv'}")


if __name__ == "__main__":
    main()
