import os
import glob
import json
import argparse
import time
import shutil
import csv
from compressed_domain_tracker import CompressedDomainTracker
from generate_video_output import generate_video


def _video_name_from_path(video_file):
    return os.path.splitext(os.path.basename(video_file))[0]


def _output_paths(output_dir, video_name):
    return {
        "motion_json": os.path.join(output_dir, f"{video_name}_motion_features.json"),
        "tracking_csv": os.path.join(output_dir, f"{video_name}_tracking_results.csv"),
        "visualization": os.path.join(output_dir, f"{video_name}_visualization.mp4"),
    }


def is_video_complete(output_dir, video_name):
    """A video is complete when it has tracking + non-empty ROI motion data."""
    paths = _output_paths(output_dir, video_name)
    if not os.path.exists(paths["tracking_csv"]) or not os.path.exists(paths["motion_json"]):
        return False

    try:
        with open(paths["motion_json"], "r", encoding="utf-8") as f:
            motion_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if not motion_data:
        return False

    try:
        with open(paths["tracking_csv"], "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except OSError:
        return False

    if len(lines) <= 1:
        return False

    body = "".join(lines[1:])
    if "DET" not in body:
        return False
    if "PROP" not in body:
        return False

    return True


def assemble_master_csv(output_dir, video_files):
    master_csv_path = os.path.join(output_dir, "master_tracking_results.csv")
    print("[INFO] Saving master aggregated tracking CSV dataset...")

    with open(master_csv_path, "w", newline="", encoding="utf-8") as master_csv:
        writer = csv.writer(master_csv)
        writer.writerow([
            "video_name", "pts", "frame_type", "source",
            "confidence", "cx", "cy", "w", "h",
        ])

        for idx, video_file in enumerate(video_files, 1):
            video_name = _video_name_from_path(video_file)
            single_csv = _output_paths(output_dir, video_name)["tracking_csv"]
            if not os.path.exists(single_csv):
                continue

            with open(single_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 8:
                        writer.writerow([video_name] + row[:8])

            if idx % 25 == 0 or idx == len(video_files):
                pct = 100.0 * idx / len(video_files)
                print(f"[PROGRESS] Master CSV: [{idx}/{len(video_files)}] ({pct:.1f}%)")

    print(f"[INFO] Master CSV saved: {master_csv_path}")
    return master_csv_path


def assemble_master_json(output_dir, video_files):
    master_json_path = os.path.join(output_dir, "master_folder_features.json")
    print("[INFO] Saving master aggregated JSON feature dataset (zero-RAM disk-streaming mode)...")

    total = len(video_files)
    with open(master_json_path, "w", encoding="utf-8") as master_file:
        master_file.write("{\n")
        first_entry = True

        for idx, video_file in enumerate(video_files, 1):
            video_name = _video_name_from_path(video_file)
            motion_json = _output_paths(output_dir, video_name)["motion_json"]
            if not os.path.exists(motion_json):
                continue

            try:
                with open(motion_json, "r", encoding="utf-8") as f:
                    video_roi_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if not video_roi_data:
                continue

            total_p_frames = len(video_roi_data)
            total_roi_macroblocks = sum(len(mbs) for mbs in video_roi_data.values())

            if not first_entry:
                master_file.write(",\n")
            first_entry = False

            master_file.write(f'  {json.dumps(video_name)}: {{\n')
            master_file.write(f'    "video_path": {json.dumps(video_file)},\n')
            master_file.write(f'    "total_p_frames": {total_p_frames},\n')
            master_file.write(f'    "total_roi_macroblocks": {total_roi_macroblocks},\n')
            master_file.write('    "frames": ')
            with open(motion_json, "r", encoding="utf-8") as motion_file:
                master_file.write(motion_file.read().strip())
            master_file.write("\n  }")

            if idx % 10 == 0 or idx == total:
                pct = 100.0 * idx / total
                print(f"[PROGRESS] Disk streaming master JSON: [{idx}/{total}] ({pct:.1f}%) complete...")

        master_file.write("\n}\n")

    print(f"[INFO] Master JSON saved: {master_json_path}")
    return master_json_path


def build_summary(output_dir, video_files):
    summary_report = []
    for video_file in video_files:
        video_name = _video_name_from_path(video_file)
        paths = _output_paths(output_dir, video_name)
        status = "SUCCESS" if is_video_complete(output_dir, video_name) else "INCOMPLETE"

        frames_processed = 0
        total_macroblocks = 0
        if os.path.exists(paths["motion_json"]):
            try:
                with open(paths["motion_json"], "r", encoding="utf-8") as f:
                    motion_data = json.load(f)
                frames_processed = len(motion_data)
                total_macroblocks = sum(len(mbs) for mbs in motion_data.values())
            except (json.JSONDecodeError, OSError):
                pass

        summary_report.append({
            "video_name": video_name,
            "frames_processed": frames_processed,
            "total_macroblocks": total_macroblocks,
            "status": status,
        })

    summary_path = os.path.join(output_dir, "folder_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_report, summary_file, indent=2)

    return summary_report, summary_path


def process_folder(
    folder_path=r"Human Activity Recognition - Video Dataset/Walking",
    model_path=r"best_model .h5",
    output_dir=r"tracking_outputs\Walking_dataset",
    max_videos=0,
    render_visuals=False,
    force_reprocess=False,
    assemble_only=False,
    retry_failed=False,
    conf_threshold=0.5,
):
    print("=== BATCH COMPRESSED-DOMAIN EXTRACTION ===")
    print(f"Target Folder: {folder_path}")
    print(f"Model Path: {model_path}")
    print(f"Output Directory: {output_dir}")

    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder does not exist: {folder_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(folder_path, "*.mp4")))
    if not video_files:
        print(f"[ERROR] No .mp4 videos found in {folder_path}")
        return

    print(f"Found {len(video_files)} video files in folder.")
    if max_videos and max_videos > 0:
        video_files = video_files[:max_videos]
        print(f"Processing first {len(video_files)} video(s)...")
    else:
        print(f"Processing ALL {len(video_files)} videos in folder...")

    if assemble_only:
        assemble_master_csv(output_dir, video_files)
        assemble_master_json(output_dir, video_files)
        summary_report, summary_path = build_summary(output_dir, video_files)
        complete = sum(1 for item in summary_report if item["status"] == "SUCCESS")
        print(f"\n[SUCCESS] Assembly complete: {complete}/{len(summary_report)} videos have full tracking + motion data.")
        print(f"Summary saved: {summary_path}")
        return

    start_time = time.time()
    processed = 0
    skipped = 0

    for idx, video_file in enumerate(video_files, 1):
        video_name = _video_name_from_path(video_file)
        paths = _output_paths(output_dir, video_name)

        print("\n---------------------------------------------------------")
        print(f"[{idx}/{len(video_files)}] Processing Video: {video_name}")
        print(f"File: {video_file}")
        print("---------------------------------------------------------")

        if (
            not force_reprocess
            and not retry_failed
            and is_video_complete(output_dir, video_name)
        ):
            print(f"[SKIP] {video_name} already processed. Appending existing tracking CSV to master...")
            skipped += 1
            if render_visuals and not os.path.exists(paths["visualization"]):
                single_csv = paths["tracking_csv"]
                if os.path.exists(single_csv):
                    print(f"[RENDER] Missing visualization for {video_name}, rendering now...")
                    generate_video(
                        video_path=video_file,
                        csv_path=single_csv,
                        json_path=paths["motion_json"],
                        output_mp4_paths=[paths["visualization"]],
                    )
            continue

        if retry_failed and is_video_complete(output_dir, video_name):
            print(f"[SKIP] {video_name} already complete.")
            skipped += 1
            continue

        tracker = CompressedDomainTracker(
            video_path=video_file,
            model_path=model_path,
            conf_threshold=conf_threshold,
        )

        video_roi_data = tracker.run() or {}

        single_json_path = os.path.join(tracker.output_dir, "roi_motion_data.json")
        single_csv_path = os.path.join(tracker.output_dir, "tracking_results.csv")

        with open(paths["motion_json"], "w", encoding="utf-8") as feature_file:
            json.dump(video_roi_data, feature_file, indent=2)

        if os.path.exists(single_csv_path):
            shutil.copyfile(single_csv_path, paths["tracking_csv"])

        if render_visuals and os.path.exists(single_csv_path):
            print(f"\n[RENDER] Rendering visualization video for {video_name}...")
            generate_video(
                video_path=video_file,
                csv_path=single_csv_path,
                json_path=single_json_path,
                output_mp4_paths=[paths["visualization"]],
            )

        p_frames = len(video_roi_data)
        roi_mbs = sum(len(mbs) for mbs in video_roi_data.values())
        print(f"[DONE] {video_name}: {p_frames} P-frame motion records, {roi_mbs} ROI macroblocks.")
        processed += 1

    assemble_master_csv(output_dir, video_files)
    assemble_master_json(output_dir, video_files)
    summary_report, summary_path = build_summary(output_dir, video_files)

    total_duration = time.time() - start_time
    complete = sum(1 for item in summary_report if item["status"] == "SUCCESS")
    incomplete = [item["video_name"] for item in summary_report if item["status"] != "SUCCESS"]

    print("\n=========================================================")
    print("[SUCCESS] FOLDER PROCESSING COMPLETE!")
    print(f"Videos Reprocessed: {processed}")
    print(f"Videos Skipped (already complete): {skipped}")
    print(f"Videos With Full Tracking + Motion Data: {complete}/{len(summary_report)}")
    if incomplete:
        print(f"Incomplete Videos ({len(incomplete)}): {', '.join(incomplete[:10])}" +
              (" ..." if len(incomplete) > 10 else ""))
    print(f"Total Time Taken: {total_duration:.2f} seconds")
    print(f"Summary Saved: {summary_path}")
    print(f"Features Output Directory: {output_dir}")
    print("=========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", default=r"Human Activity Recognition - Video Dataset\Walking",
                        help="Input folder containing MP4 videos")
    parser.add_argument("--model", "-m", default=r"best_model .h5", help="Model path")
    parser.add_argument("--out", "-o", default=r"tracking_outputs\Walking_dataset", help="Output folder")
    parser.add_argument("--limit", "-l", type=int, default=0,
                        help="Number of videos to process (0 for ALL videos in folder)")
    parser.add_argument("--render", action="store_true",
                        help="Render visualization MP4 for each processed video")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess all videos even if outputs already exist")
    parser.add_argument("--assemble-only", action="store_true",
                        help="Only rebuild master CSV/JSON from existing per-video outputs")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Reprocess only videos missing tracking or motion data")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    args = parser.parse_args()

    process_folder(
        folder_path=args.folder,
        model_path=args.model,
        output_dir=args.out,
        max_videos=args.limit,
        render_visuals=args.render,
        force_reprocess=args.force,
        assemble_only=args.assemble_only,
        retry_failed=args.retry_failed,
        conf_threshold=args.conf,
    )
