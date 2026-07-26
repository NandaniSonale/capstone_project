#!/usr/bin/env python3
"""List all P-frames in roi_motion_data.json (fast, no full load)."""

import json
import os
import re
import sys

ROI_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tracking_outputs",
    "roi_motion_data.json",
)


def list_frames_fast(path):
    """Scan file for frame keys without parsing 5M lines of JSON."""
    pattern = re.compile(r'^\s{2}"(frame_\d+)":\s*\[')
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            m = pattern.match(line)
            if m:
                frames.append((line_no, m.group(1)))
    return frames


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ROI_PATH
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return

    print(f"Scanning: {path}\n")
    frames = list_frames_fast(path)
    print(f"Total P-frames in file: {len(frames)}\n")

    if not frames:
        print("No frame_* keys found.")
        return

    print("First 10 frames (line number in JSON file):")
    for line_no, key in frames[:10]:
        print(f"  line {line_no:>8}  ->  {key}")

    if len(frames) > 10:
        print("  ...")
        print("Last 5 frames:")
        for line_no, key in frames[-5:]:
            print(f"  line {line_no:>8}  ->  {key}")

    print("\nTip: In the editor, press Ctrl+G and jump to line", frames[1][0] if len(frames) > 1 else "?", "to see frame_1024")


if __name__ == "__main__":
    main()
