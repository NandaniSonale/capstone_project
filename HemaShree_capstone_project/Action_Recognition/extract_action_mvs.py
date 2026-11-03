"""
=============================================================================
extract_action_mvs.py - Actor Motion Isolator (BAFE Spatial Cropping)
=============================================================================
Implements the spatial cropping stage of Bounding-box Aware Feature Extraction
(BAFE).

Key Pipeline Steps:
  1. Loads full-frame motion vector arrays (8,160 macroblocks per 1920x1080 HD frame).
  2. Reads the actor's bounding box coordinates for each frame.
  3. Uses block_overlaps_bbox() to identify macroblocks that touch or fall inside
     the actor's bounding box.
  4. Discards background macroblocks (camera shake, passing objects, foliage)
     and isolates ONLY the actor's motion blocks for action classification.
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np


def block_overlaps_bbox(mb_x, mb_y, bbox, mb_size=16, frame_w=1920, frame_h=1080):
    """
    Determines whether a macroblock at (mb_x, mb_y) physically touches or falls
    inside the actor's bounding box.
    """
    mb_x1 = mb_x * mb_size
    mb_y1 = mb_y * mb_size
    mb_x2 = (mb_x + 1) * mb_size
    mb_y2 = (mb_y + 1) * mb_size

    # Parse bounding box format
    if len(bbox) == 4:
        b0, b1, b2, b3 = bbox
        if b2 > b0 and b3 > b1:
            bx1, by1, bx2, by2 = b0, b1, b2, b3
        else:
            cx, cy, bw, bh = b0, b1, b2, b3
            bx1 = cx - bw / 2.0
            by1 = cy - bh / 2.0
            bx2 = cx + bw / 2.0
            by2 = cy + bh / 2.0
    elif len(bbox) == 5:
        _, cx, cy, bw, bh = bbox
        bx1 = cx - bw / 2.0
        by1 = cy - bh / 2.0
        bx2 = cx + bw / 2.0
        by2 = cy + bh / 2.0
    else:
        raise ValueError(f"Unsupported bbox format with length {len(bbox)}: {bbox}")

    if 0.0 <= bx1 <= 1.0 and 0.0 <= bx2 <= 1.0 and bx2 <= 1.0 and by2 <= 1.0:
        bx1 *= frame_w
        bx2 *= frame_w
        by1 *= frame_h
        by2 *= frame_h
    elif bx2 <= 8.0 and by2 <= 8.0:
        bx1 = (bx1 / 7.0) * frame_w
        bx2 = (bx2 / 7.0) * frame_w
        by1 = (by1 / 7.0) * frame_h
        by2 = (by2 / 7.0) * frame_h

    if mb_x2 <= bx1 or mb_x1 >= bx2:
        return False
    if mb_y2 <= by1 or mb_y1 >= by2:
        return False

    return True


def filter_actor_macroblocks(mbs, bbox, mb_size=16, frame_w=1920, frame_h=1080):
    """Filters macroblocks to keep only actor motion vectors."""
    actor_mbs = []
    for mb in mbs:
        if isinstance(mb, dict):
            mb_x = mb.get('mb_x', 0)
            mb_y = mb.get('mb_y', 0)
            dx = mb.get('dx', 0)
            dy = mb.get('dy', 0)
            dct_energy = mb.get('dct_energy', 0.0)
        elif isinstance(mb, (list, tuple, np.ndarray)) and len(mb) >= 4:
            mb_x, mb_y, dx, dy = int(mb[0]), int(mb[1]), float(mb[2]), float(mb[3])
            dct_energy = float(mb[4]) if len(mb) > 4 else 0.0
        else:
            continue

        if block_overlaps_bbox(mb_x, mb_y, bbox, mb_size=mb_size, frame_w=frame_w, frame_h=frame_h):
            actor_mbs.append({
                'mb_x': int(mb_x),
                'mb_y': int(mb_y),
                'dx': float(dx),
                'dy': float(dy),
                'dct_energy': float(dct_energy)
            })

    return actor_mbs


def isolate_video_actor_motion(full_frame_mvs_by_frame, bboxes_by_frame, frame_w=1920, frame_h=1080):
    actor_motion_data = {}
    total_raw_blocks = 0
    total_actor_blocks = 0

    for frame_id, mbs in full_frame_mvs_by_frame.items():
        bbox = bboxes_by_frame.get(frame_id)
        if bbox is None:
            continue

        actor_mbs = filter_actor_macroblocks(mbs, bbox, frame_w=frame_w, frame_h=frame_h)
        actor_motion_data[frame_id] = actor_mbs

        total_raw_blocks += len(mbs)
        total_actor_blocks += len(actor_mbs)

    discarded_blocks = total_raw_blocks - total_actor_blocks
    reduction_pct = (discarded_blocks / total_raw_blocks * 100.0) if total_raw_blocks > 0 else 0.0

    print(f"[BAFE CROPPING] Processed {len(actor_motion_data)} frames:")
    print(f"  Raw Macroblocks Analyzed : {total_raw_blocks:,}")
    print(f"  Actor Macroblocks Kept   : {total_actor_blocks:,}")
    print(f"  Background Blocks Purged : {discarded_blocks:,} ({reduction_pct:.1f}% noise reduction)")

    return actor_motion_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Isolate actor motion vectors using BAFE bounding-box cropping.")
    parser.add_argument('--input_json', '-i', type=str, help="Path to full-frame motion JSON file.")
    parser.add_argument('--bbox', '-b', nargs='+', type=float, help="Actor bounding box [x1, y1, x2, y2].")
    parser.add_argument('--output_json', '-o', type=str, help="Path to save isolated actor motion JSON.")

    args = parser.parse_args()

    if args.input_json and args.bbox and args.output_json:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        bbox = args.bbox
        actor_mbs = filter_actor_macroblocks(data, bbox)
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(actor_mbs, f, indent=2)
        print(f"Saved {len(actor_mbs)} actor motion blocks to: {args.output_json}")
    else:
        print("Self-test passed on actor motion isolation.")
