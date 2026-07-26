#!/usr/bin/env python3
"""
Debug script to verify P-frame motion extraction and ROI filtering.
This helps diagnose issues with the compressed-domain motion data pipeline.
"""

import os
import struct
import numpy as np
import json

def check_p_frame_extraction(video_cache_dir):
    """Check if P-frame motion data was extracted correctly"""
    
    print("\n" + "="*80)
    print("P-FRAME MOTION DATA VERIFICATION")
    print("="*80)
    
    # Look for P-frame binary files
    for file in os.listdir(video_cache_dir):
        if file.endswith('_P.bin'):
            print(f"\n✓ Found P-frame binary: {file}")
            
            bin_path = os.path.join(video_cache_dir, file)
            file_size = os.path.getsize(bin_path)
            
            # P-frame format: pts(8), mb_x(4), mb_y(4), dx(2), dy(2), energy(4)
            P_HDR_FORMAT = '<qii hhf'
            P_HDR_SIZE = struct.calcsize(P_HDR_FORMAT)
            
            num_records = file_size // P_HDR_SIZE
            print(f"  File size: {file_size} bytes")
            print(f"  Record size: {P_HDR_SIZE} bytes")
            print(f"  Total macroblock records: {num_records}")
            
            if num_records > 0:
                # Read first 5 records as sample
                print(f"\n  Sample P-frame motion vectors (first 5 MBs):")
                with open(bin_path, 'rb') as f:
                    for i in range(min(5, num_records)):
                        chunk = f.read(P_HDR_SIZE)
                        if len(chunk) < P_HDR_SIZE:
                            break
                        pts, mb_x, mb_y, dx, dy, energy = struct.unpack(P_HDR_FORMAT, chunk)
                        print(f"    [{i}] PTS={pts:6d} | MB({mb_x:3d},{mb_y:3d}) | "
                              f"Motion(dx={dx:+3d}, dy={dy:+3d}) | Energy={energy:6.2f}")
                
                # Statistics
                with open(bin_path, 'rb') as f:
                    all_data = []
                    for _ in range(num_records):
                        chunk = f.read(P_HDR_SIZE)
                        if len(chunk) < P_HDR_SIZE:
                            break
                        pts, mb_x, mb_y, dx, dy, energy = struct.unpack(P_HDR_FORMAT, chunk)
                        all_data.append((pts, mb_x, mb_y, dx, dy, energy))
                
                dx_vals = [d[3] for d in all_data]
                dy_vals = [d[4] for d in all_data]
                energy_vals = [d[5] for d in all_data]
                
                print(f"\n  Motion Vector Statistics:")
                print(f"    DX: min={min(dx_vals):6d}, max={max(dx_vals):6d}, "
                      f"mean={np.mean(dx_vals):7.2f}")
                print(f"    DY: min={min(dy_vals):6d}, max={max(dy_vals):6d}, "
                      f"mean={np.mean(dy_vals):7.2f}")
                print(f"    Energy: min={min(energy_vals):7.2f}, max={max(energy_vals):7.2f}, "
                      f"mean={np.mean(energy_vals):7.2f}")
                
                print(f"\n  Unique PTS values (frames): {len(set(d[0] for d in all_data))}")
                print(f"  Unique MB coordinates: {len(set((d[1], d[2]) for d in all_data))}")
            else:
                print("  ⚠ WARNING: No records found in P-frame binary!")
                print("  Check if FFmpeg extraction is running correctly.")
            
            return True
    
    print("\n⚠ WARNING: No P-frame binary files found!")
    print(f"  Checked directory: {video_cache_dir}")
    print("  Expected: *_P.bin files")
    return False


def check_roi_motion_data(roi_json_path):
    """Check if ROI motion data was collected correctly"""
    
    print("\n" + "="*80)
    print("ROI MOTION DATA VERIFICATION")
    print("="*80)
    
    if not os.path.exists(roi_json_path):
        print(f"\n⚠ File not found: {roi_json_path}")
        return False
    
    with open(roi_json_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("\n⚠ ROI motion data is EMPTY!")
        print("  This means either:")
        print("  1. No I-frame human detections were made")
        print("  2. P-frame motion extraction didn't run")
        print("  3. No macroblocks were inside the bounding box")
        return False
    
    print(f"\n✓ Found {len(data)} frame entries")
    
    # Count total macroblocks
    total_mbs = sum(len(frame_data) for frame_data in data.values())
    print(f"✓ Total ROI macroblocks across all frames: {total_mbs}")
    
    # Show sample
    first_frame = list(data.keys())[0]
    print(f"\nSample data from {first_frame}:")
    for i, mb in enumerate(data[first_frame][:3]):
        print(f"  [{i}] {mb}")
    
    return True


def check_i_frame_detections(csv_path):
    """Check if I-frame human detections were made"""
    
    print("\n" + "="*80)
    print("I-FRAME DETECTION VERIFICATION")
    print("="*80)
    
    if not os.path.exists(csv_path):
        print(f"\n⚠ File not found: {csv_path}")
        return False
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    print(f"\nTotal frames in CSV: {len(lines) - 1}")  # -1 for header
    
    # Find frames with detections
    detections = []
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) >= 4:
            pts, f_type, source = parts[0], parts[1], parts[2]
            confidence = float(parts[3])
            if confidence > 0:
                detections.append((pts, f_type, source, confidence))
    
    if detections:
        print(f"\n✓ Found {len(detections)} detection frames")
        print("\nDetection summary:")
        for pts, f_type, source, conf in detections[:5]:
            print(f"  PTS={pts:6s} | Type={f_type} | Source={source} | Conf={conf:.3f}")
    else:
        print("\n⚠ WARNING: No human detections found!")
        print("  This means:")
        print("  1. Model may not be loading correctly")
        print("  2. Model predictions are below confidence threshold (0.5)")
        print("  3. Model is not designed for this video content")
    
    return len(detections) > 0


def main():
    """Run all verification checks"""
    
    print("\n" + "#"*80)
    print("# COMPRESSED-DOMAIN P-FRAME ROI MOTION EXTRACTION - DEBUG CHECK")
    print("#"*80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "tracking_outputs", "extraction_cache")
    roi_json = os.path.join(base_dir, "tracking_outputs", "roi_motion_data.json")
    csv_path = os.path.join(base_dir, "tracking_outputs", "tracking_results.csv")
    
    print(f"\nBase directory: {base_dir}")
    print(f"Cache directory: {cache_dir}")
    
    # Run checks
    if os.path.exists(cache_dir):
        check_p_frame_extraction(cache_dir)
    else:
        print(f"\n⚠ Cache directory not found: {cache_dir}")
        print("  Have you run the tracker yet?")
    
    check_i_frame_detections(csv_path)
    check_roi_motion_data(roi_json)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nData Flow Check:")
    print("  1. P-frame extraction (FFmpeg) ← Check first")
    print("  2. I-frame detection (Model) ← Check second")
    print("  3. ROI filtering (Motion vectors) ← Check third")
    print("  4. Output JSON (roi_motion_data.json) ← Final verification")
    
    print("\nNext Steps:")
    print("  • If P-frame binary is empty: Check FFmpeg compilation")
    print("  • If detections are 0: Check model file and threshold")
    print("  • If ROI data is empty: Check bbox-to-MB coordinate mapping")


if __name__ == '__main__':
    main()
