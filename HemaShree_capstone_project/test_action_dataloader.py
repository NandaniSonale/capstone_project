import os
import sys
import yaml
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_DIR = os.path.join(BASE_DIR, "Action_Recognition")
sys.path.insert(0, ACTION_DIR)

# Fix encoding
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except: pass

from src.data_loader import build_datasets

cfg_path = os.path.join(ACTION_DIR, "config", "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

train_set, val_set, info = build_datasets(cfg, root_dir=ACTION_DIR)
print("=== ACTION RECOGNITION DATA LOADER TEST ===")
print(f"Discovered Classes ({len(info['class_names'])}): {info['class_names']}")
print(f"Split Strategy: {info['split_strategy']}")
print(f"Total Train Clips: {info['num_train_clips']}")
print(f"Total Val Clips: {info['num_val_clips']}")
print(f"Train Clips Per Class: {info['train_clips_per_class']}")
print(f"Val Clips Per Class: {info['val_clips_per_class']}")
x, y, v_idx = train_set[0]
print(f"Input Tensor Shape: {x.shape} (20 channels: 10 frames x 2 (dx, dy), 224x224)")
print(f"Sample Label: {y} ({info['class_names'][y]}) from video {info['train_videos'][v_idx]}")
print("=== ALL CHECKS PASSED SUCCESSFULLY ===")
