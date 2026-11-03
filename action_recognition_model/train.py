import os
import re
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from models import TemporalCNNActionClassifier, BiLSTMActionClassifier

NUM_FEATURES = 8
MAX_TIMESTEPS = 60


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


class ActionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def extract_features_from_annotation_file(ann_path, max_timesteps=MAX_TIMESTEPS):
    """Extract features from HAR_annotations text file."""
    boxes = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('frame'):
                continue
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    f_idx = int(parts[0])
                    x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = max(x2 - x1, 0.02)
                    h = max(y2 - y1, 0.02)
                    boxes.append((f_idx, cx, cy, w, h))
                except ValueError:
                    continue

    if not boxes:
        return np.zeros((max_timesteps, NUM_FEATURES), dtype=np.float32)

    boxes.sort(key=lambda x: x[0])
    sequence = []

    for i in range(len(boxes)):
        curr = boxes[i]
        prev = boxes[i - 1] if i > 0 else curr

        dx = (curr[1] - prev[1]) * 100.0
        dy = (curr[2] - prev[2]) * 100.0
        mag = np.sqrt(dx**2 + dy**2)

        w_mb = max(curr[3] * 1080.0 / 16.0, 1.0)
        h_mb = max(curr[4] * 1920.0 / 16.0, 1.0)
        density = w_mb * h_mb

        energy = np.log1p(15.0 + abs(dx) * 4.0 + abs(dy) * 4.0)

        feat = np.array([
            dx, dy, abs(dx) * 0.3, abs(dy) * 0.3, mag, energy, energy * 0.25, density / 100.0
        ], dtype=np.float32)
        sequence.append(feat)

    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence) > max_timesteps:
        indices = np.linspace(0, len(sequence) - 1, max_timesteps).astype(int)
        sequence = sequence[indices]
    elif len(sequence) < max_timesteps:
        pad_len = max_timesteps - len(sequence)
        padding = np.zeros((pad_len, NUM_FEATURES), dtype=np.float32)
        sequence = np.vstack([sequence, padding])

    return sequence


def extract_features_from_roi_json(roi_json_path, max_timesteps=MAX_TIMESTEPS):
    """Extract features directly from extracted H.264 roi_motion_data.json."""
    try:
        with open(roi_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    if not data:
        return None

    sequence = []
    frame_keys = sorted(data.keys(), key=lambda k: int(re.sub(r'\D', '', k) or 0))

    for k in frame_keys:
        mbs = data[k]
        if not mbs:
            sequence.append(np.zeros(NUM_FEATURES, dtype=np.float32))
            continue

        dxs = [mb.get('dx', 0) for mb in mbs]
        dys = [mb.get('dy', 0) for mb in mbs]
        energies = [mb.get('dct_energy', 0.0) for mb in mbs]
        mags = [np.sqrt(dx**2 + dy**2) for dx, dy in zip(dxs, dys)]

        mean_dx = float(np.mean(dxs))
        mean_dy = float(np.mean(dys))
        std_dx = float(np.std(dxs))
        std_dy = float(np.std(dys))
        mean_mag = float(np.mean(mags))
        mean_energy = float(np.mean(energies))
        std_energy = float(np.std(energies))
        active_mbs = float(len(mbs))

        sequence.append([mean_dx, mean_dy, std_dx, std_dy, mean_mag, mean_energy, std_energy, active_mbs])

    if not sequence:
        return None

    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence) > max_timesteps:
        indices = np.linspace(0, len(sequence) - 1, max_timesteps).astype(int)
        sequence = sequence[indices]
    elif len(sequence) < max_timesteps:
        pad_len = max_timesteps - len(sequence)
        padding = np.zeros((pad_len, NUM_FEATURES), dtype=np.float32)
        sequence = np.vstack([sequence, padding])

    return sequence


ALL_ACTION_CLASSES = [
    'Walking',
    'Clapping',
    'Sitting',
    'Standing Still',
    'Walking While Reading Book',
    'Walking While Using Phone',
    'Meet and Split'
]


def load_dataset(
    annotations_dir,
    output_dir="output",
    target_classes=None,
    samples_per_class=120,
    force_reload=False,
    cache_copy_dir="action_recognition_model"
):
    if target_classes is None:
        target_classes = ALL_ACTION_CLASSES

    cache_file = os.path.join(output_dir, "action_dataset_cache.npz")
    if os.path.exists(cache_file) and not force_reload:
        try:
            cached = np.load(cache_file, allow_pickle=True)
            X = cached['X']
            y = cached['y']
            class_names = list(cached['class_names'])
            mean = cached['mean']
            std = cached['std']
            print(f"[CACHE] Loaded pre-compiled dataset from: {cache_file}", flush=True)
            print(f"  Total Sequences : {X.shape[0]} | Timesteps: {X.shape[1]} | Features: {X.shape[2]}", flush=True)
            print(f"  Classes ({len(class_names)}): {class_names}\n", flush=True)
            return X, y, class_names, mean, std
        except Exception:
            pass

    X = []
    y = []
    class_names = [c for c in target_classes if os.path.exists(os.path.join(output_dir, c)) or os.path.exists(os.path.join(annotations_dir, c))]

    print("\n" + "=" * 65)
    print("COMPILING COMPRESSED-DOMAIN MULTI-CLASS DATASET (ALL 7 CLASSES)")
    print("=" * 65)

    real_counts = {}
    if os.path.exists(output_dir):
        # 1. First scan per-class directories: output/<class_name>/<video_name>/
        for c_name in class_names:
            c_dir = os.path.join(output_dir, c_name)
            if os.path.exists(c_dir) and os.path.isdir(c_dir):
                vdirs = [d for d in os.listdir(c_dir) if os.path.isdir(os.path.join(c_dir, d))]
                vdirs.sort(key=natural_sort_key)
                for vdir in vdirs:
                    possible_json_paths = [
                        os.path.join(c_dir, vdir, "motion", "roi_motion_data.json"),
                        os.path.join(c_dir, vdir, "roi_motion_data.json")
                    ]
                    for rpath in possible_json_paths:
                        if os.path.exists(rpath):
                            feat = extract_features_from_roi_json(rpath, MAX_TIMESTEPS)
                            if feat is not None:
                                X.append(feat)
                                y.append(class_names.index(c_name))
                                real_counts[c_name] = real_counts.get(c_name, 0) + 1
                            break

        # 2. Also check flat video directories in output_dir (for backward compatibility)
        flat_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d not in class_names]
        flat_dirs.sort(key=natural_sort_key)
        for vdir in flat_dirs:
            matched_class = None
            # Match longest class name first
            for c_cand in sorted(class_names, key=len, reverse=True):
                if vdir.startswith(c_cand):
                    matched_class = c_cand
                    break
            if matched_class and real_counts.get(matched_class, 0) < 5:
                possible_json_paths = [
                    os.path.join(output_dir, vdir, "motion", "roi_motion_data.json"),
                    os.path.join(output_dir, vdir, "roi_motion_data.json")
                ]
                for rpath in possible_json_paths:
                    if os.path.exists(rpath):
                        feat = extract_features_from_roi_json(rpath, MAX_TIMESTEPS)
                        if feat is not None:
                            X.append(feat)
                            y.append(class_names.index(matched_class))
                            real_counts[matched_class] = real_counts.get(matched_class, 0) + 1
                        break

    for c_name in class_names:
        print(f"  [Real H.264 Features] {c_name:28s}: {real_counts.get(c_name, 0)} sequences", flush=True)

    # 3. Fallback to annotation files only if real sequences are fewer than 30
    for c_name in class_names:
        already_have = real_counts.get(c_name, 0)
        if already_have >= 30:
            continue
        c_dir = os.path.join(annotations_dir, c_name)
        if not os.path.exists(c_dir):
            continue

        files = [f for f in os.listdir(c_dir) if f.lower().endswith('.txt')]
        files.sort(key=natural_sort_key)
        needed = max(samples_per_class - already_have, 30)
        selected_files = files[:needed]

        print(f"  [Annotations Data] {c_name:28s}: {len(selected_files)} sequences", flush=True)
        for fname in selected_files:
            fpath = os.path.join(c_dir, fname)
            feat = extract_features_from_annotation_file(fpath, MAX_TIMESTEPS)
            X.append(feat)
            y.append(class_names.index(c_name))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std

    try:
        np.savez_compressed(cache_file, X=X, y=y, class_names=np.array(class_names), mean=mean, std=std)
        # Keep the normalization cache alongside the checkpoint it belongs to.
        os.makedirs(cache_copy_dir, exist_ok=True)
        np.savez_compressed(os.path.join(cache_copy_dir, "action_dataset_cache.npz"),
                            X=X, y=y, class_names=np.array(class_names), mean=mean, std=std)
        print(f"[CACHE] Saved compiled dataset cache to: {cache_file}", flush=True)
    except Exception as e:
        print(f"[CACHE] Warning: Failed to save cache: {e}", flush=True)

    print(f"\nDataset compiled successfully!")
    print(f"  Total Sequences : {X.shape[0]} | Timesteps: {X.shape[1]} | Features: {X.shape[2]}")
    print(f"  Classes ({len(class_names)}): {class_names}\n")

    return X, y, class_names, mean, std


def train(
    annotations_dir=r"HAR_annotations",
    output_dir="output",
    save_dir="action_recognition_model",
    epochs=35,
    batch_size=16,
    lr=0.001,
    model_type="tcnn",
    target_classes=None,
    force_reload=False
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if target_classes is None:
        target_classes = ALL_ACTION_CLASSES

    X, y, class_names, mean, std = load_dataset(
        annotations_dir, output_dir=output_dir, target_classes=target_classes,
        samples_per_class=120, force_reload=force_reload, cache_copy_dir=save_dir
    )

    num_classes = len(class_names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    train_dataset = ActionDataset(X_train, y_train)
    val_dataset = ActionDataset(X_val, y_val)
    test_dataset = ActionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_tcnn = model_type.lower() in ('tcnn', 'temporal_cnn', 'cnn')
    if is_tcnn:
        model_arch_name = "Temporal Convolutional Network (Temporal 1D-CNN)"
        model = TemporalCNNActionClassifier(input_dim=NUM_FEATURES, num_classes=num_classes, dropout=0.3).to(device)
        model_filename = 'action_recognition_tcnn.pt'
        metrics_filename = 'action_recognition_metrics_tcnn.json'
        cm_filename = 'action_confusion_matrix_tcnn.png'
    else:
        model_arch_name = "Bidirectional LSTM (Bi-LSTM)"
        model = BiLSTMActionClassifier(input_dim=NUM_FEATURES, hidden_dim=64, num_layers=2, num_classes=num_classes, dropout=0.3).to(device)
        model_filename = 'action_recognition_bilstm.pt'
        metrics_filename = 'action_recognition_metrics.json'
        cm_filename = 'action_confusion_matrix.png'

    model_save_path = os.path.join(save_dir, model_filename)
    metrics_json_path = os.path.join(save_dir, metrics_filename)
    cm_plot_path = os.path.join(save_dir, cm_filename)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    best_model_state = None

    print("\n" + "=" * 65)
    print(f"TRAINING {model_arch_name.upper()} ({epochs} Epochs)")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        epoch_train_loss = total_loss / total
        epoch_train_acc = correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                v_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                v_correct += (preds == batch_y).sum().item()
                v_total += batch_y.size(0)

        epoch_val_loss = v_loss / v_total
        epoch_val_acc = v_correct / v_total
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.1f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.1f}%")

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Evaluation on Test Set
    model.eval()
    y_true, y_pred = [], []
    test_loss, t_total = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            t_total += batch_y.size(0)

    test_loss = test_loss / t_total
    test_acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Action Recognition ({model_arch_name}) Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # Save weights also to output_dir for backward compatibility
    torch.save(model.state_dict(), model_save_path)
    torch.save(model.state_dict(), os.path.join(output_dir, model_filename))

    metrics_report = {
        "model_architecture": model_arch_name,
        "num_classes": num_classes,
        "classes": class_names,
        "test_metrics": {
            "test_accuracy": round(float(test_acc), 4),
            "test_loss": round(float(test_loss), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1_score": round(float(f1_macro), 4),
            "weighted_f1_score": round(float(f1_weighted), 4)
        },
        "per_class_metrics": {},
        "confusion_matrix": cm.tolist()
    }

    p_per, r_per, f_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    for i, c_name in enumerate(class_names):
        metrics_report["per_class_metrics"][c_name] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1_score": round(float(f_per[i]), 4)
        }

    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_report, f, indent=2)
    with open(os.path.join(output_dir, metrics_filename), 'w') as f:
        json.dump(metrics_report, f, indent=2)

    print("\n" + "=" * 65)
    print("ACTION RECOGNITION DEEP LEARNING MODEL EVALUATION")
    print("=" * 65)
    print(f"Model Architecture  : {model_arch_name}")
    print(f"Input Feature Space : (60 Timesteps, 8 Motion Features)")
    print(f"Number of Classes   : {num_classes} ({', '.join(class_names)})\n")

    print(f"--- Global Performance ---")
    print(f"  Test Accuracy     : {test_acc * 100:.2f}%")
    print(f"  Test Loss         : {test_loss:.4f}")
    print(f"  Macro Precision   : {prec_macro * 100:.2f}%")
    print(f"  Macro Recall      : {rec_macro * 100:.2f}%")
    print(f"  Macro F1-Score    : {f1_macro * 100:.2f}%\n")

    print(f"--- Per-Class Performance ---")
    for c_name in class_names:
        stats = metrics_report["per_class_metrics"][c_name]
        print(f"  {c_name:28s} | Prec: {stats['precision']*100:.1f}% | Rec: {stats['recall']*100:.1f}% | F1: {stats['f1_score']*100:.1f}%")

    print(f"\nTrained Model saved to    : {model_save_path}")
    print(f"Metrics Report saved to   : {metrics_json_path}")
    print(f"Confusion Matrix saved to : {cm_plot_path}")
    print("=" * 65 + "\n")

    return metrics_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Action Recognition Model (Temporal CNN or BiLSTM)")
    parser.add_argument('--annotations', default=r"HAR_annotations", help="Annotations root folder")
    parser.add_argument('--output-dir', default="output", help="Extracted features output directory")
    parser.add_argument('--save-dir', default="action_recognition_model", help="Where to save models & reports")
    parser.add_argument('--epochs', type=int, default=35, help="Number of training epochs")
    parser.add_argument('--model-type', choices=['tcnn', 'bilstm'], default='tcnn', help="Model architecture ('tcnn' or 'bilstm')")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--force-reload', action='store_true', help="Force recompiling the dataset cache")
    args = parser.parse_args()

    train(
        annotations_dir=args.annotations,
        output_dir=args.output_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model_type,
        force_reload=args.force_reload
    )
