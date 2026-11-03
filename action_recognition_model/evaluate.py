import os
import sys
import json
import argparse
import numpy as np
import torch

from models import TemporalCNNActionClassifier, BiLSTMActionClassifier

NUM_FEATURES = 8
MAX_TIMESTEPS = 60
DEFAULT_CLASSES = [
    'Walking',
    'Clapping',
    'Sitting',
    'Standing Still',
    'Walking While Reading Book',
    'Walking While Using Phone',
    'Meet and Split'
]


def extract_features_from_json(roi_json_path, max_timesteps=MAX_TIMESTEPS):
    """Convert an roi_motion_data.json into a (1, 60, 8) tensor."""
    with open(roi_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        return None

    sequence = []
    frame_keys = sorted(data.keys(), key=lambda k: int(''.join(filter(str.isdigit, k)) or 0))

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

    return np.expand_dims(sequence, axis=0)  # Shape: (1, 60, 8)


def predict_action(json_path, weights_path, model_type="tcnn", classes=None):
    if classes is None:
        classes = DEFAULT_CLASSES

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type.lower() in ('tcnn', 'temporal_cnn', 'cnn'):
        model = TemporalCNNActionClassifier(input_dim=NUM_FEATURES, num_classes=len(classes))
    else:
        model = BiLSTMActionClassifier(input_dim=NUM_FEATURES, hidden_dim=64, num_layers=2, num_classes=len(classes))

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    X = extract_features_from_json(json_path)
    if X is None:
        print(f"[ERROR] Could not extract features from {json_path}")
        return None

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    pred_label = classes[pred_idx]
    confidence = float(probs[pred_idx])

    print(f"\n==========================================")
    print(f"PREDICTED ACTION: {pred_label} ({confidence * 100:.2f}%)")
    print(f"==========================================")
    for c_name, p in zip(classes, probs):
        print(f"  {c_name:28s}: {p * 100:.2f}%")
    print(f"==========================================\n")

    return {
        "predicted_action": pred_label,
        "confidence": confidence,
        "class_probabilities": {c: float(p) for c, p in zip(classes, probs)}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate / Run Inference on Compressed-Domain Motion JSON")
    parser.add_argument('--input', '-i', required=True, help="Path to roi_motion_data.json")
    parser.add_argument('--weights', '-w', default="action_recognition_model/action_recognition_tcnn.pt", help="Path to trained model weights (.pt)")
    parser.add_argument('--model-type', choices=['tcnn', 'bilstm'], default='tcnn', help="Model architecture")
    args = parser.parse_args()

    predict_action(args.input, args.weights, model_type=args.model_type)
