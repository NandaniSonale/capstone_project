import numpy as np
import cv2
import tensorflow as tf

# --- CONFIG ---
MODEL_PATH = "/content/best_model.h5"   # your .h5 file
INPUT_FILE = "/content/0.npy"      # your .npy file
CONF_THRESHOLD = 0.5
GRID_SIZE = 7

# --- LOAD MODEL ---
print("🛠️ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model Loaded")

# --- LOAD INPUT ---
img = np.load(INPUT_FILE)

# Normalize (same as training)
img = (img - img.mean()) / (img.std() + 1e-8)

# Resize to 300x300
img = cv2.resize(img, (300, 300))

# Add batch dimension
img = np.expand_dims(img, axis=0)

# --- PREDICT ---
pred = model.predict(img)[0]   # (7,7,5)

print("\n📊 Output shape:", pred.shape)

# --- DECODE OUTPUT ---
boxes = []

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):

        conf = pred[i, j, 0]

        if conf < CONF_THRESHOLD:
            continue

        cx, cy, w, h = pred[i, j, 1:]

        abs_cx = (j + cx) / GRID_SIZE
        abs_cy = (i + cy) / GRID_SIZE

        xmin = (abs_cx - w/2) * 300
        ymin = (abs_cy - h/2) * 300
        xmax = (abs_cx + w/2) * 300
        ymax = (abs_cy + h/2) * 300

        boxes.append({
            "confidence": float(conf),
            "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
        })

# --- PRINT RESULTS ---
print("\n🎯 DETECTIONS:")

if len(boxes) == 0:
    print("❌ No objects detected")
else:
    for i, b in enumerate(boxes):
        print(f"\nObject {i+1}")
        print(f"Confidence: {b['confidence']:.3f}")
        print(f"BBox: {b['bbox']}")
