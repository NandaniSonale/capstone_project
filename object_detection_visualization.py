import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# --- CONFIG ---
MODEL_PATH = "/content/best_model.h5"
INPUT_FILE = "/content/158.npy"
CONF_THRESHOLD = 0.5
GRID_SIZE = 7

# --- LOAD MODEL ---
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- LOAD NPY ---
data = np.load(INPUT_FILE)

# =========================
# 🔹 PART 1: MODEL INPUT
# =========================
img_model = data.copy()

img_model = (img_model - img_model.mean()) / (img_model.std() + 1e-8)
img_model = cv2.resize(img_model, (300, 300))
img_model = np.expand_dims(img_model, axis=0)

# --- PREDICT ---
pred = model.predict(img_model)[0]

# =========================
# 🔹 PART 2: VISUALIZATION (YOUR CODE)
# =========================
img = data[:, :, 0]

img = np.abs(img)
img = (img - img.min()) / (img.max() + 1e-8)
img = img.astype(np.float32)

smooth = cv2.GaussianBlur(img, (9, 9), 0)

high_freq = img - smooth
img_balanced = smooth + 0.2 * high_freq

p2, p98 = np.percentile(img_balanced, (3, 97))
img_enhanced = np.clip((img_balanced - p2) / (p98 - p2 + 1e-8), 0, 1)

img_final = cv2.GaussianBlur(img_enhanced, (3, 3), 0)

# Resize to match 300x300 (IMPORTANT)
img_display = cv2.resize(img_final, (300, 300))

# Convert to RGB for plotting boxes
img_display = (img_display * 255).astype(np.uint8)
img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)

# =========================
# 🔹 PART 3: DRAW BOXES
# =========================
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):

        conf = pred[i, j, 0]
        if conf < CONF_THRESHOLD:
            continue

        cx, cy, w, h = pred[i, j, 1:]

        abs_cx = (j + cx) / GRID_SIZE
        abs_cy = (i + cy) / GRID_SIZE

        xmin = int((abs_cx - w/2) * 300)
        ymin = int((abs_cy - h/2) * 300)
        xmax = int((abs_cx + w/2) * 300)
        ymax = int((abs_cy + h/2) * 300)

        # Draw box
        cv2.rectangle(img_display, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

        # Confidence text
        cv2.putText(img_display, f"{conf:.2f}", (xmin, ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# =========================
# 🔹 SHOW RESULT
# =========================
plt.figure(figsize=(6,6))
plt.imshow(img_display)
plt.title("DCT Visualization + Predictions")
plt.axis('off')
plt.show()
