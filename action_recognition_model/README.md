# Compressed-Domain Action Recognition Model

This module contains all deep learning architectures, training pipelines, evaluation routines, and model checkpoints for **Compressed-Domain Human Action Recognition** directly from H.264 macroblock motion vectors $(dx, dy)$ and transform DCT residual energies without decoding full RGB pixels.

---

## 📁 Folder Contents

```
action_recognition_model/
├── models.py                          # Neural network architectures: Temporal 1D-CNN & BiLSTM
├── train.py                           # Training & validation pipeline with metrics and confusion matrix
├── evaluate.py                        # Single-video inference and evaluation script
├── train_tf_keras.py                  # TensorFlow/Keras training alternative
├── action_recognition_tcnn.pt         # Trained Temporal CNN model weights (PyTorch)
├── action_recognition_bilstm.pt       # Trained BiLSTM model weights (PyTorch)
├── action_recognition_model.h5        # Trained Keras/TensorFlow model weights
├── action_recognition_metrics_tcnn.json# Performance report for Temporal CNN
├── action_recognition_metrics.json    # Performance report for BiLSTM
├── action_confusion_matrix_tcnn.png   # Confusion matrix plot for Temporal CNN
├── action_confusion_matrix.png        # Confusion matrix plot for BiLSTM
├── action_dataset_cache.npz           # Pre-compiled dataset cache (instant loading)
└── README.md                          # Documentation and usage guide
```

---

## 🧠 Model Architectures

### 1. Temporal Convolutional Network (Temporal 1D-CNN) - *Recommended*
- **Architecture**: 1D Temporal Convolutions with dilated residual blocks ($d=1, 2, 4$) and Dual Global Temporal Pooling (Avg + Max).
- **Why it's better**:
  - Full parallelization across 60 timesteps (trains and infers much faster than sequential RNNs).
  - Multi-scale receptive fields capture both rapid micro-movements (gestures, reading a book) and broad periodic motions (walking cadence).
  - Residual connections prevent vanishing gradients.

### 2. Bidirectional LSTM (Bi-LSTM)
- **Architecture**: 2-Layer BiLSTM with 64 and 32 hidden units, Batch Normalization, and Dropout (0.3).
- **Role**: Sequential baseline processing temporal dependencies in forward and backward directions.

---

## 📐 Input Feature Specification

The models take sequence tensors of shape:

$$\mathbf{X} \in \mathbb{R}^{B \times 60 \times 8}$$

- **$B$**: Batch Size (e.g. 16 or 32)
- **$60$**: Timesteps (frames sampled across video duration)
- **$8$**: Compressed-domain motion features per frame:
  1. `mean_dx`: Average horizontal motion vector across macroblocks in ROI.
  2. `mean_dy`: Average vertical motion vector across macroblocks in ROI.
  3. `std_dx`: Horizontal motion vector standard deviation.
  4. `std_dy`: Vertical motion vector standard deviation.
  5. `mean_mag`: Mean motion vector magnitude $\sqrt{dx^2 + dy^2}$ (movement speed).
  6. `mean_energy`: Mean DCT transform residual energy ($\sum |\text{coeffs}|$).
  7. `std_energy`: Standard deviation of DCT energy across macroblocks.
  8. `active_mbs`: Total count of active macroblocks inside the bounding box.

---

## 🚀 How to Run

### 1. Train the Temporal CNN (Default)
```powershell
python action_recognition_model/train.py --model-type tcnn --epochs 35
```

### 2. Train the BiLSTM Model
```powershell
python action_recognition_model/train.py --model-type bilstm --epochs 35
```

### 3. Run Inference on a Video's Motion JSON
```powershell
python action_recognition_model/evaluate.py --input "output/Walking (1)/motion/roi_motion_data.json" --weights "action_recognition_model/action_recognition_tcnn.pt" --model-type tcnn
```
