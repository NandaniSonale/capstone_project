import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from compressed_domain_tracker import CompressedDomainTracker

# =====================================================================
# CONFIGURATION
# =====================================================================
DATASET_ROOT = r"Human Activity Recognition - Video Dataset"
MODEL_PATH = r"best_model .h5"
CACHE_DIR = r"tracking_outputs/features_cache"
MAX_SEQUENCE_LENGTH = 60  # Number of P-frames to keep per sequence
NUM_FEATURES = 8         # Number of features extracted per frame

# Create directories if they do not exist
os.makedirs(CACHE_DIR, exist_ok=True)


# =====================================================================
# STEP 1: SPATIO-TEMPORAL FEATURE EXTRACTION FROM ROI DATA
# =====================================================================
def extract_temporal_features(json_path, max_frames=60):
    """
    Loads raw macroblock ROI motion data from JSON and aggregates the 
    variable number of macroblocks per frame into a fixed-size 
    feature vector per frame: [mean_dx, mean_dy, std_dx, std_dy, mean_mag, mean_energy, std_energy, density]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    sequence = []
    
    # Sort frames chronologically by numerical PTS
    sorted_frames = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    
    for frame_key in sorted_frames:
        mbs = data[frame_key]
        if not mbs:
            # If frame has no ROI macroblocks, pad with zero features
            sequence.append(np.zeros(NUM_FEATURES))
            continue
            
        dxs = np.array([mb['dx'] for mb in mbs])
        dys = np.array([mb['dy'] for mb in mbs])
        energies = np.array([mb['dct_energy'] for mb in mbs])
        
        # Calculate spatial aggregations for the frame
        mean_dx = np.mean(dxs)
        mean_dy = np.mean(dys)
        std_dx = np.std(dxs)
        std_dy = np.std(dys)
        mean_mag = np.mean(np.sqrt(dxs**2 + dys**2))
        
        # Use log scale for DCT energy since it spans several orders of magnitude
        log_energies = np.log1p(energies)
        mean_energy = np.mean(log_energies)
        std_energy = np.std(log_energies)
        
        density = len(mbs)  # Total macroblocks inside the bounding box
        
        feature_vector = np.array([
            mean_dx, mean_dy, std_dx, std_dy, mean_mag, mean_energy, std_energy, density
        ])
        sequence.append(feature_vector)
        
    # Standardize length: Pad or Truncate sequence to max_frames
    if len(sequence) > max_frames:
        sequence = sequence[:max_frames]
    elif len(sequence) < max_frames:
        padding = [np.zeros(NUM_FEATURES) for _ in range(max_frames - len(sequence))]
        sequence.extend(padding)
        
    return np.array(sequence)


# =====================================================================
# STEP 2: COMPILE DATASET (EXTRACT FEATURES FOR ALL VIDEOS)
# =====================================================================
def load_or_extract_dataset():
    """
    Walks through the activity categories, processes each video through the
    compressed domain tracker (if not cached), and compiles the features and labels.
    """
    X = []
    y = []
    video_paths = []
    
    print("\n" + "="*80)
    print("🎬 COMPILED DATASET EXTRACTION")
    print("="*80)
    
    categories = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    print(f"Detected categories: {categories}\n")
    
    for category in categories:
        category_dir = os.path.join(DATASET_ROOT, category)
        videos = [f for f in os.listdir(category_dir) if f.lower().endswith(('.mp4', '.avi'))]
        
        print(f"📁 Processing Category: {category} ({len(videos)} videos)")
        
        for video_file in videos:
            video_path = os.path.join(category_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            # Create a unique cache filename for the extracted JSON
            cache_json_path = os.path.join(CACHE_DIR, f"{category}_{video_name}_roi.json")
            
            # If not cached, run the extraction pipeline
            if not os.path.exists(cache_json_path):
                print(f"  ⚡ Running tracker on: {video_file}...")
                tracker = CompressedDomainTracker(video_path, MODEL_PATH)
                
                # Divert output of the tracker to our cache path
                tracker.output_dir = CACHE_DIR
                tracker.run()
                
                # Move output file to its specific cache name
                default_output = os.path.join(CACHE_DIR, "roi_motion_data.json")
                if os.path.exists(default_output):
                    if os.path.exists(cache_json_path):
                        os.remove(cache_json_path)
                    os.rename(default_output, cache_json_path)
            
            # Load and aggregate features from JSON
            if os.path.exists(cache_json_path) and os.path.getsize(cache_json_path) > 2:
                features = extract_temporal_features(cache_json_path, MAX_SEQUENCE_LENGTH)
                X.append(features)
                y.append(category)
                video_paths.append(video_path)
            else:
                print(f"  ⚠️ Warning: No valid extraction cache found for {video_file}")
                
    return np.array(X), np.array(y), video_paths


# =====================================================================
# STEP 3: CREATE ACTION RECOGNITION MODEL
# =====================================================================
def build_lstm_model(input_shape=(60, 8), num_classes=5):
    """
    Builds a temporal sequence classifier network using LSTM layers
    designed to accept aggregated ROI motion features.
    """
    inputs = Input(shape=input_shape)
    
    # First LSTM Layer
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Second LSTM Layer
    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Fully Connected Output Layers
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =====================================================================
# MAIN PIPELINE RUNNER
# =====================================================================
def main():
    # 1. Load dataset features
    X, y, video_paths = load_or_extract_dataset()
    
    if len(X) == 0:
        print("\n❌ Error: No features extracted. Ensure the dataset folder contains valid H.264 mp4 videos.")
        return
        
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Input shape (X): {X.shape} -> (Samples, Timesteps, Features)")
    print(f"   Labels shape (y): {y.shape}")
    
    # 2. Encode Labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    # Save label mapping for inference
    label_map = {int(i): str(name) for i, name in enumerate(encoder.classes_)}
    with open("tracking_outputs/action_labels.json", "w") as f:
        json.dump(label_map, f, indent=2)
        
    print(f"   Classes mapped: {label_map}")
    
    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_split=0.2, random_state=42, stratify=y_encoded
    )
    print(f"   Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # 4. Build Model
    model = build_lstm_model(input_shape=(MAX_SEQUENCE_LENGTH, NUM_FEATURES), num_classes=num_classes)
    model.summary()
    
    # 5. Train Model
    print("\n🚀 Training Action Recognition Model...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=8,
        callbacks=callbacks
    )
    
    # 6. Save final model
    model.save("tracking_outputs/action_recognition_lstm.h5")
    print("\n🎉 Training Complete! Model saved to: tracking_outputs/action_recognition_lstm.h5")
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"📊 Test Accuracy: {accuracy*100:.2f}% | Test Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
