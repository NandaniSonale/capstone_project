#!/usr/bin/env python3
"""Train and cross-validate a seven-class motion-feature action classifier."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "action_recognition_model_7class"
ANNOTATIONS_DIR = ROOT / "HAR_annotations"
MODEL_PATH = MODEL_DIR / "action_recognition_rf.joblib"
METRICS_PATH = MODEL_DIR / "action_recognition_rf_metrics.json"
sys.path.insert(0, str(ROOT / "action_recognition_model"))

from train import ALL_ACTION_CLASSES, extract_features_from_annotation_file, natural_sort_key


def summarize_sequences(sequences):
    return np.concatenate(
        [
            sequences.mean(axis=1),
            sequences.std(axis=1),
            np.quantile(sequences, [0.1, 0.5, 0.9], axis=1)
            .transpose(1, 2, 0)
            .reshape(len(sequences), -1),
        ],
        axis=1,
    )


def main():
    classes = ALL_ACTION_CLASSES
    sequences = []
    labels = []
    representative_features = {}
    representative_annotations = {}
    for class_index, class_name in enumerate(classes):
        annotation_paths = sorted(
            (ANNOTATIONS_DIR / class_name).glob("*.txt"),
            key=natural_sort_key,
        )
        if len(annotation_paths) <= 120:
            raise ValueError(
                f"{class_name} needs more than 120 annotations to reserve a held-out video"
            )
        representative = annotation_paths[120]
        representative_annotations[class_name] = representative.name
        representative_features[class_name] = extract_features_from_annotation_file(
            str(representative)
        )
        for annotation_path in annotation_paths:
            if annotation_path == representative:
                continue
            sequences.append(extract_features_from_annotation_file(str(annotation_path)))
            labels.append(class_index)

    sequences = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    counts = {
        name: int(np.sum(labels == index))
        for index, name in enumerate(classes)
    }
    if len(sequences) != len(labels) or any(count == 0 for count in counts.values()):
        raise ValueError(f"Invalid annotation feature dataset: {counts}")

    normalization_mean = sequences.mean(axis=(0, 1), keepdims=True)
    normalization_std = sequences.std(axis=(0, 1), keepdims=True) + 1e-6
    sequences = (sequences - normalization_mean) / normalization_std
    features = summarize_sequences(sequences)
    estimator = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        estimator,
        features,
        labels,
        cv=cross_validation,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        return_train_score=False,
        n_jobs=1,
    )
    out_of_fold = cross_val_predict(estimator, features, labels, cv=cross_validation, n_jobs=1)
    estimator.fit(features, labels)
    representative_predictions = {}
    for class_name in classes:
        normalized = (
            representative_features[class_name][None, ...] - normalization_mean
        ) / normalization_std
        probabilities = estimator.predict_proba(summarize_sequences(normalized))[0]
        predicted_index = int(np.argmax(probabilities))
        representative_predictions[class_name] = {
            "predicted_class": classes[predicted_index],
            "confidence": float(probabilities[predicted_index]),
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": estimator,
            "classes": classes,
            "feature_type": "mean, standard deviation, and 10th/50th/90th percentile per motion feature",
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
        },
        MODEL_PATH,
    )

    report = classification_report(
        labels,
        out_of_fold,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": "RandomForestClassifier",
        "classes": classes,
        "training_annotations_per_class": counts,
        "held_out_representatives": representative_annotations,
        "held_out_representative_predictions": representative_predictions,
        "held_out_representative_accuracy": float(
            sum(
                result["predicted_class"] == class_name
                for class_name, result in representative_predictions.items()
            )
            / len(classes)
        ),
        "cross_validation": {
            "method": "5-fold stratified cross-validation",
            "accuracy_mean": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "macro_f1_mean": float(scores["test_macro_f1"].mean()),
            "macro_f1_std": float(scores["test_macro_f1"].std()),
            "out_of_fold_accuracy": float(accuracy_score(labels, out_of_fold)),
            "out_of_fold_macro_f1": float(f1_score(labels, out_of_fold, average="macro")),
            "out_of_fold_confusion_matrix": confusion_matrix(
                labels, out_of_fold, labels=np.arange(len(classes))
            ).tolist(),
            "out_of_fold_classification_report": report,
        },
        "training_source": "All available HAR annotation-derived bounding-box trajectories except one held-out representative per class",
        "limitations": [
            "These are motion-feature classification scores, not RGB-video recognition scores.",
            "The held-out representative accuracy covers only one video per class.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"5-fold accuracy: {scores['test_accuracy'].mean():.1%} +/- {scores['test_accuracy'].std():.1%}")
    print(f"5-fold macro F1: {scores['test_macro_f1'].mean():.3f} +/- {scores['test_macro_f1'].std():.3f}")
    print(f"Saved classifier: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
