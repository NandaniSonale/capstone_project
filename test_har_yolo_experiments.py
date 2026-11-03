import unittest

import numpy as np

from har_yolo_benchmark import (
    add_detection_frame,
    action_features,
    classification_summary,
    detection_summary,
    iou_xyxy,
)


class HARExperimentMetricTests(unittest.TestCase):
    def test_action_features_have_fixed_model_input_shape(self):
        rows = [
            {"frame": 0, "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.6},
            {"frame": 1, "x1": 0.2, "y1": 0.2, "x2": 0.4, "y2": 0.6},
        ]
        features = action_features(rows)
        self.assertEqual(features.shape, (60, 8))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertEqual(features[1, 0], np.float32(10.0))

    def test_iou_for_identical_and_disjoint_boxes(self):
        first = np.asarray([0, 0, 10, 10], dtype=np.float32)
        self.assertEqual(iou_xyxy(first, first), 1.0)
        self.assertEqual(
            iou_xyxy(first, np.asarray([12, 12, 20, 20], dtype=np.float32)),
            0.0,
        )

    def test_detection_summary_confidence_threshold_and_ap(self):
        state = {
            "confidence_threshold": 0.25,
            "ground_truth_count": 0,
            "prediction_count": 0,
            "confidence_prediction_count": 0,
            "confidence_true_positives": 0,
            "confidence_false_positives": 0,
            "ranked_predictions": [],
        }
        truth = np.asarray([0, 0, 10, 10], dtype=np.float32)
        add_detection_frame(
            state,
            [
                (truth.copy(), 0.9),
                (np.asarray([20, 20, 30, 30], dtype=np.float32), 0.8),
            ],
            [truth],
        )
        result = detection_summary(state)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 2.0 / 3.0)
        self.assertEqual(result["ap50"], 1.0)

    def test_unsupported_action_class_remains_in_seven_class_metrics(self):
        result = classification_summary(
            [
                {"true_action": "Walking", "predicted_action": "Walking"},
                {"true_action": "Meet and Split", "predicted_action": "Walking"},
            ]
        )
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["per_class"]["Meet and Split"]["support"], 1)
        self.assertFalse(result["per_class"]["Meet and Split"]["supported_by_model"])
        self.assertEqual(result["confusion_matrix"][4][4], 1)
        self.assertEqual(result["confusion_matrix"][1][4], 1)


if __name__ == "__main__":
    unittest.main()
