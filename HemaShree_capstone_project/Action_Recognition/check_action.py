"""
check_action.py
===============
Runs Action Recognition inference on any video or roi_motion_data.json:
Usage:
    python check_action.py
    python check_action.py --input "output/Walking (1)/motion/roi_motion_data.json"
"""

import os
import sys

# Find root directory containing action_recognition_model
curr = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(curr, "action_recognition_model")):
    ROOT_DIR = curr
elif os.path.exists(os.path.join(os.path.dirname(curr), "action_recognition_model")):
    ROOT_DIR = os.path.dirname(curr)
elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(curr)), "action_recognition_model")):
    ROOT_DIR = os.path.dirname(os.path.dirname(curr))
else:
    ROOT_DIR = r"C:\Users\newuser\capstone_project"

sys.path.insert(0, os.path.join(ROOT_DIR, "action_recognition_model"))

from evaluate import predict_action

if __name__ == '__main__':
    default_json = os.path.join(ROOT_DIR, r"output\Walking (1)\motion\roi_motion_data.json")
    default_weights = os.path.join(ROOT_DIR, r"action_recognition_model\action_recognition_tcnn.pt")

    json_path = sys.argv[1] if len(sys.argv) > 1 else default_json
    weights_path = sys.argv[2] if len(sys.argv) > 2 else default_weights

    predict_action(json_path, weights_path, model_type="tcnn")
