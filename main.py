"""
=============================================================================
main.py - Master Entry Point (Capstone Project / Action Recognition)
=============================================================================
Provides a root-level master entry point that routes to the Action Recognition
pipeline in Action_Recognition/main.py:

Usage Examples:
  python main.py preprocess
  python main.py train
  python main.py inference --sample Action_Recognition/data/processed/sample_00000.npy
  python main.py inference --video "output/Walking (1)"
=============================================================================
"""

import os
import sys

# Add Action_Recognition folder to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_REC_DIR = os.path.join(CURRENT_DIR, "Action_Recognition")

if ACTION_REC_DIR not in sys.path:
    sys.path.insert(0, ACTION_REC_DIR)

# Switch working directory to Action_Recognition for relative paths if needed
try:
    from Action_Recognition.main import main as action_main
except ImportError:
    import sys
    sys.path.append(ACTION_REC_DIR)
    from main import main as action_main

if __name__ == '__main__':
    # Forward execution to Action_Recognition master CLI
    action_main()
