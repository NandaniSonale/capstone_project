# HAR versus YOLOv8 experiments

This add-on evaluates the existing five-output HAR BiLSTM and YOLOv8n without
changing the project's training or inference code. Run commands from the
repository root in PowerShell.

## What was inspected

- Dataset: `Human Activity Recognition - Video Dataset/`, seven class
  directories and 1,113 videos in total.
- Person boxes: one matching annotation text file exists for each inspected
  dataset video in `HAR_annotations/<class>/<video>.txt`. It contains frame
  indices and normalized person boxes.
- HAR checkpoint used: `action_recognition_model/action_recognition_bilstm.pt`.
  Its output is five classes in the order stored in the normalization cache.
  The matching training mean/std are in
  `action_recognition_model/action_dataset_cache.npz`.
- YOLO checkpoint used: `yolov8n.pt`, COCO person detector (class 0), not a
  seven-action classifier.
- The large root `best_model .h5` has a convolutional image-model configuration
  and is not the checkpoint paired with the five-class HAR metrics or PyTorch
  model definition. It is not treated as an action classifier.
- Environment observed at implementation: Windows 11, Python 3.12.10,
  AMD Ryzen 3 5300U, 7.87 GB RAM, PyTorch 2.9.0 CPU build, no CUDA GPU.

The actual supported action classes are `Walking`, `Clapping`, `Sitting`,
`Standing Still`, and `Walking While Using Phone`. The dataset additionally
contains `Meet and Split` and `Walking While Reading Book`. The HAR model
cannot predict those two classes; the evaluator does not expand or fake its
output layer.

## Annotation and metric validity

The repository's annotation-generation script runs YOLOv8n and writes only
person boxes above confidence 0.5. Those files are therefore pseudo-labels,
not independent ground truth. By default, the evaluator writes box precision,
recall, F1, AP50, and mAP50:95 as **annotation agreement diagnostics**, and
the report explicitly says they are not valid ground-truth detection scores.
Only pass `--annotations-are-ground-truth` if the boxes have been independently
verified; that flag marks the results as ground-truth metrics but does not
change the source annotations.

The HAR model consumes annotation-derived 60x8 features. Feature construction
reproduces the existing annotation feature converter and uses the saved
training normalization cache. Consequently, this is a compatible checkpoint
inference experiment, not a raw-video end-to-end HAR pipeline. The checkpoint
metrics and dataset cache do not include a held-out video-ID manifest, so
overlap with training videos cannot be ruled out. HAR metrics are reported over
all seven dataset labels, including zero support/prediction behavior for the
two unsupported labels. mAP is not an action-classification metric and is not
reported for the HAR classifier.

## Environment setup

Use the repository's existing environment if the listed imports are already
available. Otherwise, create an isolated environment and install the listed
dependencies. For CUDA, install the PyTorch build appropriate for the target
CUDA version before installing this requirements file.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_har_yolo.txt
```

The scripts use the model and dataset paths relative to the repository root.
Both experiments run on CPU with the currently available weights; GPU support
is used automatically if a CUDA-enabled PyTorch install is present.

## Experiment 1: complete dataset

Evaluate every video and every decoded frame:

```powershell
python evaluate_har_yolo.py complete `
  --dataset-root "Human Activity Recognition - Video Dataset" `
  --annotations-root HAR_annotations `
  --har-weights action_recognition_model\action_recognition_bilstm.pt `
  --normalization-cache action_recognition_model\action_dataset_cache.npz `
  --yolo-weights yolov8n.pt `
  --output-dir evaluation_results\complete_dataset
```

This discovers the seven dataset classes and evaluates all videos (currently
1,113); it does not cap frames or videos. To declare independently verified
annotations, add `--annotations-are-ground-truth`.

## Experiment 2: 70-video long-duration video

Select ten videos per class deterministically with seed 42, merge them with
aligned frame indices, and save the video, source boundaries, labels, and
merged annotations:

```powershell
python merge_har_videos.py `
  --dataset-root "Human Activity Recognition - Video Dataset" `
  --annotations-root HAR_annotations `
  --output-dir evaluation_results\long_duration `
  --samples-per-class 10 `
  --seed 42
```

Evaluate the whole merged stream as a continuous YOLO input, while evaluating
HAR independently on each of the 70 labeled source segments:

```powershell
python evaluate_har_yolo.py long `
  --manifest evaluation_results\long_duration\segments.csv `
  --merged-video evaluation_results\long_duration\long_duration_70_videos.mp4 `
  --merged-annotations evaluation_results\long_duration\merged_annotations.csv `
  --har-weights action_recognition_model\action_recognition_bilstm.pt `
  --normalization-cache action_recognition_model\action_dataset_cache.npz `
  --yolo-weights yolov8n.pt `
  --output-dir evaluation_results\long_duration\evaluation
```

The merge defaults to the first selected clip's dimensions and FPS; all clips
are resized and FPS-normalized to those output settings. `segments.csv` uses
zero-based `[start_frame, end_frame_exclusive)` boundaries. `merged_annotations.csv`
maps source boxes to the output frame timeline.

## Smoke test before a full run

This evaluates one video per class and at most 12 video frames. It writes
partial, explicitly limited outputs to a separate directory:

```powershell
python evaluate_har_yolo.py complete `
  --max-videos-per-class 1 `
  --max-frames-per-video 12 `
  --output-dir evaluation_results\smoke_test
```

For a quick merge/manifest smoke test without encoding the full 70 clips, use
the regular merge command with `--samples-per-class 1` and a separate output
directory. The main 70-video command retains its default of ten per class.

## Measurements and outputs

Each evaluation directory includes:

- `results.json`: settings, model metadata, hardware, runtime, action metrics,
  and detection/annotation-agreement metrics.
- `video_results.csv`: per-video (or merged stream) frame counts, wall time,
  decoded-video FPS, YOLO wrapper time, and framework preprocessing/inference/
  postprocessing timing.
- `action_predictions.csv`: per-video/segment HAR result and annotation,
  feature, model, and postprocessing latency.
- `comparison_table.csv`: side-by-side model size, parameter count, MACs,
  FLOPs, memory, speed, and task metrics. FLOPs use the stated `2 x THOP MACs`
  convention; operator coverage is tool-dependent.
- `detection_metrics.json` and `har_classification_metrics.json`.
- `runtime_comparison.png`, `model_resources.png`, `detection_metrics.png`, and
  `har_confusion_matrix.png`.
- `research_report.md`: generated report with methods, measured hardware,
  results, and limitations.

Memory is process peak RSS sampled every 100 ms; it is not GPU-only memory.
HAR throughput is reported as classified 60-step sequences per second, not raw
video frames per second, because this checkpoint does not receive raw frames.
YOLO latency is measured per decoded frame and framework-reported timing
separates preprocessing, network inference, and postprocessing. HAR latency is
for annotation parsing/feature preparation and one 60-step classification;
the different input modalities mean the latency values are not an equivalent
raw-video throughput comparison. Video decode time is included in YOLO wall
FPS. THOP profiles the concrete model input shapes (HAR `1x60x8`, YOLO
`1x3x640x640` by default).

## Reproducibility and limitations

- Long-video selection is deterministic for the seed and dataset file listing;
  the exact sample list is recorded in `segments.csv` and `segments.json`.
- The YOLO image size is 640, confidence threshold 0.25, and NMS IoU 0.70 by
  default. These settings are recorded in the result JSON.
- YOLOv8n does not classify the seven human activities. It can only be compared
  to person-box annotations; HAR class scores apply only to the HAR classifier.
- Pseudo-label-derived box scores are agreement with labels produced using
  YOLOv8n, not an unbiased model-quality assessment. Independent/manual box
  labels are needed for defensible detection metrics.
- Training overlap, feature-source differences, low CPU throughput, and
  possible people/track annotation omissions limit action-metric claims.
- A complete run processes all frames on the detected device. With the
  supplied CPU-only PyTorch runtime, expect substantial runtime; smoke-test
  first, and do not interpret a capped smoke test as the complete experiment.
