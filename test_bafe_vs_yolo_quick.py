import os
import cv2
import av
import numpy as np
import pandas as pd
from ultralytics import YOLO

video_path = r'Human Activity Recognition - Video Dataset/Walking/Walking (1).mp4'
csv_path = r'output/Walking (1)/tracking/tracking_results.csv'

df = pd.read_csv(csv_path)
yolo = YOLO('yolov8n')

container = av.open(video_path)
stream = container.streams.video[0]
w_img, h_img = stream.width, stream.height
grid_size = 7.0

print(f"Video Resolution: {w_img}x{h_img}")

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

tested = 0
for frame in container.decode(video=0):
    pts = frame.pts
    sub = df[df['pts'] == pts]
    if sub.empty:
        continue

    r = sub.iloc[0]
    cx_rel = r['cx'] / grid_size
    cy_rel = r['cy'] / grid_size
    w_rel = r['w'] / grid_size
    h_rel = r['h'] / grid_size

    bafe_box = [
        (cx_rel - w_rel/2) * w_img,
        (cy_rel - h_rel/2) * h_img,
        (cx_rel + w_rel/2) * w_img,
        (cy_rel + h_rel/2) * h_img
    ]

    img = frame.to_ndarray(format='bgr24')
    results = yolo(img, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    f_type = r['frame_type']
    source = r['source']

    if len(boxes) > 0:
        ious = [compute_iou(bafe_box, b) for b in boxes]
        best_idx = int(np.argmax(ious))
        best_iou = ious[best_idx]
        best_box = boxes[best_idx]
        print(f"PTS {pts:5d} [{f_type:1s} {source:4s}] | BAFE: {[round(x,1) for x in bafe_box]} | YOLO: {[round(x,1) for x in best_box]} | IoU: {best_iou:.3f}")
    else:
        print(f"PTS {pts:5d} [{f_type:1s} {source:4s}] | No YOLO detection")

    tested += 1
    if tested >= 15:
        break
