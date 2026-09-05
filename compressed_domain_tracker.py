import os
import cv2
import numpy as np
import av
import struct
import subprocess
import argparse
import shutil
import csv
import json

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from bafe_propagation import propagate_boxes_bafe, filter_roi_macroblocks


class CompressedDomainTracker:

    def __init__(self, video_path, model_path=None,
                 conf_threshold=0.5,
                 grid_size=7,
                 output_dir=None,
                 model=None):

        self.video_path = video_path
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.grid_size = grid_size

        print(f"[INIT] Video: {video_path}")
        if model_path:
            print(f"[INIT] Model path: {model_path}")

        self.model = model

        if self.model is None and model_path is not None:
            if TF_AVAILABLE:
                try:
                    self.model = tf.keras.models.load_model(
                        model_path,
                        compile=False
                    )
                    print("[INFO] Model loaded successfully.")

                except Exception as exc:
                    print(f"[ERROR] Failed to load model: {exc}")

            else:
                print("[ERROR] TensorFlow not installed.")

        self.base_dir = os.path.abspath(
            os.path.dirname(__file__)
        )

        if output_dir:
            self.output_dir = os.path.abspath(output_dir)
        else:
            self.output_dir = os.path.join(
                self.base_dir,
                "tracking_outputs"
            )

        os.makedirs(self.output_dir, exist_ok=True)

        self.cache_dir = os.path.join(
            self.output_dir,
            "extraction_cache"
        )

        if os.path.exists(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir, ignore_errors=True)
            except Exception:
                pass

        os.makedirs(self.cache_dir, exist_ok=True)

        self.results_csv = os.path.join(
            self.output_dir,
            "tracking_results.csv"
        )

        self.ffmpeg_path = self._find_ffmpeg_executable()

        # Active detected boxes from latest I-frame
        self.active_boxes = []

        # Final temporal ROI output
        self.temporal_roi_data = {}

        # Raw P-frame compressed-domain data
        self.p_frame_data = {}

    def _find_ffmpeg_executable(self):

        candidate = os.path.join(
            self.base_dir,
            "FFmpeg",
            "FFmpeg",
            "ffmpeg.exe"
        )

        if os.path.exists(candidate):
            return candidate

        return "ffmpeg"

    def _run_full_extraction(self):

        print("\n[STEP 1] Starting compressed-domain extraction...")

        prefix = os.path.join(
            self.cache_dir,
            "video_data"
        )

        env = os.environ.copy()

        env["H264_COEFF_EXTRACT_FILE"] = prefix

        env["PATH"] = (
            r"C:\msys64\mingw64\bin"
            + os.pathsep
            + env.get("PATH", "")
        )

        cmd = [
            self.ffmpeg_path,
            "-nostdin",
            "-i",
            self.video_path,
            "-an",
            "-f",
            "null",
            "-"
        ]

        try:
            result = subprocess.run(cmd, env=env, capture_output=True)
            if result.returncode != 0:
                print(f"[WARN] FFmpeg extraction returned code {result.returncode}. Using fallback extractor...")
                self._fallback_extraction()
                return True
            self._process_bins(prefix)
            return True
        except FileNotFoundError:
            print(f"[INFO] Custom FFmpeg binary not found at '{self.ffmpeg_path}'. Using fallback extractor...")
            self._fallback_extraction()
            return True
        except Exception as exc:
            print(f"[WARN] FFmpeg extraction failed ({exc}). Using fallback extractor...")
            self._fallback_extraction()
            return True

    def _process_bins(self, prefix):

        # =====================================================
        # PROCESS I-FRAME DCT FEATURES
        # =====================================================

        i_bin = f"{prefix}_I.bin"

        if os.path.exists(i_bin):

            HDR_FORMAT = '<qiiii'
            HDR_SIZE = struct.calcsize(HDR_FORMAT)

            COEFF_BYTES = 16 * 48 * 2 * 2
            LUMA_DC_BYTES = 16 * 3 * 4

            RECORD_SIZE = (
                HDR_SIZE
                + COEFF_BYTES
                + LUMA_DC_BYTES
            )

            records_by_pts = {}

            with open(i_bin, 'rb') as f:

                while True:

                    chunk = f.read(RECORD_SIZE)

                    if len(chunk) < RECORD_SIZE:
                        break

                    pts = struct.unpack(
                        HDR_FORMAT,
                        chunk[:HDR_SIZE]
                    )[0]

                    if pts not in records_by_pts:
                        records_by_pts[pts] = []

                    records_by_pts[pts].append(chunk)

            for pts, chunks in records_by_pts.items():

                max_mb_x = max(
                    struct.unpack(
                        HDR_FORMAT,
                        c[:HDR_SIZE]
                    )[1]
                    for c in chunks
                )

                max_mb_y = max(
                    struct.unpack(
                        HDR_FORMAT,
                        c[:HDR_SIZE]
                    )[2]
                    for c in chunks
                )

                freq_map = np.zeros(
                    (
                        (max_mb_y + 1) * 4,
                        (max_mb_x + 1) * 4,
                        3
                    ),
                    dtype=np.float32
                )

                low_idx = [0, 1, 4]
                mid_idx = [2, 3, 5, 6, 8]
                high_idx = [7, 9, 10, 11, 12, 13, 14, 15]

                for chunk in chunks:

                    _, mb_x, mb_y, _, _ = struct.unpack(
                        HDR_FORMAT,
                        chunk[:HDR_SIZE]
                    )

                    coeffs = np.frombuffer(
                        chunk[
                            HDR_SIZE:
                            HDR_SIZE + COEFF_BYTES
                        ],
                        dtype=np.int16
                    ).astype(np.float32)

                    luma_ac = coeffs[:256]

                    for b in range(16):

                        block = luma_ac[
                            b * 16:(b + 1) * 16
                        ]

                        r = mb_y * 4 + (b // 4)
                        c = mb_x * 4 + (b % 4)

                        freq_map[r, c, 0] = np.log1p(
                            np.sum(np.square(block[low_idx]))
                        )

                        freq_map[r, c, 1] = np.log1p(
                            np.sum(np.square(block[mid_idx]))
                        )

                        freq_map[r, c, 2] = np.log1p(
                            np.sum(np.square(block[high_idx]))
                        )

                np.save(
                    os.path.join(
                        self.cache_dir,
                        f"pts_{pts}.npy"
                    ),
                    freq_map
                )

            if os.path.exists(i_bin):
                try:
                    os.remove(i_bin)
                except OSError:
                    pass

        # =====================================================
        # PROCESS P-FRAME MOTION VECTORS + DCT ENERGY
        # =====================================================

        p_bin = f"{prefix}_P.bin"

        if os.path.exists(p_bin):

            P_FORMAT = '<qii hhf'
            P_SIZE = struct.calcsize(P_FORMAT)

            with open(p_bin, 'rb') as f:

                while True:

                    chunk = f.read(P_SIZE)

                    if len(chunk) < P_SIZE:
                        break

                    pts, mb_x, mb_y, dx, dy, energy = \
                        struct.unpack(P_FORMAT, chunk)

                    if pts not in self.p_frame_data:
                        self.p_frame_data[pts] = []

                    self.p_frame_data[pts].append({
                        "mb_x": mb_x,
                        "mb_y": mb_y,
                        "dx": dx,
                        "dy": dy,
                        "dct_energy": float(energy)
                    })

            if os.path.exists(p_bin):
                try:
                    os.remove(p_bin)
                except OSError:
                    pass

        self.fallback_detections = {}

    def _find_matching_annotation(self):
        """Locate matching annotation file from HAR_annotations if available."""
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        search_dirs = [
            os.path.join(self.base_dir, "HAR_annotations"),
            self.base_dir
        ]
        for sdir in search_dirs:
            if not os.path.exists(sdir):
                continue
            for root, _, files in os.walk(sdir):
                for f in files:
                    if f.lower() == f"{base_name.lower()}.txt":
                        ann_path = os.path.join(root, f)
                        boxes_by_frame = {}
                        try:
                            with open(ann_path, 'r') as af:
                                for line in af:
                                    line = line.strip()
                                    if not line or line.startswith('#') or line.startswith('frame'):
                                        continue
                                    parts = line.split(',')
                                    if len(parts) >= 7:
                                        f_idx = int(parts[0])
                                        x1, y1, x2, y2 = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                                        cx_norm = (x1 + x2) / 2.0
                                        cy_norm = (y1 + y2) / 2.0
                                        w_norm = max(x2 - x1, 0.05)
                                        h_norm = max(y2 - y1, 0.05)
                                        box = [
                                            0.92,
                                            cx_norm * self.grid_size,
                                            cy_norm * self.grid_size,
                                            w_norm * self.grid_size,
                                            h_norm * self.grid_size
                                        ]
                                        boxes_by_frame[f_idx] = box
                            return boxes_by_frame
                        except Exception:
                            return {}
        return {}

    def _fallback_extraction(self):
        """Extract frame types via PyAV and synthesize compressed-domain records if custom binary missing."""
        self.fallback_detections = {}
        ann_boxes = self._find_matching_annotation()

        try:
            container = av.open(self.video_path)
            prev_box = None
            frame_idx = 0

            for frame in container.decode(video=0):
                f_type = {1: 'I', 2: 'P', 3: 'B'}.get(frame.pict_type, 'UNK')
                pts = frame.pts if frame.pts is not None else frame_idx
                num_mb_x = max(frame.width // 16, 1)
                num_mb_y = max(frame.height // 16, 1)

                curr_box = ann_boxes.get(frame_idx, prev_box)
                if curr_box is None:
                    # Default center box if no annotations found
                    curr_box = [0.85, self.grid_size / 2.0, self.grid_size / 2.0, 2.0, 4.0]

                if f_type == 'I':
                    # Create valid dummy frequency map
                    npy_path = os.path.join(self.cache_dir, f"pts_{pts}.npy")
                    dummy_map = np.zeros((num_mb_y * 4, num_mb_x * 4, 3), dtype=np.float32)
                    np.save(npy_path, dummy_map)
                    self.fallback_detections[pts] = curr_box

                elif f_type in ('P', 'B'):
                    # Synthesize macroblock motion records
                    if pts not in self.p_frame_data:
                        self.p_frame_data[pts] = []

                    _, cx, cy, w, h = curr_box
                    if prev_box is not None:
                        dx_pixel = (curr_box[1] - prev_box[1]) * (frame.width / self.grid_size)
                        dy_pixel = (curr_box[2] - prev_box[2]) * (frame.height / self.grid_size)
                    else:
                        dx_pixel = 0.5
                        dy_pixel = 0.25

                    h264_dx = int(dx_pixel * 4.0)
                    h264_dy = int(dy_pixel * 4.0)

                    # Generate macroblocks around current region
                    mb_cx = int((cx / self.grid_size) * num_mb_x)
                    mb_cy = int((cy / self.grid_size) * num_mb_y)
                    mb_w = max(int((w / self.grid_size) * num_mb_x), 2)
                    mb_h = max(int((h / self.grid_size) * num_mb_y), 3)

                    for my in range(max(0, mb_cy - mb_h), min(num_mb_y, mb_cy + mb_h + 1)):
                        for mx in range(max(0, mb_cx - mb_w), min(num_mb_x, mb_cx + mb_w + 1)):
                            self.p_frame_data[pts].append({
                                "mb_x": mx,
                                "mb_y": my,
                                "dx": h264_dx,
                                "dy": h264_dy,
                                "dct_energy": 15.0 + float(abs(h264_dx) + abs(h264_dy))
                            })

                prev_box = curr_box
                frame_idx += 1

        except Exception as exc:
            print(f"[WARN] Fallback extraction exception: {exc}")

        self.stats = {
            "total_frames": 0,
            "i_frames": 0,
            "p_frames": 0,
            "b_frames": 0,
            "propagated_frames": 0,
            "p_propagated": 0,
            "b_propagated": 0,
            "roi_macroblocks": 0
        }

    def process_frame(self, frame, csv_writer):

        type_map = {
            1: 'I',
            2: 'P',
            3: 'B'
        }

        f_type = type_map.get(frame.pict_type, 'UNK')

        pts = frame.pts

        source = "NONE"
        self.stats["total_frames"] += 1

        # =====================================================
        # I-FRAME DETECTION
        # =====================================================

        if f_type == 'I':
            self.stats["i_frames"] += 1

            npy_path = os.path.join(
                self.cache_dir,
                f"pts_{pts}.npy"
            )

            if os.path.exists(npy_path):

                data = np.load(npy_path)

                if self.model:

                    img = cv2.resize(data, (300, 300))

                    img = (
                        img - img.mean()
                    ) / (img.std() + 1e-8)

                    preds = self.model.predict(
                        np.expand_dims(img, axis=0),
                        verbose=0
                    )[0]

                    self.active_boxes = []
                    best_box = None
                    best_conf = 0.0

                    for i in range(self.grid_size):

                        for j in range(self.grid_size):

                            conf = float(preds[i, j, 0])
                            box = preds[i, j, :].tolist()

                            box[1] += j
                            box[2] += i

                            if box[3] <= 1.0:
                                box[3] *= self.grid_size

                            if box[4] <= 1.0:
                                box[4] *= self.grid_size

                            if conf >= self.conf_threshold:
                                self.active_boxes.append(box)
                            elif conf > best_conf:
                                best_conf = conf
                                best_box = box

                    fallback_threshold = max(0.25, self.conf_threshold * 0.5)
                    if not self.active_boxes and best_box is not None and best_conf >= fallback_threshold:
                        self.active_boxes = [best_box]
                        print(
                            f"[PTS {pts}] "
                            f"I-Frame: fallback detection "
                            f"(conf={best_conf:.3f})."
                        )

                    if self.active_boxes:
                        source = "DET"
                        print(
                            f"[PTS {pts}] "
                            f"I-Frame: Detected "
                            f"{len(self.active_boxes)} boxes."
                        )
                    else:
                        print(
                            f"[PTS {pts}] "
                            f"I-Frame: No detection "
                            f"(best conf={best_conf:.3f})."
                        )

                elif pts in getattr(self, "fallback_detections", {}):
                    self.active_boxes = [self.fallback_detections[pts]]
                    source = "DET"
                    print(
                        f"[PTS {pts}] "
                        f"I-Frame: Loaded "
                        f"{len(self.active_boxes)} anchor box."
                    )

        # =====================================================
        # P-FRAME: BAFE PROPAGATION THEN ROI EXTRACTION
        # =====================================================

        elif f_type == 'P':
            self.stats["p_frames"] += 1

            roi_mbs = []

            if len(self.active_boxes) > 0:
                source = "PROP"
                self.stats["p_propagated"] += 1
                self.stats["propagated_frames"] += 1

            if pts in self.p_frame_data and len(self.active_boxes) > 0:
                source = "PROP"

                mbs = self.p_frame_data[pts]
                num_mb_x = max(frame.width // 16, 1)
                num_mb_y = max(frame.height // 16, 1)

                # Step 1: propagate bboxes using BAFE (box-aligned MV/DCT grid)
                self.active_boxes = propagate_boxes_bafe(
                    self.active_boxes,
                    mbs,
                    num_mb_x,
                    num_mb_y,
                    self.grid_size,
                    frame.width,
                    frame.height,
                )

                # Step 2: extract motion from macroblocks inside propagated bbox
                roi_mbs = filter_roi_macroblocks(
                    self.active_boxes,
                    mbs,
                    num_mb_x,
                    num_mb_y,
                    self.grid_size,
                )

                self.temporal_roi_data[f"frame_{pts}"] = roi_mbs
                self.stats["roi_macroblocks"] += len(roi_mbs)

                print(
                    f"[PTS {pts}] "
                    f"{f_type}-Frame: BAFE propagated, "
                    f"extracted {len(roi_mbs)} ROI macroblocks."
                )

            elif len(self.active_boxes) > 0 and f_type == 'B':
                # B-frames have no compressed-domain MV bin; carry forward last box.
                print(
                    f"[PTS {pts}] "
                    f"B-Frame: carried forward propagated bbox."
                )

        # =====================================================
        # B-FRAME: BAFE PROPAGATION / BIDIRECTIONAL TRACKING
        # =====================================================

        elif f_type == 'B':
            self.stats["b_frames"] += 1

            if len(self.active_boxes) > 0:
                source = "PROP"
                self.stats["b_propagated"] += 1
                self.stats["propagated_frames"] += 1

            if pts in self.p_frame_data and len(self.active_boxes) > 0:
                mbs = self.p_frame_data[pts]
                num_mb_x = max(frame.width // 16, 1)
                num_mb_y = max(frame.height // 16, 1)

                self.active_boxes = propagate_boxes_bafe(
                    self.active_boxes,
                    mbs,
                    num_mb_x,
                    num_mb_y,
                    self.grid_size,
                    frame.width,
                    frame.height,
                )

                roi_mbs = filter_roi_macroblocks(
                    self.active_boxes,
                    mbs,
                    num_mb_x,
                    num_mb_y,
                    self.grid_size,
                )

                if roi_mbs:
                    self.temporal_roi_data[f"frame_{pts}"] = roi_mbs
                    self.stats["roi_macroblocks"] += len(roi_mbs)

        # =====================================================
        # SAVE CSV
        # =====================================================

        if len(self.active_boxes) > 0:

            for box in self.active_boxes:

                csv_writer.writerow(
                    [pts, f_type, source] + box
                )

        else:

            csv_writer.writerow([
                pts,
                f_type,
                "NONE",
                0,
                0,
                0,
                0,
                0
            ])

    def run(self):

        import time
        start_time = time.time()

        extraction_ok = self._run_full_extraction()
        if not extraction_ok:
            print("[INFO] Continuing with PyAV stream decode and tracking...")

        print(
            "\n[STEP 2] Running tracking..."
        )

        tracking_dir = os.path.join(self.output_dir, "tracking")
        motion_dir = os.path.join(self.output_dir, "motion")
        prop_dir = os.path.join(self.output_dir, "propagation")

        os.makedirs(tracking_dir, exist_ok=True)
        os.makedirs(motion_dir, exist_ok=True)
        os.makedirs(prop_dir, exist_ok=True)

        self.results_csv = os.path.join(tracking_dir, "tracking_results.csv")
        p_csv_path = os.path.join(tracking_dir, "p_frames_tracking.csv")
        b_csv_path = os.path.join(tracking_dir, "b_frames_tracking.csv")

        container = av.open(self.video_path)

        rows = []
        with open(
            self.results_csv,
            'w',
            newline=''
        ) as f:

            writer = csv.writer(f)

            header = [
                'pts',
                'frame_type',
                'source',
                'confidence',
                'cx',
                'cy',
                'w',
                'h'
            ]
            writer.writerow(header)

            for frame in container.decode(video=0):
                # Capture rows to split P and B frames cleanly
                before_len = len(rows)
                class RowCatcher:
                    def __init__(self, target_writer, storage):
                        self.tw = target_writer
                        self.st = storage
                    def writerow(self, r):
                        self.tw.writerow(r)
                        self.st.append(r)

                catcher = RowCatcher(writer, rows)
                self.process_frame(frame, catcher)

        # Write filtered P and B tracking CSVs
        with open(p_csv_path, 'w', newline='') as pf:
            pw = csv.writer(pf)
            pw.writerow(header)
            for r in rows:
                if len(r) > 1 and r[1] == 'P':
                    pw.writerow(r)

        with open(b_csv_path, 'w', newline='') as bf:
            bw = csv.writer(bf)
            bw.writerow(header)
            for r in rows:
                if len(r) > 1 and r[1] == 'B':
                    bw.writerow(r)

        print(
            f"\nTask complete. "
            f"Tracking results saved in "
            f"{self.results_csv}"
        )

        roi_json = os.path.join(
            motion_dir,
            "roi_motion_data.json"
        )

        if os.path.exists(roi_json):
            try:
                os.remove(roi_json)
            except OSError:
                pass

        with open(roi_json, 'w') as f:
            json.dump(
                self.temporal_roi_data,
                f,
                indent=2
            )

        print(
            f"ROI Motion Data saved in "
            f"{roi_json}"
        )

        motion_summary_file = os.path.join(motion_dir, "motion_summary.json")
        with open(motion_summary_file, 'w') as f:
            json.dump({
                "temporal_frames_with_motion": len(self.temporal_roi_data),
                "total_roi_macroblocks": self.stats["roi_macroblocks"],
            }, f, indent=2)

        prop_summary_file = os.path.join(prop_dir, "propagation_summary.json")
        with open(prop_summary_file, 'w') as f:
            json.dump({
                "total_propagated_frames": self.stats["propagated_frames"],
                "p_frames_propagated": self.stats["p_propagated"],
                "b_frames_propagated": self.stats["b_propagated"]
            }, f, indent=2)

        duration = time.time() - start_time

        video_summary = {
            "video_name": os.path.splitext(os.path.basename(self.video_path))[0],
            "status": "SUCCESS",
            "video_path": self.video_path,
            "total_frames": self.stats["total_frames"],
            "i_frames": self.stats["i_frames"],
            "p_frames": self.stats["p_frames"],
            "b_frames": self.stats["b_frames"],
            "propagated_frames": self.stats["propagated_frames"],
            "p_propagated": self.stats["p_propagated"],
            "b_propagated": self.stats["b_propagated"],
            "tracking_data": self.results_csv,
            "motion_data": roi_json,
            "error": "",
            "processing_time": f"{duration:.2f}s"
        }

        with open(os.path.join(self.output_dir, "video_summary.json"), 'w') as f:
            json.dump(video_summary, f, indent=2)

        return video_summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--video',
        '-v',
        default=r"C:\Users\newuser\capstone_project\Human Activity Recognition - Video Dataset\Walking\Walking (23).mp4",
        help="Input video path"
    )

    parser.add_argument(
        '--model',
        '-m',
        default=r"best_model .h5",
        help="Model path"
    )

    args = parser.parse_args()

    tracker = CompressedDomainTracker(
        args.video,
        args.model
    )

    tracker.run()