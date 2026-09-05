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

    def __init__(self, video_path, model_path,
                 conf_threshold=0.5,
                 grid_size=7):

        self.video_path = video_path
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.grid_size = grid_size

        print(f"[INIT] Video: {video_path}")
        print(f"[INIT] Model: {model_path}")

        self.model = None

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

        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print("[ERROR] FFmpeg extraction failed.")
            return False

        self._process_bins(prefix)

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

    def process_frame(self, frame, csv_writer):

        type_map = {
            1: 'I',
            2: 'P',
            3: 'B'
        }

        f_type = type_map.get(frame.pict_type, 'UNK')

        pts = frame.pts

        source = "NONE"

        # =====================================================
        # I-FRAME DETECTION
        # =====================================================

        if f_type == 'I':

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

        # =====================================================
        # P-FRAME: BAFE PROPAGATION THEN ROI EXTRACTION
        # =====================================================

        elif f_type in ('P', 'B'):

            roi_mbs = []

            if len(self.active_boxes) > 0 and f_type == 'B':
                source = "PROP"

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

        if self.model is None:

            print(
                "\n[ERROR] No model loaded."
            )

            return

        if not self._run_full_extraction():
            return

        print(
            "\n[STEP 2] Running tracking..."
        )

        container = av.open(self.video_path)

        with open(
            self.results_csv,
            'w',
            newline=''
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                'pts',
                'frame_type',
                'source',
                'confidence',
                'cx',
                'cy',
                'w',
                'h'
            ])

            for frame in container.decode(video=0):

                self.process_frame(
                    frame,
                    writer
                )

        print(
            f"\nTask complete. "
            f"Tracking results saved in "
            f"{self.results_csv}"
        )

        roi_json = os.path.join(
            self.output_dir,
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

        return self.temporal_roi_data


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