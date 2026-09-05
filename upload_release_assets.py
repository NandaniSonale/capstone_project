import os
import subprocess
import sys

def upload_videos(start_idx=1, end_idx=10, tag="v1.0.0-artifacts", repo="NandaniSonale/capstone_project"):
    for i in range(start_idx, end_idx + 1):
        vid_name = f"Walking ({i})_visualized.mp4"
        vid_path = os.path.join("output", f"Walking ({i})", vid_name)
        if os.path.exists(vid_path):
            print(f"[{i}/{end_idx}] Uploading {vid_name} ({os.path.getsize(vid_path)/1024/1024:.1f} MB)...", flush=True)
            cmd = ["gh", "release", "upload", tag, vid_path, "--repo", repo, "--clobber"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  -> Uploaded successfully.", flush=True)
            else:
                print(f"  -> Upload failed: {res.stderr}", flush=True)
        else:
            print(f"[{i}/{end_idx}] {vid_path} not found, skipping.", flush=True)

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    upload_videos(1, count)
