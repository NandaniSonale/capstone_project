import os
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

TAG = "v1.0.0-artifacts"
REPO = "NandaniSonale/capstone_project"

def get_already_uploaded():
    try:
        res = subprocess.run(
            ["gh", "release", "view", TAG, "--repo", REPO, "--json", "assets"],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(res.stdout)
        uploaded = set()
        for a in data.get("assets", []):
            name = a.get("name", "")
            # Example: Walking.1._visualized.mp4 -> 1
            if name.startswith("Walking.") and "._visualized.mp4" in name:
                num_str = name.replace("Walking.", "").replace("._visualized.mp4", "")
                if num_str.isdigit():
                    uploaded.add(int(num_str))
            elif name.startswith("Walking (") and ")_visualized.mp4" in name:
                num_str = name.replace("Walking (", "").replace(")_visualized.mp4", "")
                if num_str.isdigit():
                    uploaded.add(int(num_str))
        return uploaded
    except Exception as e:
        print(f"Warning: Could not fetch asset list: {e}")
        return set()

def upload_single_video(i):
    vid_name = f"Walking ({i})_visualized.mp4"
    vid_path = os.path.join("output", f"Walking ({i})", vid_name)
    if not os.path.exists(vid_path):
        return i, False, "File not found"
    
    size_mb = os.path.getsize(vid_path) / (1024 * 1024)
    cmd = ["gh", "release", "upload", TAG, vid_path, "--repo", REPO, "--clobber"]
    
    for attempt in range(1, 4):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return i, True, f"{size_mb:.1f} MB"
        time.sleep(2 * attempt)
    
    return i, False, res.stderr.strip()

def main():
    uploaded = get_already_uploaded()
    print(f"Found {len(uploaded)} videos already uploaded.")
    
    pending = [i for i in range(1, 172) if i not in uploaded]
    print(f"Total videos to upload: {len(pending)}")
    
    if not pending:
        print("All 171 videos are already uploaded!")
        return

    completed_count = len(uploaded)
    total = 171

    # Using 3 workers for fast, stable concurrent upload
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(upload_single_video, i): i for i in pending}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                i, success, msg = future.result()
                if success:
                    completed_count += 1
                    print(f"[{completed_count}/{total}] Successfully uploaded Walking ({i})_visualized.mp4 ({msg})", flush=True)
                else:
                    print(f"[FAIL] Walking ({i})_visualized.mp4 failed: {msg}", flush=True)
            except Exception as exc:
                print(f"[ERROR] Walking ({idx}) exception: {exc}", flush=True)

    print(f"\nUpload process finished! Total uploaded: {completed_count}/{total}")

if __name__ == "__main__":
    main()
