import json
import os
import subprocess
from tqdm import tqdm

METADATA_PATH = "data/processed/metadata/msrvtt_metadata.json"
FRAMES_ROOT = "data/processed/video/frames"
FPS = 1
MAX_FRAMES = 15

os.makedirs(FRAMES_ROOT, exist_ok=True)

# Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Get unique videos
videos = {}
for item in metadata:
    videos[item["video_id"]] = item["video_path"]

print(f"Unique videos to process: {len(videos)}")

failed_videos = []

for video_id, video_path in tqdm(videos.items(), desc="Extracting frames"):
    out_dir = os.path.join(FRAMES_ROOT, video_id)

    # Skip if already processed
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= 5:
        continue

    os.makedirs(out_dir, exist_ok=True)

    output_pattern = os.path.join(out_dir, "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={FPS}",
        "-frames:v", str(MAX_FRAMES),
        output_pattern
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        failed_videos.append(video_id)

# Save failures (if any)
if failed_videos:
    with open("data/processed/video/frame_extraction_failed.txt", "w") as f:
        for vid in failed_videos:
            f.write(vid + "\n")

print("===================================")
print(f"Frame extraction complete")
print(f"Failed videos: {len(failed_videos)}")
