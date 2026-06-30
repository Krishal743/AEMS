import json
import os
import subprocess
import sys
from tqdm import tqdm

METADATA_PATH = "data/processed/metadata/msrvtt_metadata.json"
FRAMES_ROOT = "data/processed/video/frames_uniform"
NUM_FRAMES = 16

os.makedirs(FRAMES_ROOT, exist_ok=True)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

unique_videos = list(set(item["video_id"] for item in metadata))
videos = {}
for vid in unique_videos:
    for item in metadata:
        if item["video_id"] == vid:
            videos[vid] = item["video_path"]
            break

print(f"Total unique videos: {len(videos)}")

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        return float(result.stdout.strip())
    except:
        return None

def extract_uniform_frames(video_path, output_dir, num_frames=16):
    video_path = str(video_path).replace('\\', '/')
    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_frames):
        timestamp = (i + 0.5) * duration / num_frames
        output_file = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        
        if os.path.exists(output_file):
            continue
            
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_file
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=5)
        except:
            return False
    
    return True

failed = []
success = 0
skipped = 0

for video_id, video_path in tqdm(list(videos.items()), desc="Extracting frames"):
    output_dir = os.path.join(FRAMES_ROOT, video_id)
    
    if os.path.exists(output_dir):
        existing = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        if existing == NUM_FRAMES:
            success += 1
            continue
        elif existing > 0:
            for f in os.listdir(output_dir):
                if f.endswith('.jpg'):
                    os.remove(os.path.join(output_dir, f))
    
    if extract_uniform_frames(video_path, output_dir, NUM_FRAMES):
        success += 1
    else:
        failed.append(video_id)

print(f"\nExtraction complete:")
print(f"  Success: {success}")
print(f"  Failed: {len(failed)}")

if failed:
    with open("data/processed/video_uniform_failed.txt", "w") as f:
        for vid in failed:
            f.write(vid + "\n")