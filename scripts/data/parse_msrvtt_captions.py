import json
import os
from tqdm import tqdm

# Paths (relative to project root)
ANN_PATH = "data/raw/msrvtt/annotations/msrvtt_annotations.json"
TRAIN_LIST = "data/raw/msrvtt/annotations/train_list.txt"
TEST_LIST = "data/raw/msrvtt/annotations/test_list.txt"
VIDEO_DIR = "data/raw/msrvtt/videos"
OUT_PATH = "data/processed/metadata/msrvtt_metadata.json"

# Load train/test splits
with open(TRAIN_LIST) as f:
    train_videos = set(line.strip() for line in f)

with open(TEST_LIST) as f:
    test_videos = set(line.strip() for line in f)

print(f"Train videos: {len(train_videos)}")
print(f"Test videos: {len(test_videos)}")

# Load annotation JSON
with open(ANN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]

metadata = []
skipped = 0

for idx, ann in enumerate(tqdm(annotations, desc="Parsing captions")):
    video_id = ann["image_id"]
    caption = ann["caption"].lower().strip()

    # Determine split
    if video_id in train_videos:
        split = "train"
    elif video_id in test_videos:
        split = "test"
    else:
        skipped += 1
        continue  # safety: ignore unknown videos

    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        skipped += 1
        continue

    record = {
        "id": f"msrvtt_{video_id}_cap_{idx}",
        "dataset": "MSR-VTT",
        "video_id": video_id,
        "video_path": video_path,
        "frames_dir": f"data/processed/video/frames/{video_id}/",
        "text": caption,
        "split": split
    }

    metadata.append(record)

os.makedirs("data/processed/metadata", exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("===================================")
print(f"Total captions processed: {len(annotations)}")
print(f"Metadata entries written: {len(metadata)}")
print(f"Skipped entries: {skipped}")
print(f"Saved to: {OUT_PATH}")
