import json, os, random, sys, time, argparse, glob
from src.config import AEMS_DATASET_ROOT, AEMS_MANIFEST_PATH, set_seeds
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Build AEMS manifest with stratified 85/15 split")
parser.add_argument("--pilot", action="store_true", help="Build a 500-video pilot subset")
parser.add_argument("--output", type=str, default=None, help="Override output manifest path")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

set_seeds(args.seed)
if args.pilot:
    OUTPUT_MANIFEST = args.output or "data/processed/aems_pilot/metadata/aems_manifest_v1.json"
    PILOT_PREFIX = "aems_pilot"
else:
    OUTPUT_MANIFEST = args.output or AEMS_MANIFEST_PATH
    PILOT_PREFIX = "aems"
MANIFEST_DIR = os.path.dirname(OUTPUT_MANIFEST)
os.makedirs(MANIFEST_DIR, exist_ok=True)
BASE_DATA_DIR = os.path.dirname(MANIFEST_DIR)
BUILD_LOG = os.path.join(BASE_DATA_DIR, "_build_log.txt")
FAILED_LOG = os.path.join(BASE_DATA_DIR, "_manifest_failed.txt")

log_lines = []

categories = sorted(os.listdir(AEMS_DATASET_ROOT))
records = []
skipped_videos = []

for cat_idx, cat in enumerate(categories):
    cat_dir = os.path.join(AEMS_DATASET_ROOT, cat)
    if not os.path.isdir(cat_dir):
        continue
    json_files = sorted(glob.glob(os.path.join(cat_dir, "*.json")))
    log_lines.append(f"Category '{cat}': {len(json_files)} JSON files")

    cat_records = []
    for jf in json_files:
        video_id = os.path.splitext(os.path.basename(jf))[0]
        mp4_path = jf.replace(".json", ".mp4")
        if not os.path.exists(mp4_path):
            skipped_videos.append((video_id, cat, "missing MP4"))
            continue
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            skipped_videos.append((video_id, cat, f"JSON parse error: {e}"))
            continue

        cm = data.get("content_metadata") or {}
        description = cm.get("description", "") or ""
        qa_list = cm.get("qAndA") or []
        qa_questions = [q.get("question", "") for q in qa_list if q.get("question")]
        qa_answers = [q.get("answer", "") for q in qa_list if q.get("answer")]
        text_to_speech = data.get("text_to_speech") or ""

        if not description.strip():
            skipped_videos.append((video_id, cat, "empty description"))
            continue
        if len(qa_questions) == 0:
            skipped_videos.append((video_id, cat, "no Q&A questions"))
            continue

        frames_dir_rel = os.path.join("data", "processed", PILOT_PREFIX, "frames_uniform", video_id)
        audio_path_rel = os.path.join("data", "processed", PILOT_PREFIX, "audio", f"{video_id}.wav")
        record = {
            "video_id": video_id,
            "source_dataset": "aems",
            "split": "train",
            "content_fine_category": cat,
            "content_parent_category": data.get("content_parent_category", ""),
            "duration_seconds": data.get("duration_seconds", 0),
            "resolution": data.get("resolution", ""),
            "fps": cm.get("fps", 30.0),
            "original_video_filename": data.get("original_video_filename", ""),
            "original_json_filename": data.get("original_json_filename", ""),
            "video_path": mp4_path,
            "json_path": jf,
            "frames_dir": frames_dir_rel,
            "audio_path": audio_path_rel,
            "text_description": description,
            "text_transcript": text_to_speech,
            "text_transcript_word_count": data.get("text_to_speech_word_count", 0),
            "timecoded_text_to_speech": data.get("timecoded_text_to_speech", []),
            "youtube_title": data.get("youtube_title", ""),
            "youtube_tags": data.get("youtube_tags", []),
            "youtube_description": data.get("youtube_description", ""),
            "qa_questions": qa_questions,
            "qa_answers": qa_answers,
        }
        cat_records.append(record)

    if args.pilot:
        target = min(30, len(cat_records))
        random.Random(args.seed + cat_idx).shuffle(cat_records)
        cat_records = cat_records[:target]

    rng = random.Random(args.seed + cat_idx)
    rng.shuffle(cat_records)
    train_size = max(1, int(0.85 * len(cat_records)))
    for i, rec in enumerate(cat_records):
        rec["split"] = "train" if i < train_size else "test"
    cat_records.sort(key=lambda r: (r["split"], r["video_id"]))
    records.extend(cat_records)

records.sort(key=lambda r: (r["content_fine_category"], r["video_id"]))
total = len(records)
train_count = sum(1 for r in records if r["split"] == "train")
test_count = sum(1 for r in records if r["split"] == "test")

tmp_path = OUTPUT_MANIFEST + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
os.replace(tmp_path, OUTPUT_MANIFEST)

log_lines.append(f"\nTotal records: {total}")
log_lines.append(f"  Train: {train_count}")
log_lines.append(f"  Test:  {test_count}")
log_lines.append(f"  Skipped: {len(skipped_videos)}")
for vid, cat, reason in skipped_videos:
    log_lines.append(f"    {vid} ({cat}): {reason}")

with open(BUILD_LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

with open(FAILED_LOG, "w", encoding="utf-8") as f:
    for vid, cat, reason in skipped_videos:
        f.write(f"{vid},{cat},{reason}\n")

print(f"Manifest written: {OUTPUT_MANIFEST} ({total} records)")
print(f"  Train: {train_count}, Test: {test_count}, Skipped: {len(skipped_videos)}")
print(f"  Build log: {BUILD_LOG}")
