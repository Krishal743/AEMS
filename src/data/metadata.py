import json


def load_metadata(path="data/processed/aems/metadata/aems_manifest_v1.json"):
    with open(path) as f:
        return json.load(f)


def filter_by_split(metadata, split="test"):
    return [m for m in metadata if m["split"] == split]


def get_unique_video_ids(metadata):
    return set(m["video_id"] for m in metadata)


def get_common_video_ids(video_db_v, video_db_a, caption_db, test_vids):
    return [
        vid
        for vid in test_vids
        if vid in video_db_v and vid in video_db_a and vid in caption_db
    ]


def filter_items_by_videos(items, valid_video_set):
    return [item for item in items if item["video_id"] in valid_video_set]


def get_unique_texts(items):
    unique_texts = []
    unique_video_ids = []
    seen = set()
    for item in items:
        if item["video_id"] not in seen:
            seen.add(item["video_id"])
            unique_texts.append(item["text"])
            unique_video_ids.append(item["video_id"])
    return unique_texts, unique_video_ids
