import json


def load_metadata(path="data/processed/aems/metadata/aems_manifest_v1.json"):
    with open(path) as f:
        return json.load(f)


def filter_by_split(metadata, split="test"):
    return [m for m in metadata if m["split"] == split]


def get_common_video_ids(video_db_v, video_db_a, caption_db, test_vids):
    return [
        vid
        for vid in test_vids
        if vid in video_db_v and vid in video_db_a and vid in caption_db
    ]
