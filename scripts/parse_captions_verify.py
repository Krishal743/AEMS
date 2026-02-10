import json

with open("data/processed/metadata/msrvtt_metadata.json") as f:
    data = json.load(f)

print("Total samples:", len(data))
print("First sample:", data[0])
print("Last sample:", data[-1])
print("Unique videos:", len(set(d["video_id"] for d in data)))

splits = {}
for d in data:
    splits[d["split"]] = splits.get(d["split"], 0) + 1
print("Split counts:", splits)
