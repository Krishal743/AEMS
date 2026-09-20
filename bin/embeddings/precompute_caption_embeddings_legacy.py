import json
import os
import argparse
import torch
import clip
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METADATA = "data/processed/metadata/msrvtt_metadata.json"
BATCH_SIZE = 64

parser = argparse.ArgumentParser(description="Precompute caption embeddings for MSR-VTT")
parser.add_argument("--split", default="test", choices=["train", "test"],
                    help="Which split to encode (default: test)")
args = parser.parse_args()

split = args.split
OUT_PATH = f"embeddings/caption_embeddings_{split}.pt"

print("[SETUP] Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

print("[DATA] Loading metadata...")
with open(METADATA) as f:
    data = json.load(f)

split_data = [d for d in data if d["split"] == split]
print(f"[DATA] {split} entries: {len(split_data)}")

video_captions = {}
for item in split_data:
    vid = item["video_id"]
    text = item["text"]
    if vid not in video_captions:
        video_captions[vid] = []
    video_captions[vid].append(text)

print(f"[DATA] Unique videos: {len(video_captions)}")
for vid, caps in list(video_captions.items())[:3]:
    print(f"  {vid}: {len(caps)} captions")

print("[CAPTION] Encoding captions (batched)...")
caption_embeddings = {}

for vid, captions in tqdm(video_captions.items(), desc="Encoding"):
    all_embeds = []
    
    for i in range(0, len(captions), BATCH_SIZE):
        batch = captions[i:i + BATCH_SIZE]
        tokens = clip.tokenize(batch).to(DEVICE)
        
        with torch.no_grad():
            embeds = model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        
        all_embeds.append(embeds.cpu())
    
    all_embeds = torch.cat(all_embeds, dim=0)
    caption_embeddings[vid] = all_embeds

torch.save(caption_embeddings, OUT_PATH)
print(f"[DONE] Saved caption embeddings for {len(caption_embeddings)} videos")
print(f"[DONE] Output: {OUT_PATH}")
