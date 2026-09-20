import argparse, os, gc
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm
from src.config import (AEMS_MANIFEST_PATH, AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
                         AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
                         CHUNK_TOKEN_BUDGET, DEVICE, set_seeds)
from src.data.metadata import load_metadata, filter_by_split

parser = argparse.ArgumentParser(description="Precompute AEMS text embeddings")
parser.add_argument("--split", required=True, choices=["train", "test"])
parser.add_argument("--fusion", required=True, choices=["description", "transcript", "fused"])
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seeds(args.seed)
OUT_TEMPLATES = {
    "description": AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
    "transcript": AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE,
    "fused": AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
}
OUT_PATH = OUT_TEMPLATES[args.fusion].format(split=args.split)
os.makedirs("embeddings", exist_ok=True)

print(f"[INIT] Loading CLIP on {DEVICE}")
model, _ = clip.load("ViT-B/32", device=DEVICE)
model.eval()

print(f"[DATA] Loading manifest: {args.manifest}")
records = filter_by_split(load_metadata(args.manifest), split=args.split)
print(f"[DATA] {len(records)} records")


def encode_one(text):
    if not text or not text.strip():
        return None
    tokens = clip.tokenize([text], truncate=True).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1).squeeze(0).cpu()
    return emb


def encode_chunks_and_mean(text):
    if not text or not text.strip():
        return None
    words = text.split()
    chunks = []
    current = []
    current_tokens = 0
    for word in words:
        word_len = len(word) // 4 + 1
        if current_tokens + word_len + 1 > CHUNK_TOKEN_BUDGET and current:
            chunks.append(" ".join(current))
            current = [word]
            current_tokens = word_len
        else:
            current.append(word)
            current_tokens += word_len + 1
    if current:
        chunks.append(" ".join(current))
    if not chunks:
        return None
    all_embeds = []
    for chunk in chunks:
        tokens = clip.tokenize([chunk], truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1).squeeze(0).cpu()
        all_embeds.append(emb)
    result = torch.stack(all_embeds).mean(dim=0)
    result = F.normalize(result, dim=0)
    return result


out = {}
skipped = 0

for item in tqdm(records, desc=f"Encoding {args.fusion} {args.split}"):
    vid = item["video_id"]
    desc = item.get("text_description", "")
    trans = item.get("text_transcript", "")

    if args.fusion == "description":
        e = encode_one(desc)
    elif args.fusion == "transcript":
        e = encode_chunks_and_mean(trans)
    elif args.fusion == "fused":
        e_desc = encode_one(desc)
        e_trans = encode_chunks_and_mean(trans)
        if e_desc is None and e_trans is None:
            e = None
        elif e_desc is None:
            e = e_trans
        elif e_trans is None:
            e = e_desc
        else:
            e = F.normalize(0.5 * e_desc + 0.5 * e_trans, dim=0)

    if e is not None:
        out[vid] = e
    else:
        skipped += 1

torch.save(out, OUT_PATH)
print(f"\n[DONE] Saved {len(out)} embeddings to {OUT_PATH}")
print(f"       Skipped: {skipped}")
