"""Precompute dense passage embeddings with a retrieval-trained text encoder.

The CLIP passage branch scores 0.397 R@1 on validation; the same passages
encoded by E5 score 0.639, because E5 was trained for retrieval while CLIP's
text tower was trained to match image captions through a 77-token window.
Chunks are correspondingly larger here (400 tokens, not 72).

Saves {video_id: (n_chunks, dim)}, scored at query time by ChunkIndex max-sim,
exactly like the CLIP passage branch.
"""

import argparse, os
import torch
from tqdm import tqdm
from src.config import (AEMS_MANIFEST_PATH, AEMS_DENSE_PASSAGES_PATH_TEMPLATE,
                        AEMS_DENSE_TEXT_MODEL, AEMS_DENSE_CHUNK_TOKENS, DEVICE, set_seeds)
from src.data.metadata import load_metadata, filter_by_split
from src.data.text_chunks import video_chunks
from src.encoders.text_retrieval import load_dense_encoder

parser = argparse.ArgumentParser(description="Precompute dense (E5/BGE/GTE) passage embeddings")
parser.add_argument("--split", required=True, choices=["train", "test"])
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--model", default=AEMS_DENSE_TEXT_MODEL)
parser.add_argument("--output", default=None)
parser.add_argument("--weights", default=None,
                    help="checkpoint to load instead of the stock model; defaults to the "
                         "AEMS fine-tuned retriever when it exists")
parser.add_argument("--chunk-tokens", type=int, default=AEMS_DENSE_CHUNK_TOKENS)
parser.add_argument("--max-chunks", type=int, default=32)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seeds(args.seed)

out_path = args.output or AEMS_DENSE_PASSAGES_PATH_TEMPLATE.format(split=args.split)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

encoder = load_dense_encoder(DEVICE, args.model, name=args.weights)
print(f"[INIT] Loaded {encoder.config['name']} ({args.model} conventions) on {DEVICE}")

records = filter_by_split(load_metadata(args.manifest), split=args.split)
print(f"[DATA] {len(records)} records, {args.chunk_tokens}-token chunks")

# Flatten every video's passages into one list so the encoder runs in big batches,
# then slice the result back per video. Passages are emitted contiguously, so a
# running cursor splits them in one pass.
texts, counts, video_ids = [], [], []
for record in tqdm(records, desc="Chunking"):
    parts = video_chunks(record, max_chunks=args.max_chunks, budget=args.chunk_tokens)
    if not parts:
        continue
    video_ids.append(record["video_id"])
    counts.append(len(parts))
    texts.extend(parts)

print(f"[ENC] Encoding {len(texts)} passages from {len(video_ids)} videos")
embeddings = encoder.encode_passages(texts, batch_size=args.batch_size)

out, cursor = {}, 0
for video, count in zip(video_ids, counts):
    out[video] = embeddings[cursor:cursor + count]
    cursor += count
assert cursor == len(texts), f"split {cursor} passages but encoded {len(texts)}"

torch.save(out, out_path)
print(f"\n[DONE] {len(out)} videos, {len(texts)} passages, dim={embeddings.shape[1]} -> {out_path}")
