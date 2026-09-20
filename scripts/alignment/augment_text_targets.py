#!/usr/bin/env python3
"""
augment_text_targets.py -- Multi-positive CLIP-text targets for TRAIN split
============================================================================
Expands the single per-clip description target into a rich, diverse set of
CLIP-text positives, all derived from EXISTING manifest text (no new caption
model, per the '2-lite' augmentation plan):

  per train clip: description
                  + each qa_question
                  + each qa_answer
                  + youtube_title
                  + each youtube_tag (prompted)
                  + category prompts (fine + parent)

TESTS ARE UNTOUCHED (train split only -> no leakage). Targets are embedded with
the SAME CLIP ViT-B/32 used everywhere and L2-normalized.

Writes embeddings/aems_text_targets_augmented_train.pt ({video_id: [M,512]}).

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/alignment/augment_text_targets.py
"""

import argparse
import time

import torch
import torch.nn.functional as F
import clip

from src.config import AEMS_MANIFEST_PATH, DEVICE, set_seeds
from src.data.metadata import load_metadata, filter_by_split

OUT_PATH = "embeddings/aems_text_targets_augmented_train.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)

    recs = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="train")
    print(f"[DATA] {len(recs)} train records")

    model, _ = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    t0 = time.time()

    def encode_one(text):
        if not text or not str(text).strip():
            return None
        tokens = clip.tokenize([str(text)], truncate=True).to(DEVICE)
        with torch.no_grad():
            e = model.encode_text(tokens)
        return F.normalize(e, dim=-1).squeeze(0).cpu()

    out = {}
    total_targets = 0
    for r in recs:
        vid = r["video_id"]
        texts = []
        if r.get("text_description"):
            texts.append(str(r["text_description"]))
        for q in r.get("qa_questions", []) or []:
            texts.append(str(q))
        for a in r.get("qa_answers", []) or []:
            texts.append(str(a))
        if r.get("youtube_title"):
            texts.append(str(r["youtube_title"]))
        for tg in r.get("youtube_tags", []) or []:
            texts.append(f"a video tagged {tg}")
        cats = [c for c in (r.get("content_fine_category"), r.get("content_parent_category")) if c]
        for c in cats:
            texts.append(f"a video about {c}")

        embs = [e for e in (encode_one(t) for t in texts) if e is not None]
        if embs:
            out[vid] = F.normalize(torch.stack(embs), dim=-1)
            total_targets += len(embs)

    mu = sum(t.shape[0] for t in out.values()) / max(len(out), 1)
    print(f"[STATS] clips={len(out)} total_targets={total_targets} "
          f"mean_targets/clip={mu:.1f}")
    torch.save(out, OUT_PATH)
    print(f"[DONE] {OUT_PATH} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()