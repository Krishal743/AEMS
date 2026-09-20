#!/usr/bin/env python3
"""
M3: Shared-text-anchor alignment (ImageBind/Girdhar 2023; Grave 2019).
=====================================================================
Both CLAP-audio and CLAP-text live in the SAME CLAP joint embedding space
(CLAP aligns audio<->text during pretraining). CLIP-text (description) lives
in the CLIP space. Strategy:
  1. Encode the train description sentences with the CLAP *text* encoder
     -> CLAP-text points in CLAP space.
  2. Fit an orthogonal Procrustes map CLAP-text -> CLIP-text on the shared
     sentences (CLIP-text already precomputed for the identical sentences).
  3. Apply that same map to the precomputed CLAP-audio embeddings (same CLAP
     space) to pull audio into CLIP-text space.

Also evaluate against the pure CLIP-text target (description).

Fit/CLAP-text encode on TRAIN anchors only; evaluate on held-out TEST.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from scripts.alignment.compare_utils import (
    load_anchor_pairs, procrustes_map, evaluate_method, save_db,
)
from src.config import AEMS_MANIFEST_PATH, AEMS_AUDIO_EMBEDDINGS_PATH, set_seeds
from src.data.metadata import load_metadata, filter_by_split
from src.encoders.clap_encode import CLAPEncoder

OUTPUT_DIR = "outputs/alignment"
ALIGNED_DIR = "embeddings"
os.makedirs(ALIGNED_DIR, exist_ok=True)


def main():
    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("M3: SHARED-TEXT-ANCHOR ALIGNMENT (CLAP-text -> CLIP-text -> audio)")
    print("=" * 72)

    recs_tr = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="train")
    recs_te = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")

    # Gather description sentences for train/test, aligned with anchor video ids
    def desc_for(recs):
        return {r["video_id"]: r["text_description"] for r in recs}
    desc_tr = desc_for(recs_tr)
    desc_te = desc_for(recs_te)

    # Anchor video sets
    X_tr, Y_tr, vids_tr = load_anchor_pairs("train")   # CLAP-audio, CLIP-text
    X_te, Y_te, vids_te = load_anchor_pairs("test")

    # --- 1. Encode descriptions with CLAP-text (train + test) ---
    print(f"\nLoading CLAP ({device}) and encoding descriptions...")
    clap = CLAPEncoder(device=device)
    def clap_text_for(vids):
        text_list = [desc_tr.get(v, desc_te.get(v, "")) for v in vids]
        emb = clap.encode_text(text_list)
        e = torch.as_tensor(emb, dtype=torch.float32)
        return F.normalize(e.reshape(-1, 512), dim=-1)
    clap_text_tr = clap_text_for(vids_tr)   # CLAP space
    clap_text_te = clap_text_for(vids_te)
    del clap

    # --- 2. Fit Procrustes CLAP-text -> CLIP-text on TRAIN ---
    print("Fitting Procrustes CLAP-text -> CLIP-text ...")
    Wp, bp = procrustes_map(clap_text_tr, Y_tr)
    def map_to_clip(Z):
        return F.normalize((Z - clap_text_tr.mean(0, keepdim=True)) @ Wp.T +
                           bp, dim=-1)

    # Validate the text-side map: CLAP-text(query) -> CLIP space, retrieve audio
    mapped_clap_text_te = map_to_clip(clap_text_te)
    eval_text = evaluate_method(X_te, mapped_clap_text_te, "m3", None, vids_te)
    print(f"  text-map R@1(clapText->audio)={eval_text['cliptext_to_audio']['R@1']:.4f}")

    # --- 3. Apply map to precomputed CLAP-audio embeddings ---
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, map_location="cpu",
                          weights_only=False)
    full_vids = sorted(audio_db.keys())
    X_full = F.normalize(torch.stack([audio_db[v].float() for v in full_vids]), dim=-1)
    audio_full = map_to_clip(X_full)

    # Test-set audio mapped
    audio_te = map_to_clip(X_te)

    # Evaluate CLIP-text(desc) query -> mapped audio gallery (the operative test)
    cat = {r["video_id"]: r["content_fine_category"] for r in recs_te}
    eval_m3 = evaluate_method(audio_te, Y_te, "m3", cat, vids_te)
    path = save_db(audio_full, "m3_shared_text", full_vids)
    eval_m3["export_path"] = path
    eval_m3["text_side"] = eval_text
    print(f"\n[M3 result] CLIP-text->audio: R@1={eval_m3['cliptext_to_audio']['R@1']:.4f} "
          f"MRR={eval_m3['cliptext_to_audio']['MRR']:.4f}")
    print(f"  alignment={eval_m3['alignment_cosine']:.4f} "
          f"modality_gap={eval_m3['modality_gap']:.4f}")
    print(f"  saved: {path}")

    out_path = os.path.join(OUTPUT_DIR, "m3_shared_text_results.json")
    with open(out_path, "w") as f:
        json.dump({"m3_shared_text": eval_m3}, f, indent=2, default=str)
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
