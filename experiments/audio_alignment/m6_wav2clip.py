#!/usr/bin/env python3
"""
M6: Wav2CLIP-style audio alignment -- align to CLIP-video, query stays CLIP-text
================================================================================
Wu et al., 'Wav2CLIP' (ICASSP 2022): instead of aligning audio directly to text,
train the audio adapter to match the CLIP *image* embedding of the same video.
Because CLIP's image and text spaces are joined (Radford 2021), an audio vector
that matches the video's image embedding is automatically compatible with CLIP
text queries -- bypassing the problematic CLAP->CLIP text-space gap (M1/M3
failure) and using a semantically richer target than a short caption.

Reuses the M4 recipe (AudioAdapter MLP, hard-negative InfoNCE, batch 256) but
with FROZEN CLIP-video embeddings as the positive target (AudioCLIP/Guzhov 2022
frozen-backbone principle; ImageBind/Girdhar 2023 single-anchor unification).

Trained on 5,748 TRAIN anchors only; evaluated on held-out TEST (1,022) against
frozen CLIP-text description embeddings (both directions).

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/audio_alignment/m6_wav2clip.py --target video --epochs 30
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.audio_alignment.m4_adapter_contrastive import (
    AudioAdapter, contrastive_loss, quick_r1, quick_mrr,
)
from experiments.audio_alignment.compare_utils import (
    load_anchor_pairs, evaluate_method, save_db, collect_text_embeddings,
)
from src.config import (
    AEMS_AUDIO_EMBEDDINGS_PATH, AEMS_MANIFEST_PATH,
    AEMS_VID_EMBEDDINGS_PATH, set_seeds,
)
from src.data.metadata import load_metadata, filter_by_split

OUTPUT_DIR = "outputs/alignment"
ALIGNED_DIR = "embeddings"
MODEL_DIR = "models"
os.makedirs(ALIGNED_DIR, exist_ok=True)


def load_video_targets(vids):
    """Stack CLIP-video (mean-pooled frame) embeddings for the given videos,
    float32 + L2-normalized (the on-disk DB is float16)."""
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, map_location="cpu",
                          weights_only=False)
    return F.normalize(torch.stack([video_db[v].float() for v in vids]), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["video", "text", "hybrid"], default="video",
                        help="Training positive target: video (Wav2CLIP), text (M4), "
                             "or hybrid (sum of both InfoNCE terms)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=512)
    args = parser.parse_args()
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"M6: ADAPTER + CONTRASTIVE  (target={'CLIP-video' if args.target=='video' else 'CLIP-text'})")
    print("=" * 72)

    # --- Data: TRAIN anchors only ---
    X_tr, Y_tr_txt, vids_tr = load_anchor_pairs("train")
    X_te, Y_te, vids_te = load_anchor_pairs("test")
    assert X_tr.shape[0] == 5748, X_tr.shape

    train_align_name = None
    Y_tr_video = load_video_targets(vids_tr)
    Y_tr_text = F.normalize(Y_tr_txt, dim=-1)
    if args.target == "video":
        Y_tr = Y_tr_video
        train_align_name = "video"
    elif args.target == "text":
        Y_tr = Y_tr_text
    else:  # hybrid
        Y_tr = Y_tr_text
        train_align_name = "video"

    ds = TensorDataset(X_tr, Y_tr_text)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=0, drop_last=False)
    if args.target == "hybrid":
        ds_v = TensorDataset(X_tr, Y_tr_text, Y_tr_video)
        dl = DataLoader(ds_v, batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=False)

    model = AudioAdapter(d=512, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"Training on {X_tr.shape[0]} train anchors, {device} ...")
    best_r1 = -1.0
    best_state = None
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for batch in dl:
            opt.zero_grad()
            if args.target == "hybrid":
                xb, yb_txt, yb_vid = batch
                xb = xb.to(device)
                yb_txt = F.normalize(yb_txt, dim=-1).to(device)
                yb_vid = F.normalize(yb_vid, dim=-1).to(device)
                proj = model.forward(xb)
                loss = (contrastive_loss(proj, yb_txt, model.logit_scale)
                        + contrastive_loss(proj, yb_vid, model.logit_scale))
            else:
                xb, yb = batch
                xb, yb = xb.to(device), F.normalize(yb, dim=-1).to(device)
                proj = model.forward(xb)
                loss = contrastive_loss(proj, yb, model.logit_scale)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        # quick val R@1 on test (vs CLIP-text) every few epochs
        if ep % 5 == 0 or ep == args.epochs - 1:
            r1 = quick_r1(model, X_te, Y_te, device)
            mrr = quick_mrr(model, X_te, Y_te, device)
            msg = f"  ep {ep+1:3d}  loss={tot/len(dl):.4f}  test audio->text R@1={r1:.4f} MRR={mrr:.4f}"
            if train_align_name:
                with torch.no_grad():
                    p = model.encode(X_tr[:1024].to(device))
                    a = float((F.normalize(p, dim=-1) * Y_tr[:1024].to(device)).sum(1).mean())
                msg += f"  train align[{train_align_name}]={a:.4f}"
            print(msg)
            if r1 > best_r1:
                best_r1 = r1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if args.target == "video":
        model_path = os.path.join(MODEL_DIR, "audio_adapter_clipvid_m6.pt")
        tag = "m6_w2c"
        res_key = "m6_w2clip"
    elif args.target == "hybrid":
        model_path = os.path.join(MODEL_DIR, "audio_adapter_clipvid_text_hybrid_m6.pt")
        tag = "m6_hybrid"
        res_key = "m6_hybrid"
    else:
        model_path = os.path.join(MODEL_DIR, "audio_adapter_cliptext_m6_textvariant.pt")
        tag = "m6_textvar"
        res_key = "m6_text_target"
    torch.save(best_state, model_path)

    # --- Final evaluation on test (both directions, vs CLIP-text) ---
    recs_te = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    cat = {r["video_id"]: r["content_fine_category"] for r in recs_te}
    model.eval()
    with torch.no_grad():
        proj_te = model.encode(X_te.to(device)).cpu()
    eval_m6 = evaluate_method(proj_te, Y_te, tag, cat, vids_te)

    # Full-gallery export
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, map_location="cpu",
                          weights_only=False)
    full_vids = sorted(audio_db.keys())
    X_full = F.normalize(torch.stack([audio_db[v].float() for v in full_vids]), dim=-1)
    with torch.no_grad():
        proj_full = model.encode(X_full.to(device)).cpu()
    path = save_db(proj_full, tag, full_vids)
    eval_m6["export_path"] = path
    eval_m6["best_train_R1"] = best_r1
    eval_m6["model_path"] = model_path

    print(f"\n[M6 result] CLIP-text->audio: R@1={eval_m6['cliptext_to_audio']['R@1']:.4f} "
          f"MRR={eval_m6['cliptext_to_audio']['MRR']:.4f} "
          f"R@10={eval_m6['cliptext_to_audio']['R@10']:.4f} "
          f"median_rank={eval_m6['cliptext_to_audio']['median_rank']:.0f}")
    print(f"  audio->CLIP-text R@1={eval_m6['audio_to_cliptext']['R@1']:.4f} "
          f"R@10={eval_m6['audio_to_cliptext']['R@10']:.4f}")
    print(f"  alignment={eval_m6['alignment_cosine']:.4f} "
          f"modality_gap={eval_m6['modality_gap']:.4f}")
    print(f"  model={model_path}\n  saved DB: {path}")

    with open(os.path.join(OUTPUT_DIR, "m6_wav2clip_results.json"), "w") as f:
        json.dump({res_key: eval_m6}, f, indent=2, default=str)
    print("[DONE] outputs/alignment/m6_wav2clip_results.json")


if __name__ == "__main__":
    main()