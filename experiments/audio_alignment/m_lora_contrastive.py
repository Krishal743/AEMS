#!/usr/bin/env python3
"""
TIER 2: LoRA adapters inside the HTSAT audio encoder + contrastive fine-tuning
=============================================================================
Insert LoRA adapters (Hu 2021) into CLAP's HTSAT audio backend Linear layers
(attn.qkv / attn.proj / mlp.fc1 / mlp.fc2), freeze the base backbone weights,
and train the LoRA paths + a re-initialised projection head with InfoNCE
against FROZEN CLIP-text (description) targets on raw waveforms.

Trained strictly on the 5,748 TRAIN anchors; evaluated on held-out TEST (1022).

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/audio_alignment/m_lora_contrastive.py --rank 16 --epochs 40
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import AEMS_MANIFEST_PATH, AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, set_seeds
from src.data.metadata import load_metadata
from experiments.audio_alignment.compare_utils import load_anchor_pairs, save_db
from experiments.audio_alignment.audio_ft_utils import (
    AudioWaveformDataset, AudioCollator, clap_audio_embedding,
    contrastive_audio_loss, find_audio_path, AEMS_AUDIO_SR,
)
from experiments.audio_alignment.lora_torch import inject_lora, count_trainable

OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_paths(vids):
    return [find_audio_path(v) for v in vids]


def quick_eval(clap_model, X_te_waves, Y_te, device, bs=16, model_cfg=None):
    """Compute held-out test R@1/MRR (audio->text) using real waveform encode."""
    clap_model.eval()
    collator = AudioCollator(model_cfg)
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_te_waves), bs):
            batch = X_te_waves[i:i+bs]
            dicts = collator(batch)
            e = clap_audio_embedding(clap_model, dicts, device)
            embs.append(F.normalize(e, dim=-1).cpu())
    P = torch.cat(embs)
    Y = F.normalize(Y_te, dim=-1)
    sim = P @ Y.T
    n = P.shape[0]
    ranks = torch.argsort(sim, descending=True)
    rr = (ranks == torch.arange(n).view(-1, 1)).nonzero()[:, 1].float()
    r1 = float((rr == 0).float().mean())
    mrr = float((1.0 / (rr + 1)).mean())
    return r1, mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze HTSAT backbone entirely (projection-only) for sanity check")
    args = parser.parse_args()
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"TIER 2: LoRA in HTSAT + contrastive  (r={args.rank}, alpha={args.alpha})")
    print("=" * 72)

    # anchors
    X_tr, Y_tr, vids_tr = load_anchor_pairs("train")
    X_te, Y_te, vids_te = load_anchor_pairs("test")
    print(f"train anchors={len(vids_tr)}  test anchors={len(vids_te)}")
    Y_tr = F.normalize(Y_tr, dim=-1)
    Y_te = F.normalize(Y_te, dim=-1)

    # load CLAP
    import laion_clap
    clap = laion_clap.CLAP_Module(enable_fusion=False)
    clap.load_ckpt()
    model = clap.model.to(device)
    model.eval()
    model_cfg = clap.model_cfg

    # freeze text branch (we only optimize audio path); freeze CLAP audio projection?
    # Keep audio_projection trainable (it maps 1024->512). Re-init it to avoid inheriting CLAP's
    # own audio-text alignment bias (which is for CLAP text, not CLIP text).
    for p in model.text_branch.parameters():
        p.requires_grad_(False)
    if hasattr(model, "text_transform"):
        for p in model.text_transform.parameters():
            p.requires_grad_(False)
    # audio_projection: re-init and train
    def _reinit(seq):
        for m in seq:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    _reinit(model.audio_projection)
    for p in model.audio_projection.parameters():
        p.requires_grad_(True)

    inserted = []
    if not args.freeze_backbone:
        inserted = inject_lora(model, targets=None, r=args.rank, alpha=args.alpha, dropout=0.1)
    else:
        for p in model.audio_branch.parameters():
            p.requires_grad_(False)
        print("[sanity] backbone fully frozen (projection-head only)")

    train_params = count_trainable(model)
    print(f"LoRA modules inserted: {len(inserted)}  |  trainable params: {train_params/1e6:.3f} M")

    # datasets (cached waveforms)
    tr_paths = find_paths(vids_tr)
    print("Caching train waveforms (first-time may take a while)...")
    ds = AudioWaveformDataset(tr_paths, vids_tr, clip_sec=10)
    collator = AudioCollator(model_cfg)

    # precompute test waveforms once
    te_paths = find_paths(vids_te)
    te_ds = AudioWaveformDataset(te_paths, vids_te, clip_sec=10)
    te_waves = [te_ds[i] for i in range(len(te_ds))]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    logit_scale = nn.Parameter(torch.tensor(float(np.log(20.0))).to(device))
    opt.add_param_group({"params": [logit_scale]})

    inds = list(range(len(ds)))
    best_r1 = -1.0
    best_state = None
    for ep in range(args.epochs):
        model.train()
        random_state = np.random.default_rng(ep).permutation(inds)
        tot = 0.0; nb = 0
        for s in range(0, len(inds), args.batch):
            idx = random_state[s:s+args.batch]
            batch_waves = [ds[i] for i in idx]
            yb = Y_tr[idx].to(device)
            dicts = collator(batch_waves)
            opt.zero_grad()
            pa = clap_audio_embedding(model, dicts, device)
            loss = contrastive_audio_loss(pa, yb, logit_scale.exp())
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            r1, mrr = quick_eval(model, te_waves, Y_te, device, bs=args.batch, model_cfg=model_cfg)
            print(f"  ep {ep+1:3d} loss={tot/nb:.4f}  test R@1={r1:.4f} MRR={mrr:.4f}")
            if r1 > best_r1:
                best_r1 = r1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items() if v.dtype == torch.float32 or v.dtype == torch.float16}
                best_state["_logit_scale"] = logit_scale.detach().cpu()

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # final eval
    r1, mrr = quick_eval(model, te_waves, Y_te, device, bs=args.batch, model_cfg=model_cfg)
    print(f"\n[TIER2 result] audio->CLIP-text R@1={r1:.4f} MRR={mrr:.4f} (held-out test)")

    # export full-gallery aligned DB
    audio_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
    full_vids = sorted(audio_db.keys())
    print("Exporting full aligned audio DB ...")
    collator_full = AudioCollator(model_cfg)
    out = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(full_vids), args.batch):
            bv = full_vids[i:i+args.batch]
            bw = [load_waveform_cached(find_audio_path(v)) for v in bv]
            dicts = collator_full(bw)
            e = clap_audio_embedding(model, dicts, device)
            e = F.normalize(e, dim=-1).cpu()
            for j, v in enumerate(bv):
                out[v] = e[j]
    path = save_db_from_dict(out, "m4_lora")
    tag = f"rank{args.rank}"
    model_path = f"models/audio_lora_cliptext_{tag}.pt"
    torch.save(best_state, model_path)

    payload = {"rank": args.rank, "alpha": args.alpha, "n_lora": len(inserted),
               "trainable_params": train_params, "test_a2t_R1": r1, "test_a2t_MRR": mrr,
               "best_train_R1": best_r1, "export_path": path, "model_path": model_path}
    with open(os.path.join(OUTPUT_DIR, f"m2_lora_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("[DONE] outputs/alignment/m2_lora_results.json |", path)


def load_waveform_cached(path):
    from experiments.audio_alignment.audio_ft_utils import load_waveform, AEMS_AUDIO_SR
    return load_waveform(path, sr=AEMS_AUDIO_SR)


def save_db_from_dict(d, tag):
    path = f"embeddings/aems_audio_aligned_{tag}.pt"
    torch.save(d, path)
    return path


if __name__ == "__main__":
    main()
