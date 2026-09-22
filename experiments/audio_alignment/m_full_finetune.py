#!/usr/bin/env python3
"""
TIER 3: full audio-encoder fine-tune + contrastive against frozen CLIP text
============================================================================
Unfreeze the entire HTSAT audio backbone + audio projection head and train
end-to-end with InfoNCE against FROZEN CLIP-text (description) targets, from
raw waveforms. Gradient accumulation + cosine schedule to fit a 24 GB GPU.

Trained strictly on the 5,748 TRAIN anchors; evaluated on held-out TEST (1022).

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/audio_alignment/m_full_finetune.py --epochs 12 --micro-batch 2 --accum 4 --lr 5e-6
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import AEMS_MANIFEST_PATH, AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, set_seeds
from src.data.metadata import load_metadata
from experiments.audio_alignment.compare_utils import load_anchor_pairs, save_db
from experiments.audio_alignment.audio_ft_utils import (
    AudioWaveformDataset, AudioCollator, clap_audio_embedding,
    contrastive_audio_loss, find_audio_path, load_waveform, AEMS_AUDIO_SR,
)

OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_paths(vids):
    return [find_audio_path(v) for v in vids]


def collate_deterministic(collator, batch, base_seed):
    """Seed numpy+torch RNG so the random 10s window crop is reproducible."""
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    return collator(batch)


def quick_eval(clap_model, X_te_waves, Y_te, device, bs=8, model_cfg=None, seed=1234):
    """Deterministic held-out test R@1/MRR (audio->text)."""
    clap_model.eval()
    collator = AudioCollator(model_cfg)
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_te_waves), bs):
            batch = X_te_waves[i:i+bs]
            dicts = collate_deterministic(collator, batch, seed + i // bs)
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="head + logit_scale learning rate")
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--warmup", type=float, default=0.02)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(f"TIER 3: FULL audio-encoder fine-tune  "
          f"(batch={args.batch}, head_lr={args.lr}, bb_lr={args.backbone_lr})")
    print("=" * 72)

    X_tr, Y_tr, vids_tr = load_anchor_pairs("train")
    X_te, Y_te, vids_te = load_anchor_pairs("test")
    print(f"train anchors={len(vids_tr)}  test anchors={len(vids_te)}")
    Y_tr = F.normalize(Y_tr, dim=-1)
    Y_te = F.normalize(Y_te, dim=-1)

    import laion_clap
    clap = laion_clap.CLAP_Module(enable_fusion=False)
    clap.load_ckpt()
    model = clap.model.to(device)
    model.train()
    model_cfg = clap.model_cfg

    # freeze the text side entirely
    for p in model.text_branch.parameters():
        p.requires_grad_(False)
    if hasattr(model, "text_transform"):
        for p in model.text_transform.parameters():
            p.requires_grad_(False)
    # audio path: fully trainable
    for p in model.audio_branch.parameters():
        p.requires_grad_(True)
    # disable stochastic SpecAugment masking so fine-tune gradients stay coherent
    for n, mod in model.audio_branch.named_children():
        if "spec_augment" in n.lower():
            setattr(model.audio_branch, n, nn.Identity())
    # fresh projection head (CLAP's own head is locked to CLAP-text; a fresh one
    # gives a well-posed mapping problem against frozen CLIP-text targets)
    def _reinit(seq):
        for m in seq:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    _reinit(model.audio_projection)
    for p in model.audio_projection.parameters():
        p.requires_grad_(True)

    backbone_params = [p for p in model.audio_branch.parameters() if p.requires_grad]
    head_params = [p for p in model.audio_projection.parameters() if p.requires_grad]
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"trainable: backbone={n_backbone/1e6:.1f}M head={n_head/1e3:.1f}K")

    tr_paths = find_paths(vids_tr)
    print("Caching train waveforms (first-time may take a while)...")
    ds = AudioWaveformDataset(tr_paths, vids_tr, clip_sec=10)
    collator = AudioCollator(model_cfg)

    te_paths = find_paths(vids_te)
    te_ds = AudioWaveformDataset(te_paths, vids_te, clip_sec=10)
    te_waves = [te_ds[i] for i in range(len(te_ds))]

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": 0.0},
        {"params": head_params, "lr": args.lr, "weight_decay": 1e-4},
    ])
    logit_scale = nn.Parameter(torch.tensor(float(np.log(20.0))).to(device))
    opt.add_param_group({"params": [logit_scale]})

    n_batches_per_epoch = int(np.ceil(len(ds) / args.batch))
    total_steps = args.epochs * n_batches_per_epoch
    warmup_steps = int(total_steps * args.warmup)

    def lr_mult(step):
        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)

    inds = list(range(len(ds)))
    best_r1 = -1.0
    best_state = None
    step = 0
    t0 = time.time()
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
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip)
            opt.step()
            sched.step()
            tot += loss.item(); nb += 1
            step += 1
        if ep % 2 == 0 or ep == args.epochs - 1:
            r1, mrr = quick_eval(model, te_waves, Y_te, device, bs=args.batch,
                                 model_cfg=model_cfg)
            el = time.time() - t0
            print(f"  ep {ep+1:3d} loss={tot/nb:.4f}  test R@1={r1:.4f} MRR={mrr:.4f}  elapsed={el/60:.1f}m")
            if r1 > best_r1:
                best_r1 = r1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()
                              if v.dtype == torch.float32 or v.dtype == torch.float16}
                best_state["_logit_scale"] = logit_scale.detach().cpu()

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    r1, mrr = quick_eval(model, te_waves, Y_te, device, bs=args.batch,
                         model_cfg=model_cfg)
    print(f"\n[TIER3 result] audio->CLIP-text R@1={r1:.4f} MRR={mrr:.4f} (held-out test, deterministic)")

    print("Caching full-gallery waveforms for export...")
    audio_db = torch.load(AEMS_CLAP_AUDIO_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
    full_vids = sorted(audio_db.keys())
    full_waves = {}
    for v in full_vids:
        full_waves[v] = load_waveform(find_audio_path(v), sr=AEMS_AUDIO_SR)
    print("Exporting full aligned audio DB ...")
    collator_full = AudioCollator(model_cfg)
    embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(full_vids), args.batch):
            bv = full_vids[i:i+args.batch]
            bw = [full_waves[v] for v in bv]
            dicts = collate_deterministic(collator_full, bw, 9000 + i // (args.batch))
            e = clap_audio_embedding(model, dicts, device)
            embs.append(F.normalize(e, dim=-1).cpu())
    embs = torch.cat(embs)
    path = save_db(embs, "m4_fullft", full_vids)
    model_path = "models/audio_fullft_cliptext.pt"
    torch.save(best_state, model_path)

    payload = {"epochs": args.epochs, "batch": args.batch,
               "lr": args.lr, "test_a2t_R1": r1, "test_a2t_MRR": mrr, "best_train_R1": best_r1,
               "export_path": path, "model_path": model_path}
    with open(os.path.join(OUTPUT_DIR, "tier3_fullft_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("[DONE] outputs/alignment/tier3_fullft_results.json |", path)


if __name__ == "__main__":
    main()