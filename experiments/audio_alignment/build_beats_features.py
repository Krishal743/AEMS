#!/usr/bin/env python3
"""
build_beats_features.py -- BEATs (iter3+ AS2M) audio features
==============================================================
Official Microsoft BEATs model (90.3M, 768-d, AudioSet-2M pretraining) recovered
from microsoft/unilm becausete the standalone microsoft/BEATs repo was deleted.
Checkpoint: lpepino/beats_ckpts (public mirror of the official weights, no HF
token required). Mirrors the cached CLAP 3-segment-mean protocol:

    for each 30s clip: 3 fixed 10s segments [0, mid, end] @16kHz
        -> BEATs patch-embedding transformer (last layer) -> mean pool
        -> mean over the 3 segments -> L2-normalized 768-d

Writes embeddings/aems_audio_embeddings_beats_v1.pt ({video_id: tensor}).

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/audio_alignment/build_beats_features.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "beats_code"))
from BEATs import BEATs, BEATsConfig

from src.config import AEMS_MANIFEST_PATH, AEMS_AUDIO_DIR, set_seeds
from src.data.metadata import load_metadata

SR = 16000
CLIP_SEC = 10
SEG_SAMPLES = SR * CLIP_SEC
CHECKPOINT = os.path.expanduser("~/.cache/beats/BEATs_iter3_plus_AS2M.pt")
OUT_PATH = "embeddings/aems_audio_embeddings_beats_v1.pt"


def segments_from_wave(wav, sr):
    """3 fixed 10s segments (+pad) mirroring the cached CLAP recipe."""
    dur = wav.shape[-1] / sr
    seg_samples = SEG_SAMPLES
    if dur <= CLIP_SEC:
        offsets = [0]
    else:
        offsets = [0,
                   int((dur - CLIP_SEC) / 2 * sr),
                   int((dur - CLIP_SEC) * sr)]
    segs = []
    for off in offsets:
        s = wav[off:off + seg_samples]
        if s.shape[-1] < seg_samples:
            s = np.pad(s, (0, seg_samples - s.shape[-1]))
        segs.append(s.astype("float32")[:seg_samples])
    return segs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default=AEMS_AUDIO_DIR)
    parser.add_argument("--seg-per-fwd", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recs = load_metadata(AEMS_MANIFEST_PATH)
    vids = sorted({r["video_id"] for r in recs})
    if args.limit:
        vids = vids[:args.limit]
    print(f"[DATA] {len(vids)} videos | sr={SR} | segs_per_fwd={args.seg_per_fwd} | {device}")

    ck = torch.load(CHECKPOINT, map_location="cpu")
    m = BEATs(BEATsConfig(ck["cfg"]))
    m.load_state_dict(ck["model"])
    m.eval().to(device)
    del ck
    print("[MODEL] BEATs_iter3_plus_AS2M loaded")

    seg_emb = {v: [] for v in vids}
    skipped = 0
    pending = []
    n_seg = 0
    t0 = time.time()

    def flush():
        nonlocal pending, n_seg
        if not pending:
            return
        waves = torch.stack([p[2] for p in pending])
        with torch.no_grad():
            rep, _ = m.extract_features(waves.to(device))
        feats = F.normalize(rep.mean(dim=1), dim=-1).cpu()
        for (vid, ei, _), f in zip(pending, feats):
            seg_emb[vid].append(f)
            n_seg += 1
        pending = []

    for i, vid in enumerate(vids):
        path = os.path.join(args.audio_dir, f"{vid}.wav")
        if not os.path.exists(path):
            skipped += 1
            continue
        try:
            wav, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            skipped += 1
            continue
        for ei, s in enumerate(segments_from_wave(wav, SR)):
            pending.append((vid, ei, torch.from_numpy(s).float()))
            if len(pending) >= args.seg_per_fwd:
                flush()
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(vids)} ({n_seg} segs, {time.time()-t0:.0f}s) skipped={skipped}")
    flush()

    final = {v: F.normalize(torch.stack(seg_emb[v]).mean(dim=0), dim=0)
             for v in vids if seg_emb[v]}
    torch.save(final, OUT_PATH)
    print(f"[DONE] {len(final)} videos -> {OUT_PATH} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()