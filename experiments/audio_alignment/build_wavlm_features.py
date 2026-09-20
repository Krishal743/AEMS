#!/usr/bin/env python3
"""
build_wavlm_features.py -- Strong self-supervised audio features (WavLM-Large)
==============================================================================
BEATs and ImageBind checkpoints are gated/unavailable in this environment, so
this uses torchaudio's WavLM-Large (24-layer self-supervised transformer,
1024-d) as the stronger-audio-encoder stand-in. Mirrors the CLAP cached
pipeline exactly so the adapter sees the same 3-segment-mean input semantics:

    for each 30s clip: 3 fixed segments [0, mid, end] (10s each)
        -> WavLM frame features (mean-pooled over time, last layer)
        -> mean over the 3 segments -> L2-normalized 1024-d

Writes embeddings/aems_audio_embeddings_wavlm_v1.pt ({video_id: tensor}).
Existing files/DBs are not modified.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/alignment/build_wavlm_features.py
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa

from src.config import AEMS_MANIFEST_PATH, AEMS_AUDIO_DIR, set_seeds
from src.data.metadata import load_metadata

SR = 16000
CLIP_SEC = 10
SEG_SAMPLES = SR * CLIP_SEC
OUT_PATH = "embeddings/aems_audio_embeddings_wavlm_v1.pt"


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
    parser.add_argument("--seg-per-fwd", type=int, default=32,
                        help="segments per WavLM forward pass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="optional cap on videos processed (debugging)")
    args = parser.parse_args()
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recs = load_metadata(AEMS_MANIFEST_PATH)
    seen = {}
    for r in recs:
        seen.setdefault(r["video_id"], True)
    vids = sorted(seen.keys())
    if args.limit:
        vids = vids[:args.limit]
    print(f"[DATA] {len(vids)} videos | sr={SR} | segs_per_fwd={args.seg_per_fwd} | {device}")

    bundle = torchaudio.pipelines.WAVLM_LARGE
    t0 = time.time()
    model = bundle.get_model().to(device)
    model.eval()
    print(f"[MODEL] WavLM-Large ready in {time.time()-t0:.0f}s")

    out = {}
    skipped = 0
    pending = []  # (vid, segment_index, seg_tensor)
    n_seg_done = 0

    def flush():
        nonlocal pending, n_seg_done
        if not pending:
            return
        waves = torch.stack([p[2] for p in pending]).to(device)
        waves = waves / (waves.abs().amax(dim=1, keepdim=True) + 1e-8)
        with torch.no_grad():
            hs, _ = model.extract_features(waves)
        feats = F.normalize(hs[-1].mean(dim=1), dim=-1).cpu()
        for (vid, si, _), f in zip(pending, feats):
            seg_emb[vid][si] = f
            n_seg_done += 1
        pending = []

    seg_emb = {v: [None, None, None] for v in vids}
    t0 = time.time()
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
        segs = segments_from_wave(wav, SR)
        for si, s in enumerate(segs):
            pending.append((vid, si, torch.from_numpy(s).float()))
            if len(pending) >= args.seg_per_fwd:
                flush()
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(vids)} videos  ({n_seg_done} segs, "
                  f"{time.time()-t0:.0f}s)  skipped={skipped}")
    flush()

    final = {}
    for v in vids:
        vals = [seg_emb[v][si] for si in range(3) if seg_emb[v][si] is not None]
        if not vals:
            continue
        e = F.normalize(torch.stack(vals).mean(dim=0), dim=0)
        final[v] = e
    torch.save(final, OUT_PATH)
    print(f"[DONE] {len(final)} videos -> {OUT_PATH} (skipped={skipped}) "
          f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()