#!/usr/bin/env python3
"""
build_imagebind_features.py -- ImageBind (huge) audio features
==============================================================
Official facebookresearch/ImageBind checkpoint (public dl.fbaipublicfiles.com).
The audio branch is a 12-layer transformer fed with a fixed 204-frame
log-mel (2 s) input (learnable pos-embed is fixed-size), so unlike CLAP/WavLM
the 3-window scheme uses 3 fixed **2 s** windows at the same anchors
[0, mid, end]; the three per-window embeddings are meaned and L2-normalized.
Preprocessing mirrors the official demo (imagebind/data.py) exactly:
kaldi fbank (htk, use_energy=False, hanning, dither=0, 25ms/10ms, 128 mel,
target_length=204) then Normalize(mean=-4.268, std=9.138) over the mel image.

Writes embeddings/aems_audio_embeddings_imagebind_v1.pt ({video_id: 1024-d}).

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/alignment/build_imagebind_features.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi
import librosa
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagebind_deps"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagebind_code"))
from imagebind.models.imagebind_model import ImageBindModel, ModalityType

from src.config import AEMS_MANIFEST_PATH, AEMS_AUDIO_DIR, set_seeds
from src.data.metadata import load_metadata

SR = 16000
WIN_SEC = 2
WIN_SAMPLES = SR * WIN_SEC
NUM_MEL_BINS = 128
TARGET_LENGTH = 204
MEAN, STD = -4.268, 9.138
CHECKPOINT = os.path.expanduser("~/.cache/beats/imagebind_huge.pth")
OUT_PATH = "embeddings/aems_audio_embeddings_imagebind_v1.pt"


def waveform2melspec(waveform, target_length=TARGET_LENGTH):
    waveform = waveform - waveform.mean()
    fbank = ta_kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=SR,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=NUM_MEL_BINS,
        dither=0.0,
        frame_length=25,
        frame_shift=10,
    )
    fbank = fbank.transpose(0, 1)          # [128, T]
    n_frames = fbank.size(1)
    p = target_length - n_frames
    if p > 0:
        fbank = F.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    return fbank.unsqueeze(0)              # [1, 128, 204]


def win_offsets(dur_sec):
    if dur_sec <= WIN_SEC:
        return [0]
    last = int((dur_sec - WIN_SEC) * SR)
    return [0, int((dur_sec - WIN_SEC) / 2 * SR), last]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default=AEMS_AUDIO_DIR)
    parser.add_argument("--win-per-fwd", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    recs = load_metadata(AEMS_MANIFEST_PATH)
    vids = sorted({r["video_id"] for r in recs})
    if args.limit:
        vids = vids[:args.limit]
    print(f"[DATA] {len(vids)} videos | sr={SR} | win(s)={WIN_SEC} | {device}")

    model = ImageBindModel(
        vision_embed_dim=1280, vision_num_blocks=32, vision_num_heads=16,
        text_embed_dim=1024, text_num_blocks=24, text_num_heads=16,
        out_embed_dim=1024, audio_drop_path=0.1, imu_drop_path=0.7).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    print("[MODEL] imagebind_huge loaded")

    norm = transforms.Normalize(mean=MEAN, std=STD)
    win_emb = {v: [] for v in vids}
    skipped = 0
    pending = []
    n_win = 0
    t0 = time.time()

    def flush():
        nonlocal pending, n_win
        if not pending:
            return
        specs = torch.stack([p[2] for p in pending]).to(device)
        with torch.no_grad():
            out = model({ModalityType.AUDIO: specs})
        embs = out[ModalityType.AUDIO].cpu()          # [B,1024] normalized
        for (vid, _, _), e in zip(pending, embs):
            win_emb[vid].append(e)
            n_win += 1
        pending = []

    for i, vid in enumerate(vids):
        path = os.path.join(args.audio_dir, f"{vid}.wav")
        if not os.path.exists(path):
            skipped += 1
            continue
        wav, _ = librosa.load(path, sr=SR, mono=True)
        for off in win_offsets(wav.shape[-1] / SR):
            seg = wav[off:off + WIN_SAMPLES]
            if seg.shape[-1] < WIN_SAMPLES:
                seg = np.pad(seg, (0, WIN_SAMPLES - seg.shape[-1]))
            seg = torch.from_numpy(seg.astype("float32"))
            spec = norm(waveform2melspec(seg.unsqueeze(0)))
            pending.append((vid, off, spec))
            if len(pending) >= args.win_per_fwd:
                flush()
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(vids)} ({n_win} wins, {time.time()-t0:.0f}s) skipped={skipped}")
    flush()

    final = {v: F.normalize(torch.stack(win_emb[v]).mean(dim=0), dim=0)
             for v in vids if win_emb[v]}
    torch.save(final, OUT_PATH)
    print(f"[DONE] {len(final)} videos -> {OUT_PATH} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()