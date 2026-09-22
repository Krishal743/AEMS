"""Precompute raw WavLM-Large audio features (1024-d, 3 fixed 10 s segments).

These are adapter inputs, not the searchable audio branch: run
bin/training/train_audio_adapter.py afterwards to project them into CLIP space
(AEMS_AUDIO_EMBEDDINGS_PATH).
"""

import argparse, os
import librosa
import torch
from tqdm import tqdm
from src.encoders.wavlm_encode import WavLMEncoder
from src.config import (AEMS_MANIFEST_PATH, AEMS_AUDIO_DIR, AEMS_WAVLM_FEATURES_PATH,
                        AEMS_WAVLM_SR, DEVICE, set_seeds)
from src.data.metadata import load_metadata

parser = argparse.ArgumentParser(description="Precompute AEMS WavLM audio features")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--audio-dir", type=str, default=AEMS_AUDIO_DIR)
parser.add_argument("--output", type=str, default=AEMS_WAVLM_FEATURES_PATH)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seeds(args.seed)
os.makedirs(os.path.dirname(args.output), exist_ok=True)

print(f"[INIT] Loading WavLM-Large on {DEVICE}")
encoder = WavLMEncoder(device=DEVICE)

print(f"[DATA] Loading manifest: {args.manifest}")
videos = sorted({r["video_id"] for r in load_metadata(args.manifest)})

out = {}
skipped = 0
for vid in tqdm(videos, desc="Encoding audio"):
    path = os.path.join(args.audio_dir, f"{vid}.wav")
    if not os.path.exists(path):
        skipped += 1
        continue
    try:
        wav, _ = librosa.load(path, sr=AEMS_WAVLM_SR, mono=True)
    except Exception:
        skipped += 1
        continue
    out[vid] = encoder.encode_wave(wav)

torch.save(out, args.output)
print(f"\n[DONE] Saved {len(out)} features to {args.output}")
print(f"       Skipped: {skipped}")
