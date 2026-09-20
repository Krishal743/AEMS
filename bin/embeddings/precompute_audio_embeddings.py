import argparse, os, gc
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from tqdm import tqdm
from src.encoders.clap_encode import CLAPEncoder
from src.config import (AEMS_MANIFEST_PATH, AEMS_AUDIO_DIR, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_AUDIO_SR, AEMS_AUDIO_CLIP_SEC, DEVICE, set_seeds)
from src.data.metadata import load_metadata

parser = argparse.ArgumentParser(description="Precompute AEMS audio embeddings (3-segment CLAP)")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--audio-dir", type=str, default=AEMS_AUDIO_DIR)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seeds(args.seed)
os.makedirs("embeddings", exist_ok=True)
SEGMENT_DURATION = AEMS_AUDIO_CLIP_SEC
NUM_SEGMENTS = 3

print(f"[INIT] Loading CLAP on {DEVICE}")
encoder = CLAPEncoder(device=DEVICE)

print(f"[DATA] Loading manifest: {args.manifest}")
records = load_metadata(args.manifest)
videos = {}
for item in records:
    vid = item["video_id"]
    if vid not in videos:
        videos[vid] = os.path.join(args.audio_dir, f"{vid}.wav")

out = {}
skipped = 0

for vid, audio_path in tqdm(videos.items(), desc="Encoding audio"):
    if not os.path.exists(audio_path):
        skipped += 1
        continue
    try:
        audio, sr = librosa.load(audio_path, sr=AEMS_AUDIO_SR, mono=True)
    except Exception:
        skipped += 1
        continue
    duration_sec = len(audio) / AEMS_AUDIO_SR
    segment_samples = SEGMENT_DURATION * AEMS_AUDIO_SR

    if duration_sec <= SEGMENT_DURATION:
        segment = audio[:segment_samples]
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        x = segment.astype("float32").reshape(1, -1)
        with torch.no_grad():
            emb = encoder.model.get_audio_embedding_from_data(x=x)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        emb = F.normalize(emb, dim=-1).squeeze(0).cpu()
    else:
        seg_offsets = [
            0,
            int((duration_sec - SEGMENT_DURATION) / 2 * AEMS_AUDIO_SR),
            int((duration_sec - SEGMENT_DURATION) * AEMS_AUDIO_SR),
        ]
        segment_embeds = []
        for offset in seg_offsets:
            segment = audio[offset:offset + segment_samples]
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))
            x = segment.astype("float32").reshape(1, -1)
            with torch.no_grad():
                emb = encoder.model.get_audio_embedding_from_data(x=x)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            emb = F.normalize(emb, dim=-1).squeeze(0).cpu()
            segment_embeds.append(emb)
        audio_embed = torch.stack(segment_embeds).mean(dim=0)
        audio_embed = F.normalize(audio_embed, dim=0)
    out[vid] = audio_embed

torch.save(out, AEMS_AUDIO_EMBEDDINGS_PATH)
print(f"\n[DONE] Saved {len(out)} embeddings to {AEMS_AUDIO_EMBEDDINGS_PATH}")
print(f"       Skipped: {skipped}")
