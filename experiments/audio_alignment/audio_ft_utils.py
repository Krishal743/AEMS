#!/usr/bin/env python3
"""
Shared utilities for gradient-based CLAP audio fine-tuning (Tiers 2/3).
Provides:
  - waveform loading (librosa, 48k mono) matching CLAPEncoder.encode_audio
  - batch feature preparation via CLAP's own get_audio_features (exact inference
    parity), keeping gradients flowing through the audio backbone
  - a pre-cached waveform dataset to avoid re-reading 5.7k audio files per epoch
  - the frozen-CLIP-text contrastive loss (InfoNCE) + saving aligned DBs
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import librosa

from laion_clap.training.data import get_audio_features
from experiments.audio_alignment.compare_utils import save_db
from src.config import set_seeds

AEMS_AUDIO_SR = 48000
AEMS_AUDIO_CLIP_SEC = 10
AEMS_AUDIO_NUM_SEGMENTS = 3


def load_waveform(audio_path, sr=AEMS_AUDIO_SR):
    """Mono float32 audio tensor (len = sr*clip_sec)."""
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    wav = wav.astype("float32")
    return torch.from_numpy(wav).float()


class AudioWaveformDataset(torch.utils.data.Dataset):
    """Pre-cached waveforms keyed by (video_id, audio_path, clip_sec).
    Eagerly loads all waveforms once (RAM: 5748 * 10s * 48k * 4B ~= 11 GB)."""

    def __init__(self, paths, vids, clip_sec=10, sr=48000):
        self.paths = paths
        self.vids = vids
        self.sr = sr
        self.target_len = sr * clip_sec
        self.waves = []
        for p in paths:
            w = load_waveform(p, sr=sr)
            self.waves.append(w)

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        return self.waves[i]


class AudioCollator:
    """Converts a list of waveforms into CLAP audio input dicts."""

    def __init__(self, model_cfg, sr=48000, clip_sec=10, seed=42):
        self.model_cfg = model_cfg
        self.sr = sr
        self.clip_sec = clip_sec
        self.seed = seed

    def __call__(self, batch_waves):
        dicts = []
        for w in batch_waves:
            d = {}
            d = get_audio_features(
                d, w, self.sr * self.clip_sec,
                data_truncating="rand_trunc",
                data_filling="repeatpad",
                audio_cfg=self.model_cfg["audio_cfg"],
                require_grad=True,
            )
            dicts.append(d)
        return dicts


def clap_audio_embedding(clap_model, audio_dicts, device):
    """Run the differentiable audio projection path (mirrors get_audio_embedding)."""
    keys = audio_dicts[0].keys()
    input_dict = {}
    for k in keys:
        input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in audio_dicts], dim=0).to(device)
    embeds = clap_model.encode_audio(input_dict, device=device)["embedding"]
    embeds = clap_model.audio_projection(embeds)
    return embeds


def contrastive_audio_loss(proj_audio, text_tgt, logit_scale):
    """InfoNCE with identity positives + in-batch negatives.
    proj_audio, text_tgt: [B,d] not yet normalized (we normalize here).
    Returns scalar loss. logit_scale is a scalar tensor (exp applied)."""
    pa = F.normalize(proj_audio, dim=-1)
    tt = F.normalize(text_tgt, dim=-1)
    n = pa.shape[0]
    L = logit_scale.clamp(max=50)
    sim = pa @ tt.T * L
    diag = torch.arange(n, device=sim.device)
    pos = sim[diag, diag]
    return -pos.mean() + torch.logsumexp(sim, dim=1).mean()


def build_audio_db_from_model(clap_model, vids, device, bs=32):
    """Encode a set of videos' waveforms into an aligned audio DB {vid: tensor}."""
    clap_model.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(vids), bs):
            batch_vids = vids[i:i+bs]
            waves = [load_waveform(find_audio_path(v), sr=AEMS_AUDIO_SR) for v in batch_vids]
            collator = AudioCollator(clap_model.model_cfg, sr=AEMS_AUDIO_SR)
            dicts = collator(waves)
            emb = clap_audio_embedding(clap_model, dicts, device)
            emb = F.normalize(emb, dim=-1).cpu()
            for j, v in enumerate(batch_vids):
                out[v] = emb[j]
    return out


_audio_path_cache = {}


def find_audio_path(vid):
    if vid not in _audio_path_cache:
        from src.config import AEMS_MANIFEST_PATH
        from src.data.metadata import load_metadata
        if not _audio_path_cache:
            for r in load_metadata(AEMS_MANIFEST_PATH):
                _audio_path_cache[r["video_id"]] = r["audio_path"]
    return _audio_path_cache.get(vid)
