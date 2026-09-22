"""WavLM-Large audio encoder.

Recipe (matches embeddings/aems_audio_embeddings_wavlm_v1.pt): resample to
16 kHz mono, take three fixed 10 s segments [start, middle, end] (zero-padded
when shorter), peak-normalize each, mean-pool WavLM's last layer over time,
L2-normalize, average the segments and L2-normalize again -> 1024-d.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa

from src.config import AEMS_WAVLM_SR, AEMS_AUDIO_CLIP_SEC

WAVLM_DIM = 1024


def fixed_segments(wav, sr=AEMS_WAVLM_SR, clip_sec=AEMS_AUDIO_CLIP_SEC):
    seg = sr * clip_sec
    dur = len(wav) / sr
    if dur <= clip_sec:
        offsets = [0]
    else:
        offsets = [0, int((dur - clip_sec) / 2 * sr), int((dur - clip_sec) * sr)]
    segs = []
    for off in offsets:
        s = wav[off:off + seg]
        if len(s) < seg:
            s = np.pad(s, (0, seg - len(s)))
        segs.append(s.astype("float32")[:seg])
    return segs


class WavLMEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = torchaudio.pipelines.WAVLM_LARGE.get_model().to(device).eval()

    @torch.no_grad()
    def encode_segments(self, segments):
        """(n, samples) float waveforms at 16 kHz -> (n, 1024) L2-normalized."""
        waves = torch.as_tensor(np.stack(segments)).float().to(self.device)
        waves = waves / (waves.abs().amax(dim=1, keepdim=True) + 1e-8)
        hs, _ = self.model.extract_features(waves)
        return F.normalize(hs[-1].mean(dim=1), dim=-1).cpu()

    def encode_wave(self, wav):
        """1-D 16 kHz waveform -> (1024,) L2-normalized clip embedding."""
        segs = self.encode_segments(fixed_segments(wav))
        return F.normalize(segs.mean(dim=0), dim=0)

    def encode_file(self, path):
        wav, _ = librosa.load(path, sr=AEMS_WAVLM_SR, mono=True)
        return self.encode_wave(wav)
