"""Projects raw audio features (e.g. WavLM, 1024-d) into CLIP text space."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioAdapter(nn.Module):
    """MLP with a linear skip path; outputs L2-normalized 512-d CLIP-space vectors."""

    def __init__(self, input_dim=1024, output_dim=512, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, x):
        return F.normalize(self.net(x) + self.skip(x), dim=-1)


def load_audio_adapter(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    input_dim = state["skip.weight"].shape[1]
    adapter = AudioAdapter(input_dim=input_dim).to(device)
    adapter.load_state_dict(state)  # strict: a mismatched checkpoint must fail
    adapter.eval()
    return adapter
