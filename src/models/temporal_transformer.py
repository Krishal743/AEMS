import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        num_frames=16,
        hidden_dim=512,
        num_heads=4,
        num_layers=2,
        ffn_hidden=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames + 1, hidden_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        cls_output = self.projection(x[:, 0])
        return F.normalize(cls_output, dim=1)
