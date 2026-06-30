import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(self, text_dim=512, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, text_embed):
        x = text_embed.float() if text_embed.dtype == torch.float16 else text_embed
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x
