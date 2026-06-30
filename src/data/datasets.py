import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import clip

class MSRVTTDataset(Dataset):
    def __init__(self, metadata_path, split="train", device="cpu"):
        with open(metadata_path) as f:
            data = json.load(f)

        self.data = [d for d in data if d["split"] == split]
        self.device = device
        self.preprocess = clip.load("ViT-B/32", device=device)[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        frames_dir = item["frames_dir"]

        frames = sorted(os.listdir(frames_dir))
        images = []

        for f in frames:
            img = Image.open(os.path.join(frames_dir, f)).convert("RGB")
            images.append(self.preprocess(img))

        images = torch.stack(images)          # [T, 3, H, W]
        text = item["text"]

        return images, text, item["video_id"]
