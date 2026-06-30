import torch, os, json, gc
from PIL import Image
import clip
from tqdm import tqdm

DEVICE = "cuda"
NUM_FRAMES = 16
HIDDEN_DIM = 512
FRAMES_DIR = "data/processed/video/frames_uniform"
METADATA_PATH = "data/processed/metadata/msrvtt_metadata.json"
OUTPUT_PATH = "embeddings/video_embeddings_transformer.pt"
CHECKPOINT = "temporal_transformer_best.pth"

import torch.nn as nn
import torch.nn.functional as F

class TemporalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, NUM_FRAMES + 1, HIDDEN_DIM) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=4, dim_feedforward=2048,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        cls_output = self.projection(x[:, 0])
        return F.normalize(cls_output, dim=1)

print("[INIT] Loading CLIP...")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad_(False)

print("[DATA] Finding videos with frames...")
valid_video_ids = set()
for vid in os.listdir(FRAMES_DIR):
    frame_dir = os.path.join(FRAMES_DIR, vid)
    if os.path.isdir(frame_dir):
        frame_count = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        if frame_count == NUM_FRAMES:
            valid_video_ids.add(vid)
print(f"Videos with {NUM_FRAMES} frames: {len(valid_video_ids)}")

def get_video_frames(video_id):
    frame_dir = os.path.join(FRAMES_DIR, video_id)
    if not os.path.exists(frame_dir):
        return None
    frames = []
    for i in range(NUM_FRAMES):
        frame_path = os.path.join(frame_dir, "frame_%04d.jpg" % i)
        if not os.path.exists(frame_path):
            return None
        img = Image.open(frame_path).convert("RGB")
        img = preprocess(img)
        frames.append(img)
    return torch.stack(frames)

print("[MODEL] Loading transformer checkpoint...")
model = TemporalTransformer().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

@torch.no_grad()
def encode_clip_frames(frame_tensors):
    bsz = frame_tensors.shape[0]
    flat = frame_tensors.flatten(0, 1)
    embeds = clip_model.encode_image(flat)
    embeds = embeds.view(bsz, NUM_FRAMES, HIDDEN_DIM).float()
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds

print("\n[EXPORT] Re-exporting embeddings (with squeeze fix)...")
all_video_embeds = {}
unique_videos = sorted(valid_video_ids)
for video_id in tqdm(unique_videos, desc="Encoding"):
    frames = get_video_frames(video_id)
    if frames is None:
        continue
    frames = frames.to(DEVICE, non_blocking=True)
    with torch.no_grad():
        frame_embeds = encode_clip_frames(frames.unsqueeze(0))
        video_embed = model(frame_embeds)
    all_video_embeds[video_id] = video_embed.squeeze(0).cpu()

    if (len(all_video_embeds)) % 1000 == 0:
        torch.save(all_video_embeds, OUTPUT_PATH)

    del frames, frame_embeds, video_embed
    gc.collect()
    torch.cuda.empty_cache()

torch.save(all_video_embeds, OUTPUT_PATH)
print(f"[DONE] Saved {len(all_video_embeds)} embeddings to {OUTPUT_PATH}")

# Verify shape
sample = next(iter(all_video_embeds.values()))
print(f"[VERIFY] Sample embedding shape: {sample.shape} (expected: torch.Size([512]))")
