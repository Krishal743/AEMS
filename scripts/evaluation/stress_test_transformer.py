import torch
import torch.nn as nn
import torch.nn.functional as F
import os, random, gc, numpy as np
from PIL import Image
import clip

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
DEVICE = "cuda"

NUM_FRAMES = 16
FRAMES_DIR = "data/processed/video/frames_uniform"
HIDDEN_DIM = 512

# Find valid videos
valid_video_ids = sorted(os.listdir(FRAMES_DIR))
random.shuffle(valid_video_ids)

print(f"[INIT] Found {len(valid_video_ids)} videos with frames")

# Load CLIP
print("[INIT] Loading CLIP ViT-B/32...")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model = clip_model.eval()

# Minimal transformer (same arch as the real one)
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
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.projection(x[:, 0])
        return F.normalize(x, dim=1)

def load_batch(video_ids, batch_size):
    frames_list, texts = [], []
    for vid in video_ids[:batch_size]:
        frame_dir = os.path.join(FRAMES_DIR, vid)
        fpaths = [os.path.join(frame_dir, "frame_%04d.jpg" % i) for i in range(NUM_FRAMES)]
        if not all(os.path.exists(p) for p in fpaths):
            continue
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in fpaths]
        frames_list.append(torch.stack(imgs))
        texts.append("a video")
    if len(frames_list) < 2:
        return None, None
    return torch.stack(frames_list), texts

def info_nce_loss(v_emb, t_emb, temp=0.07):
    logits = v_emb @ t_emb.T / temp
    labels = torch.arange(len(logits), device=DEVICE)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

batch_sizes = [32, 64, 128, 256, 512, 1024]
results = []
torch.cuda.reset_peak_memory_stats()

for bsz in batch_sizes:
    print(f"\n{'='*60}")
    print(f"[TEST] Batch size = {bsz}")
    print(f"{'='*60}")

    try:
        model = TemporalTransformer().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Load batch
        frames, texts = load_batch(valid_video_ids, bsz)
        if frames is None:
            print(f"[SKIP] Not enough valid videos for bsz={bsz}")
            results.append((bsz, "SKIP", 0))
            continue

        # Move to GPU
        frames = frames.to(DEVICE)

        # CLIP encode
        with torch.no_grad():
            frame_embeds = clip_model.encode_image(frames.flatten(0, 1))
            frame_embeds = frame_embeds.view(bsz, NUM_FRAMES, HIDDEN_DIM)
            frame_embeds = F.normalize(frame_embeds, dim=-1)

            text_tokens = clip.tokenize(texts).to(DEVICE)
            text_embeds = clip_model.encode_text(text_tokens)
            text_embeds = F.normalize(text_embeds, dim=-1)

        # Transformer forward
        video_embeds = model(frame_embeds.float())

        # InfoNCE + backward
        loss = info_nce_loss(video_embeds, text_embeds.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        peak = torch.cuda.max_memory_allocated() / 1024**3
        used = torch.cuda.memory_allocated() / 1024**3
        print(f"[ OK ] Loss={loss.item():.4f} | Peak={peak:.2f}GB | Used={used:.2f}GB")
        results.append((bsz, "OK", peak))

        # Cleanup
        del model, optimizer, frames, frame_embeds, text_embeds, video_embeds, loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[OOM] Peak was {peak:.2f}GB before failure")
        results.append((bsz, "OOM", peak))
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Don't try larger sizes
        break

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
for bsz, status, peak in results:
    print(f"  Batch {bsz:>4d}: {status:>4s}  (peak {peak:.2f}GB)")
