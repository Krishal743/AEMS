import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
import clip
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_QUERIES = 50
MAX_VIDEOS = 50


class GatingNetwork(nn.Module):
    def __init__(self, text_dim=512, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, text_embed):
        x = text_embed.float() if text_embed.dtype == torch.float16 else text_embed
        return self.net(x)


def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()


def main():
    print("[INIT] Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    
    print("[DATA] Loading embeddings...")
    video_db_v = torch.load("embeddings/video_embeddings.pt", weights_only=False)
    video_db_a = torch.load("embeddings/audio_embeddings.pt", weights_only=False)
    caption_db = torch.load("embeddings/caption_embeddings.pt", weights_only=False)
    
    common_vids = [vid for vid in video_db_v.keys() if vid in video_db_a and vid in caption_db][:MAX_VIDEOS]
    
    video_embeds_v = F.normalize(torch.stack([to_float(video_db_v[vid]) for vid in common_vids]).to(DEVICE), p=2, dim=1).float()
    video_embeds_a = F.normalize(torch.stack([to_float(video_db_a[vid]) for vid in common_vids]).to(DEVICE), p=2, dim=1).float()
    caption_embeds = F.normalize(torch.stack([to_float(caption_db[vid]).max(dim=0)[0] for vid in common_vids]).to(DEVICE), p=2, dim=1).float()
    
    with open("data/processed/metadata/msrvtt_metadata.json") as f:
        metadata = json.load(f)
    
    test_items = [m for m in metadata if m["split"] == "test"]
    common_vid_set = set(common_vids)
    filtered_items = [item for item in test_items if item["video_id"] in common_vid_set]
    
    unique_texts = []
    unique_video_ids = []
    seen = set()
    for item in filtered_items:
        if item["video_id"] not in seen:
            seen.add(item["video_id"])
            unique_texts.append(item["text"])
            unique_video_ids.append(item["video_id"])
    
    unique_texts = unique_texts[:MAX_QUERIES]
    unique_video_ids = unique_video_ids[:MAX_QUERIES]
    
    print(f"[TEXT] Encoding...")
    tokens = clip.tokenize(unique_texts).to(DEVICE)
    text_embeds = clip_model.encode_text(tokens).float()
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
    del clip_model
    gc.collect()
    
    num_queries = len(unique_video_ids)
    num_videos = len(common_vids)
    
    print("\n" + "="*70)
    print("LOSS BEHAVIOR ANALYSIS")
    print("="*70)
    
    gating_net = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(gating_net.parameters(), lr=1e-3)
    
    sample_indices = random.sample(range(num_queries), min(5, num_queries))
    
    print("\n--- BEFORE TRAINING ---")
    gating_net.eval()
    for q_idx in sample_indices:
        q = text_embeds[q_idx:q_idx+1]
        
        with torch.no_grad():
            w = gating_net(q)[0]
        
        correct_idx = common_vids.index(unique_video_ids[q_idx])
        
        s_v = (q @ video_embeds_v.T)[0, correct_idx].item()
        s_t = (q @ caption_embeds.T)[0, correct_idx].item()
        s_a = (q @ video_embeds_a.T)[0, correct_idx].item()
        
        print(f"\nQuery: \"{unique_texts[q_idx][:35]}\"")
        print(f"  Raw scores: v={s_v:.4f}, t={s_t:.4f}, a={s_a:.4f}")
        print(f"  Weights: {w[0].item():.4f}, {w[1].item():.4f}, {w[2].item():.4f}")
    
    print("\n--- TRAINING 1 EPOCH ---")
    gating_net.train()
    indices = torch.randperm(num_queries)
    
    for i in range(0, min(50, num_queries), 10):
        batch = indices[i:i+10]
        
        q_batch = text_embeds[batch]
        
        sim_v = q_batch @ video_embeds_v.T
        sim_t = q_batch @ caption_embeds.T
        sim_a = q_batch @ video_embeds_a.T
        
        weights = gating_net(q_batch)
        
        sim_gated = (weights[:, 0:1] * sim_v + weights[:, 1:2] * sim_t + weights[:, 2:3] * sim_a)
        
        gt = torch.arange(len(batch)).long().to(DEVICE)
        loss = nn.CrossEntropyLoss()(sim_gated, gt)
        
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
    
    print("\n--- AFTER TRAINING ---")
    gating_net.eval()
    for q_idx in sample_indices:
        q = text_embeds[q_idx:q_idx+1]
        
        with torch.no_grad():
            w = gating_net(q)[0]
        
        correct_idx = common_vids.index(unique_video_ids[q_idx])
        
        s_v = (q @ video_embeds_v.T)[0, correct_idx].item()
        s_t = (q @ caption_embeds.T)[0, correct_idx].item()
        s_a = (q @ video_embeds_a.T)[0, correct_idx].item()
        
        equal_fused = (s_v + s_t + s_a) / 3
        weighted_fused = w[0]*s_v + w[1]*s_t + w[2]*s_a
        best_single = max(s_v, s_t, s_a)
        
        print(f"\nQuery: \"{unique_texts[q_idx][:35]}\"")
        print(f"  Raw scores: v={s_v:.4f}, t={s_t:.4f}, a={s_a:.4f}")
        print(f"  Weights: {w[0].item():.4f}, {w[1].item():.4f}, {w[2].item():.4f}")
        print(f"  Equal={equal_fused:.4f}, Weighted={weighted_fused:.4f}, Best={best_single:.4f}")
    
    print("\n--- WEIGHT RANGES ---")
    all_w = []
    for i in range(num_queries):
        with torch.no_grad():
            all_w.append(gating_net(text_embeds[i:i+1])[0].tolist())
    all_w = torch.tensor(all_w)
    print(f"w_v: {all_w[:,0].min():.4f} - {all_w[:,0].max():.4f}")
    print(f"w_t: {all_w[:,1].min():.4f} - {all_w[:,1].max():.4f}")
    print(f"w_a: {all_w[:,2].min():.4f} - {all_w[:,2].max():.4f}")
    print(f"\nNarrow range (<0.1) = averaging behavior")


if __name__ == "__main__":
    main()