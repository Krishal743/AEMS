import json
import torch
import clip
import numpy as np
from tqdm import tqdm
from src.encoders.clap_encode import CLAPEncoder
from src.evaluation.evaluate_retrieval import evaluate_retrieval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METADATA = "data/processed/metadata/msrvtt_metadata.json"

def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)

def main():
    print("[SETUP] Loading embeddings...")
    clip_db = torch.load("embeddings/video_embeddings.pt", weights_only=False)
    audio_db = torch.load("embeddings/audio_embeddings.pt", weights_only=False)
    caption_db = torch.load("embeddings/caption_embeddings.pt", weights_only=False)
    
    print("[SETUP] Loading metadata...")
    with open(METADATA) as f:
        data = json.load(f)
    
    test_data = [d for d in data if d["split"] == "test"]
    print(f"[DATA] Test entries: {len(test_data)}")
    
    video_ids = sorted(set(clip_db.keys()) & set(audio_db.keys()) & set(caption_db.keys()))
    print(f"[DATA] Common videos: {len(video_ids)}")
    
    clip_video = torch.stack([clip_db[v] for v in video_ids])
    clap_audio = torch.stack([torch.tensor(audio_db[v]) for v in video_ids])
    
    clip_video = normalize(clip_video)
    clap_audio = normalize(clap_audio)
    
    texts = []
    text_video_ids = []
    for item in test_data:
        vid = item["video_id"]
        if vid not in video_ids:
            continue
        texts.append(item["text"])
        text_video_ids.append(vid)
    
    print(f"[TEXT] Queries: {len(texts)}")
    
    print("[MODEL] Loading CLIP...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    
    print("[MODEL] Loading CLAP...")
    clap_encoder = CLAPEncoder(device=DEVICE)
    
    print("[CLIP] Encoding query text...")
    BATCH_SIZE = 64
    clip_text_list = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="CLIP encoding"):
        batch = texts[i:i + BATCH_SIZE]
        tokens = clip.tokenize(batch).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        clip_text_list.append(emb.cpu())
    
    clip_text = torch.cat(clip_text_list, dim=0)
    
    print("[CLAP] Encoding query text...")
    clap_text_list = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="CLAP encoding"):
        batch = texts[i:i + BATCH_SIZE]
        with torch.no_grad():
            emb = clap_encoder.encode_text(batch)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        clap_text_list.append(emb.cpu())
    
    clap_text = torch.cat(clap_text_list, dim=0)
    
    print("[CAPTION] Computing sim_t (MAX aggregation)...")
    clip_text_device = clip_text.to(DEVICE)
    sim_t_list = []
    for vid in tqdm(video_ids, desc="sim_t"):
        cap_embeds = caption_db[vid].to(DEVICE)
        sims = clip_text_device @ cap_embeds.T
        max_sims = sims.max(dim=1).values
        sim_t_list.append(max_sims.cpu())
    
    sim_t = torch.stack(sim_t_list, dim=1)
    
    print("[SIM] Computing sim_v and sim_a...")
    clip_video_dev = clip_video.to(DEVICE).float()
    clap_audio_dev = clap_audio.to(DEVICE).float()
    clip_text_device = clip_text.to(DEVICE).float()
    clap_text_device = clap_text.to(DEVICE).float()
    sim_v = clip_text_device @ clip_video_dev.T
    sim_a = clap_text_device @ clap_audio_dev.T
    
    sim_v = sim_v.cpu()
    sim_a = sim_a.cpu()
    
    print("\n[VALIDATION] Sample rankings for query 0:")
    print(f"  sim_v top-5:  {torch.argsort(sim_v[0], descending=True)[:5].tolist()}")
    print(f"  sim_t top-5:  {torch.argsort(sim_t[0], descending=True)[:5].tolist()}")
    print(f"  sim_a top-5:  {torch.argsort(sim_a[0], descending=True)[:5].tolist()}")
    
    print("\n[EVAL] Evaluating each branch:")
    for name, sim in [("sim_v", sim_v), ("sim_t", sim_t), ("sim_a", sim_a)]:
        results = evaluate_retrieval(sim, text_video_ids, video_ids)
        print(f"  {name}: R@1={results['R@1']:.4f}, R@5={results['R@5']:.4f}, R@10={results['R@10']:.4f}")
    
    print("\n[EVAL] Testing fusion (equal weights)...")
    sim_fused = (sim_v + sim_t + sim_a) / 3
    results = evaluate_retrieval(sim_fused, text_video_ids, video_ids)
    print(f"  Equal fusion: R@1={results['R@1']:.4f}, R@5={results['R@5']:.4f}, R@10={results['R@10']:.4f}")

if __name__ == "__main__":
    main()
