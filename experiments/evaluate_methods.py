#!/usr/bin/env python3
"""
Modified evaluation script that can test different fusion methods
"""

import argparse
import torch
import torch.nn.functional as F
import clip
import json
from src.encoders.clap_encode import CLAPEncoder
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, DEVICE, set_seeds)

def run_evaluation_only():
    """Run evaluation without training - compare all methods"""
    print("[INFO] Starting evaluation-only comparison...")
    
    # Load metadata
    from src.data.metadata import load_metadata, filter_by_split
    metadata = load_metadata(AEMS_MANIFEST_PATH)
    test_items = filter_by_split(metadata, "test")
    
    # Load embeddings
    print("[INFO] Loading embeddings...")
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
    text_db_test = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"), weights_only=False)
    
    # Get test video IDs
    test_video_ids = set(item["video_id"] for item in test_items)
    common_vids_test = sorted(
        set(video_db.keys()) & set(audio_db.keys()) & set(text_db_test.keys()) & test_video_ids
    )
    
    print(f"[INFO] Test videos: {len(common_vids_test)}")
    
    # Build test matrices
    video_matrix = torch.stack([F.normalize(video_db[vid].float(), dim=0) for vid in common_vids_test])
    audio_matrix = torch.stack([F.normalize(audio_db[vid].float(), dim=0) for vid in common_vids_test])
    text_matrix = torch.stack([F.normalize(text_db_test[vid].float(), dim=0) for vid in common_vids_test])
    
    # Load test queries
    test_queries = []
    test_query_video_ids = []
    for item in test_items:
        vid = item["video_id"]
        if vid not in common_vids_test:
            continue
        for q in item["qa_questions"]:
            test_queries.append(q)
            test_query_video_ids.append(vid)
    
    print(f"[INFO] Test queries: {len(test_queries)}")
    
    # Encode queries using CLIP
    print("[INFO] Encoding test queries with CLIP...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    
    def encode_clip_queries(texts, batch_size=32):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(DEVICE)
            with torch.no_grad():
                emb = clip_model.encode_text(tokens)
                emb = F.normalize(emb, dim=-1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, dim=0)
    
    test_query_clip = encode_clip_queries(test_queries)
    print(f"[INFO] Encoded queries: {test_query_clip.shape}")
    
    # Encode queries using CLAP for audio
    print("[INFO] Encoding test queries with CLAP...")
    clap_encoder = CLAPEncoder(device=DEVICE)
    
    def encode_clap_queries(texts, batch_size=32):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                emb = clap_encoder.encode_text(batch)
            if isinstance(emb, torch.Tensor):
                emb = F.normalize(emb, dim=-1)
                all_emb.append(emb.cpu())
        return torch.cat(all_emb, dim=0) if all_emb else torch.zeros(len(texts), 512)
    
    test_query_clap = encode_clap_queries(test_queries)
    print(f"[INFO] Encoded CLAP queries: {test_query_clap.shape}")
    
    # Compute similarity matrices
    print("[INFO] Computing similarity matrices...")
    sim_v = test_query_clip.float() @ video_matrix.T
    sim_a = test_query_clap.float() @ audio_matrix.T
    sim_t = test_query_clip.float() @ text_matrix.T
    
    # Normalize similarities
    sim_v = F.normalize(sim_v, dim=1)
    sim_a = F.normalize(sim_a, dim=1)
    sim_t = F.normalize(sim_t, dim=1)
    
    # Evaluate different methods
    systems = {
        "Visual only": sim_v,
        "Text only": sim_t,
        "Audio only": sim_a,
        "Equal fusion": (sim_v + sim_t + sim_a) / 3,
    }
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (NO GATING)")
    print("=" * 70)
    
    for name, sim in systems.items():
        metrics = evaluate_retrieval(sim, test_query_video_ids, common_vids_test, ks=[1, 5, 10])
        print(f"  {name:>20}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
    
    # Load and test gating network if it exists
    import os
    gating_weights_path = "models/aems_gating_weights_v1.pth"
    if os.path.exists(gating_weights_path):
        print("\n[INFO] Testing adaptive gating with trained weights...")
        from src.models.gating_network import GatingNetwork
        
        gating_net = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
        gating_net.load_state_dict(torch.load(gating_weights_path), strict=False)
        gating_net.eval()
        
        sim_gated_list = []
        for i in range(0, len(test_queries), 32):
            q = test_query_clip[i:i+32].float().to(DEVICE)
            with torch.no_grad():
                w = gating_net(q)
            gated = (w[:, 0:1] * sim_v[i:i+32] +
                     w[:, 1:2] * sim_t[i:i+32] + 
                     w[:, 2:3] * sim_a[i:i+32])
            sim_gated_list.append(gated.cpu())
        
        sim_gated = torch.cat(sim_gated_list, dim=0)
        metrics = evaluate_retrieval(sim_gated, test_query_video_ids, common_vids_test, ks=[1, 5, 10])
        print(f"  {'Adaptive gating':>20}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
    else:
        print(f"\n[INFO] No gating weights found at {gating_weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation comparison")
    parser.add_argument("--mode", type=str, default="evaluation", choices=["training", "evaluation"])
    args = parser.parse_args()
    
    if args.mode == "evaluation":
        run_evaluation_only()
    else:
        print("Use the original training script for training mode")