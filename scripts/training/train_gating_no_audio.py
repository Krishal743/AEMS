#!/usr/bin/env python3
"""
Modified gating network that only uses visual + text (no audio)
"""

import argparse
import json
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import random
import numpy as np
from src.encoders.clap_encode import CLAPEncoder
from src.models.gating_network import GatingNetwork
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, AEMS_GATING_WEIGHTS_PATH,
                         DEVICE, set_seeds)
from src.data.metadata import load_metadata, filter_by_split

set_seeds(42)

# Modified parameters - remove audio
QUERY_BATCH_SIZE = 32
VIDEO_BATCH_SIZE = 100
TOP_K = 10
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
RANKING_MARGIN = 0.2
NUM_NEGATIVES = 10

# New gating network with only 2 modalities
class GatingNetworkNoAudio(nn.Module):
    def __init__(self, text_dim=512, hidden_dim=128):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Only visual + text
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        weights = torch.softmax(self.fc2(x), dim=1)
        return weights

def train_gating_no_audio():
    print("[INFO] Training gating network with visual + text only (no audio)")
    
    # Load metadata
    metadata = load_metadata(AEMS_MANIFEST_PATH)
    train_items = filter_by_split(metadata, "train")
    test_items = filter_by_split(metadata, "test")
    
    # Load embeddings (no audio needed)
    print("[EMB] Loading embeddings...")
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
    text_db_train = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"), weights_only=False)
    text_db_test = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"), weights_only=False)
    
    # Get common video IDs
    all_video_ids = set()
    for item in train_items + test_items:
        all_video_ids.add(item["video_id"])
    
    common_vids_all = sorted(
        set(video_db.keys()) & set(text_db_train.keys()) & all_video_ids
    )
    common_vids_test = sorted(
        set(video_db.keys()) & set(text_db_test.keys()) &
        set(item["video_id"] for item in test_items)
    )
    
    print(f"[DATA] All videos: {len(common_vids_all)}")
    print(f"[DATA] Test videos: {len(common_vids_test)}")
    
    # Load training queries
    train_queries = []
    train_query_video_ids = []
    for item in train_items:
        vid = item["video_id"]
        if vid not in common_vids_all:
            continue
        for q in item["qa_questions"]:
            train_queries.append(q)
            train_query_video_ids.append(vid)
    
    print(f"[DATA] Training queries: {len(train_queries)}")
    
    # Build matrices
    print("[EMB] Building matrices...")
    video_matrix = torch.stack([F.normalize(video_db[vid].float(), dim=0) for vid in common_vids_all])
    text_matrix_full = torch.stack([F.normalize(text_db_train[vid].float(), dim=0) for vid in common_vids_all])
    text_matrix_test = torch.stack([F.normalize(text_db_test[vid].float(), dim=0) for vid in common_vids_test])
    
    # Encode queries
    print("[CLIP] Encoding queries...")
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
    
    train_query_clip = encode_clip_queries(train_queries)
    
    # Setup training
    video_id_to_idx = {vid: i for i, vid in enumerate(common_vids_all)}
    gating_net = GatingNetworkNoAudio().to(DEVICE)
    optimizer = torch.optim.Adam(gating_net.parameters(), lr=LEARNING_RATE)
    
    num_train = len(train_queries)
    num_candidates = len(common_vids_all)
    
    print(f"[TRAIN] Training for {NUM_EPOCHS} epochs ({num_candidates} candidates)...")
    
    for epoch in range(NUM_EPOCHS):
        gating_net.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        num_batches = 0
        indices = torch.randperm(num_train)
        
        for q_start in range(0, num_train, QUERY_BATCH_SIZE):
            q_end = min(q_start + QUERY_BATCH_SIZE, num_train)
            batch_len = q_end - q_start
            if batch_len < QUERY_BATCH_SIZE // 2:
                continue
            
            batch_indices = indices[q_start:q_end]
            query_batch = train_query_clip[batch_indices].float().to(DEVICE)
            
            # Compute similarities
            all_sim_v, all_sim_t = [], []
            for v_start in range(0, num_candidates, VIDEO_BATCH_SIZE):
                v_end = min(v_start + VIDEO_BATCH_SIZE, num_candidates)
                vb_v = video_matrix[v_start:v_end].float().to(DEVICE)
                vb_t = text_matrix_full[v_start:v_end].float().to(DEVICE)
                all_sim_v.append(query_batch @ vb_v.T)
                all_sim_t.append(query_batch @ vb_t.T)
            
            all_sim_v = torch.cat(all_sim_v, dim=1)
            all_sim_t = torch.cat(all_sim_t, dim=1)
            
            # Normalize similarities
            all_sim_v = F.normalize(all_sim_v, dim=1)
            all_sim_t = F.normalize(all_sim_t, dim=1)
            
            # Get weights and compute gated similarities
            weights = gating_net(query_batch)
            sim_gated = weights[:, 0:1] * all_sim_v + weights[:, 1:2] * all_sim_t
            
            # Compute ranking loss
            correct_indices = torch.tensor([video_id_to_idx[train_query_video_ids[batch_indices[i_idx]]] 
                                            for i_idx in range(batch_len)], device=DEVICE)
            correct_scores = sim_gated[torch.arange(batch_len), correct_indices]
            
            neg_scores = sim_gated.clone()
            neg_scores[torch.arange(batch_len), correct_indices] = -float('inf')
            topk_neg_scores, _ = torch.topk(neg_scores, min(NUM_NEGATIVES, num_candidates - 1), dim=1)
            
            target = torch.ones_like(topk_neg_scores)
            loss = F.margin_ranking_loss(
                correct_scores.unsqueeze(1).expand(-1, NUM_NEGATIVES),
                topk_neg_scores,
                target,
                RANKING_MARGIN,
                reduction='mean'
            )
            
            loss = loss * 3.0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss
            num_batches += 1
        
        avg_loss = (total_loss / max(num_batches, 1)).item()
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  Loss: {avg_loss:.4f}", flush=True)
        
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_weights = gating_net(train_query_clip[:100].float().to(DEVICE)).cpu()
                print(f"    Debug weights - w_v: {sample_weights[:, 0].mean():.3f}, w_t: {sample_weights[:, 1].mean():.3f}", flush=True)
    
    # Save model
    save_path = "models/aems_gating_weights_no_audio.pth"
    print(f"[SAVE] Saving to {save_path}")
    torch.save(gating_net.state_dict(), save_path)
    
    # Evaluate
    print("[EVAL] Evaluating...")
    test_queries = []
    test_query_video_ids = []
    for item in test_items:
        vid = item["video_id"]
        if vid not in common_vids_test:
            continue
        for q in item["qa_questions"]:
            test_queries.append(q)
            test_query_video_ids.append(vid)
    
    test_query_clip = encode_clip_queries(test_queries)
    
    # Compute test similarities
    sim_v = test_query_clip.float() @ video_matrix_test.T
    sim_t = test_query_clip.float() @ text_matrix_test.T
    
    sim_v = F.normalize(sim_v, dim=1)
    sim_t = F.normalize(sim_t, dim=1)
    
    # Evaluate different methods
    systems = {
        "Visual only": sim_v,
        "Text only": sim_t,
        "Equal fusion": (sim_v + sim_t) / 2,
    }
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (NO AUDIO)")
    print("=" * 70)
    
    for name, sim in systems.items():
        metrics = evaluate_retrieval(sim, test_query_video_ids, common_vids_test, ks=[1, 5, 10])
        print(f"  {name:>20}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
    
    # Test adaptive gating
    gating_net.eval()
    sim_gated_list = []
    for i in range(0, len(test_queries), QUERY_BATCH_SIZE):
        q = test_query_clip[i:i+QUERY_BATCH_SIZE].float().to(DEVICE)
        with torch.no_grad():
            w = gating_net(q)
        gated = w[:, 0:1] * sim_v[i:i+QUERY_BATCH_SIZE] + w[:, 1:2] * sim_t[i:i+QUERY_BATCH_SIZE]
        sim_gated_list.append(gated.cpu())
    
    sim_gated = torch.cat(sim_gated_list, dim=0)
    metrics = evaluate_retrieval(sim_gated, test_query_video_ids, common_vids_test, ks=[1, 5, 10])
    print(f"  {'Adaptive gating (no audio)':>20}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
    
    # Show final weights
    with torch.no_grad():
        sample_weights = gating_net(test_query_clip[:100].float().to(DEVICE)).cpu()
        print(f"\n  Final gate weights (mean over 100 queries):")
        print(f"    w_v: {sample_weights[:, 0].mean():.4f} +/- {sample_weights[:, 0].std():.4f}")
        print(f"    w_t: {sample_weights[:, 1].mean():.4f} +/- {sample_weights[:, 1].std():.4f}")

if __name__ == "__main__":
    train_gating_no_audio()