#!/usr/bin/env python3
"""
Evaluation script for Priority 1 optimizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import clip
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.encoders.clap_encode import CLAPEncoder
from src.models.gating_network import GatingNetwork
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, AEMS_GATING_WEIGHTS_PATH,
                         DEVICE, set_seeds)
from src.data.metadata import load_metadata, filter_by_split

# Priority 1 Optimizations
ANGULAR_SIMILARITY = True  # Use angular similarity instead of cosine
TEMPERATURE_AUDIO = 0.5  # Temperature scaling for audio similarities
TEMPERANCE_TEXT = 1.0  # Temperature scaling for text similarities
TEMPERANCE_VISUAL = 1.0  # Temperature scaling for visual similarities
MODALITY_SCALE_AUDIO = 0.8  # Scale factor for audio normalization
MODALITY_SCALE_TEXT = 1.0  # Scale factor for text normalization
MODALITY_SCALE_VISUAL = 1.0  # Scale factor for visual normalization

def angular_similarity(query_emb, video_emb, temperature=1.0):
    """Compute angular similarity between query and video embeddings"""
    cosine_sim = query_emb @ video_emb.T
    angular_sim = 1 - torch.acos(torch.clamp(cosine_sim, -1, 1)) / np.pi
    if temperature != 1.0:
        angular_sim = angular_sim / temperature
    return angular_sim

def main():
    print("[EVAL] Starting Priority 1 evaluation...")
    
    set_seeds(42)
    
    # Load metadata
    print("[DATA] Loading manifest...")
    metadata = load_metadata(AEMS_MANIFEST_PATH)
    train_queries = filter_by_split(metadata, 'train')
    test_queries = filter_by_split(metadata, 'test')
    
    print(f"[DATA] Test queries: {len(test_queries)}")
    print(f"[DATA] Training queries: {len(train_queries)}")
    
    # Load embeddings first
    print("[EMB] Loading embeddings...")
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH)
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH)
    text_db_train = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"))
    text_db_test = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
    
    # Get common video IDs
    all_video_ids = set()
    for item in train_queries + test_queries:
        all_video_ids.add(item["video_id"])
    
    common_vids_all = sorted(
        set(video_db.keys()) & set(audio_db.keys()) & set(text_db_train.keys()) & all_video_ids
    )
    common_vids_test = sorted(
        set(video_db.keys()) & set(audio_db.keys()) & set(text_db_test.keys()) &
        set(item["video_id"] for item in test_queries)
    )
    
    print(f"[DATA] All videos with complete embeddings: {len(common_vids_all)}")
    print(f"[DATA] Test candidate videos: {len(common_vids_test)}")
    
    # Build matrices from databases
    print("[EMB] Building full video matrices on CPU...")
    video_matrix = torch.stack([F.normalize(video_db[vid].float(), dim=0) for vid in common_vids_all])
    audio_matrix = torch.stack([F.normalize(audio_db[vid].float(), dim=0) for vid in common_vids_all])
    text_matrix_full = torch.stack([F.normalize(text_db_train[vid].float(), dim=0) for vid in common_vids_all])
    
    video_matrix_test = torch.stack([F.normalize(video_db[vid].float(), dim=0) for vid in common_vids_test])
    audio_matrix_test = torch.stack([F.normalize(audio_db[vid].float(), dim=0) for vid in common_vids_test])
    text_matrix_test = torch.stack([F.normalize(text_db_test[vid].float(), dim=0) for vid in common_vids_test])
    
    print(f"[EMB] Video matrix: {video_matrix.shape}")
    print(f"[EMB] Audio matrix: {audio_matrix.shape}")
    print(f"[EMB] Text matrix: {text_matrix_full.shape}")
    
    # Load trained gating network
    print("[LOAD] Loading trained gating network...")
    gating_net = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    checkpoint = torch.load(AEMS_GATING_WEIGHTS_PATH)
    gating_net.load_state_dict(checkpoint)
    gating_net.eval()
    
    # Encode test queries
    print("[CLIP] Encoding test queries...")
    clip_encoder = CLAPEncoder(device=DEVICE)
    test_query_clip = clip_encoder.encode_text([q['qa_questions'][0] for q in test_queries])
    
    print("[CLAP] Encoding test queries...")
    clap_encoder = CLAPEncoder(device=DEVICE)
    test_query_clap = clap_encoder.encode_text([q['qa_questions'][0] for q in test_queries])
    
    # Convert numpy arrays to torch tensors if needed
    if isinstance(test_query_clip, np.ndarray):
        test_query_clip = torch.from_numpy(test_query_clip).float()
    if isinstance(test_query_clap, np.ndarray):
        test_query_clap = torch.from_numpy(test_query_clap).float()
    
    print(f"[EMB] Test query CLIP: {test_query_clip.shape}")
    print(f"[EMB] Test query CLAP: {test_query_clap.shape}")
    
    # Compute similarities with angular similarity
    print("[SIM] Computing similarities with angular similarity...")
    if ANGULAR_SIMILARITY:
        sim_v = angular_similarity(test_query_clip.float(), video_matrix_test, temperature=TEMPERANCE_VISUAL).to(DEVICE)
        sim_a = angular_similarity(test_query_clap.float(), audio_matrix_test, temperature=TEMPERATURE_AUDIO).to(DEVICE)
        sim_t = angular_similarity(test_query_clip.float(), text_matrix_test, temperature=TEMPERANCE_TEXT).to(DEVICE)
    else:
        sim_v = F.normalize(test_query_clip.float() @ video_matrix_test.T, dim=1).to(DEVICE)
        sim_a = F.normalize(test_query_clap.float() @ audio_matrix_test.T, dim=1).to(DEVICE)
        sim_t = F.normalize(test_query_clip.float() @ text_matrix_test.T, dim=1).to(DEVICE)
    
    print(f"[SIM] Visual similarity: {sim_v.shape}")
    print(f"[SIM] Audio similarity: {sim_a.shape}")
    print(f"[SIM] Text similarity: {sim_t.shape}")
    
    # Compute adaptive gating
    print("[GATING] Computing adaptive gating...")
    sim_gated_list = []
    for i in range(0, len(test_queries), 32):
        q = test_query_clip[i:i+32].float().to(DEVICE)
        with torch.no_grad():
            w = gating_net(q)
            sim_v_batch = sim_v[i:i+32]
            sim_t_batch = sim_t[i:i+32]
            sim_a_batch = sim_a[i:i+32]
            
            # Apply modality-specific scaling
            sim_v_scaled = sim_v_batch * MODALITY_SCALE_VISUAL
            sim_t_scaled = sim_t_batch * MODALITY_SCALE_TEXT
            sim_a_scaled = sim_a_batch * MODALITY_SCALE_AUDIO
            
            gated = (w[:, 0:1] * sim_v_scaled +
                     w[:, 1:2] * sim_t_scaled +
                     w[:, 2:3] * sim_a_scaled)
        sim_gated_list.append(gated.cpu())
    
    sim_gated = torch.cat(sim_gated_list, dim=0)
    print(f"[GATING] Gated similarity: {sim_gated.shape}")
    
    # Create systems for evaluation
    systems = {
        "Visual only": sim_v,
        "Text only": sim_t,
        "Audio only": sim_a,
        "Equal fusion": (sim_v + sim_t + sim_a) / 3,
        "Adaptive gating": sim_gated,
    }
    
    # Build video ID lists for evaluation
    test_video_ids_list = [test_queries[i]['video_id'] for i in range(len(test_queries))]
    candidate_video_ids = common_vids_test
    
    # Evaluate all systems
    print("[EVAL] Evaluating all systems...")
    results = {}
    for system_name, system_sim in systems.items():
        print(f"[EVAL] Evaluating {system_name}...")
        # Compute R@1, R@5, R@10 directly
        r_at = {1: 0.0, 5: 0.0, 10: 0.0}
        for i in range(system_sim.size(0)):
            gt_video = test_video_ids_list[i]
            ranked_indices = torch.argsort(system_sim[i], descending=True)
            for k in [1, 5, 10]:
                topk_vids = [candidate_video_ids[j] for j in ranked_indices[:k].tolist()]
                if gt_video in topk_vids:
                    r_at[k] += 1.0
        num_q = system_sim.size(0)
        for k in r_at:
            r_at[k] /= num_q
        results[system_name] = {'R@1': r_at[1], 'R@5': r_at[5], 'R@10': r_at[10]}
        print(f"[EVAL] {system_name}: R@1={r_at[1]:.4f}, R@5={r_at[5]:.4f}, R@10={r_at[10]:.4f}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PRIORITY 1 OPTIMIZATION EVALUATION RESULTS")
    print("=" * 70)
    print(f"Angular Similarity: {ANGULAR_SIMILARITY}")
    print(f"Temperature (Audio): {TEMPERATURE_AUDIO}")
    print(f"Temperature (Text): {TEMPERANCE_TEXT}")
    print(f"Temperature (Visual): {TEMPERANCE_VISUAL}")
    print(f"Modality Scale (Audio): {MODALITY_SCALE_AUDIO}")
    print(f"Modality Scale (Text): {MODALITY_SCALE_TEXT}")
    print(f"Modality Scale (Visual): {MODALITY_SCALE_VISUAL}")
    print("-" * 70)
    
    # Sort results by R@1
    sorted_results = sorted(results.items(), key=lambda x: x[1]['R@1'], reverse=True)
    for system_name, result in sorted_results:
        print(f"{system_name:15} | R@1: {result['R@1']:7.4f} | R@5: {result['R@5']:7.4f} | R@10: {result['R@10']:7.4f}")
    
    # Save results
    import json
    with open('priority1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVE] Results saved to priority1_results.json")
    
    # Print improvement analysis
    baseline_r1 = results['Visual only']['R@1']
    optimized_r1 = results['Adaptive gating']['R@1']
    improvement = (optimized_r1 - baseline_r1) / baseline_r1 * 100
    
    print(f"\n[IMPROVEMENT] Analysis:")
    print(f"Baseline (Visual only): {baseline_r1:.4f}")
    print(f"Optimized (Adaptive gating): {optimized_r1:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    
    if improvement > 0:
        print("✅ SUCCESS: Priority 1 optimizations improved performance!")
    else:
        print("❌ ISSUE: Priority 1 optimizations did not improve performance.")

if __name__ == "__main__":
    main()