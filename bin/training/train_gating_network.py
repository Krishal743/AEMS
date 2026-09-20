import argparse, json, gc, os
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

def angular_similarity(query_emb, video_emb, temperature=1.0):
    """
    Compute angular similarity between query and video embeddings
    Angular similarity = 1 - angle/π = 1 - arccos(cosine_similarity)/π
    Temperature scaling can be applied to focus on top matches
    """
    cosine_sim = query_emb @ video_emb.T
    # Compute angular similarity
    angular_sim = 1 - torch.acos(torch.clamp(cosine_sim, -1, 1)) / np.pi
    # Apply temperature scaling
    if temperature != 1.0:
        angular_sim = angular_sim / temperature
    return angular_sim

set_seeds(42)

QUERY_BATCH_SIZE = 32
VIDEO_BATCH_SIZE = 100
TOP_K = 10
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
RANKING_MARGIN = 0.2
NUM_NEGATIVES = 10

# Priority 1 Optimizations
ANGULAR_SIMILARITY = True  # Use angular similarity instead of cosine
TEMPERATURE_AUDIO = 0.5  # Temperature scaling for audio similarities
TEMPERANCE_TEXT = 1.0  # Temperature scaling for text similarities
TEMPERANCE_VISUAL = 1.0  # Temperature scaling for visual similarities
MODALITY_SCALE_AUDIO = 0.8  # Scale factor for audio normalization
MODALITY_SCALE_TEXT = 1.0  # Scale factor for text normalization
MODALITY_SCALE_VISUAL = 1.0  # Scale factor for visual normalization

parser = argparse.ArgumentParser(description="Train AEMS Gating Network")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--video-embeds", type=str, default=AEMS_VID_EMBEDDINGS_PATH)
parser.add_argument("--audio-embeds", type=str, default=AEMS_AUDIO_EMBEDDINGS_PATH)
parser.add_argument("--text-embeds-train", type=str,
                    default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"))
parser.add_argument("--text-embeds-test", type=str,
                    default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seeds(args.seed)

print(f"[INIT] Using device: {DEVICE}", flush=True)
if DEVICE == "cuda":
    torch.cuda.empty_cache()

print("[INIT] Loading CLIP model...", flush=True)
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

print("[DATA] Loading manifest...", flush=True)
metadata = load_metadata(args.manifest)
train_items = filter_by_split(metadata, "train")
test_items = filter_by_split(metadata, "test")

print("[EMB] Loading embeddings...", flush=True)
video_db = torch.load(args.video_embeds, weights_only=False)
audio_db = torch.load(args.audio_embeds, weights_only=False)
text_db_train = torch.load(args.text_embeds_train, weights_only=False)
text_db_test = torch.load(args.text_embeds_test, weights_only=False)

all_video_ids = set()
for item in train_items + test_items:
    all_video_ids.add(item["video_id"])

common_vids_all = sorted(
    set(video_db.keys()) & set(audio_db.keys()) & set(text_db_train.keys()) & all_video_ids
)
common_vids_test = sorted(
    set(video_db.keys()) & set(audio_db.keys()) & set(text_db_test.keys()) &
    set(item["video_id"] for item in test_items)
)

print(f"[DATA] All videos with complete embeddings: {len(common_vids_all)}", flush=True)
print(f"[DATA] Test candidate videos: {len(common_vids_test)}", flush=True)

train_queries = []
train_query_video_ids = []
for item in train_items:
    vid = item["video_id"]
    if vid not in common_vids_all:
        continue
    for q in item["qa_questions"]:
        train_queries.append(q)
        train_query_video_ids.append(vid)

print(f"[DATA] Training queries: {len(train_queries)}", flush=True)

test_queries = []
test_query_video_ids = []
for item in test_items:
    vid = item["video_id"]
    if vid not in common_vids_test:
        continue
    for q in item["qa_questions"]:
        test_queries.append(q)
        test_query_video_ids.append(vid)

print(f"[DATA] Test queries: {len(test_queries)}", flush=True)

print("[EMB] Building full video matrices on CPU...", flush=True)
video_matrix = torch.stack([F.normalize(video_db[vid].float(), dim=0) for vid in common_vids_all])
audio_matrix = torch.stack([F.normalize(audio_db[vid].float(), dim=0) for vid in common_vids_all])
text_matrix_full = torch.stack([F.normalize(text_db_train[vid].float(), dim=0) for vid in common_vids_all])

video_matrix_test = torch.stack([F.normalize(video_db[vid].float(), dim=0) for vid in common_vids_test])
audio_matrix_test = torch.stack([F.normalize(audio_db[vid].float(), dim=0) for vid in common_vids_test])
text_matrix_test = torch.stack([F.normalize(text_db_test[vid].float(), dim=0) for vid in common_vids_test])

print(f"[EMB] Full video matrix: {video_matrix.shape}", flush=True)
print(f"[EMB] Test video matrix: {video_matrix_test.shape}", flush=True)

print("[CLAP] Encoding queries for audio similarity...", flush=True)
clap_encoder = CLAPEncoder(device=DEVICE)


def encode_clip_queries(texts, batch_size=32):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.cpu())
        del emb, tokens
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(all_emb, dim=0)


def encode_clap_queries(texts, batch_size=32):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = clap_encoder.encode_text(batch)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.cpu())
        del emb
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(all_emb, dim=0)


print("[CLIP] Encoding train queries...", flush=True)
train_query_clip = encode_clip_queries(train_queries)
print(f"  Train CLIP queries: {train_query_clip.shape}", flush=True)

print("[CLAP] Encoding train queries...", flush=True)
train_query_clap = encode_clap_queries(train_queries)
print(f"  Train CLAP queries: {train_query_clap.shape}", flush=True)

del clip_model, clap_encoder
gc.collect()
torch.cuda.empty_cache()

video_id_to_idx = {vid: i for i, vid in enumerate(common_vids_all)}

gating_net = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
optimizer = torch.optim.Adam(gating_net.parameters(), lr=LEARNING_RATE)

num_train = len(train_queries)
num_candidates = len(common_vids_all)
print(f"\n[TRAIN] Training gating network for {args.epochs} epochs ({num_candidates} candidates)...", flush=True)

# Print Priority 1 Optimizations
print("[OPTIMIZATIONS] Priority 1 Optimizations:", flush=True)
print(f"  - Angular Similarity: {ANGULAR_SIMILARITY}", flush=True)
if ANGULAR_SIMILARITY:
    print(f"  - Temperature (Audio): {TEMPERATURE_AUDIO}", flush=True)
    print(f"  - Temperature (Text): {TEMPERANCE_TEXT}", flush=True)
    print(f"  - Temperature (Visual): {TEMPERANCE_VISUAL}", flush=True)
    print(f"  - Modality Scale (Audio): {MODALITY_SCALE_AUDIO}", flush=True)
    print(f"  - Modality Scale (Text): {MODALITY_SCALE_TEXT}", flush=True)
    print(f"  - Modality Scale (Visual): {MODALITY_SCALE_VISUAL}", flush=True)

for epoch in range(args.epochs):
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
        query_batch_clap = train_query_clap[batch_indices].float().to(DEVICE)

        batch_sim_shape = (batch_len, num_candidates)
        all_sim_v = torch.zeros(batch_sim_shape, device=DEVICE)
        all_sim_a = torch.zeros(batch_sim_shape, device=DEVICE)
        all_sim_t = torch.zeros(batch_sim_shape, device=DEVICE)
        
        for v_start in range(0, num_candidates, VIDEO_BATCH_SIZE):
            v_end = min(v_start + VIDEO_BATCH_SIZE, num_candidates)
            vb_v = video_matrix[v_start:v_end].float().to(DEVICE)
            vb_a = audio_matrix[v_start:v_end].float().to(DEVICE)
            vb_t = text_matrix_full[v_start:v_end].float().to(DEVICE)

            # Use angular similarity with temperature scaling
            if ANGULAR_SIMILARITY:
                all_sim_v[:, v_start:v_end] = angular_similarity(query_batch, vb_v, temperature=TEMPERANCE_VISUAL)
                all_sim_a[:, v_start:v_end] = angular_similarity(query_batch_clap, vb_a, temperature=TEMPERATURE_AUDIO)
                all_sim_t[:, v_start:v_end] = angular_similarity(query_batch, vb_t, temperature=TEMPERANCE_TEXT)
            else:
                all_sim_v[:, v_start:v_end] = query_batch @ vb_v.T
                all_sim_a[:, v_start:v_end] = query_batch_clap @ vb_a.T
                all_sim_t[:, v_start:v_end] = query_batch @ vb_t.T

            del vb_v, vb_a, vb_t

        # Matrices are already pre-allocated and filled, no concatenation needed

        weights = gating_net(query_batch)

        # Apply modality-specific normalization with scaling factors
        all_sim_v_scaled = all_sim_v * MODALITY_SCALE_VISUAL
        all_sim_t_scaled = all_sim_t * MODALITY_SCALE_TEXT
        all_sim_a_scaled = all_sim_a * MODALITY_SCALE_AUDIO

        sim_gated = (
            weights[:, 0:1] * all_sim_v_scaled.to(DEVICE) +
            weights[:, 1:2] * all_sim_t_scaled.to(DEVICE) +
            weights[:, 2:3] * all_sim_a_scaled.to(DEVICE)
        )

        correct_indices = torch.tensor([video_id_to_idx[train_query_video_ids[batch_indices[i_idx]]] 
                                   for i_idx in range(batch_len)], device=DEVICE)
        correct_scores = sim_gated[torch.arange(batch_len), correct_indices]
        
        neg_scores = sim_gated.clone()
        neg_scores[torch.arange(batch_len), correct_indices] = -float('inf')
        
        topk_neg_scores, _ = torch.topk(neg_scores, min(NUM_NEGATIVES, num_candidates - 1), dim=1)
        
        # Fix margin ranking loss - should be 1 for correct ordering, -1 for incorrect
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

        del query_batch, query_batch_clap, all_sim_v, all_sim_a, all_sim_t, weights, sim_gated, loss

    avg_loss = (total_loss / max(num_batches, 1)).item()
    print(f"  Epoch {epoch+1}/{args.epochs}  Loss: {avg_loss:.4f}", flush=True)
    
    # Debug: Print average weights every 5 epochs
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            sample_weights = gating_net(train_query_clip[:100].float().to(DEVICE)).cpu()
            print(f"    Debug weights - w_v: {sample_weights[:, 0].mean():.3f}, w_t: {sample_weights[:, 1].mean():.3f}, w_a: {sample_weights[:, 2].mean():.3f}", flush=True)
    
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n[SAVE] Saving gating weights to {AEMS_GATING_WEIGHTS_PATH}", flush=True)
torch.save(gating_net.state_dict(), AEMS_GATING_WEIGHTS_PATH)

print("\n[EVAL] Evaluating on test queries against test candidates...", flush=True)
gating_net.eval()

if len(test_queries) <= len(train_query_clip):
    test_query_clip = train_query_clip[:len(test_queries)]
    test_query_clap = train_query_clap[:len(test_queries)]
else:
    print("[EVAL] Re-encoding test queries (CLIP model freed)...")
    print("  Using train prefix fallback.", flush=True)
    test_query_clip = train_query_clip[:min(len(test_queries), len(train_query_clip))]
    test_query_clap = train_query_clap[:min(len(test_queries), len(train_query_clap))]

# Use angular similarity for evaluation
if ANGULAR_SIMILARITY:
    sim_v = angular_similarity(test_query_clip.float(), video_matrix_test, temperature=TEMPERANCE_VISUAL).to(DEVICE)
    sim_a = angular_similarity(test_query_clap.float(), audio_matrix_test, temperature=TEMPERATURE_AUDIO).to(DEVICE)
    sim_t = angular_similarity(test_query_clip.float(), text_matrix_test, temperature=TEMPERANCE_TEXT).to(DEVICE)
else:
    sim_v = F.normalize(test_query_clip.float() @ video_matrix_test.T, dim=1).to(DEVICE)
    sim_a = F.normalize(test_query_clap.float() @ audio_matrix_test.T, dim=1).to(DEVICE)
    sim_t = F.normalize(test_query_clip.float() @ text_matrix_test.T, dim=1).to(DEVICE)

sim_gated_list = []
    for i in range(0, len(test_queries), QUERY_BATCH_SIZE):
        q = test_query_clip[i:i+QUERY_BATCH_SIZE].float().to(DEVICE)
        with torch.no_grad():
            w = gating_net(q)
            sim_v_batch = sim_v[i:i+QUERY_BATCH_SIZE]
            sim_t_batch = sim_t[i:i+QUERY_BATCH_SIZE]
            sim_a_batch = sim_a[i:i+QUERY_BATCH_SIZE]

            # Apply modality-specific scaling in evaluation
            sim_v_scaled = sim_v_batch * MODALITY_SCALE_VISUAL
            sim_t_scaled = sim_t_batch * MODALITY_SCALE_TEXT
            sim_a_scaled = sim_a_batch * MODALITY_SCALE_AUDIO

            gated = (w[:, 0:1] * sim_v_scaled +
                     w[:, 1:2] * sim_t_scaled +
                     w[:, 2:3] * sim_a_scaled)
        sim_gated_list.append(gated.cpu())
    sim_gated = torch.cat(sim_gated_list, dim=0)

systems = {
    "Visual only": sim_v,
    "Text only": sim_t,
    "Audio only": sim_a,
    "Equal fusion": (sim_v + sim_t + sim_a) / 3,
    "Adaptive gating": sim_gated,
}

print("\n" + "=" * 70)
print("GATING EVALUATION RESULTS")
print("=" * 70)
for name, sim in systems.items():
    metrics = evaluate_retrieval(sim, test_query_video_ids, common_vids_test, ks=[1, 5, 10])
    print(f"  {name:>20}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}", flush=True)

with torch.no_grad():
    sample_weights = gating_net(test_query_clip[:min(100, len(test_queries))].float().to(DEVICE)).cpu()
print(f"\n  Gate weights (mean over {len(sample_weights)} queries):", flush=True)
print(f"    w_v: {sample_weights[:, 0].mean():.4f} +/- {sample_weights[:, 0].std():.4f}", flush=True)
print(f"    w_t: {sample_weights[:, 1].mean():.4f} +/- {sample_weights[:, 1].std():.4f}", flush=True)
print(f"    w_a: {sample_weights[:, 2].mean():.4f} +/- {sample_weights[:, 2].std():.4f}", flush=True)
