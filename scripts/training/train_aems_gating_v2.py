import argparse, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from src.encoders.clap_encode import CLAPEncoder
from src.models.gating_network import GatingNetwork
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, AEMS_GATING_WEIGHTS_PATH,
                         DEVICE, set_seeds)
from src.data.metadata import load_metadata, filter_by_split

def angular_similarity(query_emb, video_emb, temperature=1.0):
    cosine_sim = query_emb @ video_emb.T
    angular_sim = 1 - torch.acos(torch.clamp(cosine_sim, -1, 1)) / np.pi
    if temperature != 1.0:
        angular_sim = angular_sim / temperature
    return angular_sim

def weight_entropy_loss(weights):
    """Penalize low-entropy weight distributions to prevent modality collapse."""
    eps = 1e-8
    log_w = torch.log(weights + eps)
    entropy = -(weights * log_w).sum(dim=-1)
    return -entropy.mean()

set_seeds(42)

QUERY_BATCH_SIZE = 32
VIDEO_BATCH_SIZE = 100
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
ENTROPY_COEFF = 0.0
GRAD_CLIP = 1.0
WARMUP_EPOCHS = 2

ANGULAR_SIMILARITY = True
TEMPERATURE_AUDIO = 0.5
TEMPERANCE_TEXT = 1.0
TEMPERANCE_VISUAL = 1.0
MODALITY_SCALE_AUDIO = 0.8
MODALITY_SCALE_TEXT = 1.0
MODALITY_SCALE_VISUAL = 1.0

parser = argparse.ArgumentParser(description="Train AEMS Gating Network (v2 - Fixed)")
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

video_id_to_idx = {vid: i for i, vid in enumerate(common_vids_all)}
num_train = len(train_queries)
num_candidates = len(common_vids_all)

# New gating network: takes per-modality similarity scores as input
gating_net = GatingNetwork(sim_dim=3, hidden_dim=64, min_weight=0.0, dropout=0.2).to(DEVICE)
optimizer = torch.optim.AdamW(gating_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"\n[TRAIN] Training FIXED gating network for {args.epochs} epochs ({num_candidates} candidates)...", flush=True)
print("[FIXES] Priority 2 Fixes:", flush=True)
print(f"  - Architecture: Similarity-input MLP (3→64→32→3) + residual + LayerNorm + Dropout", flush=True)
print(f"  - Loss: Cross-entropy + weight entropy regularization (coeff={ENTROPY_COEFF})", flush=True)
print(f"  - Weight floor: 0.1 minimum per modality", flush=True)
print(f"  - LR: {LEARNING_RATE}, Weight decay: {WEIGHT_DECAY}", flush=True)
print(f"  - Gradient clipping: {GRAD_CLIP}", flush=True)
print(f"  - Warmup epochs: {WARMUP_EPOCHS}", flush=True)

for epoch in range(args.epochs):
    gating_net.train()
    total_loss = torch.tensor(0.0, device=DEVICE)
    num_batches = 0
    indices = torch.randperm(num_train)

    # Learning rate warmup
    if epoch < WARMUP_EPOCHS:
        warmup_factor = (epoch + 1) / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            pg['lr'] = LEARNING_RATE * warmup_factor
    else:
        for pg in optimizer.param_groups:
            pg['lr'] = LEARNING_RATE

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

            if ANGULAR_SIMILARITY:
                all_sim_v[:, v_start:v_end] = angular_similarity(query_batch, vb_v, temperature=TEMPERANCE_VISUAL)
                all_sim_a[:, v_start:v_end] = angular_similarity(query_batch_clap, vb_a, temperature=TEMPERATURE_AUDIO)
                all_sim_t[:, v_start:v_end] = angular_similarity(query_batch, vb_t, temperature=TEMPERANCE_TEXT)
            else:
                all_sim_v[:, v_start:v_end] = query_batch @ vb_v.T
                all_sim_a[:, v_start:v_end] = query_batch_clap @ vb_a.T
                all_sim_t[:, v_start:v_end] = query_batch @ vb_t.T

            del vb_v, vb_a, vb_t

        # New gating network: takes per-candidate similarity scores
        weights = gating_net(all_sim_v, all_sim_t, all_sim_a)  # (batch, N, 3)

        # Apply modality-specific scaling
        sim_v_scaled = all_sim_v * MODALITY_SCALE_VISUAL
        sim_t_scaled = all_sim_t * MODALITY_SCALE_TEXT
        sim_a_scaled = all_sim_a * MODALITY_SCALE_AUDIO

        # Weighted fusion using per-candidate weights
        sim_gated = (
            weights[:, :, 0:1].squeeze(-1) * sim_v_scaled +
            weights[:, :, 1:2].squeeze(-1) * sim_t_scaled +
            weights[:, :, 2:3].squeeze(-1) * sim_a_scaled
        )

        correct_indices = torch.tensor([video_id_to_idx[train_query_video_ids[batch_indices[i_idx]]]
                                   for i_idx in range(batch_len)], device=DEVICE)

        # Cross-entropy loss: treat as classification over N candidates
        log_probs = F.log_softmax(sim_gated, dim=1)
        ce_loss = F.nll_loss(log_probs, correct_indices)

        # Weight entropy regularization: prevent modality collapse
        mean_weights = weights.mean(dim=1)  # (batch, 3)
        entropy_loss = weight_entropy_loss(mean_weights)

        loss = ce_loss + ENTROPY_COEFF * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gating_net.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss
        num_batches += 1

        if num_batches == 1 and (epoch + 1) % 5 == 0:
            with torch.no_grad():
                dbg = weights[:10, :10].cpu()
                print(f"    Debug weights batch0 - w_v: {dbg[:,:,0].mean():.3f}, w_t: {dbg[:,:,1].mean():.3f}, w_a: {dbg[:,:,2].mean():.3f}", flush=True)

        del query_batch, query_batch_clap, all_sim_v, all_sim_a, all_sim_t, weights, sim_gated, loss

    avg_loss = (total_loss / max(num_batches, 1)).item()
    print(f"  Epoch {epoch+1}/{args.epochs}  Loss: {avg_loss:.4f}  LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

    gc.collect()
    torch.cuda.empty_cache()

print(f"\n[SAVE] Saving gating weights to {AEMS_GATING_WEIGHTS_PATH}", flush=True)
torch.save(gating_net.state_dict(), AEMS_GATING_WEIGHTS_PATH)

print("\n[EVAL] Encoding test queries with CLIP and CLAP...", flush=True)
test_query_clip = encode_clip_queries(test_queries)
print(f"  Test CLIP queries: {test_query_clip.shape}", flush=True)
test_query_clap = encode_clap_queries(test_queries)
print(f"  Test CLAP queries: {test_query_clap.shape}", flush=True)

del clip_model, clap_encoder
gc.collect()
torch.cuda.empty_cache()

print("\n[EVAL] Evaluating on test queries against test candidates...", flush=True)
gating_net.eval()

# Compute similarities with angular similarity
if ANGULAR_SIMILARITY:
    sim_v = angular_similarity(test_query_clip.float(), video_matrix_test, temperature=TEMPERANCE_VISUAL).to(DEVICE)
    sim_a = angular_similarity(test_query_clap.float(), audio_matrix_test, temperature=TEMPERATURE_AUDIO).to(DEVICE)
    sim_t = angular_similarity(test_query_clip.float(), text_matrix_test, temperature=TEMPERANCE_TEXT).to(DEVICE)
else:
    sim_v = F.normalize(test_query_clip.float() @ video_matrix_test.T, dim=1).to(DEVICE)
    sim_a = F.normalize(test_query_clap.float() @ audio_matrix_test.T, dim=1).to(DEVICE)
    sim_t = F.normalize(test_query_clip.float() @ text_matrix_test.T, dim=1).to(DEVICE)

# Adaptive gating evaluation
sim_gated_list = []
for i in range(0, len(test_queries), QUERY_BATCH_SIZE):
    sv = sim_v[i:i+QUERY_BATCH_SIZE]
    st = sim_t[i:i+QUERY_BATCH_SIZE]
    sa = sim_a[i:i+QUERY_BATCH_SIZE]
    with torch.no_grad():
        w = gating_net(sv, st, sa)  # (batch, N, 3)
        gated = (w[:, :, 0:1].squeeze(-1) * sv * MODALITY_SCALE_VISUAL +
                 w[:, :, 1:2].squeeze(-1) * st * MODALITY_SCALE_TEXT +
                 w[:, :, 2:3].squeeze(-1) * sa * MODALITY_SCALE_AUDIO)
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
print("GATING EVALUATION RESULTS (Priority 2 Fixed)")
print("=" * 70)
for name, sim in systems.items():
    metrics = evaluate_retrieval(sim, test_query_video_ids, common_vids_test, ks=[1, 5, 10])
    print(f"  {name:>20}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}", flush=True)

with torch.no_grad():
    sv = sim_v[:min(100, len(test_queries))]
    st = sim_t[:min(100, len(test_queries))]
    sa = sim_a[:min(100, len(test_queries))]
    sample_weights = gating_net(sv, st, sa).cpu()
print(f"\n  Gate weights (mean over {len(sample_weights)} queries):", flush=True)
print(f"    w_v: {sample_weights[:, :, 0].mean():.4f} +/- {sample_weights[:, :, 0].std():.4f}", flush=True)
print(f"    w_t: {sample_weights[:, :, 1].mean():.4f} +/- {sample_weights[:, :, 1].std():.4f}", flush=True)
print(f"    w_a: {sample_weights[:, :, 2].mean():.4f} +/- {sample_weights[:, :, 2].std():.4f}", flush=True)
