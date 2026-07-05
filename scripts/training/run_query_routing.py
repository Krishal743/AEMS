LEXICON_LOGGING = False
from src.models.gating_network import GatingNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
import clip
import random
import numpy as np
from src.encoders.clap_encode import CLAPEncoder

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_QUERIES = 100
MAX_VIDEOS = 100
QUERY_BATCH_SIZE = 32
VIDEO_BATCH_SIZE = 100
TOP_K = 10

NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_TRAIN_QUERIES = 300
CURRICULUM_EPOCHS = 0.0  # Disable curriculum (use ranking only)
MODALITY_DROPOUT_PROB = 0.0  # Disable for clean ranking loss

# All zero — heuristic penalties removed per recovery plan
ENTROPY_PENALTY = 0.0
ALIGNMENT_REWARD = 0.0
CAPTION_PENALTY = 0.0

VISUAL_KEYWORDS = {"car", "person", "scene", "red", "blue", "building", "sky", "dog", "cat", "water", "face", "road", "tree", "mountain", "indoor", "outdoor", "city", "street", "room", "beach", "field", "forest", "desk", "chair", "table", "window", "floor", "wall", "car", "truck", "bus", "bike", "motorcycle", "boat", "airplane", "helicopter", "plane", "bird", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "fish", "snake", "lizard", "frog", "butterfly", "bee", "ant", "spider", "crab", "shark", "whale", "dolphin", "child", "adult", "man", "woman", "boy", "girl", "people", "crowd", "soldier", "police", "doctor", "nurse", "chef", "driver", "pilot", "singer", "dancer", "actor", "athlete", "student", "teacher"}

AUDIO_KEYWORDS = {"music", "song", "sound", "loud", "quiet", "explosion", "crash", "bang", "fire", "water", "rain", "thunder", "wind", "voice", "speech", "talking", "singing", "laughing", "crying", "scream", "shout", "whisper", "applause", "cheering", "music", "beat", "rhythm", "drum", "guitar", "piano", "violin", "trumpet", "horn", "bell", "chime", "horn", "alarm", "siren", "bell", "gun", "shot", "firework", "engine", "motor", "tire", "footstep", "running", "walking", "jumping", "climbing", "swimming", "diving", "flying", "driving", "riding", "landing", "taking off", "breaking", "crashing", "hitting", "kicking", "punching", "slapping", "clapping", "snapping", "clicking", "ticking", "clock", "timer", "bell"}

TEXT_KEYWORDS = {"talk", "lecture", "explain", "describe", "tell", "show", "how to", "what is", "about", "regarding", "concerning", "tutorial", "lesson", "class", "course", "teaching", "learning", "study", "reading", "writing", "speaking", "discuss", "analysis", "review", "summary", "explanation", "demonstration", "instruction", "guide", "introduction", "conclusion", "result", "finding", "discovery", "knowledge", "information", "fact", "concept", "theory", "principle", "method", "approach", "technique", "strategy", "process", "procedure", "step", "stage", "phase", "level", "degree", "extent", "amount", "number", "point", "issue", "problem", "question", "answer", "solution"}



def get_common_video_ids(video_db_v, video_db_a, caption_db, test_vids):
    return [vid for vid in test_vids if vid in video_db_v and vid in video_db_a and vid in caption_db]


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()


def encode_queries_batched_gpu(clip_model, texts, batch_size=16):
    all_embeds = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = clip.tokenize(batch_texts).to(DEVICE)
        
        embeds = clip_model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=1, keepdim=True)
        
        all_embeds.append(embeds.cpu())
        
        del embeds, tokens
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    return torch.cat(all_embeds, dim=0)


def main():
    print(f"[INIT] Using device: {DEVICE}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print("[MEM] Cleared GPU memory")
    
    print("[INIT] Loading CLIP model on GPU...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    
    print("[DATA] Loading dataset...")
    with open("data/processed/metadata/msrvtt_metadata.json") as f:
        metadata = json.load(f)
    
    test_items = [m for m in metadata if m["split"] == "test"]
    test_video_ids = set(m["video_id"] for m in test_items)
    print(f"[DATA] {len(test_video_ids)} unique test videos")
    
    print("[EMB] Loading embeddings from disk...")
    video_db_v = torch.load("embeddings/video_embeddings.pt", weights_only=False)
    video_db_a = torch.load("embeddings/audio_embeddings.pt", weights_only=False)
    caption_db = torch.load("embeddings/caption_embeddings.pt", weights_only=False)
    
    common_vids = get_common_video_ids(video_db_v, video_db_a, caption_db, test_video_ids)
    print(f"[FILTER] {len(common_vids)} videos with all 3 modalities")
    
    print(f"[EMB] Limiting to {MAX_VIDEOS} videos...")
    common_vids = common_vids[:MAX_VIDEOS]
    common_vid_set = set(common_vids)
    print(f"[FILTER] Using {len(common_vids)} videos")
    
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
    print(f"[TEXT] Using {len(unique_texts)} unique test captions...")
    
    print("[TEXT] Encoding queries on GPU...")
    text_embeds = encode_queries_batched_gpu(clip_model, unique_texts, batch_size=QUERY_BATCH_SIZE)
    print(f"[TEXT] Encodings complete: {text_embeds.shape}")

    print("[CLAP] Encoding queries for audio similarity...")
    clap_encoder = CLAPEncoder(device=DEVICE)
    clap_text_list = []
    for i in range(0, len(unique_texts), QUERY_BATCH_SIZE):
        batch = unique_texts[i:i+QUERY_BATCH_SIZE]
        emb = clap_encoder.encode_text(batch)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        emb = emb / emb.norm(dim=1, keepdim=True)
        clap_text_list.append(emb.cpu())
        del emb
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    text_embeds_clap = torch.cat(clap_text_list, dim=0)
    print(f"[CLAP] Audio query encodings: {text_embeds_clap.shape}")
    
    del clip_model, clap_encoder
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"[MEM] After CLIP: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("[EMB] Building embeddings on CPU (permanent)...")

    video_embeds_v_cpu = torch.stack([to_tensor(video_db_v[vid]) for vid in common_vids])
    video_embeds_a_cpu = torch.stack([to_tensor(video_db_a[vid]) for vid in common_vids])
    # Fix: keep all 20 captions per video
    caption_tensor_cpu = torch.stack([
        F.normalize(to_tensor(caption_db[vid]), p=2, dim=1) for vid in common_vids
    ])

    del video_db_v, video_db_a, caption_db
    gc.collect()

    video_embeds_v_cpu = F.normalize(video_embeds_v_cpu, p=2, dim=1)
    video_embeds_a_cpu = F.normalize(video_embeds_a_cpu, p=2, dim=1)
    
    print(f"[MEM] Embeddings on CPU: {video_embeds_v_cpu.element_size() * video_embeds_v_cpu.nelement() * 3 / 1e6:.2f} MB")
    
    print("[GATE] Initializing gating network...")
    gating_net = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(gating_net.parameters(), lr=LEARNING_RATE)
    
    text_embeds_cpu = text_embeds.cpu()
    text_embeds_clap_cpu = text_embeds_clap.cpu()
    num_queries = len(unique_video_ids)
    num_videos = len(common_vids)
    
    del text_embeds, text_embeds_clap
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    print("[TRAIN] Training gating network (streaming)...")
    
    gating_net.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        total_curriculum_loss = 0.0
        num_batches = 0
        
        curriculum_epochs = max(1, int(NUM_EPOCHS * CURRICULUM_EPOCHS))
        in_curriculum_phase = epoch < curriculum_epochs
        
        indices = torch.randperm(num_queries)[:NUM_TRAIN_QUERIES]
        
        curriculum_epochs = int(NUM_EPOCHS * CURRICULUM_EPOCHS)
        print(f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS}", end="", flush=True)
        if epoch < curriculum_epochs and LEXICON_LOGGING:
            print(f" [CURRICULUM active (first {curriculum_epochs} epochs)]", end="", flush=True)
        print(f" with {len(indices)} queries...", flush=True)
        
        for q_start in range(0, num_queries, QUERY_BATCH_SIZE):
            q_end = min(q_start + QUERY_BATCH_SIZE, num_queries)
            batch_len = q_end - q_start
            
            if batch_len < QUERY_BATCH_SIZE // 2:
                continue
            
            epoch_loss = 0.0
            
                query_batch = text_embeds_cpu[q_start:q_end].float().to(DEVICE)
                query_batch_clap = text_embeds_clap_cpu[q_start:q_end].float().to(DEVICE)
                query_batch.requires_grad_(True)
                
                all_sim_v = []
                all_sim_a = []
                all_sim_t = []
                
                for v_start in range(0, num_videos, VIDEO_BATCH_SIZE):
                    v_end = min(v_start + VIDEO_BATCH_SIZE, num_videos)
                    
                    video_batch_v = video_embeds_v_cpu[v_start:v_end].float().to(DEVICE)
                    video_batch_a = video_embeds_a_cpu[v_start:v_end].float().to(DEVICE)
                    c_tensor_batch = caption_tensor_cpu[v_start:v_end].float().to(DEVICE)
                    
                    sim_v = query_batch @ video_batch_v.T
                    # Fix: Use CLAP text encoder for audio similarity
                    sim_a = query_batch_clap @ video_batch_a.T
                    sim_qc = (query_batch.unsqueeze(1).unsqueeze(1) * c_tensor_batch.unsqueeze(0)).sum(dim=-1)
                    sim_t = sim_qc.max(dim=-1).values
                    
                    all_sim_v.append(sim_v)
                    all_sim_a.append(sim_a)
                    all_sim_t.append(sim_t)
                    
                    del video_batch_v, video_batch_a, c_tensor_batch
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            
            all_sim_v = torch.cat(all_sim_v, dim=1)
            all_sim_a = torch.cat(all_sim_a, dim=1)
            all_sim_t = torch.cat(all_sim_t, dim=1)
            
            # Modality dropout (15% probability)
            mask_v = torch.ones_like(all_sim_v)
            mask_a = torch.ones_like(all_sim_a)
            mask_t = torch.ones_like(all_sim_t)
            
            if torch.rand(1).item() < MODALITY_DROPOUT_PROB:
                modality_to_drop = torch.randint(0, 3, (1,)).item()
                if modality_to_drop == 0:
                    mask_v.zero_()
                elif modality_to_drop == 1:
                    mask_a.zero_()
                else:
                    mask_t.zero_()
            
            weights = gating_net(query_batch)
            
            sim_gated = (
                weights[:, 0:1] * (all_sim_v * mask_v) +
                weights[:, 1:2] * (all_sim_t * mask_t) +
                weights[:, 2:3] * (all_sim_a * mask_a)
            )
            
            # Ranking loss: compare correct video against hard negatives
            # For each query, correct video is at diagonal index
            # FIXED: Map query to correct video index
            video_id_to_idx = {vid: i for i, vid in enumerate(common_vids)}
            
            ranking_losses = []
            num_negatives = 10  # Use more negatives
            
            for i in range(batch_len):
                query_idx = q_start + i
                # CORRECT POSITIVE INDEX
                correct_vid = unique_video_ids[query_idx]
                correct_idx = video_id_to_idx[correct_vid]
                correct_score = sim_gated[i, correct_idx].unsqueeze(0)
                
                # Get top-k hard negatives (exclude correct video at correct_idx)
                neg_scores = sim_gated[i].clone()
                neg_scores[correct_idx] = -float('inf')  # Exclude correct
                topk_scores, topk_indices = torch.topk(neg_scores, min(num_negatives, batch_len - 1))
                
                # Margin ranking loss: correct - negative > margin
                for neg_score in topk_scores:
                    margin = 0.2  # Larger margin
                    loss = torch.clamp(margin - (correct_score - neg_score.unsqueeze(0)), min=0)
                    ranking_losses.append(loss)
            
            ranking_loss = torch.stack(ranking_losses).mean() if ranking_losses else 0
            ranking_loss = ranking_loss * 3.0  # Stronger weight
            
            # All heuristic penalties removed per recovery plan (zero)
            epoch_loss = ranking_loss
            
            epoch_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += epoch_loss.item()
            num_batches += 1
            
            del query_batch, all_sim_v, all_sim_a, all_sim_t, weights, sim_gated, epoch_loss
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        print(f"[EPOCH {epoch+1}] Loss: {total_loss/max(num_batches,1):.4f}", end="")
        if in_curriculum_phase and num_batches > 0:
            print(f" (curriculum: {total_curriculum_loss/max(num_batches,1):.4f})", end="")
        print("")
    
    print("[SAVE] Saving gating network weights...")
    torch.save(gating_net.state_dict(), "models/gating_weights.pth")
    print("[SAVE] Saved to gating_weights.pth")
    
    print("[EVAL] Evaluating with learned gating (streaming)...")
    
    gating_net.eval()
    all_topk_results = [None] * num_queries
    
    with torch.no_grad():
        for q_start in range(0, num_queries, QUERY_BATCH_SIZE):
            q_end = min(q_start + QUERY_BATCH_SIZE, num_queries)
            batch_len = q_end - q_start
            
            query_batch = text_embeds_cpu[q_start:q_end].float().to(DEVICE)
            
            weights = gating_net(query_batch)
            weights_cpu = weights.cpu()
            
            del weights
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            
            for v_start in range(0, num_videos, VIDEO_BATCH_SIZE):
                v_end = min(v_start + VIDEO_BATCH_SIZE, num_videos)
                
                video_batch_v = video_embeds_v_cpu[v_start:v_end].float().to(DEVICE)
                video_batch_a = video_embeds_a_cpu[v_start:v_end].float().to(DEVICE)
                c_tensor_batch = caption_tensor_cpu[v_start:v_end].float().to(DEVICE)
                query_batch_clap = text_embeds_clap_cpu[q_start:q_end].float().to(DEVICE)
                
                sim_v = query_batch @ video_batch_v.T
                # Fix: Use CLAP text encoder for audio similarity
                sim_a = query_batch_clap @ video_batch_a.T
                sim_qc = (query_batch.unsqueeze(1).unsqueeze(1) * c_tensor_batch.unsqueeze(0)).sum(dim=-1)
                sim_t = sim_qc.max(dim=-1).values
                
                # Keyword override DISABLED - use learned gating only
                # query_texts_batch = unique_texts[q_start:q_end]
                # for idx, qtext in enumerate(query_texts_batch):
                #     qtext_lower = qtext.lower()
                #     if any(word in qtext_lower for word in VISUAL_KEYWORDS):
                #         weights_cpu[idx, 0] = 0.6
                #         weights_cpu[idx, 1] = 0.2
                #         weights_cpu[idx, 2] = 0.2
                #     elif any(word in qtext_lower for word in AUDIO_KEYWORDS):
                #         weights_cpu[idx, 0] = 0.2
                #         weights_cpu[idx, 1] = 0.2
                #         weights_cpu[idx, 2] = 0.6
                #     elif any(word in qtext_lower for word in TEXT_KEYWORDS):
                #         weights_cpu[idx, 0] = 0.2
                #         weights_cpu[idx, 1] = 0.6
                #         weights_cpu[idx, 2] = 0.2

                sim_gated = (
                weights_cpu[:, 0:1].to(DEVICE) * sim_v +
                weights_cpu[:, 1:2].to(DEVICE) * sim_t +
                weights_cpu[:, 2:3].to(DEVICE) * sim_a
            )
                
                for i in range(batch_len):
                    query_idx = q_start + i
                    scores, indices = torch.topk(sim_gated[i], TOP_K)
                    video_indices = indices.cpu().tolist()
                    candidate_vids = [common_vids[v_start + idx] for idx in video_indices]
                    
                    if all_topk_results[query_idx] is None:
                        all_topk_results[query_idx] = candidate_vids
                    else:
                        all_topk_results[query_idx].extend(candidate_vids)
                        all_topk_results[query_idx] = all_topk_results[query_idx][:TOP_K]
                
                del video_batch_v, video_batch_a, c_tensor_batch
                del sim_v, sim_a, sim_t, sim_gated
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            
            del query_batch, weights_cpu
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    
    print("[EVAL] Computing recall metrics...")
    
    results = {f"R@{k}": 0 for k in [1, 5, 10]}
    
    for i in range(num_queries):
        gt_video = unique_video_ids[i]
        topk_vids = all_topk_results[i] if all_topk_results[i] else []
        
        for k in [1, 5, 10]:
            if gt_video in topk_vids[:k]:
                results[f"R@{k}"] += 1
    
    for k in [1, 5, 10]:
        results[f"R@{k}"] /= num_queries
    
    print("\n===== Gated Retrieval Results =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    gating_net_cpu = gating_net.cpu()
    with torch.no_grad():
        sample_text = text_embeds_cpu[:100]
        sample_weights = gating_net_cpu(sample_text)
    
    print("\n===== Branch Weights (mean) =====")
    print(f"sim_v weight: {sample_weights[:, 0].mean().item():.4f}")
    print(f"sim_t weight: {sample_weights[:, 1].mean().item():.4f}")
    print(f"sim_a weight: {sample_weights[:, 2].mean().item():.4f}")
    
    print("\n===== Baseline Comparison =====")
    
    test_sample = min(100, num_queries)
    vid_sample = min(100, num_videos)
    
    baseline_correct = 0
    sim_v = text_embeds_cpu[:test_sample].float() @ video_embeds_v_cpu[:vid_sample].float().T
    for i in range(sim_v.shape[0]):
        gt = unique_video_ids[i]
        ranked = torch.argsort(sim_v[i], descending=True)
        top1_idx = ranked[0].item()
        top1_vid = common_vids[top1_idx]
        if gt == top1_vid:
            baseline_correct += 1
    
    print(f"sim_v baseline: R@1={baseline_correct/test_sample:.4f}")


if __name__ == "__main__":
    main()