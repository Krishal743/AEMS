from src.models.gating_network import GatingNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, gc, os, argparse, random, numpy as np
import clip
import numpy as np
from tqdm import tqdm
from src.encoders.clap_encode import CLAPEncoder

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_BATCH_SIZE = 32
VIDEO_BATCH_SIZE = 100
NUM_NEGATIVES = 10
RANKING_MARGIN = 0.2

DEFAULT_METADATA = "data/processed/metadata/msrvtt_metadata.json"
DEFAULT_VIDEO = "embeddings/video_embeddings.pt"
DEFAULT_AUDIO = "embeddings/audio_embeddings.pt"
DEFAULT_CAPTION_TRAIN = "embeddings/caption_embeddings_train.pt"
DEFAULT_CAPTION_TEST = "embeddings/caption_embeddings_test.pt"

def parse_args():
    parser = argparse.ArgumentParser(description="Train gating network with ranking loss")
    parser.add_argument("--video-embeds", default=DEFAULT_VIDEO)
    parser.add_argument("--audio-embeds", default=DEFAULT_AUDIO)
    parser.add_argument("--caption-embeds-train", default=DEFAULT_CAPTION_TRAIN)
    parser.add_argument("--caption-embeds-test", default=DEFAULT_CAPTION_TEST)
    parser.add_argument("--gate-weights", default="models/gating_weights.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-queries", type=int, default=5000)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    return parser.parse_args()


def get_common_video_ids(db_v, db_a, db_c, candidate_set):
    return [vid for vid in candidate_set if vid in db_v and vid in db_a and vid in db_c]


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()


@torch.no_grad()
def encode_queries(clip_model, texts, batch_size=32):
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = clip.tokenize(batch).to(DEVICE)
        embeds = clip_model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=1, keepdim=True)
        all_embeds.append(embeds.cpu())
        del embeds, tokens
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(all_embeds, dim=0)


def compute_ranking_loss(weights, sim_v, sim_t, sim_a, pos_indices):
    sim_gated = (
        weights[:, 0:1] * sim_v +
        weights[:, 1:2] * sim_t +
        weights[:, 2:3] * sim_a
    )
    ranking_losses = []
    for i in range(len(pos_indices)):
        pos_idx = pos_indices[i]
        correct_score = sim_gated[i, pos_idx].unsqueeze(0)
        neg_scores = sim_gated[i].clone()
        neg_scores[pos_idx] = -float("inf")
        topk_scores, _ = torch.topk(neg_scores, min(NUM_NEGATIVES, len(pos_indices) - 1))
        for neg_score in topk_scores:
            ranking_losses.append(
                torch.clamp(RANKING_MARGIN - (correct_score - neg_score.unsqueeze(0)), min=0)
            )
    loss = torch.stack(ranking_losses).mean() if ranking_losses else torch.tensor(0.0, device=DEVICE)
    return loss * 3.0


def main():
    args = parse_args()
    print("=" * 60)
    print("GATING NETWORK TRAINING (ranking loss)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Video embeds: {args.video_embeds}")
    print(f"Epochs: {args.epochs}, Train queries/epoch: {args.train_queries}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\n[DATA] Loading metadata...")
    with open(args.metadata) as f:
        metadata = json.load(f)

    train_items = [m for m in metadata if m["split"] == "train"]
    test_items = [m for m in metadata if m["split"] == "test"]
    train_video_ids = set(m["video_id"] for m in train_items)
    test_video_ids = set(m["video_id"] for m in test_items)
    print(f"  Train items: {len(train_items)} ({len(train_video_ids)} unique videos)")
    print(f"  Test items:  {len(test_items)} ({len(test_video_ids)} unique videos)")

    print("\n[EMB] Loading embeddings...")
    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_train_db = torch.load(args.caption_embeds_train, weights_only=False)
    caption_test_db = torch.load(args.caption_embeds_test, weights_only=False)
    print(f"  Video: {len(video_db)}  Audio: {len(audio_db)}")
    print(f"  Caption train: {len(caption_train_db)}  Caption test: {len(caption_test_db)}")

    train_common_vids = get_common_video_ids(video_db, audio_db, caption_train_db, train_video_ids)
    train_vid_to_idx = {vid: i for i, vid in enumerate(train_common_vids)}
    test_common_vids = get_common_video_ids(video_db, audio_db, caption_test_db, test_video_ids)
    print(f"  Common train videos: {len(train_common_vids)}")
    print(f"  Common test videos:  {len(test_common_vids)}")

    if len(train_common_vids) == 0:
        print("[ERROR] No overlapping train videos.")
        return 1

    train_vid_set = set(train_common_vids)
    train_items_filtered = [m for m in train_items if m["video_id"] in train_vid_set]
    test_vid_set = set(test_common_vids)
    test_items_filtered = [m for m in test_items if m["video_id"] in test_vid_set]

    random.shuffle(train_items_filtered)
    val_size = int(len(train_items_filtered) * 0.15)
    val_items = train_items_filtered[:val_size]
    train_items_final = train_items_filtered[val_size:]
    print(f"  Train queries: {len(train_items_final)}  Val queries: {len(val_items)}")

    print("\n[CLIP] Loading model...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    print("[TEXT] Encoding training queries...")
    train_texts = [m["text"] for m in train_items_final]
    train_vids = [m["video_id"] for m in train_items_final]
    train_text_embeds = encode_queries(clip_model, train_texts, batch_size=QUERY_BATCH_SIZE)
    print(f"  Train text embeds: {train_text_embeds.shape}")

    print("[TEXT] Encoding validation queries...")
    val_texts = [m["text"] for m in val_items]
    val_vids = [m["video_id"] for m in val_items]
    val_text_embeds = encode_queries(clip_model, val_texts, batch_size=QUERY_BATCH_SIZE)
    print(f"  Val text embeds: {val_text_embeds.shape}")

    print("[TEXT] Encoding test queries...")
    test_texts = [m["text"] for m in test_items_filtered]
    test_text_vids = [m["video_id"] for m in test_items_filtered]
    test_text_embeds = encode_queries(clip_model, test_texts, batch_size=QUERY_BATCH_SIZE)
    print(f"  Test text embeds: {test_text_embeds.shape}")

    del clip_model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("[CLAP] Encoding queries for audio similarity...")
    clap_encoder = CLAPEncoder(device=DEVICE)

    def encode_queries_clap(encoder, texts_list, batch_size=32):
        all_embeds = []
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i:i+batch_size]
            emb = encoder.encode_text(batch)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            emb = emb / emb.norm(dim=1, keepdim=True)
            all_embeds.append(emb.cpu())
            del emb
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(all_embeds, dim=0)

    train_text_embeds_clap = encode_queries_clap(clap_encoder, train_texts, batch_size=QUERY_BATCH_SIZE)
    val_text_embeds_clap = encode_queries_clap(clap_encoder, val_texts, batch_size=QUERY_BATCH_SIZE)
    test_text_embeds_clap = encode_queries_clap(clap_encoder, test_texts, batch_size=QUERY_BATCH_SIZE)
    print(f"  Train CLAP: {train_text_embeds_clap.shape}  Val CLAP: {val_text_embeds_clap.shape}  Test CLAP: {test_text_embeds_clap.shape}")

    del clap_encoder
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\n[EMB] Building modality matrices on CPU...")
    num_videos = len(train_common_vids)
    video_matrix = torch.stack([to_tensor(video_db[v]) for v in train_common_vids])
    audio_matrix = torch.stack([to_tensor(audio_db[v]) for v in train_common_vids])
    # Fix: keep all captions per video (handle variable caption counts)
    caption_list = [F.normalize(to_tensor(caption_train_db[v]), p=2, dim=1) for v in train_common_vids]
    max_caps_train = max(c.shape[0] for c in caption_list)
    cap_padded_train = []
    for c in caption_list:
        n = c.shape[0]
        if n < max_caps_train:
            pad = torch.zeros(max_caps_train - n, c.shape[1])
            c = torch.cat([c, pad], dim=0)
        cap_padded_train.append(c)
    caption_tensor = torch.stack(cap_padded_train)
    video_matrix = F.normalize(video_matrix, p=2, dim=1)
    audio_matrix = F.normalize(audio_matrix, p=2, dim=1)
    print(f"  Train modality matrices: {video_matrix.shape} each")
    print(f"  Caption tensor (per-video, per-caption): {caption_tensor.shape}")

    num_test_videos = len(test_common_vids)
    video_matrix_test = torch.stack([to_tensor(video_db[v]) for v in test_common_vids])
    audio_matrix_test = torch.stack([to_tensor(audio_db[v]) for v in test_common_vids])
    # Fix: keep all test caption per video (handle variable caption counts)
    caption_list_test = [F.normalize(to_tensor(caption_test_db[v]), p=2, dim=1) for v in test_common_vids]
    max_caps = max(c.shape[0] for c in caption_list_test)
    cap_padded = []
    for c in caption_list_test:
        n = c.shape[0]
        if n < max_caps:
            pad = torch.zeros(max_caps - n, c.shape[1])
            c = torch.cat([c, pad], dim=0)
        cap_padded.append(c)
    caption_tensor_test = torch.stack(cap_padded)
    video_matrix_test = F.normalize(video_matrix_test, p=2, dim=1)
    audio_matrix_test = F.normalize(audio_matrix_test, p=2, dim=1)
    print(f"  Test modality matrices:  {video_matrix_test.shape} each")
    print(f"  Test caption tensor: {caption_tensor_test.shape}")

    del video_db, audio_db, caption_train_db, caption_test_db
    gc.collect()

    print("\n[MODEL] Initializing gating network...")
    gate = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    print(f"  Parameters: {sum(p.numel() for p in gate.parameters()):,}")

    train_text_cpu = train_text_embeds.cpu()
    val_text_cpu = val_text_embeds.cpu()
    test_text_cpu = test_text_embeds.cpu()
    train_text_clap_cpu = train_text_embeds_clap.cpu()
    val_text_clap_cpu = val_text_embeds_clap.cpu()
    test_text_clap_cpu = test_text_embeds_clap.cpu()
    del train_text_embeds, val_text_embeds, test_text_embeds
    del train_text_embeds_clap, val_text_embeds_clap, test_text_embeds_clap
    gc.collect()

    num_train = len(train_texts)
    best_val_r1 = 0.0
    train_log = []

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(args.epochs):
        gate.train()
        total_loss = 0.0
        num_batches = 0
        indices = torch.randperm(num_train)[:min(args.train_queries, num_train)]

        pbar = tqdm(range(0, len(indices), QUERY_BATCH_SIZE),
                    desc=f"Epoch {epoch+1}/{args.epochs}")

        for q_start_idx in pbar:
            q_end_idx = min(q_start_idx + QUERY_BATCH_SIZE, len(indices))
            batch_len = q_end_idx - q_start_idx
            if batch_len < 2:
                continue

            batch_global_idx = indices[q_start_idx:q_end_idx]
            q_batch = train_text_cpu[batch_global_idx].float().to(DEVICE)
            q_batch_clap = train_text_clap_cpu[batch_global_idx].float().to(DEVICE)
            q_batch.requires_grad_(True)

            all_sim_v, all_sim_t, all_sim_a = [], [], []
            for v_start in range(0, num_videos, VIDEO_BATCH_SIZE):
                v_end = min(v_start + VIDEO_BATCH_SIZE, num_videos)
                v_batch = video_matrix[v_start:v_end].float().to(DEVICE)
                a_batch = audio_matrix[v_start:v_end].float().to(DEVICE)
                c_tensor = caption_tensor[v_start:v_end].float().to(DEVICE)
                all_sim_v.append(q_batch @ v_batch.T)
                # Fix: max over captions per video (broadcast over queries and videos)
                sim_qc = (q_batch.unsqueeze(1).unsqueeze(1) * c_tensor.unsqueeze(0)).sum(dim=-1)
                all_sim_t.append(sim_qc.max(dim=-1).values)
                # Fix: Use CLAP text encoder for audio similarity
                all_sim_a.append(q_batch_clap @ a_batch.T)
                del v_batch, a_batch, c_tensor
                gc.collect()

            sim_v = torch.cat(all_sim_v, dim=1)
            sim_t = torch.cat(all_sim_t, dim=1)
            sim_a = torch.cat(all_sim_a, dim=1)
            del all_sim_v, all_sim_t, all_sim_a
            gc.collect()

            weights = gate(q_batch)
            batch_vids = [train_vids[idx] for idx in batch_global_idx.tolist()]
            pos_indices = torch.tensor([train_vid_to_idx[vid] for vid in batch_vids], device=DEVICE)
            loss = compute_ranking_loss(weights, sim_v, sim_t, sim_a, pos_indices)

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            del q_batch, sim_v, sim_t, sim_a, weights, loss
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)

        from src.evaluation.evaluate_retrieval import evaluate_retrieval

        gate.eval()
        print(f"  Validating...", end="", flush=True)
        with torch.no_grad():
            val_q = val_text_cpu.float()
            val_q_clap = val_text_clap_cpu.float()
            
            all_sim_v, all_sim_t, all_sim_a = [], [], []
            for v_start in range(0, num_videos, VIDEO_BATCH_SIZE):
                v_end = min(v_start + VIDEO_BATCH_SIZE, num_videos)
                v_batch = video_matrix[v_start:v_end].float()
                a_batch = audio_matrix[v_start:v_end].float()
                c_batch = caption_tensor[v_start:v_end].float()
                
                # Process queries in mini-batches to avoid OOM
                q_batch_sim_v, q_batch_sim_t, q_batch_sim_a = [], [], []
                for q_start in range(0, val_q.size(0), QUERY_BATCH_SIZE):
                    q_end = min(q_start + QUERY_BATCH_SIZE, val_q.size(0))
                    q_part = val_q[q_start:q_end].to(DEVICE)
                    q_clap_part = val_q_clap[q_start:q_end].to(DEVICE)
                    v_part = v_batch.to(DEVICE)
                    a_part = a_batch.to(DEVICE)
                    c_part = c_batch.to(DEVICE)
                    
                    sim_v = q_part @ v_part.T
                    sim_qc = (q_part.unsqueeze(1).unsqueeze(1) * c_part.unsqueeze(0)).sum(dim=-1)
                    sim_t = sim_qc.max(dim=-1).values
                    sim_a = q_clap_part @ a_part.T
                    
                    q_batch_sim_v.append(sim_v.cpu())
                    q_batch_sim_t.append(sim_t.cpu())
                    q_batch_sim_a.append(sim_a.cpu())
                    del q_part, q_clap_part, v_part, a_part, c_part, sim_v, sim_t, sim_a
                    gc.collect()
                
                all_sim_v.append(torch.cat(q_batch_sim_v, dim=0))
                all_sim_t.append(torch.cat(q_batch_sim_t, dim=0))
                all_sim_a.append(torch.cat(q_batch_sim_a, dim=0))
                del v_batch, a_batch, c_batch, q_batch_sim_v, q_batch_sim_t, q_batch_sim_a
                gc.collect()
            
            val_sim_v = torch.cat(all_sim_v, dim=1)
            val_sim_t = torch.cat(all_sim_t, dim=1)
            val_sim_a = torch.cat(all_sim_a, dim=1)
            del all_sim_v, all_sim_t, all_sim_a
            gc.collect()
            
            w = gate(val_q.to(DEVICE)).cpu()
            val_sim_gated = (
                w[:, 0:1] * val_sim_v +
                w[:, 1:2] * val_sim_t +
                w[:, 2:3] * val_sim_a
            )
            val_metrics = evaluate_retrieval(
                val_sim_gated, val_vids, train_common_vids, ks=[1, 5, 10]
            )
            del val_q, val_q_clap, w, val_sim_v, val_sim_t, val_sim_a, val_sim_gated
            gc.collect()

        print(f"  Val R@1: {val_metrics['R@1']:.4f}  R@5: {val_metrics['R@5']:.4f}  R@10: {val_metrics['R@10']:.4f}")

        train_log.append({"epoch": epoch + 1, "loss": avg_loss, **val_metrics})

        if val_metrics["R@1"] > best_val_r1:
            best_val_r1 = val_metrics["R@1"]
            torch.save(gate.state_dict(), args.gate_weights)
            print(f"  [BEST] Saved {args.gate_weights} (R@1={best_val_r1:.4f})")

        print(f"  Epoch {epoch+1} complete — loss: {avg_loss:.4f}")

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — Best val R@1: {best_val_r1:.4f}")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("EPOCH LOG")
    print("-" * 60)
    print(f"{'Epoch':>6} {'Loss':>8} {'R@1':>6} {'R@5':>6} {'R@10':>7}")
    for entry in train_log:
        print(f"{entry['epoch']:>6} {entry['loss']:>8.4f} {entry['R@1']:>6.4f} {entry['R@5']:>6.4f} {entry['R@10']:>7.4f}")

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (test split candidates)")
    print("=" * 60)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE))
    gate.eval()

    with torch.no_grad():
        for label, (t_cpu, v_list) in [
            ("Gated", (test_text_cpu, test_text_vids)),
        ]:
            q = t_cpu.float()
            q_clap = test_text_clap_cpu.float()
            
            all_sim_v, all_sim_t, all_sim_a = [], [], []
            num_test_videos = len(test_common_vids)
            for v_start in range(0, num_test_videos, VIDEO_BATCH_SIZE):
                v_end = min(v_start + VIDEO_BATCH_SIZE, num_test_videos)
                v_batch = video_matrix_test[v_start:v_end].float()
                a_batch = audio_matrix_test[v_start:v_end].float()
                c_batch = caption_tensor_test[v_start:v_end].float()
                
                q_batch_sim_v, q_batch_sim_t, q_batch_sim_a = [], [], []
                for q_start in range(0, q.size(0), QUERY_BATCH_SIZE):
                    q_end = min(q_start + QUERY_BATCH_SIZE, q.size(0))
                    q_part = q[q_start:q_end].to(DEVICE)
                    q_clap_part = q_clap[q_start:q_end].to(DEVICE)
                    v_part = v_batch.to(DEVICE)
                    a_part = a_batch.to(DEVICE)
                    c_part = c_batch.to(DEVICE)
                    
                    sim_v = q_part @ v_part.T
                    sim_qc = (q_part.unsqueeze(1).unsqueeze(1) * c_part.unsqueeze(0)).sum(dim=-1)
                    sim_t = sim_qc.max(dim=-1).values
                    sim_a = q_clap_part @ a_part.T
                    
                    q_batch_sim_v.append(sim_v.cpu())
                    q_batch_sim_t.append(sim_t.cpu())
                    q_batch_sim_a.append(sim_a.cpu())
                    del q_part, q_clap_part, v_part, a_part, c_part, sim_v, sim_t, sim_a
                    gc.collect()
                
                all_sim_v.append(torch.cat(q_batch_sim_v, dim=0))
                all_sim_t.append(torch.cat(q_batch_sim_t, dim=0))
                all_sim_a.append(torch.cat(q_batch_sim_a, dim=0))
                del v_batch, a_batch, c_batch, q_batch_sim_v, q_batch_sim_t, q_batch_sim_a
                gc.collect()
            
            t_sim_v = torch.cat(all_sim_v, dim=1)
            t_sim_t = torch.cat(all_sim_t, dim=1)
            t_sim_a = torch.cat(all_sim_a, dim=1)
            del all_sim_v, all_sim_t, all_sim_a
            gc.collect()
            
            w = gate(q.to(DEVICE)).cpu()
            t_sim_gated = (
                w[:, 0:1] * t_sim_v +
                w[:, 1:2] * t_sim_t +
                w[:, 2:3] * t_sim_a
            )
            test_metrics = evaluate_retrieval(
                t_sim_gated, test_text_vids, test_common_vids, ks=[1, 5, 10]
            )
            print(f"\n  {label} on test split:")
            print(f"    R@1: {test_metrics['R@1']:.4f}  R@5: {test_metrics['R@5']:.4f}  R@10: {test_metrics['R@10']:.4f}")
            
            del q, q_clap, w, t_sim_v, t_sim_t, t_sim_a, t_sim_gated
            gc.collect()

    mean_weights = []
    with torch.no_grad():
        sample_text = train_text_cpu[:500].float().to(DEVICE)
        sample_weights = gate(sample_text).cpu()
        mean_weights = sample_weights.mean(dim=0)

    print("\n  Learned gate weights (mean across 500 train queries):")
    print(f"    w_v (video):   {mean_weights[0].item():.4f}")
    print(f"    w_t (caption): {mean_weights[1].item():.4f}")
    print(f"    w_a (audio):   {mean_weights[2].item():.4f}")

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
