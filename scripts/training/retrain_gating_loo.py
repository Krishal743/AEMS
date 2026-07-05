"""
Retrain gating network with leave-one-out caption similarity.
Prevents exact caption matching during training and evaluation.
"""
from src.models.gating_network import GatingNetwork
import torch
import torch.nn.functional as F
import json, gc, os, argparse, random, numpy as np
import clip
from tqdm import tqdm
from src.encoders.clap_encode import CLAPEncoder
from collections import defaultdict

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
    parser = argparse.ArgumentParser(description="Train gating network with leave-one-out caption similarity")
    parser.add_argument("--video-embeds", default=DEFAULT_VIDEO)
    parser.add_argument("--audio-embeds", default=DEFAULT_AUDIO)
    parser.add_argument("--caption-embeds-train", default=DEFAULT_CAPTION_TRAIN)
    parser.add_argument("--caption-embeds-test", default=DEFAULT_CAPTION_TEST)
    parser.add_argument("--gate-weights", default="models/gating_weights_loo.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-queries", type=int, default=5000)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    return parser.parse_args()

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

def build_caption_index_map(metadata, split="train"):
    """Build (video_id, text) -> index_in_video mapping matching precompute order."""
    split_data = [m for m in metadata if m["split"] == split]
    video_captions = defaultdict(list)
    for item in split_data:
        video_captions[item["video_id"]].append(item["text"])
    index_map = {}
    for vid, captions in video_captions.items():
        for i, text in enumerate(captions):
            index_map[(vid, text)] = i
    return index_map

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

def mask_own_caption(sim_qc, q_batch_vids, q_batch_texts, v_start, v_end, cap_index_map, vid_to_idx):
    for qi, vid in enumerate(q_batch_vids):
        if vid not in vid_to_idx:
            continue
        global_v_idx = vid_to_idx[vid]
        if v_start <= global_v_idx < v_end:
            local_v_idx = global_v_idx - v_start
            cap_idx = cap_index_map.get((vid, q_batch_texts[qi]))
            if cap_idx is not None and cap_idx < sim_qc.shape[2]:
                sim_qc[qi, local_v_idx, cap_idx] = -float('inf')

def main():
    args = parse_args()
    print("=" * 60)
    print("GATING NETWORK TRAINING (leave-one-out caption)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
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

    print(f"  Train entries: {len(train_items)} ({len(train_video_ids)} videos)")
    print(f"  Test entries:  {len(test_items)} ({len(test_video_ids)} videos)")

    # Build caption index maps for LOO masking
    train_caption_index = build_caption_index_map(metadata, "train")
    test_caption_index = build_caption_index_map(metadata, "test")
    print(f"  Train caption index entries: {len(train_caption_index)}")
    print(f"  Test caption index entries:  {len(test_caption_index)}")

    print("\n[EMB] Loading embeddings...")
    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_train_db = torch.load(args.caption_embeds_train, weights_only=False)
    caption_test_db = torch.load(args.caption_embeds_test, weights_only=False)

    print("\n[DATA] Filtering to common videos...")
    train_common = [v for v in video_db if v in audio_db and v in caption_train_db and v in train_video_ids]
    test_common = [v for v in video_db if v in audio_db and v in caption_test_db and v in test_video_ids]
    print(f"  Train common: {len(train_common)}")
    print(f"  Test common:  {len(test_common)}")

    # Build video->index maps
    train_vid_to_idx = {vid: i for i, vid in enumerate(train_common)}
    test_vid_to_idx = {vid: i for i, vid in enumerate(test_common)}

    # Split train into train/val (video-level)
    random.shuffle(train_common)
    val_size = max(1, int(len(train_common) * 0.15))
    val_vids = set(train_common[:val_size])
    train_vids_final = train_common[val_size:]

    train_items_final = [m for m in train_items if m["video_id"] in train_vids_final]
    val_items = [m for m in train_items if m["video_id"] in val_vids]

    print(f"  Train: {len(train_items_final)} items ({len(train_vids_final)} videos)")
    print(f"  Val:   {len(val_items)} items ({len(val_vids)} videos)")

    train_texts = [m["text"] for m in train_items_final]
    train_vids_list = [m["video_id"] for m in train_items_final]
    val_texts = [m["text"] for m in val_items]
    val_vids_list = [m["video_id"] for m in val_items]

    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    print("\n[ENC] Encoding train queries (CLIP)...")
    train_text_embeds = encode_queries(clip_model, train_texts)
    print(f"  Train CLIP: {train_text_embeds.shape}")

    print("[ENC] Encoding val queries (CLIP)...")
    val_text_embeds = encode_queries(clip_model, val_texts)
    print(f"  Val CLIP: {val_text_embeds.shape}")

    print("[CLAP] Encoding queries for audio similarity...")
    clap_encoder = CLAPEncoder(device=DEVICE)

    for name, texts in [("Train", train_texts), ("Val", val_texts)]:
        emb_list = []
        for i in range(0, len(texts), QUERY_BATCH_SIZE):
            batch = texts[i:i+QUERY_BATCH_SIZE]
            emb = clap_encoder.encode_text(batch)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            emb = emb / emb.norm(dim=1, keepdim=True)
            emb_list.append(emb.cpu())
        e = torch.cat(emb_list, dim=0)
        if name == "Train":
            train_text_embeds_clap = e
        else:
            val_text_embeds_clap = e
        print(f"  {name} CLAP: {e.shape}")

    del clap_encoder
    gc.collect()

    print("\n[EMB] Building modality matrices on CPU...")
    train_num_videos = len(train_common)
    video_matrix = torch.stack([to_tensor(video_db[v]) for v in train_common])
    audio_matrix = torch.stack([to_tensor(audio_db[v]) for v in train_common])
    caption_list = [F.normalize(to_tensor(caption_train_db[v]), p=2, dim=1) for v in train_common]
    max_caps = max(c.shape[0] for c in caption_list)
    cap_padded = []
    for c in caption_list:
        n = c.shape[0]
        if n < max_caps:
            pad = torch.zeros(max_caps - n, c.shape[1])
            c = torch.cat([c, pad], dim=0)
        cap_padded.append(c)
    caption_tensor = torch.stack(cap_padded)
    video_matrix = F.normalize(video_matrix, p=2, dim=1)
    audio_matrix = F.normalize(audio_matrix, p=2, dim=1)
    print(f"  Train modality matrices: {video_matrix.shape} each")
    print(f"  Train caption tensor: {caption_tensor.shape}")

    test_num_videos = len(test_common)
    video_matrix_test = torch.stack([to_tensor(video_db[v]) for v in test_common])
    audio_matrix_test = torch.stack([to_tensor(audio_db[v]) for v in test_common])
    caption_list_test = [F.normalize(to_tensor(caption_test_db[v]), p=2, dim=1) for v in test_common]
    max_caps_test = max(c.shape[0] for c in caption_list_test)
    cap_padded_test = []
    for c in caption_list_test:
        n = c.shape[0]
        if n < max_caps_test:
            pad = torch.zeros(max_caps_test - n, c.shape[1])
            c = torch.cat([c, pad], dim=0)
        cap_padded_test.append(c)
    caption_tensor_test = torch.stack(cap_padded_test)
    video_matrix_test = F.normalize(video_matrix_test, p=2, dim=1)
    audio_matrix_test = F.normalize(audio_matrix_test, p=2, dim=1)
    print(f"  Test modality matrices: {video_matrix_test.shape} each")
    print(f"  Test caption tensor: {caption_tensor_test.shape}")

    del video_db, audio_db, caption_train_db, caption_test_db
    gc.collect()

    print("\n[MODEL] Initializing gating network...")
    gate = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    print(f"  Parameters: {sum(p.numel() for p in gate.parameters()):,}")

    train_text_cpu = train_text_embeds.cpu()
    val_text_cpu = val_text_embeds.cpu()
    train_text_clap_cpu = train_text_embeds_clap.cpu()
    val_text_clap_cpu = val_text_embeds_clap.cpu()
    del train_text_embeds, val_text_embeds
    del train_text_embeds_clap, val_text_embeds_clap
    gc.collect()

    num_train = len(train_texts)
    best_val_r1 = 0.0

    print("\n" + "=" * 60)
    print("TRAINING (leave-one-out caption)")
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

            batch_vids = [train_vids_list[idx] for idx in batch_global_idx.tolist()]
            batch_texts = [train_texts[idx] for idx in batch_global_idx.tolist()]

            all_sim_v, all_sim_t, all_sim_a = [], [], []
            for v_start in range(0, train_num_videos, VIDEO_BATCH_SIZE):
                v_end = min(v_start + VIDEO_BATCH_SIZE, train_num_videos)
                v_batch = video_matrix[v_start:v_end].float().to(DEVICE)
                a_batch = audio_matrix[v_start:v_end].float().to(DEVICE)
                c_tensor = caption_tensor[v_start:v_end].float().to(DEVICE)
                all_sim_v.append(q_batch @ v_batch.T)
                sim_qc = (q_batch.unsqueeze(1).unsqueeze(1) * c_tensor.unsqueeze(0)).sum(dim=-1)
                # Leave-one-out: mask out each query's own caption from its video
                mask_own_caption(sim_qc, batch_vids, batch_texts,
                                 v_start, v_end, train_caption_index, train_vid_to_idx)
                all_sim_t.append(sim_qc.max(dim=-1).values)
                all_sim_a.append(q_batch_clap @ a_batch.T)
                del v_batch, a_batch, c_tensor, sim_qc
                gc.collect()

            sim_v = torch.cat(all_sim_v, dim=1)
            sim_t = torch.cat(all_sim_t, dim=1)
            sim_a = torch.cat(all_sim_a, dim=1)
            del all_sim_v, all_sim_t, all_sim_a
            gc.collect()

            weights = gate(q_batch)
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

        # ---- Validation ----
        gate.eval()
        print(f"  Validating...", end="", flush=True)
        with torch.no_grad():
            val_q = val_text_cpu.float()
            val_q_clap = val_text_clap_cpu.float()
            all_sim_v, all_sim_t, all_sim_a = [], [], []
            for v_start in range(0, train_num_videos, VIDEO_BATCH_SIZE):
                v_end = min(v_start + VIDEO_BATCH_SIZE, train_num_videos)
                v_batch = video_matrix[v_start:v_end].float()
                a_batch = audio_matrix[v_start:v_end].float()
                c_batch = caption_tensor[v_start:v_end].float()
                q_batch_sim_v, q_batch_sim_t, q_batch_sim_a = [], [], []
                for q_start in range(0, val_q.size(0), QUERY_BATCH_SIZE):
                    q_end = min(q_start + QUERY_BATCH_SIZE, val_q.size(0))
                    q_part = val_q[q_start:q_end].to(DEVICE)
                    q_clap_part = val_q_clap[q_start:q_end].to(DEVICE)
                    v_part = v_batch.to(DEVICE)
                    a_part = a_batch.to(DEVICE)
                    c_part = c_batch.to(DEVICE)
                    q_batch_sim_v.append(q_part @ v_part.T)
                    sim_qc = (q_part.unsqueeze(1).unsqueeze(1) * c_part.unsqueeze(0)).sum(dim=-1)
                    # LOO masking for val queries
                    val_batch_texts = [val_texts[i] for i in range(q_start, q_end)]
                    val_batch_vids = [val_vids_list[i] for i in range(q_start, q_end)]
                    mask_own_caption(sim_qc, val_batch_vids, val_batch_texts,
                                     v_start, v_end, train_caption_index, train_vid_to_idx)
                    q_batch_sim_t.append(sim_qc.max(dim=-1).values)
                    q_batch_sim_a.append(q_clap_part @ a_part.T)
                    del q_part, q_clap_part, v_part, a_part, c_part, sim_qc
                    gc.collect()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                all_sim_v.append(torch.cat(q_batch_sim_v, dim=0))
                all_sim_t.append(torch.cat(q_batch_sim_t, dim=0))
                all_sim_a.append(torch.cat(q_batch_sim_a, dim=0))
                del v_batch, a_batch, c_batch
                gc.collect()

            sim_v_val = torch.cat(all_sim_v, dim=1)
            sim_t_val = torch.cat(all_sim_t, dim=1)
            sim_a_val = torch.cat(all_sim_a, dim=1)
            del all_sim_v, all_sim_t, all_sim_a
            gc.collect()

            val_weights = gate(val_q.to(DEVICE))
            sim_gated_val = (
                val_weights[:, 0:1] * sim_v_val +
                val_weights[:, 1:2] * sim_t_val +
                val_weights[:, 2:3] * sim_a_val
            )
            val_metrics = evaluate_retrieval(sim_gated_val, val_vids_list, train_common, ks=[1, 5, 10])
            del sim_v_val, sim_t_val, sim_a_val, sim_gated_val, val_weights
            gc.collect()

        print(f"  Val R@1={val_metrics['R@1']:.4f} R@5={val_metrics['R@5']:.4f} (train loss={avg_loss:.4f})")

        if val_metrics["R@1"] > best_val_r1:
            best_val_r1 = val_metrics["R@1"]
            torch.save(gate.state_dict(), args.gate_weights)
            print(f"  [BEST] New best R@1: {best_val_r1:.4f} -> saved to {args.gate_weights}")

    # ---- Final test evaluation ----
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (leave-one-out)")
    print("=" * 60)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE, weights_only=False))
    gate.eval()

    test_items_common = [m for m in test_items if m["video_id"] in test_common]
    test_texts_list = [m["text"] for m in test_items_common]
    test_vids_list_full = [m["video_id"] for m in test_items_common]

    with torch.no_grad():
        print("[ENC] Encoding test queries (CLIP)...")
        clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        clip_model.eval()
        test_text_embeds = encode_queries(clip_model, test_texts_list)
        print(f"  Test CLIP: {test_text_embeds.shape}")

        print("[CLAP] Encoding test queries for audio...")
        clap_encoder = CLAPEncoder(device=DEVICE)
        test_clap_list = []
        for i in range(0, len(test_texts_list), QUERY_BATCH_SIZE):
            batch = test_texts_list[i:i+QUERY_BATCH_SIZE]
            emb = clap_encoder.encode_text(batch)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            emb = emb / emb.norm(dim=1, keepdim=True)
            test_clap_list.append(emb.cpu())
        test_text_clap = torch.cat(test_clap_list, dim=0)
        print(f"  Test CLAP: {test_text_clap.shape}")
        del clap_encoder
        gc.collect()

        test_q = test_text_embeds.float()
        test_q_clap = test_text_clap.float()

        all_sim_v, all_sim_t, all_sim_a = [], [], []
        for v_start in range(0, test_num_videos, VIDEO_BATCH_SIZE):
            v_end = min(v_start + VIDEO_BATCH_SIZE, test_num_videos)
            v_batch = video_matrix_test[v_start:v_end].float()
            a_batch = audio_matrix_test[v_start:v_end].float()
            c_batch = caption_tensor_test[v_start:v_end].float()
            q_batch_sim_v, q_batch_sim_t, q_batch_sim_a = [], [], []
            for q_start in range(0, test_q.size(0), QUERY_BATCH_SIZE):
                q_end = min(q_start + QUERY_BATCH_SIZE, test_q.size(0))
                q_part = test_q[q_start:q_end].to(DEVICE)
                q_clap_part = test_q_clap[q_start:q_end].to(DEVICE)
                v_part = v_batch.to(DEVICE)
                a_part = a_batch.to(DEVICE)
                c_part = c_batch.to(DEVICE)
                q_batch_sim_v.append(q_part @ v_part.T)
                sim_qc = (q_part.unsqueeze(1).unsqueeze(1) * c_part.unsqueeze(0)).sum(dim=-1)
                # LOO masking for test queries
                test_batch_texts = [test_texts_list[i] for i in range(q_start, q_end)]
                test_batch_vids = [test_vids_list_full[i] for i in range(q_start, q_end)]
                mask_own_caption(sim_qc, test_batch_vids, test_batch_texts,
                                 v_start, v_end, test_caption_index, test_vid_to_idx)
                q_batch_sim_t.append(sim_qc.max(dim=-1).values)
                q_batch_sim_a.append(q_clap_part @ a_part.T)
                del q_part, q_clap_part, v_part, a_part, c_part, sim_qc
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            all_sim_v.append(torch.cat(q_batch_sim_v, dim=0))
            all_sim_t.append(torch.cat(q_batch_sim_t, dim=0))
            all_sim_a.append(torch.cat(q_batch_sim_a, dim=0))
            del v_batch, a_batch, c_batch
            gc.collect()

        sim_v_test = torch.cat(all_sim_v, dim=1)
        sim_t_test = torch.cat(all_sim_t, dim=1)
        sim_a_test = torch.cat(all_sim_a, dim=1)
        del all_sim_v, all_sim_t, all_sim_a
        gc.collect()

        test_weights = gate(test_q.to(DEVICE))
        sim_gated_test = (
            test_weights[:, 0:1] * sim_v_test +
            test_weights[:, 1:2] * sim_t_test +
            test_weights[:, 2:3] * sim_a_test
        )

        systems = {
            "Visual only": sim_v_test,
            "Caption only (LOO)": sim_t_test,
            "Audio only": sim_a_test,
            "Equal V+C (LOO)": (sim_v_test + sim_t_test) / 2,
            "Equal V+T+A (LOO)": (sim_v_test + sim_t_test + sim_a_test) / 3,
            "Adaptive gating (LOO)": sim_gated_test,
        }

        print(f"\n{'System':<35} {'R@1':<10} {'R@5':<10} {'R@10':<10}")
        print(f"{'-'*65}")
        for name, sim in systems.items():
            metrics = evaluate_retrieval(sim, test_vids_list_full, test_common, ks=[1, 5, 10])
            print(f"{name:<35} {metrics['R@1']:<10.4f} {metrics['R@5']:<10.4f} {metrics['R@10']:<10.4f}")

        # Show weight distribution
        weights_cpu = test_weights.cpu()
        print(f"\n{'Modality':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"{'-'*52}")
        for name, idx in [("Visual", 0), ("Caption", 1), ("Audio", 2)]:
            w = weights_cpu[:, idx]
            print(f"{name:<12} {w.mean():<10.6f} {w.std():<10.6f} {w.min():<10.6f} {w.max():<10.6f}")

        del clip_model
        gc.collect()

    print(f"\n[DONE] Best val R@1: {best_val_r1:.4f}")
    print(f"[DONE] Saved: {args.gate_weights}")

if __name__ == "__main__":
    main()
