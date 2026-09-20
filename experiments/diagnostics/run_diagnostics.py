"""
Comprehensive diagnostic analysis of AEMS multimodal retrieval system.
Covers 5 investigations without modifying any models.
"""
import sys, json, torch, argparse, os, csv, gc
import clip
import numpy as np
from collections import Counter, defaultdict
from src.models.gating_network import GatingNetwork
from src.encoders.clap_encode import CLAPEncoder
from src.evaluation.evaluate_retrieval import evaluate_retrieval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()

def main():
    parser = argparse.ArgumentParser(description="Diagnostic analysis")
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--video-transformer-embeds", default="embeddings/video_embeddings_transformer.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds-test", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_meanpool.pth")
    parser.add_argument("--metadata", default="data/processed/metadata/msrvtt_metadata.json")
    args = parser.parse_args()

    os.makedirs("outputs/diagnostics", exist_ok=True)

    # ======================================================================
    # DATA LOADING
    # ======================================================================
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    with open(args.metadata) as f:
        metadata = json.load(f)
    test_items = [m for m in metadata if m["split"] == "test"]
    test_video_ids = set(m["video_id"] for m in test_items)
    print(f"Test metadata entries: {len(test_items)}")
    print(f"Unique test video IDs: {len(test_video_ids)}")

    # Group captions by video_id for verbatim match analysis
    video_captions_text = defaultdict(list)
    for m in test_items:
        video_captions_text[m["video_id"]].append(m["text"])

    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_test_db = torch.load(args.caption_embeds_test, weights_only=False)
    transformer_db = torch.load(args.video_transformer_embeds, weights_only=False) if os.path.exists(args.video_transformer_embeds) else {}

    print(f"video_db entries: {len(video_db)}")
    print(f"audio_db entries: {len(audio_db)}")
    print(f"caption_test_db entries: {len(caption_test_db)}")
    print(f"transformer_db entries: {len(transformer_db)}")

    common_original = [v for v in video_db if v in audio_db and v in caption_test_db and v in test_video_ids]
    common_transformer = [v for v in transformer_db if v in audio_db and v in caption_test_db and v in test_video_ids]
    print(f"Common (original): {len(common_original)}")
    print(f"Common (transformer): {len(common_transformer)}")

    # Build filtered query lists
    test_items_orig = [m for m in test_items if m["video_id"] in common_original]
    test_texts_orig = [m["text"] for m in test_items_orig]
    test_vids_orig = [m["video_id"] for m in test_items_orig]

    test_items_trans = [m for m in test_items if m["video_id"] in common_transformer]
    test_texts_trans = [m["text"] for m in test_items_trans]
    test_vids_trans = [m["video_id"] for m in test_items_trans]

    print(f"Queries (original eval): {len(test_texts_orig)}")
    print(f"Queries (transformer eval): {len(test_texts_trans)}")

    # ======================================================================
    # INVESTIGATION 1: Exact caption matching
    # ======================================================================
    print("\n" + "=" * 70)
    print("INVESTIGATION 1: EXACT CAPTION MATCHING")
    print("=" * 70)

    exact_match_count = 0
    non_exact_queries = []
    total_checked = 0
    for i, (text, vid) in enumerate(zip(test_texts_orig, test_vids_orig)):
        stored_captions = video_captions_text.get(vid, [])
        total_checked += 1
        if text in stored_captions:
            exact_match_count += 1
        else:
            non_exact_queries.append((i, text, vid, len(stored_captions)))

    print(f"Total queries checked: {total_checked}")
    print(f"Exact verbatim matches: {exact_match_count} ({100*exact_match_count/total_checked:.2f}%)")
    print(f"Non-exact matches: {total_checked - exact_match_count} ({100*(total_checked-exact_match_count)/total_checked:.2f}%)")

    if non_exact_queries:
        print(f"\nSample of non-exact matches (showing up to 10):")
        for idx, text, vid, n_caps in non_exact_queries[:10]:
            stored = video_captions_text.get(vid, [])
            print(f"  Query[{idx}]: '{text[:60]}...'")
            print(f"    Video: {vid}, Stored captions: {n_caps}")
            # Check if any stored caption starts with the query or vice versa
            for s in stored:
                if text[:20] in s or s[:20] in text:
                    print(f"    Partial match: '{s[:60]}...'")
            print()

    # Also check how caption counts vary
    cap_counts = Counter()
    for m in test_items:
        cap_counts[len(video_captions_text[m["video_id"]])] += 1
    print(f"Caption count distribution per video:")
    for cnt, freq in sorted(cap_counts.items()):
        print(f"  {cnt} captions: {freq} videos")

    # ======================================================================
    # BUILD MODALITY MATRICES (original eval)
    # ======================================================================
    print("\n" + "=" * 70)
    print("BUILDING SIMILARITY MATRICES")
    print("=" * 70)

    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common_original])
    aud_m = torch.stack([torch.as_tensor(audio_db[v]).float() for v in common_original])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    aud_m = torch.nn.functional.normalize(aud_m, p=2, dim=1)
    del video_db, audio_db
    gc.collect()

    # CLIP text embeddings
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    text_embeds = []
    for i in range(0, len(test_texts_orig), 256):
        batch = test_texts_orig[i:i+256]
        tokens = clip.tokenize(batch).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=1, keepdim=True)
        text_embeds.append(emb.cpu())
    text_embeds = torch.cat(text_embeds, dim=0)
    print(f"CLIP text embeddings shape: {text_embeds.shape}")

    # sim_v
    sim_v = text_embeds @ vid_m.T
    print(f"sim_v shape: {sim_v.shape}")

    # sim_t (caption similarity)
    sim_t_list = []
    for vid in common_original:
        cap_embeds = torch.as_tensor(caption_test_db[vid]).float()
        cap_embeds = torch.nn.functional.normalize(cap_embeds, p=2, dim=1).to(DEVICE)
        sims = text_embeds.to(DEVICE) @ cap_embeds.T
        max_sims = sims.max(dim=1).values
        sim_t_list.append(max_sims.cpu())
    sim_t = torch.stack(sim_t_list, dim=1)
    print(f"sim_t shape: {sim_t.shape}")
    del caption_test_db
    gc.collect()

    # sim_a (CLAP text)
    clap_encoder = CLAPEncoder(device=DEVICE)
    clap_text_list = []
    for i in range(0, len(test_texts_orig), 256):
        batch = test_texts_orig[i:i+256]
        emb = clap_encoder.encode_text(batch)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        emb = emb / emb.norm(dim=1, keepdim=True)
        clap_text_list.append(emb.cpu())
    clap_text_embeds = torch.cat(clap_text_list, dim=0)
    sim_a = clap_text_embeds @ aud_m.T
    print(f"sim_a shape: {sim_a.shape}")
    del clap_encoder, clap_text_embeds
    gc.collect()

    # ======================================================================
    # INVESTIGATION 2: Gating weight distribution
    # ======================================================================
    print("\n" + "=" * 70)
    print("INVESTIGATION 2: GATING WEIGHT DISTRIBUTION")
    print("=" * 70)

    gate = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE), strict=False)
    gate.eval()

    all_weights = []
    for i in range(0, len(test_texts_orig), 256):
        q = text_embeds[i:i+256].float().to(DEVICE)
        with torch.no_grad():
            w = gate(q).cpu()
        all_weights.append(w)
    all_weights = torch.cat(all_weights, dim=0)
    print(f"Gating weights shape: {all_weights.shape}")

    w_v = all_weights[:, 0].numpy()
    w_t = all_weights[:, 1].numpy()
    w_a = all_weights[:, 2].numpy()

    print(f"\nWeight statistics:")
    print(f"  {'Modality':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print(f"  {'-'*62}")
    for name, w in [("Visual", w_v), ("Caption", w_t), ("Audio", w_a)]:
        print(f"  {name:<12} {w.mean():<10.6f} {w.std():<10.6f} {w.min():<10.6f} {w.max():<10.6f} {np.median(w):<10.6f}")

    # Weight distribution histogram
    def hist_buckets(arr, bins=5):
        counts, edges = np.histogram(arr, bins=bins)
        return [(f"[{edges[i]:.4f}, {edges[i+1]:.4f}]", counts[i]) for i in range(bins)]

    print(f"\nVisual weight distribution:")
    for bucket, cnt in hist_buckets(w_v):
        pct = 100 * cnt / len(w_v)
        bar = "█" * int(pct / 2)
        print(f"  {bucket:<20} {cnt:>6} ({pct:>5.1f}%) {bar}")

    print(f"\nCaption weight distribution:")
    for bucket, cnt in hist_buckets(w_t):
        pct = 100 * cnt / len(w_t)
        bar = "█" * int(pct / 2)
        print(f"  {bucket:<20} {cnt:>6} ({pct:>5.1f}%) {bar}")

    print(f"\nAudio weight distribution:")
    for bucket, cnt in hist_buckets(w_a):
        pct = 100 * cnt / len(w_a)
        bar = "█" * int(pct / 2)
        print(f"  {bucket:<20} {cnt:>6} ({pct:>5.1f}%) {bar}")

    # How many queries have weight_t > 0.9?
    high_caption = (w_t > 0.9).sum()
    low_caption = (w_t < 0.5).sum()
    print(f"\nQueries with caption weight > 0.9: {high_caption}/{len(w_t)} ({100*high_caption/len(w_t):.1f}%)")
    print(f"Queries with caption weight < 0.5: {low_caption}/{len(w_t)} ({100*low_caption/len(w_t):.1f}%)")

    # Dominant modality per query
    dominant = np.argmax(all_weights.numpy(), axis=1)
    for mod_name, mod_idx in [("Visual", 0), ("Caption", 1), ("Audio", 2)]:
        cnt = (dominant == mod_idx).sum()
        print(f"  {mod_name} dominant: {cnt}/{len(dominant)} ({100*cnt/len(dominant):.1f}%)")

    # ======================================================================
    # INVESTIGATION 3: Per-query retrieval breakdowns
    # ======================================================================
    print("\n" + "=" * 70)
    print("INVESTIGATION 3: PER-QUERY RETRIEVAL BREAKDOWN")
    print("=" * 70)

    # Compute sim_gated
    sim_gated_list = []
    for i in range(0, len(test_texts_orig), 256):
        q = text_embeds[i:i+256].float().to(DEVICE)
        with torch.no_grad():
            w = gate(q).cpu()
        gated = (w[:, 0:1] * sim_v[i:i+256].cpu() +
                 w[:, 1:2] * sim_t[i:i+256].cpu() +
                 w[:, 2:3] * sim_a[i:i+256].cpu())
        sim_gated_list.append(gated)
    sim_gated = torch.cat(sim_gated_list, dim=0)

    def compute_per_query(sim_matrix, test_vids, common_vids):
        """Returns array of booleans: did R@1 succeed for each query?"""
        results = []
        for i in range(sim_matrix.shape[0]):
            gt_video = test_vids[i]
            ranked_indices = torch.argsort(sim_matrix[i], descending=True)
            top1_vid = common_vids[ranked_indices[0].item()]
            results.append(gt_video == top1_vid)
        return np.array(results)

    success_v = compute_per_query(sim_v, test_vids_orig, common_original)
    success_t = compute_per_query(sim_t, test_vids_orig, common_original)
    success_a = compute_per_query(sim_a, test_vids_orig, common_original)
    success_gated = compute_per_query(sim_gated, test_vids_orig, common_original)

    print(f"\nPer-query R@1 success rates:")
    print(f"  Visual only:   {success_v.mean():.4f}")
    print(f"  Caption only:  {success_t.mean():.4f}")
    print(f"  Audio only:    {success_a.mean():.4f}")
    print(f"  Adaptive:      {success_gated.mean():.4f}")

    # Contingency table: which combinations succeed/fail?
    # Categories:
    # - None succeed
    # - Only visual
    # - Only caption
    # - Only audio
    # - Visual + Caption
    # - Visual + Audio
    # - Caption + Audio
    # - All three
    patterns = {}
    for i in range(len(success_v)):
        key = (bool(success_v[i]), bool(success_t[i]), bool(success_a[i]))
        patterns[key] = patterns.get(key, 0) + 1

    print(f"\nSuccess pattern contingency table:")
    print(f"  {'V':<6} {'T':<6} {'A':<6} {'Count':<8} {'Pct':<8} {'Label'}")
    print(f"  {'-'*60}")
    for (v, t, a), cnt in sorted(patterns.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / len(success_v)
        labels = []
        if v: labels.append("Visual")
        if t: labels.append("Caption")
        if a: labels.append("Audio")
        label = "+".join(labels) if labels else "NONE"
        print(f"  {str(v):<6} {str(t):<6} {str(a):<6} {cnt:<8} {pct:<8.2f} {label}")

    # How many queries where gate succeeds but all individual modalities fail?
    only_gate_successes = (~success_v) & (~success_t) & (~success_a) & success_gated
    print(f"\nGate succeeds where ALL individual modalities fail: {only_gate_successes.sum()}/{len(success_gated)} ({100*only_gate_successes.sum()/len(success_gated):.2f}%)")

    # How many queries where gate fails but at least one modality succeeds?
    gate_fails_but_modality_succeeds = ~success_gated & (success_v | success_t | success_a)
    print(f"Gate fails but at least one modality succeeds: {gate_fails_but_modality_succeeds.sum()}/{len(success_gated)} ({100*gate_fails_but_modality_succeeds.sum()/len(success_gated):.2f}%)")

    # ======================================================================
    # INVESTIGATION 4: Adaptive vs Visual+Caption equal-fusion
    # ======================================================================
    print("\n" + "=" * 70)
    print("INVESTIGATION 4: ADAPTIVE GATING vs VISUAL+CAPTION EQUAL FUSION")
    print("=" * 70)

    # V+C equal fusion (2 modalities, no audio)
    sim_vc_eq = (sim_v + sim_t) / 2
    success_vc_eq = compute_per_query(sim_vc_eq, test_vids_orig, common_original)
    success_eq3 = compute_per_query((sim_v + sim_t + sim_a) / 3, test_vids_orig, common_original)

    print(f"\nSystem R@1 comparison:")
    print(f"  Visual only:          {success_v.mean():.4f}")
    print(f"  Caption only:         {success_t.mean():.4f}")
    print(f"  V+C equal fusion:     {success_vc_eq.mean():.4f}")
    print(f"  V+T+A equal fusion:   {success_eq3.mean():.4f}")
    print(f"  Adaptive gating:      {success_gated.mean():.4f}")

    # Per-query rank comparison between V+C equal fusion and adaptive
    def get_rank(sim_matrix, test_vids, common_vids):
        """Returns the rank (0-indexed) of the correct video for each query."""
        ranks = []
        for i in range(sim_matrix.shape[0]):
            gt_video = test_vids[i]
            ranked_indices = torch.argsort(sim_matrix[i], descending=True)
            rank = (ranked_indices.tolist().index(
                list(common_vids).index(gt_video)
            ))
            ranks.append(rank)
        return np.array(ranks)

    rank_vc_eq = get_rank(sim_vc_eq, test_vids_orig, common_original)
    rank_gated = get_rank(sim_gated, test_vids_orig, common_original)

    improved = (rank_gated < rank_vc_eq).sum()
    unchanged = (rank_gated == rank_vc_eq).sum()
    degraded = (rank_gated > rank_vc_eq).sum()

    print(f"\nAdaptive vs V+C equal fusion (rank comparison):")
    print(f"  Improved (lower rank):  {improved:>6} ({100*improved/len(rank_vc_eq):.2f}%)")
    print(f"  Unchanged (same rank):  {unchanged:>6} ({100*unchanged/len(rank_vc_eq):.2f}%)")
    print(f"  Degraded (higher rank): {degraded:>6} ({100*degraded/len(rank_vc_eq):.2f}%)")

    # Mean rank comparison
    print(f"\n  Mean rank (V+C equal):  {np.mean(rank_vc_eq):.2f}")
    print(f"  Mean rank (Adaptive):   {np.mean(rank_gated):.2f}")
    print(f"  Median rank (V+C equal): {np.median(rank_vc_eq):.0f}")
    print(f"  Median rank (Adaptive):  {np.median(rank_gated):.0f}")

    # Rank distribution
    for name, ranks in [("V+C equal", rank_vc_eq), ("Adaptive", rank_gated)]:
        print(f"\n  {name} rank distribution:")
        for threshold in [0, 1, 5, 10, 50, 100, 500, 1000, 2635]:
            perc = (ranks <= threshold).mean() * 100
            print(f"    Rank <= {threshold:<5}: {perc:>6.2f}%")

    # Show detailed examples where adaptive improves or degrades
    n_examples = 5
    improve_indices = np.where(rank_gated < rank_vc_eq)[0][:n_examples]
    degrade_indices = np.where(rank_gated > rank_vc_eq)[0][:n_examples]

    if len(improve_indices) > 0:
        print(f"\n  Example queries IMPROVED by adaptive (showing up to {n_examples}):")
        for idx in improve_indices:
            print(f"    Query[{idx}]: '{test_texts_orig[idx][:60]}...'")
            print(f"      V+C rank: {rank_vc_eq[idx]}, Adaptive rank: {rank_gated[idx]}")
            print(f"      Weights: V={all_weights[idx,0]:.4f} T={all_weights[idx,1]:.4f} A={all_weights[idx,2]:.4f}")

    if len(degrade_indices) > 0:
        print(f"\n  Example queries DEGRADED by adaptive (showing up to {n_examples}):")
        for idx in degrade_indices:
            print(f"    Query[{idx}]: '{test_texts_orig[idx][:60]}...'")
            print(f"      V+C rank: {rank_vc_eq[idx]}, Adaptive rank: {rank_gated[idx]}")
            print(f"      Weights: V={all_weights[idx,0]:.4f} T={all_weights[idx,1]:.4f} A={all_weights[idx,2]:.4f}")

    # ======================================================================
    # INVESTIGATION 5: Temporal transformer vs mean pooling
    # ======================================================================
    print("\n" + "=" * 70)
    print("INVESTIGATION 5: TEMPORAL TRANSFORMER vs MEAN POOLING")
    print("=" * 70)

    if len(common_transformer) == 0:
        print("No overlap between transformer embeddings and test split. Skipping.")
    else:
        # Build transformer video matrix
        trans_m = torch.stack([torch.as_tensor(transformer_db[v]).float() for v in common_transformer])
        trans_m = torch.nn.functional.normalize(trans_m, p=2, dim=1)
        
        # Also build mean-pool matrix for same video set
        video_db_orig = torch.load(args.video_embeds, weights_only=False)
        mean_m = torch.stack([torch.as_tensor(video_db_orig[v]).float() for v in common_transformer])
        mean_m = torch.nn.functional.normalize(mean_m, p=2, dim=1)
        del video_db_orig
        gc.collect()

        # Compute CLIP text embeddings for transformer query set
        text_embeds_trans = []
        for i in range(0, len(test_texts_trans), 256):
            batch = test_texts_trans[i:i+256]
            tokens = clip.tokenize(batch).to(DEVICE)
            with torch.no_grad():
                emb = clip_model.encode_text(tokens).float()
                emb = emb / emb.norm(dim=1, keepdim=True)
            text_embeds_trans.append(emb.cpu())
        text_embeds_trans = torch.cat(text_embeds_trans, dim=0)

        sim_trans = text_embeds_trans @ trans_m.T
        sim_mean = text_embeds_trans @ mean_m.T

        metrics_trans = evaluate_retrieval(sim_trans, test_vids_trans, common_transformer, ks=[1, 5, 10])
        metrics_mean = evaluate_retrieval(sim_mean, test_vids_trans, common_transformer, ks=[1, 5, 10])

        print(f"\nDirect visual-only retrieval comparison (same video set):")
        print(f"  {'System':<25} {'R@1':<8} {'R@5':<8} {'R@10':<8}")
        print(f"  {'-'*50}")
        print(f"  {'Mean pool (original)':<25} {metrics_mean['R@1']:<8.4f} {metrics_mean['R@5']:<8.4f} {metrics_mean['R@10']:<8.4f}")
        print(f"  {'Temporal Transformer':<25} {metrics_trans['R@1']:<8.4f} {metrics_trans['R@5']:<8.4f} {metrics_trans['R@10']:<8.4f}")

        # Per-video embedding similarity analysis
        common_vids_set = set(common_transformer)
        overlap_vids = [v for v in common_transformer if v in common_vids_set]
        print(f"\nEmbedding similarity analysis ({len(overlap_vids)} videos):")
        
        pair_sims = []
        for v in overlap_vids:
            e1 = torch.as_tensor(transformer_db[v]).float()
            e2 = torch.as_tensor(torch.load(args.video_embeds, weights_only=False)[v]).float()
            e1 = e1 / e1.norm()
            e2 = e2 / e2.norm()
            pair_sims.append((e1 @ e2).item())
        pair_sims = np.array(pair_sims)

        print(f"  Transformer vs Mean pool cosine similarity:")
        print(f"    Mean: {pair_sims.mean():.4f}")
        print(f"    Std:  {pair_sims.std():.4f}")
        print(f"    Min:  {pair_sims.min():.4f}")
        print(f"    Max:  {pair_sims.max():.4f}")

        # Distribution
        for low, high, label in [(0.0, 0.5, "Very different (<0.5)"),
                                  (0.5, 0.7, "Somewhat similar (0.5-0.7)"),
                                  (0.7, 0.9, "Similar (0.7-0.9)"),
                                  (0.9, 0.95, "Very similar (0.9-0.95)"),
                                  (0.95, 0.99, "Nearly identical (0.95-0.99)"),
                                  (0.99, 1.0, "Almost identical (>0.99)")]:
            cnt = ((pair_sims >= low) & (pair_sims < high)).sum()
            print(f"    {label:<35} {cnt} ({100*cnt/len(pair_sims):.1f}%)")

        # Per-query analysis: how often do they disagree?
        success_trans = compute_per_query(sim_trans, test_vids_trans, common_transformer)
        success_mean = compute_per_query(sim_mean, test_vids_trans, common_transformer)

        print(f"\nPer-query agreement analysis:")
        both_succeed = (success_trans & success_mean).sum()
        both_fail = (~success_trans & ~success_mean).sum()
        trans_only = (success_trans & ~success_mean).sum()
        mean_only = (~success_trans & success_mean).sum()

        print(f"  Both succeed: {both_succeed} ({100*both_succeed/len(success_trans):.1f}%)")
        print(f"  Both fail:    {both_fail} ({100*both_fail/len(success_trans):.1f}%)")
        print(f"  Transformer only: {trans_only} ({100*trans_only/len(success_trans):.1f}%)")
        print(f"  Mean pool only:   {mean_only} ({100*mean_only/len(success_trans):.1f}%)")

        # Show some examples where transformer fails but mean pool succeeds
        if mean_only > 0:
            examples = np.where(~success_trans & success_mean)[0][:5]
            print(f"\n  Examples where Mean Pool succeeds but Transformer fails:")
            for idx in examples:
                rank_t = get_rank(sim_trans, test_vids_trans, common_transformer)[idx]
                rank_m = get_rank(sim_mean, test_vids_trans, common_transformer)[idx]
                print(f"    Query: '{test_texts_trans[idx][:60]}...'")
                print(f"      Transformer rank: {rank_t}, Mean Pool rank: {rank_m}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Full evaluation metrics for reporting
    metrics_v = evaluate_retrieval(sim_v, test_vids_orig, common_original, ks=[1, 5, 10])
    metrics_t = evaluate_retrieval(sim_t, test_vids_orig, common_original, ks=[1, 5, 10])
    metrics_a = evaluate_retrieval(sim_a, test_vids_orig, common_original, ks=[1, 5, 10])
    metrics_eq3 = evaluate_retrieval((sim_v + sim_t + sim_a) / 3, test_vids_orig, common_original, ks=[1, 5, 10])
    metrics_vc2 = evaluate_retrieval(sim_vc_eq, test_vids_orig, common_original, ks=[1, 5, 10])
    metrics_gated = evaluate_retrieval(sim_gated, test_vids_orig, common_original, ks=[1, 5, 10])

    csv_path = "outputs/diagnostics/diagnostic_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Visual", "Caption", "Audio", "Equal(V+T+A)", "Equal(V+C)", "Adaptive"])
        w.writerow(["R@1", f"{metrics_v['R@1']:.4f}", f"{metrics_t['R@1']:.4f}", f"{metrics_a['R@1']:.4f}",
                    f"{metrics_eq3['R@1']:.4f}", f"{metrics_vc2['R@1']:.4f}", f"{metrics_gated['R@1']:.4f}"])
        w.writerow(["R@5", f"{metrics_v['R@5']:.4f}", f"{metrics_t['R@5']:.4f}", f"{metrics_a['R@5']:.4f}",
                    f"{metrics_eq3['R@5']:.4f}", f"{metrics_vc2['R@5']:.4f}", f"{metrics_gated['R@5']:.4f}"])
        w.writerow(["R@10", f"{metrics_v['R@10']:.4f}", f"{metrics_t['R@10']:.4f}", f"{metrics_a['R@10']:.4f}",
                    f"{metrics_eq3['R@10']:.4f}", f"{metrics_vc2['R@10']:.4f}", f"{metrics_gated['R@10']:.4f}"])
    print(f"\nSaved: {csv_path}")

    print(f"\nFinal metrics:")
    print(f"  {'System':<25} {'R@1':<8} {'R@5':<8} {'R@10':<8}")
    print(f"  {'-'*50}")
    for name, m in [("Visual", metrics_v), ("Caption", metrics_t), ("Audio", metrics_a),
                    ("Equal (V+T+A)", metrics_eq3), ("Equal (V+C)", metrics_vc2),
                    ("Adaptive", metrics_gated)]:
        print(f"  {name:<25} {m['R@1']:<8.4f} {m['R@5']:<8.4f} {m['R@10']:<8.4f}")

if __name__ == "__main__":
    main()
