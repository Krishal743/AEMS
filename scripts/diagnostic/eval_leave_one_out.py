"""
Leave-one-out evaluation: prevent exact caption matching between queries and gallery.
For each query, its own caption embedding is excluded from its video's caption set,
forcing genuine cross-modal retrieval.
"""
import sys, json, torch, argparse, os, csv, gc
import clip
import numpy as np
from collections import defaultdict
from src.models.gating_network import GatingNetwork
from src.encoders.clap_encode import CLAPEncoder
from src.evaluation.evaluate_retrieval import evaluate_retrieval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()

def build_caption_index(metadata, split="test"):
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

def main():
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation")
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds-test", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_meanpool.pth")
    parser.add_argument("--metadata", default="data/processed/metadata/msrvtt_metadata.json")
    args = parser.parse_args()

    os.makedirs("outputs/diagnostics", exist_ok=True)

    print("=" * 70)
    print("LEAVE-ONE-OUT EVALUATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    with open(args.metadata) as f:
        metadata = json.load(f)
    test_items = [m for m in metadata if m["split"] == "test"]
    test_video_ids = set(m["video_id"] for m in test_items)

    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_test_db = torch.load(args.caption_embeds_test, weights_only=False)

    common = [v for v in video_db if v in audio_db and v in caption_test_db and v in test_video_ids]
    test_items_filtered = [m for m in test_items if m["video_id"] in common]
    test_texts = [m["text"] for m in test_items_filtered]
    test_vids = [m["video_id"] for m in test_items_filtered]
    print(f"Common videos: {len(common)}")
    print(f"Test queries:  {len(test_texts)}")

    # Build per-query caption index mapping
    caption_index = build_caption_index(metadata, "test")
    print(f"Caption index entries: {len(caption_index)}")

    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common])
    aud_m = torch.stack([torch.as_tensor(audio_db[v]).float() for v in common])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    aud_m = torch.nn.functional.normalize(aud_m, p=2, dim=1)
    del video_db, audio_db
    gc.collect()

    # ------------------------------------------------------------------
    # CLIP TEXT ENCODING
    # ------------------------------------------------------------------
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    text_embeds = []
    for i in range(0, len(test_texts), 256):
        batch = test_texts[i:i+256]
        tokens = clip.tokenize(batch).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=1, keepdim=True)
        text_embeds.append(emb.cpu())
    text_embeds = torch.cat(text_embeds, dim=0)

    sim_v = text_embeds @ vid_m.T
    print(f"sim_v: {sim_v.shape}")

    # ------------------------------------------------------------------
    # LEAVE-ONE-OUT CAPTION SIMILARITY
    # ------------------------------------------------------------------
    sim_t_loo_list = []
    sim_t_std_list = []
    for v_idx, vid in enumerate(common):
        cap_embeds = torch.as_tensor(caption_test_db[vid]).float()
        cap_embeds = torch.nn.functional.normalize(cap_embeds, p=2, dim=1).to(DEVICE)
        sims = text_embeds.to(DEVICE) @ cap_embeds.T

        # Standard: max over all captions
        max_sims_std = sims.max(dim=1).values
        sim_t_std_list.append(max_sims_std.cpu())

        # Leave-one-out: mask out query's own caption for queries belonging to this video
        max_sims_loo = sims.max(dim=1).values.clone()  # default: std max
        # Find queries whose ground truth is this video
        query_indices_in_video = [i for i, v in enumerate(test_vids) if v == vid]
        for qi in query_indices_in_video:
            text = test_texts[qi]
            cap_idx = caption_index.get((vid, text))
            if cap_idx is not None:
                # Replace the query's own caption similarity with the next-best
                row = sims[qi].clone()
                row[cap_idx] = -float('inf')
                max_sims_loo[qi] = row.max()
        sim_t_loo_list.append(max_sims_loo.cpu())

        del cap_embeds, sims
        gc.collect()

    sim_t_std = torch.stack(sim_t_std_list, dim=1)
    sim_t_loo = torch.stack(sim_t_loo_list, dim=1)
    print(f"sim_t_std:  {sim_t_std.shape}")
    print(f"sim_t_loo:  {sim_t_loo.shape}")

    del caption_test_db
    gc.collect()

    # ------------------------------------------------------------------
    # CLAP AUDIO SIMILARITY
    # ------------------------------------------------------------------
    clap_encoder = CLAPEncoder(device=DEVICE)
    clap_text_list = []
    for i in range(0, len(test_texts), 256):
        batch = test_texts[i:i+256]
        emb = clap_encoder.encode_text(batch)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        emb = emb / emb.norm(dim=1, keepdim=True)
        clap_text_list.append(emb.cpu())
    clap_text_embeds = torch.cat(clap_text_list, dim=0)
    sim_a = clap_text_embeds @ aud_m.T
    print(f"sim_a: {sim_a.shape}")
    del clap_encoder, clap_text_embeds
    gc.collect()

    # ------------------------------------------------------------------
    # GATING NETWORK
    # ------------------------------------------------------------------
    gate = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE, weights_only=False), strict=False)
    gate.eval()

    sim_gated_std_list = []
    sim_gated_loo_list = []
    for i in range(0, len(test_texts), 256):
        q = text_embeds[i:i+256].float().to(DEVICE)
        with torch.no_grad():
            w = gate(q).cpu()
        gated_std = (w[:, 0:1] * sim_v[i:i+256].cpu() +
                     w[:, 1:2] * sim_t_std[i:i+256].cpu() +
                     w[:, 2:3] * sim_a[i:i+256].cpu())
        gated_loo = (w[:, 0:1] * sim_v[i:i+256].cpu() +
                     w[:, 1:2] * sim_t_loo[i:i+256].cpu() +
                     w[:, 2:3] * sim_a[i:i+256].cpu())
        sim_gated_std_list.append(gated_std)
        sim_gated_loo_list.append(gated_loo)
    sim_gated_std = torch.cat(sim_gated_std_list, dim=0)
    sim_gated_loo = torch.cat(sim_gated_loo_list, dim=0)

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    sim_eq_vc_std = (sim_v + sim_t_std) / 2
    sim_eq_vc_loo = (sim_v + sim_t_loo) / 2
    sim_eq_vta_std = (sim_v + sim_t_std + sim_a) / 3
    sim_eq_vta_loo = (sim_v + sim_t_loo + sim_a) / 3

    systems = {
        "Visual only": sim_v,
        "Caption only (standard)": sim_t_std,
        "Caption only (leave-one-out)": sim_t_loo,
        "Audio only": sim_a,
        "Equal V+C (standard)": sim_eq_vc_std,
        "Equal V+C (leave-one-out)": sim_eq_vc_loo,
        "Equal V+T+A (standard)": sim_eq_vta_std,
        "Equal V+T+A (leave-one-out)": sim_eq_vta_loo,
        "Adaptive gating (standard)": sim_gated_std,
        "Adaptive gating (leave-one-out)": sim_gated_loo,
    }

    csv_path = "outputs/diagnostics/leave_one_out_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["System", "R@1", "R@5", "R@10"])
        print(f"\n{'System':<40} {'R@1':<10} {'R@5':<10} {'R@10':<10}")
        print(f"{'-'*70}")
        for name, sim in systems.items():
            metrics = evaluate_retrieval(sim, test_vids, common, ks=[1, 5, 10])
            w.writerow([name, f"{metrics['R@1']:.4f}", f"{metrics['R@5']:.4f}", f"{metrics['R@10']:.4f}"])
            print(f"{name:<40} {metrics['R@1']:<10.4f} {metrics['R@5']:<10.4f} {metrics['R@10']:<10.4f}")

    print(f"\nSaved: {csv_path}")

    # ------------------------------------------------------------------
    # DIAGNOSTIC: How much does caption retrieval drop without exact match?
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: STANDARD vs LEAVE-ONE-OUT GAP")
    print("=" * 70)

    def per_query_success(sim, vids, common_vids):
        successes = []
        for i in range(len(vids)):
            gt = vids[i]
            ranked = torch.argsort(sim[i], descending=True)
            top1 = common_vids[ranked[0].item()]
            successes.append(gt == top1)
        return np.array(successes)

    s_std = per_query_success(sim_t_std, test_vids, common)
    s_loo = per_query_success(sim_t_loo, test_vids, common)

    both_succeed = (s_std & s_loo).sum()
    both_fail = (~s_std & ~s_loo).sum()
    std_only = (s_std & ~s_loo).sum()
    loo_only = (~s_std & s_loo).sum()

    total = len(s_std)
    print(f"\nCaption retrieval: Standard vs Leave-one-out per-query agreement")
    print(f"  Both succeed:           {both_succeed:>6} ({100*both_succeed/total:.2f}%)")
    print(f"  Both fail:              {both_fail:>6} ({100*both_fail/total:.2f}%)")
    print(f"  Standard only (exact match carried): {std_only:>6} ({100*std_only/total:.2f}%)")
    print(f"  Leave-one-out only:     {loo_only:>6} ({100*loo_only/total:.2f}%)")

    # How much of the 92.7% standard caption R@1 is explained by exact matching?
    print(f"\n  Caption R@1 (standard):      {s_std.mean():.4f}")
    print(f"  Caption R@1 (leave-one-out): {s_loo.mean():.4f}")
    print(f"  Drop due to exact matching:   {s_std.mean() - s_loo.mean():.4f}")
    print(f"  Fraction of success from exact match: {(std_only)/(s_std.sum()):.4f}")

    # Compare adaptive gating under leave-one-out
    s_gated_std = per_query_success(sim_gated_std, test_vids, common)
    s_gated_loo = per_query_success(sim_gated_loo, test_vids, common)

    s_eq_vc_std = per_query_success(sim_eq_vc_std, test_vids, common)
    s_eq_vc_loo = per_query_success(sim_eq_vc_loo, test_vids, common)
    s_eq_vta_std = per_query_success(sim_eq_vta_std, test_vids, common)
    s_eq_vta_loo = per_query_success(sim_eq_vta_loo, test_vids, common)
    s_gated_std = per_query_success(sim_gated_std, test_vids, common)
    s_gated_loo = per_query_success(sim_gated_loo, test_vids, common)
    s_v = per_query_success(sim_v, test_vids, common)
    s_t_std = per_query_success(sim_t_std, test_vids, common)
    s_t_loo = per_query_success(sim_t_loo, test_vids, common)
    s_a = per_query_success(sim_a, test_vids, common)

    print(f"\n{'System':<40} {'Standard':<10} {'Leave-One-Out':<10}")
    print(f"{'-'*60}")
    print(f"{'Visual only':<40} {s_v.mean():<10.4f} {s_v.mean():<10.4f}")
    print(f"{'Caption only':<40} {s_t_std.mean():<10.4f} {s_t_loo.mean():<10.4f}")
    print(f"{'Audio only':<40} {s_a.mean():<10.4f} {s_a.mean():<10.4f}")
    print(f"{'Equal V+C':<40} {s_eq_vc_std.mean():<10.4f} {s_eq_vc_loo.mean():<10.4f}")
    print(f"{'Equal V+T+A':<40} {s_eq_vta_std.mean():<10.4f} {s_eq_vta_loo.mean():<10.4f}")
    print(f"{'Adaptive gating':<40} {s_gated_std.mean():<10.4f} {s_gated_loo.mean():<10.4f}")

    # Under LOO: does V+C equal fusion beat both individual modalities?
    print(f"\n--- LOO Fusion Analysis ---")
    vc_beats_c = (s_eq_vc_loo & ~s_t_loo).sum()
    vc_beats_v = (s_eq_vc_loo & ~s_v).sum()
    vc_beats_both = (s_eq_vc_loo & ~s_v & ~s_t_loo).sum()
    vc_loses_to_both = (~s_eq_vc_loo & s_v & s_t_loo).sum()
    print(f"  V+C succeeds where caption fails:  {vc_beats_c:>6} ({100*vc_beats_c/total:.2f}%)")
    print(f"  V+C succeeds where visual fails:    {vc_beats_v:>6} ({100*vc_beats_v/total:.2f}%)")
    print(f"  V+C succeeds where both fail:       {vc_beats_both:>6} ({100*vc_beats_both/total:.2f}%)")
    print(f"  V+C fails where both succeed:       {vc_loses_to_both:>6} ({100*vc_loses_to_both/total:.2f}%)")
    print(f"  V+C beats both individually:        {s_eq_vc_loo.mean() > max(s_v.mean(), s_t_loo.mean())}")

if __name__ == "__main__":
    main()
