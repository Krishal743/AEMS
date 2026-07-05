"""
Quick evaluation of the LOO-retrained gating network.
Uses the leave-one-out caption protocol on the full test set.
"""
import sys, json, torch, argparse, os, gc
import clip
import numpy as np
from collections import defaultdict
from src.models.gating_network import GatingNetwork
from src.encoders.clap_encode import CLAPEncoder
from src.evaluation.evaluate_retrieval import evaluate_retrieval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_caption_index(metadata, split="test"):
    split_data = [m for m in metadata if m["split"] == split]
    video_captions = defaultdict(list)
    for item in split_data:
        video_captions[item["video_id"]].append(item["text"])
    index_map = {}
    for vid, captions in video_captions.items():
        for i, text in enumerate(captions):
            index_map[(vid, text)] = i
    return index_map

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()

def per_query_success(sim, vids, common_vids):
    successes = []
    for i in range(len(vids)):
        gt = vids[i]
        ranked = torch.argsort(sim[i], descending=True)
        top1 = common_vids[ranked[0].item()]
        successes.append(gt == top1)
    return np.array(successes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds-test", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_loo.pth")
    parser.add_argument("--metadata", default="data/processed/metadata/msrvtt_metadata.json")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading metadata...")
    with open(args.metadata) as f:
        metadata = json.load(f)
    test_items = [m for m in metadata if m["split"] == "test"]
    test_video_ids = set(m["video_id"] for m in test_items)

    caption_index = build_caption_index(metadata, "test")
    print(f"Caption index entries: {len(caption_index)}")

    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_test_db = torch.load(args.caption_embeds_test, weights_only=False)

    common = [v for v in video_db if v in audio_db and v in caption_test_db and v in test_video_ids]
    print(f"Common videos: {len(common)}")

    test_items_filtered = [m for m in test_items if m["video_id"] in common]
    test_texts = [m["text"] for m in test_items_filtered]
    test_vids = [m["video_id"] for m in test_items_filtered]
    print(f"Test queries: {len(test_texts)}")

    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common])
    aud_m = torch.stack([torch.as_tensor(audio_db[v]).float() for v in common])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    aud_m = torch.nn.functional.normalize(aud_m, p=2, dim=1)
    del video_db, audio_db
    gc.collect()

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

    # Leave-one-out caption similarity
    sim_t_loo_list = []
    for vid in common:
        cap_embeds = torch.as_tensor(caption_test_db[vid]).float()
        cap_embeds = torch.nn.functional.normalize(cap_embeds, p=2, dim=1).to(DEVICE)
        sims = text_embeds.to(DEVICE) @ cap_embeds.T
        max_sims = sims.max(dim=1).values.clone()
        query_indices = [i for i, v in enumerate(test_vids) if v == vid]
        for qi in query_indices:
            cap_idx = caption_index.get((vid, test_texts[qi]))
            if cap_idx is not None:
                row = sims[qi].clone()
                row[cap_idx] = -float('inf')
                max_sims[qi] = row.max()
        sim_t_loo_list.append(max_sims.cpu())
        del cap_embeds, sims
        gc.collect()
    sim_t_loo = torch.stack(sim_t_loo_list, dim=1)
    print(f"sim_t_loo: {sim_t_loo.shape}")
    del caption_test_db
    gc.collect()

    # CLAP audio
    clap_encoder = CLAPEncoder(device=DEVICE)
    clap_list = []
    for i in range(0, len(test_texts), 256):
        batch = test_texts[i:i+256]
        emb = clap_encoder.encode_text(batch)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        emb = emb / emb.norm(dim=1, keepdim=True)
        clap_list.append(emb.cpu())
    clap_text_embeds = torch.cat(clap_list, dim=0)
    sim_a = clap_text_embeds @ aud_m.T
    print(f"sim_a: {sim_a.shape}")
    del clap_encoder, clap_text_embeds
    gc.collect()

    # Load LOO-trained gate
    gate = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE, weights_only=False))
    gate.eval()

    print("\n--- GATE WEIGHTS (LOO-trained) ---")
    with torch.no_grad():
        w_all = []
        for i in range(0, len(test_texts), 256):
            q = text_embeds[i:i+256].float().to(DEVICE)
            w = gate(q).cpu()
            w_all.append(w)
        w_all = torch.cat(w_all, dim=0)
        for name, idx in [("Visual", 0), ("Caption", 1), ("Audio", 2)]:
            w = w_all[:, idx]
            print(f"  {name:<12} mean={w.mean():.6f}  std={w.std():.6f}  min={w.min():.6f}  max={w.max():.6f}")

    # Simulated gated similarity
    sim_gated_loo_list = []
    for i in range(0, len(test_texts), 256):
        q = text_embeds[i:i+256].float().to(DEVICE)
        with torch.no_grad():
            w = gate(q).cpu()
        gated = (w[:, 0:1] * sim_v[i:i+256].cpu() +
                 w[:, 1:2] * sim_t_loo[i:i+256].cpu() +
                 w[:, 2:3] * sim_a[i:i+256].cpu())
        sim_gated_loo_list.append(gated)
    sim_gated_loo = torch.cat(sim_gated_loo_list, dim=0)

    # Evaluate
    sim_eq_vc_loo = (sim_v + sim_t_loo) / 2
    sim_eq_vta_loo = (sim_v + sim_t_loo + sim_a) / 3

    from src.evaluation.evaluate_retrieval import evaluate_retrieval

    systems = {
        "Visual only": sim_v,
        "Caption only (LOO)": sim_t_loo,
        "Audio only": sim_a,
        "Equal V+C (LOO)": sim_eq_vc_loo,
        "Equal V+T+A (LOO)": sim_eq_vta_loo,
        "AdaGate-LOO (LOO-trained)": sim_gated_loo,
    }

    print(f"\n{'System':<35} {'R@1':<10} {'R@5':<10} {'R@10':<10}")
    print(f"{'-'*65}")
    for name, sim in systems.items():
        metrics = evaluate_retrieval(sim, test_vids, common, ks=[1, 5, 10])
        print(f"{name:<35} {metrics['R@1']:<10.4f} {metrics['R@5']:<10.4f} {metrics['R@10']:<10.4f}")

    # Compare AdaGate-LOO vs Equal V+C under LOO
    s_gated = per_query_success(sim_gated_loo, test_vids, common)
    s_eq_vc = per_query_success(sim_eq_vc_loo, test_vids, common)
    s_v = per_query_success(sim_v, test_vids, common)
    s_t = per_query_success(sim_t_loo, test_vids, common)

    improved = (s_gated & ~s_eq_vc).sum()
    degraded = (~s_gated & s_eq_vc).sum()
    unchanged = (s_gated == s_eq_vc).sum()
    total = len(s_gated)
    print(f"\nAdaGate-LOO vs Equal V+C (LOO):")
    print(f"  Improved:   {improved:>6} ({100*improved/total:.2f}%)")
    print(f"  Unchanged:  {unchanged:>6} ({100*unchanged/total:.2f}%)")
    print(f"  Degraded:   {degraded:>6} ({100*degraded/total:.2f}%)")

    # Does AdaGate-LOO beat both individual modalities?
    both_succeed = (s_eq_vc & s_gated).sum()
    nos_t = per_query_success(sim_t_loo, test_vids, common)
    print(f"\n  AdaGate-LOO > Visual?   {s_gated.mean() > s_v.mean()}")
    print(f"  AdaGate-LOO > Caption?  {s_gated.mean() > s_t.mean()}")
    print(f"  AdaGate-LOO > Equal V+C? {s_gated.mean() > s_eq_vc.mean()}")

if __name__ == "__main__":
    main()
