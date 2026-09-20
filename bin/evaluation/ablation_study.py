"""Ablation study: remove each modality, compare equal vs adaptive fusion."""

import sys, json, torch, argparse, os, csv, gc
import clip
import numpy as np
from src.models.gating_network import GatingNetwork
from src.encoders.clap_encode import CLAPEncoder
from src.evaluation.evaluate_retrieval import evaluate_retrieval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs/ablations"


def eval_system(name, sim_matrix, query_vids, candidate_vids):
    metrics = evaluate_retrieval(sim_matrix, query_vids, candidate_vids, ks=[1, 5, 10])
    print(f"  {name:>30}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds-test", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_meanpool.pth")
    parser.add_argument("--metadata", default="data/processed/metadata/msrvtt_metadata.json")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[DATA] Loading metadata...")
    with open(args.metadata) as f:
        metadata = json.load(f)
    test_items = [m for m in metadata if m["split"] == "test"]
    print(f"  Test items: {len(test_items)}")

    print("[EMB] Loading embeddings...")
    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_test_db = torch.load(args.caption_embeds_test, weights_only=False)

    test_video_ids = set(m["video_id"] for m in test_items)
    common = [v for v in video_db if v in audio_db and v in caption_test_db and v in test_video_ids]
    print(f"  Common test videos: {len(common)}")

    test_items_filtered = [m for m in test_items if m["video_id"] in common]
    test_texts = [m["text"] for m in test_items_filtered]
    test_vids = [m["video_id"] for m in test_items_filtered]
    print(f"  Test queries: {len(test_texts)}")

    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common])
    aud_m = torch.stack([torch.as_tensor(audio_db[v]).float() for v in common])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    aud_m = torch.nn.functional.normalize(aud_m, p=2, dim=1)

    del video_db, audio_db
    gc.collect()

    print("[TEXT] Encoding test queries...")
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
        del emb, tokens
        gc.collect()
    text_embeds = torch.cat(text_embeds, dim=0)
    print(f"  Text embeds: {text_embeds.shape}")

    print("[EVAL] Computing similarities...")
    sim_v = text_embeds @ vid_m.T

    # Fix: max similarity over individual captions instead of chimeric pooling
    sim_t_list = []
    for vid in common:
        cap_embeds = torch.as_tensor(caption_test_db[vid]).float()
        cap_embeds = torch.nn.functional.normalize(cap_embeds, p=2, dim=1)
        cap_embeds = cap_embeds.to(DEVICE)
        sims = text_embeds.to(DEVICE) @ cap_embeds.T
        max_sims = sims.max(dim=1).values
        sim_t_list.append(max_sims.cpu())
    sim_t = torch.stack(sim_t_list, dim=1)

    # Fix: Use CLAP text encoder for audio similarity
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

    del caption_test_db
    gc.collect()

    print("\n[ABLATION] Results:\n")
    results = {}

    # 1. Individual modalities
    results["visual_only"] = eval_system("Visual only", sim_v, test_vids, common)
    results["caption_only"] = eval_system("Caption only", sim_t, test_vids, common)
    results["audio_only"] = eval_system("Audio only", sim_a, test_vids, common)

    # 2. Two-modality ablations
    results["visual+caption"] = eval_system("Visual + Caption (equal)", (sim_v + sim_t) / 2, test_vids, common)
    results["visual+audio"]   = eval_system("Visual + Audio (equal)", (sim_v + sim_a) / 2, test_vids, common)
    results["caption+audio"]  = eval_system("Caption + Audio (equal)", (sim_t + sim_a) / 2, test_vids, common)

    # 3. Equal fusion (all three)
    sim_equal = (sim_v + sim_t + sim_a) / 3
    results["equal_fusion"] = eval_system("Equal fusion (all 3)", sim_equal, test_vids, common)

    # 4. Adaptive gating
    gate = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE), strict=False)
    gate.eval()

    sim_gated_list = []
    for i in range(0, len(test_texts), 256):
        q = text_embeds[i:i+256].float().to(DEVICE)
        with torch.no_grad():
            w = gate(q).cpu()
        s_v = sim_v[i:i+256].cpu()
        s_t = sim_t[i:i+256].cpu()
        s_a = sim_a[i:i+256].cpu()
        gated = (w[:, 0:1] * s_v + w[:, 1:2] * s_t + w[:, 2:3] * s_a)
        sim_gated_list.append(gated)
    sim_gated = torch.cat(sim_gated_list, dim=0)
    results["adaptive_gating"] = eval_system("Adaptive gating", sim_gated, test_vids, common)

    # 5. Without audio (renormalize)
    results["no_audio"] = eval_system("No audio (equal v+t)", (sim_v + sim_t) / 2, test_vids, common)

    sim_gated_no_a = []
    for i in range(0, len(test_texts), 256):
        q = text_embeds[i:i+256].float().to(DEVICE)
        with torch.no_grad():
            w = gate(q).cpu()
        s_v = sim_v[i:i+256].cpu()
        s_t = sim_t[i:i+256].cpu()
        w_sum = w[:, 0:1] + w[:, 1:2] + 1e-8
        w_vn = w[:, 0:1] / w_sum
        w_tn = w[:, 1:2] / w_sum
        gated = (w_vn * s_v + w_tn * s_t)
        sim_gated_no_a.append(gated)
    sim_gated_no_a = torch.cat(sim_gated_no_a, dim=0)
    results["adaptive_no_audio"] = eval_system("Adaptive no audio", sim_gated_no_a, test_vids, common)

    # Save
    csv_path = os.path.join(OUTPUT_DIR, "ablation_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["System", "R@1", "R@5", "R@10"])
        for name, m in results.items():
            w.writerow([name, f"{m['R@1']:.4f}", f"{m['R@5']:.4f}", f"{m['R@10']:.4f}"])
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
