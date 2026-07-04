"""Final unified evaluation — all systems, identical conditions."""

import sys, json, torch, argparse, os, csv, gc, time
import clip
from src.models.gating_network import GatingNetwork
from src.evaluation.evaluate_retrieval import evaluate_retrieval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs/final_eval"


def measure_latency(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    return (time.perf_counter() - start) / runs


def main():
    parser = argparse.ArgumentParser(description="Final unified evaluation")
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds-test", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_meanpool.pth")
    parser.add_argument("--metadata", default="data/processed/metadata/msrvtt_metadata.json")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(args.metadata) as f:
        metadata = json.load(f)
    test_items = [m for m in metadata if m["split"] == "test"]

    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_test_db = torch.load(args.caption_embeds_test, weights_only=False)

    test_video_ids = set(m["video_id"] for m in test_items)
    common = [v for v in video_db if v in audio_db and v in caption_test_db and v in test_video_ids]

    test_items_filtered = [m for m in test_items if m["video_id"] in common]
    test_texts = [m["text"] for m in test_items_filtered]
    test_vids = [m["video_id"] for m in test_items_filtered]

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

    # sim_a temporarily uses CLIP text embeddings (Fix 4 will change this to CLAP)
    sim_a = text_embeds @ aud_m.T

    gate = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE))
    gate.eval()

    sim_gated_list = []
    for i in range(0, len(test_texts), 256):
        q = text_embeds[i:i+256].float().to(DEVICE)
        with torch.no_grad():
            w = gate(q).cpu()
        gated = (w[:, 0:1] * sim_v[i:i+256].cpu() +
                 w[:, 1:2] * sim_t[i:i+256].cpu() +
                 w[:, 2:3] * sim_a[i:i+256].cpu())
        sim_gated_list.append(gated)
    sim_gated = torch.cat(sim_gated_list, dim=0)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    systems = {
        "Visual only": sim_v,
        "Caption only": sim_t,
        "Audio only": sim_a,
        "Equal fusion": (sim_v + sim_t + sim_a) / 3,
        "Adaptive gating": sim_gated,
    }

    csv_path = os.path.join(OUTPUT_DIR, "final_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["System", "R@1", "R@5", "R@10"])
        for name, sim in systems.items():
            metrics = evaluate_retrieval(sim, test_vids, common, ks=[1, 5, 10])
            w.writerow([name, f"{metrics['R@1']:.4f}", f"{metrics['R@5']:.4f}", f"{metrics['R@10']:.4f}"])
            print(f"  {name:>20}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")

    print(f"\nSaved: {csv_path}")

    mem_before = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    print(f"  GPU memory: {mem_before:.2f} GB")
    print(f"  Candidate videos: {len(common)}")
    print(f"  Test queries: {len(test_texts)}")


if __name__ == "__main__":
    main()
