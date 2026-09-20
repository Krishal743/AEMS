"""Behavioural verification: demonstrate gating adapts per query type."""

import sys, json, torch, argparse, csv, os
import clip
from src.models.gating_network import GatingNetwork
from src.explainability.explain_retrieval import explain_gating_decision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs/behavioural"

QUERY_SETS = {
    "visual": [
        "a red car driving on a highway",
        "sunset over mountain peaks",
        "a dog running in a park",
    ],
    "audio": [
        "a loud explosion sound",
        "a guitar solo performance",
        "someone playing drums",
    ],
    "semantic": [
        "a person explaining how to cook pasta",
        "a lecture on quantum physics",
        "a news anchor reporting the weather",
    ],
    "balanced": [
        "a person walking in a room",
        "a child playing with a toy",
        "people talking at a party",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Behavioural verification of gating network")
    parser.add_argument("--video-embeds", default="embeddings/video_embeddings.pt")
    parser.add_argument("--audio-embeds", default="embeddings/audio_embeddings.pt")
    parser.add_argument("--caption-embeds", default="embeddings/caption_embeddings_test.pt")
    parser.add_argument("--gate-weights", default="models/gating_weights_meanpool.pth")
    parser.add_argument("--metadata", default="data/processed/metadata/msrvtt_metadata.json")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    caption_db = torch.load(args.caption_embeds, weights_only=False)
    common = sorted([v for v in video_db if v in audio_db and v in caption_db])

    vid_m = torch.stack([torch.as_tensor(video_db[v]).float() for v in common])
    aud_m = torch.stack([torch.as_tensor(audio_db[v]).float() for v in common])
    cap_m = torch.stack([torch.as_tensor(caption_db[v]).float().max(dim=0)[0] for v in common])
    vid_m = torch.nn.functional.normalize(vid_m, p=2, dim=1)
    aud_m = torch.nn.functional.normalize(aud_m, p=2, dim=1)
    cap_m = torch.nn.functional.normalize(cap_m, p=2, dim=1)

    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    gate = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE), strict=False)
    gate.eval()

    rows = []
    for set_name, queries in QUERY_SETS.items():
        for query in queries:
            tokens = clip.tokenize([query]).to(DEVICE)
            with torch.no_grad():
                q_emb = clip_model.encode_text(tokens).float()
                q_emb = q_emb / q_emb.norm(dim=1, keepdim=True)
                sim_v = (q_emb @ vid_m.to(DEVICE).T).squeeze(0)
                sim_t = (q_emb @ cap_m.to(DEVICE).T).squeeze(0)
                sim_a = (q_emb @ aud_m.to(DEVICE).T).squeeze(0)
                w = gate(q_emb).cpu().squeeze(0)

            g = explain_gating_decision(w)
            rows.append({
                "query_set": set_name,
                "query": query,
                "w_v": g["w_v"],
                "w_t": g["w_t"],
                "w_a": g["w_a"],
                "dominant": g["dominant_modality"],
                "spread": g["confidence_spread"],
                "max_sim_v": sim_v.max().item(),
                "max_sim_t": sim_t.max().item(),
                "max_sim_a": sim_a.max().item(),
                "fused_top1": (w[0] * sim_v + w[1] * sim_t + w[2] * sim_a).max().item(),
            })
            print(f"  [{set_name:>10}] {query[:50]:<50}  "
                  f"w_v={g['w_v']:.3f} w_t={g['w_t']:.3f} w_a={g['w_a']:.3f}  "
                  f"dominant={g['dominant_modality']}")

    csv_path = os.path.join(OUTPUT_DIR, "routing_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    dom_counts = {}
    for r in rows:
        dom_counts[r["dominant"]] = dom_counts.get(r["dominant"], 0) + 1
    print(f"\nDominant modality distribution: {dom_counts}")

    unique_doms = set(r["dominant"] for r in rows)
    print(f"Unique dominant modalities observed: {len(unique_doms)} (need >= 2 to pass)")
    for qs in ["visual", "audio", "semantic"]:
        doms = [r["dominant"] for r in rows if r["query_set"] == qs]
        print(f"  {qs}: {doms}")


if __name__ == "__main__":
    main()
