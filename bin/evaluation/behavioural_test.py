"""Behavioural verification: demonstrate gating adapts per query type."""

import argparse, csv, os
import torch
from src.encoders.clap_encode import CLAPEncoder
from src.explainability.explain_retrieval import explain_gating_decision
from src.routing.query_router import (
    add_index_args, load_search_index, load_gate, load_clip,
    encode_text_query, encode_clap_text_query, search,
)

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
    add_index_args(parser)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    index = load_search_index(args.video_embeds, args.audio_embeds, args.caption_embeds)
    gate = load_gate(args.gate_weights, DEVICE)
    clip_model = load_clip(DEVICE)
    clap_encoder = CLAPEncoder(device=DEVICE)

    rows = []
    for set_name, queries in QUERY_SETS.items():
        for query in queries:
            w, sim_v, sim_t, sim_a = search(
                index, gate,
                clip_query=encode_text_query(clip_model, query, DEVICE),
                clap_query=encode_clap_text_query(clap_encoder, query, DEVICE),
            )

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
