#!/usr/bin/env python3
"""
compare_all_methods.py -- Extended method comparison (adds M6 rows, full a2t metrics)
====================================================================================
New standalone orchestrator (compare_methods.py is left untouched). Loads every
per-method audio DB and evaluates with the SAME held-out harness, including the
M6 Wav2CLIP variants (video targets, hybrid video+text), and surfaces the full
metric set for BOTH directions (R@1/5/10, MRR, mean/median rank).

Writes outputs/alignment/method_comparison_v2.json (does not overwrite v1).

No leakage: all maps/adapter heads were fit on TRAIN; only held-out TEST is
evaluated here.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from scripts.alignment.compare_utils import load_anchor_pairs, hubness_skew
from src.config import set_seeds

OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = [
    ("raw",                "aems_audio_embeddings_v1.pt",            "RAW CLAP (no alignment)"),
    ("m1",                 "aems_audio_aligned_m1_whiten_lin.pt",    "M1 whitening + ridge"),
    ("m2",                 "aems_audio_aligned_m2_cca.pt",           "M2 CCA projection"),
    ("m5",                 "aems_audio_aligned_m5_hubness_ls.pt",    "M5 hubness (local scaling)"),
    ("m3",                 "aems_audio_aligned_m3_shared_text.pt",   "M3 shared-text anchor"),
    ("m4",                 "aems_audio_aligned_m4_adapter_ct.pt",    "M4 adapter + contrastive"),
    ("tier1_globalhn",     "aems_audio_aligned_m4_global_hn.pt",     "Tier1 M4+global hard-neg"),
    ("tier1_boosted",      "aems_audio_aligned_m4_boosted.pt",       "Tier1 M4+hubness boost"),
    ("tier2_lora",         "aems_audio_aligned_m4_lora.pt",          "Tier2 LoRA ft (r=16)"),
    ("tier3_fullft",       "aems_audio_aligned_m4_fullft.pt",        "Tier3 full ft (30ep)"),
    ("m6_w2c",             "aems_audio_aligned_m6_w2c.pt",           "M6 Wav2CLIP (video tgt)"),
    ("m6_hybrid",          "aems_audio_aligned_m6_hybrid.pt",        "M6 hybrid (video+text)"),
    ("ssl_single",         "aems_audio_aligned_ssl_single.pt",       "SSL WavLM single"),
    ("ssl_multi",          "aems_audio_aligned_ssl_multi.pt",        "SSL WavLM + multi-pos"),
    ("beats_single",       "aems_audio_aligned_beats_single.pt",     "BEATs iter3+ AS2M single"),
    ("imagebind_single",   "aems_audio_aligned_imagebind_single.pt", "ImageBind huge (audio) single"),
]


def eval_pair(G, Y):
    """Both-direction retrieval on matched rows. Returns full dict incl.
    median rank and R@10 for both directions."""
    n = G.shape[0]
    out = {}
    for dname, Q, Gal in [("t2a", Y, G), ("a2t", G, Y)]:
        sim = Q @ Gal.T
        ranks = torch.argsort(sim, descending=True)
        rr = (ranks == torch.arange(n).view(-1, 1)).nonzero()[:, 1].float()
        out[dname] = {
            "R@1": float((rr == 0).float().mean()),
            "R@5": float((rr < 5).float().mean()),
            "R@10": float((rr < 10).float().mean()),
            "MRR": float((1.0 / (rr + 1)).mean()),
            "mean_rank": float(rr.mean()),
            "median_rank": float(rr.median()),
        }
    out["alignment_cosine"] = float((G * Y).sum(dim=1).mean())
    ac = F.normalize(G.mean(0, keepdim=True), dim=-1)
    tc = F.normalize(Y.mean(0, keepdim=True), dim=-1)
    out["modality_gap"] = float(1.0 - (ac * tc).sum())
    out["hubness_skew_t2a"] = hubness_skew(G, Y)
    sub = G[torch.randperm(G.shape[0])[:1500]]
    sims = sub @ sub.T
    mask = ~torch.eye(sub.shape[0], dtype=torch.bool)
    out["gallery_anisotropy"] = float(sims[mask].mean())
    return out


def main():
    set_seeds(42)
    _, Y_te, vids_te = load_anchor_pairs("test")

    results = {}
    for mid, fname, label in METHODS:
        db = torch.load("embeddings/" + fname, weights_only=False)
        G = F.normalize(torch.stack([db[v].float() for v in vids_te]), dim=-1)
        e = eval_pair(G, Y_te)
        results[mid] = {"label": label,
                        "t2a.R@1": e["t2a"]["R@1"], "t2a.R@5": e["t2a"]["R@5"],
                        "t2a.R@10": e["t2a"]["R@10"], "t2a.MRR": e["t2a"]["MRR"],
                        "t2a.median_rank": e["t2a"]["median_rank"],
                        "a2t.R@1": e["a2t"]["R@1"], "a2t.R@5": e["a2t"]["R@5"],
                        "a2t.R@10": e["a2t"]["R@10"], "a2t.MRR": e["a2t"]["MRR"],
                        "a2t.median_rank": e["a2t"]["median_rank"],
                        "alignment_cosine": e["alignment_cosine"],
                        "modality_gap": e["modality_gap"],
                        "hubness_skew": e["hubness_skew_t2a"],
                        "gallery_anisotropy": e["gallery_anisotropy"]}

    print("=" * 130)
    print("AEMS AUDIO-ALIGNMENT METHOD COMPARISON v4  (held-out test, n=1022)")
    print("=" * 130)
    hdr = (f"{'method':<26}{'t2a R@1':>8}{'t2a R@10':>9}{'t2a MRR':>8}{'t2a medR':>9}"
           f"{'a2t R@1':>9}{'a2t R@10':>9}{'a2t MRR':>8}{'a2t medR':>10}"
           f"{'align':>7}{'gap':>7}{'hub':>7}{'aniso':>8}")
    print(hdr)
    print("-" * 130)
    for mid, _, label in METHODS:
        r = results[mid]
        print(f"{r['label']:<26}{r['t2a.R@1']:>8.4f}{r['t2a.R@10']:>9.4f}"
              f"{r['t2a.MRR']:>8.4f}{r['t2a.median_rank']:>9.0f}"
              f"{r['a2t.R@1']:>9.4f}{r['a2t.R@10']:>9.4f}{r['a2t.MRR']:>8.4f}"
              f"{r['a2t.median_rank']:>10.0f}{r['alignment_cosine']:>7.3f}"
              f"{r['modality_gap']:>7.3f}{r['hubness_skew']:>7.1f}"
              f"{r['gallery_anisotropy']:>8.3f}")
    print("-" * 130)
    print(f"Random chance R@1 for gallery of {Y_te.shape[0]} is ~{1.0/Y_te.shape[0]:.4f}")

    # Winner by t2a R@1 (the direction the live fused system queries)
    cands = ["raw", "m1", "m2", "m5", "m3", "m4",
             "tier1_globalhn", "tier1_boosted", "tier2_lora", "tier3_fullft",
             "m6_w2c", "m6_hybrid", "ssl_single", "ssl_multi",
             "beats_single", "imagebind_single"]
    winner_t2a = max(cands, key=lambda m: results[m]["t2a.R@1"])
    winner_a2t = max(cands, key=lambda m: results[m]["a2t.R@1"])
    print(f"\n>>> WINNER (t2a / text-query->audio): {results[winner_t2a]['label']} "
          f"(R@1={results[winner_t2a]['t2a.R@1']:.4f}, R@10={results[winner_t2a]['t2a.R@10']:.4f})")
    print(f">>> WINNER (a2t / audio-query->text): {results[winner_a2t]['label']} "
          f"(R@1={results[winner_a2t]['a2t.R@1']:.4f}, R@10={results[winner_a2t]['a2t.R@10']:.4f})")

    payload = {"n_test": Y_te.shape[0], "chance_R@1": 1.0 / Y_te.shape[0],
               "winner_t2a": results[winner_t2a]["label"],
               "winner_a2t": results[winner_a2t]["label"],
               "methods": results}
    out_path = os.path.join(OUTPUT_DIR, "method_comparison_v4.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[DONE] {out_path}")


if __name__ == "__main__":
    main()