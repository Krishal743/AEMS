#!/usr/bin/env python3
"""
compare_methods.py -- Orchestrator for AEMS audio-alignment method comparison.
================================================================================
Loads the per-method transformed audio DBs (already produced by the M1..M5
scripts) and evaluates every method with the SAME held-out harness, then prints
a comparison table and writes outputs/alignment/method_comparison.json.

No leakage: all maps were fit on TRAIN; only held-out TEST is evaluated here.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.audio_alignment.compare_utils import load_anchor_pairs, hubness_skew
from src.config import AEMS_MANIFEST_PATH, AEMS_AUDIO_EMBEDDINGS_PATH, set_seeds
from src.data.metadata import load_metadata, filter_by_split

OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS = [
    ("raw",                "aems_audio_embeddings_v1.pt",            "RAW CLAP (no alignment)"),
    ("m1",                 "aems_audio_aligned_m1_whiten_lin.pt",    "M1 whitening + ridge"),
    ("m2",                 "aems_audio_aligned_m2_cca.pt",           "M2 CCA projection"),
    ("m5",                 "aems_audio_aligned_m5_hubness_ls.pt",    "M5 hubness (local scaling)"),
    ("m3",                 "aems_audio_aligned_m3_shared_text.pt",   "M3 shared-text anchor"),
    ("m4",                 "aems_audio_aligned_m4_adapter_ct.pt",    "M4 adapter + contrastive"),
    ("m4procrustes",       None,                                     "Procrustes (prior round)"),
    ("tier1_globalhn",     "aems_audio_aligned_m4_global_hn.pt",     "Tier1 M4+global hard-neg"),
    ("tier1_boosted",      "aems_audio_aligned_m4_boosted.pt",       "Tier1 M4+hubness boost"),
    ("tier2_lora",         "aems_audio_aligned_m4_lora.pt",          "Tier2 LoRA ft (r=16)"),
    ("tier3_fullft",       "aems_audio_aligned_m4_fullft.pt",        "Tier3 full ft (30ep)"),
]


def eval_pair(G, Y):
    """Both-direction retrieval on matched rows. Returns dict."""
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
        if mid == "m4procrustes":
            # prior-round number (held-out test, reported in projection_validation.json)
            try:
                pv = json.load(open("outputs/alignment/projection_validation.json"))
                r = pv["retrieval"]["after_projected"]
                results[mid] = {"label": label, "t2a.R@1": r["R@1"],
                                "t2a.R@5": r["R@5"], "t2a.R@10": r["R@10"],
                                "note": "from prior Procrustes round (held-out test)"}
            except Exception:
                results[mid] = {"label": label, "note": "prior data unavailable"}
            continue

        db = torch.load("embeddings/" + fname, weights_only=False)
        G = F.normalize(torch.stack([db[v].float() for v in vids_te]), dim=-1)
        e = eval_pair(G, Y_te)
        results[mid] = {"label": label,
                        "t2a.R@1": e["t2a"]["R@1"], "t2a.R@5": e["t2a"]["R@5"],
                        "t2a.R@10": e["t2a"]["R@10"], "t2a.MRR": e["t2a"]["MRR"],
                        "a2t.R@1": e["a2t"]["R@1"], "a2t.MRR": e["a2t"]["MRR"],
                        "alignment_cosine": e["alignment_cosine"],
                        "modality_gap": e["modality_gap"],
                        "hubness_skew": e["hubness_skew_t2a"],
                        "gallery_anisotropy": e["gallery_anisotropy"]}

    print("=" * 110)
    print("AEMS AUDIO-ALIGNMENT METHOD COMPARISON  (held-out test, n=1022)")
    print("=" * 110)
    hdr = f"{'method':<26}{'t2a R@1':>8}{'t2a R@5':>8}{'t2a R@10':>9}{'MRR':>7}" \
          f"{'a2t R@1':>9}{'align':>8}{'gap':>8}{'hub':>7}{'aniso':>8}"
    print(hdr)
    print("-" * 110)
    for mid in ["raw", "m1", "m2", "m5", "m3", "m4procrustes", "m4",
                "tier1_globalhn", "tier1_boosted", "tier2_lora", "tier3_fullft"]:
        r = results[mid]
        if mid == "m4procrustes":
            print(f"{r['label']:<26}{r.get('t2a.R@1',0):>8.4f}{r.get('t2a.R@5',0):>8.4f}"
                  f"{r.get('t2a.R@10',0):>9.4f}   (prior round)")
            continue
        print(f"{r['label']:<26}{r['t2a.R@1']:>8.4f}{r['t2a.R@5']:>8.4f}{r['t2a.R@10']:>9.4f}"
              f"{r['t2a.MRR']:>7.4f}{r['a2t.R@1']:>9.4f}{r['alignment_cosine']:>8.3f}"
              f"{r['modality_gap']:>8.3f}{r['hubness_skew']:>7.1f}{r['gallery_anisotropy']:>8.3f}")
    print("-" * 110)
    print(f"Random chance R@1 for gallery of {Y_te.shape[0]} is ~{1.0/Y_te.shape[0]:.4f}")

    # Winner selection (text->audio R@1)
    winner = max(["raw", "m1", "m2", "m5", "m3", "m4",
                  "tier1_globalhn", "tier1_boosted", "tier2_lora", "tier3_fullft"],
                 key=lambda m: results[m]["t2a.R@1"])
    print(f"\n>>> WINNER: {results[winner]['label']} "
          f"(t2a R@1={results[winner]['t2a.R@1']:.4f})")

    payload = {"n_test": Y_te.shape[0], "chance_R@1": 1.0 / Y_te.shape[0],
               "winner": results[winner]["label"], "methods": results,
               "frameworks": [
                   "Wang 2019 (linear/ridge)", "Rasiwasia 2010 (CCA)",
                   "Ethayarajh 2019 (whitening)", "Girdhar 2023 (shared anchor)",
                   "Radovanović 2010 (hubness)", "Guzhov 2022 / Wu 2022 (adapter+contrastive)",
               ]}
    out_path = os.path.join(OUTPUT_DIR, "method_comparison.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
