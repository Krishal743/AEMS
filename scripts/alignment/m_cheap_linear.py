#!/usr/bin/env python3
"""
M1, M2, M5: cheap matrix-based alignment methods for CLAP-audio -> CLIP-text.
=============================================================================
  M1  Whitening + linear map (Ethayarajh 2019 anisotropy fix; Wang 2019)
  M2  CCA projection (Rasiwasia 2010; Andrew 2013)
  M5  Hubness reduction (Radovanović 2010; Zelnik-Manor 2004) -- applied as
      post-processing on a chosen input gallery.

All maps fit on TRAIN anchors only; applied to test for evaluation and to the
full gallery for export.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/alignment/m_cheap_linear.py [--which m1 m2 m5]
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from scripts.alignment.compare_utils import (
    load_anchor_pairs, ridge_fit, procrustes_map, whitening_stats, apply_whiten,
    cca_directions, local_scaling, evaluate_method,
)
from src.config import AEMS_AUDIO_EMBEDDINGS_PATH, AEMS_MANIFEST_PATH, set_seeds
from src.data.metadata import load_metadata, filter_by_split

OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ALIGNED_DIR = "embeddings"
os.makedirs(ALIGNED_DIR, exist_ok=True)


def save_db(audio_proj, tag, full_vids):
    """Save aligned audio as {video_id: tensor} for a set of videos."""
    out = {}
    for i, v in enumerate(full_vids):
        out[v] = F.normalize(audio_proj[i], dim=-1).detach().cpu()
    path = os.path.join(ALIGNED_DIR, f"aems_audio_aligned_{tag}.pt")
    torch.save(out, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", nargs="+", choices=["m1", "m2", "m5"],
                        default=["m1", "m2", "m5"])
    parser.add_argument("--cca-rank", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seeds(args.seed)

    print("=" * 72)
    print("CHEAP LINEAR ALIGNMENT METHODS (M1 whitening, M2 CCA, M5 hubness)")
    print("=" * 72)

    # Train anchors for fitting
    X_tr, Y_tr, _ = load_anchor_pairs("train")
    # Test anchors for evaluation
    X_te, Y_te, te_vids = load_anchor_pairs("test")
    # Full gallery (all videos) for export
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, map_location="cpu",
                          weights_only=False)
    full_vids = sorted(audio_db.keys())
    X_full = F.normalize(torch.stack([audio_db[v].float() for v in full_vids]), dim=-1)

    # category map for test
    recs = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
    cat = {r["video_id"]: r["content_fine_category"] for r in recs}

    results = {}

    # ------------- M1: whitening + linear map -------------
    if "m1" in args.which:
        print("\n[M1] WHITENING + LINEAR MAP")
        Ww_a, mean_a = whitening_stats(X_tr)   # on audio
        Ww_t, mean_t = whitening_stats(Y_tr)   # on text
        X_tr_w = apply_whiten(X_tr, Ww_a, mean_a)
        Y_tr_w = apply_whiten(Y_tr, Ww_t, mean_t)
        # Ridge between whitened views
        W_lin = ridge_fit(X_tr_w, Y_tr_w, lam=1e-2)
        def transform(X):
            Xw = apply_whiten(X, Ww_a, mean_a)
            return Xw @ W_lin.T
        proj_te = transform(X_te)
        eval_m1 = evaluate_method(proj_te, Y_te, "m1", cat, te_vids)
        # Full export
        proj_full = transform(X_full)
        path = save_db(proj_full, "m1_whiten_lin", full_vids)
        eval_m1["export_path"] = path
        results["m1_whiten_lin"] = eval_m1
        print(f"  test alignment={eval_m1['alignment_cosine']:.4f} "
              f"R@1(t2a)={eval_m1['cliptext_to_audio']['R@1']:.4f} "
              f"MRR={eval_m1['cliptext_to_audio']['MRR']:.4f}")
        print(f"  saved: {path}")

    # ------------- M2: CCA projection -------------
    if "m2" in args.which:
        print("\n[M2] CCA PROJECTION")
        rank = args.cca_rank
        Wx = cca_directions(X_tr, Y_tr, rank)
        # project audio into CCA subspace
        def transform_cca(X):
            Xc = X - X.mean(0, keepdim=True)
            return Xc @ Wx  # [n, rank]
        proj_te = transform_cca(X_te)
        # pad to 512 with zeros so it's a drop-in (harmless, last dims zero)
        n_te = proj_te.shape[0]
        proj_te_p = torch.zeros(n_te, 512)
        proj_te_p[:, :rank] = proj_te[:, :rank].clamp(max=0)  # placeholder avoided below
        proj_te = F.pad(proj_te, (0, 512 - rank))
        eval_m2 = evaluate_method(proj_te, Y_te, "m2", cat, te_vids)
        # Full export
        proj_full = F.pad(transform_cca(X_full), (0, 512 - rank))
        path = save_db(proj_full, "m2_cca", full_vids)
        eval_m2["export_path"] = path
        eval_m2["cca_rank"] = rank
        results["m2_cca"] = eval_m2
        print(f"  test alignment={eval_m2['alignment_cosine']:.4f} "
              f"R@1(t2a)={eval_m2['cliptext_to_audio']['R@1']:.4f} "
              f"MRR={eval_m2['cliptext_to_audio']['MRR']:.4f}")
        print(f"  saved: {path}")

    # ------------- M5: hubness reduction -------------
    if "m5" in args.which:
        # Apply local scaling to the M1 output (the strongest cheap linear map)
        print("\n[M5] HUBNESS REDUCTION (local scaling on M1 gallery)")
        Ww_a, mean_a = whitening_stats(X_tr)
        Ww_t, mean_t = whitening_stats(Y_tr)
        W_lin = ridge_fit(apply_whiten(X_tr, Ww_a, mean_a),
                          apply_whiten(Y_tr, Ww_t, mean_t), lam=1e-2)
        proj_te = apply_whiten(X_te, Ww_a, mean_a) @ W_lin.T
        ls_te = local_scaling(proj_te, k=5)
        eval_m5 = evaluate_method(ls_te, Y_te, "m5", cat, te_vids)
        # full export
        proj_full = apply_whiten(X_full, Ww_a, mean_a) @ W_lin.T
        ls_full = local_scaling(proj_full, k=5)
        path = save_db(ls_full, "m5_hubness_ls", full_vids)
        eval_m5["export_path"] = path
        results["m5_hubness_ls"] = eval_m5
        print(f"  test alignment={eval_m5['alignment_cosine']:.4f} "
              f"R@1(t2a)={eval_m5['cliptext_to_audio']['R@1']:.4f} "
              f"MRR={eval_m5['cliptext_to_audio']['MRR']:.4f}")
        print(f"  saved: {path}")

    out_path = os.path.join(OUTPUT_DIR, "cheap_linear_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[DONE] Results: {out_path}")


if __name__ == "__main__":
    main()
