#!/usr/bin/env python3
"""
eval_multi_seed_crop.py -- Rigorous M4 vs M6-hybrid decision eval
==================================================================
Settles whether the M4-vs-M6-hybrid retrieval gaps (t2a R@1 0.0323 vs 0.0254;
a2t R@1 0.0538 vs 0.0587) are real or seed/crop noise.

Two variance dimensions (both cheap, new-file-only, no existing models touched):

  MODEL SEEDS  : retrain AudioAdapter from scratch with seeds {100,200,300}
                 using the exact M6 recipe (batch 256, lr 1e-3, 30 ep, cosine,
                 hard-negative InfoNCE, best-state via dropout-OFF test eval).
                 Target = text (M4) or hybrid video+text (M6-hybrid).
  CROP SEEDS   : re-encode the 1,022 held-out test audio clips under 3
                 deterministic random 10s windows (AudioCollator/rand_trunc,
                 base_seed = crop*1000 + i//bs) VS the cached 3-segment-mean
                 features (the deployed input). Adapters were trained on the
                 cached mean-of-3-segments, so crop runs are a generalization
                 probe (documented caveat).

Also anchors to the published seed-42 models so recorded numbers
(0.0323/0.0538, 0.0254/0.0587) are cross-checked under the same protocol.

Decision rule (user-specified):
  adopt hybrid iff its a2t R@1 beats M4 outside noise AND t2a R@1/t2a MRR
  remain within noise of M4; otherwise keep M4 as deployable.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python scripts/alignment/eval_multi_seed_crop.py
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.alignment.m4_adapter_contrastive import AudioAdapter, contrastive_loss
from scripts.alignment.compare_utils import load_anchor_pairs, eval_direction
from scripts.alignment.audio_ft_utils import (
    AudioCollator, clap_audio_embedding, find_audio_path, load_waveform,
)
from src.config import AEMS_VID_EMBEDDINGS_PATH, set_seeds

OUTPUT_DIR = "outputs/alignment"
MODEL_DIR = "models"
SEEDED_MODEL_DIR = os.path.join(OUTPUT_DIR, "seeded_models")
os.makedirs(SEEDED_MODEL_DIR, exist_ok=True)

SEEDS = [100, 200, 300]
CROP_SEEDS = [400, 500, 600]
EPOCHS = 30
BATCH = 256
LR = 1e-3
HIDDEN = 512

# method -> training target ("text" = M4, "hybrid" = M6-hybrid video+text)
METHODS = {"m4": "text", "m6_hybrid": "hybrid"}
PUBLISHED = {
    "m4": os.path.join(MODEL_DIR, "audio_adapter_cliptext_m4.pt"),
    "m6_hybrid": os.path.join(MODEL_DIR, "audio_adapter_clipvid_text_hybrid_m6.pt"),
}


def load_video_targets(vids):
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, map_location="cpu",
                          weights_only=False)
    return F.normalize(torch.stack([video_db[v].float() for v in vids]), dim=-1)


def quick_a2t_r1(model, X_te, Y_te, device, bs=512):
    """Correct gallery-based audio->text R@1, dropout-OFF (matches original)."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, X_te.shape[0], bs):
            parts.append(F.normalize(model.encode(X_te[i:i+bs].to(device)), dim=-1).cpu())
    P = torch.cat(parts)
    Yn = F.normalize(Y_te, dim=-1)
    sim = P @ Yn.T
    n = P.shape[0]
    top1 = sim.argmax(dim=1)
    return float((top1 == torch.arange(n)).float().mean())


def train_adapter(target, seed, X_tr, Y_tr_text, Y_tr_video, X_te, Y_te, device):
    """Exact M6 training loop; best-state by dropout-OFF a2t R@1 on cached test."""
    set_seeds(seed)
    if target == "hybrid":
        ds = TensorDataset(X_tr, Y_tr_text, Y_tr_video)
    else:
        ds = TensorDataset(X_tr, Y_tr_text)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                    drop_last=False)
    model = AudioAdapter(d=512, hidden=HIDDEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_r1, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for batch in dl:
            opt.zero_grad()
            if target == "hybrid":
                xb, yb_txt, yb_vid = batch
                xb = xb.to(device)
                yb_txt = F.normalize(yb_txt, dim=-1).to(device)
                yb_vid = F.normalize(yb_vid, dim=-1).to(device)
                proj = model(xb)
                loss = (contrastive_loss(proj, yb_txt, model.logit_scale)
                        + contrastive_loss(proj, yb_vid, model.logit_scale))
            else:
                xb, yb = batch
                xb, yb = xb.to(device), F.normalize(yb, dim=-1).to(device)
                proj = model(xb)
                loss = contrastive_loss(proj, yb, model.logit_scale)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        if ep % 5 == 0 or ep == EPOCHS - 1:
            r1 = quick_a2t_r1(model, X_te, Y_te, device, bs=512)
            if r1 > best_r1:
                best_r1 = r1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_r1


def encode_crop(waves, crop_seed, clap_mod, device, bs=8):
    """Deterministic rand_trunc 10s-window encoding of test clips."""
    collator = AudioCollator(clap_mod.model_cfg)
    model = clap_mod.model          # nn.Module carries encode_audio/audio_projection
    embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(waves), bs):
            batch = waves[i:i+bs]
            np.random.seed(crop_seed * 1000 + i // bs)
            torch.manual_seed(crop_seed * 1000 + i // bs)
            dicts = collator(batch)
            e = clap_audio_embedding(model, dicts, device)
            embs.append(F.normalize(e, dim=-1).cpu())
    return torch.cat(embs)


def eval_model_on_features(model, X, Y, device, bs=512):
    """Both-direction metrics vs frozen CLIP-text gallery Y."""
    model.eval()
    with torch.no_grad():
        parts = []
        for i in range(0, X.shape[0], bs):
            parts.append(F.normalize(model.encode(X[i:i+bs].to(device)), dim=-1).cpu())
    P = torch.cat(parts)
    Yn = F.normalize(Y, dim=-1)
    return {"t2a": eval_direction(Yn, P), "a2t": eval_direction(P, Yn)}


def pick(metrics, dname, key):
    return metrics[dname][key]


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="skip retraining; use published seed-42 models only")
    parser.add_argument("--no-crops", action="store_true",
                        help="skip CLAP crop re-encoding (cached features only)")
    args = parser.parse_args()

    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    X_tr, Y_tr_txt, vids_tr = load_anchor_pairs("train")
    X_te_cached, Y_te, vids_te = load_anchor_pairs("test")
    Y_tr_text = F.normalize(Y_tr_txt, dim=-1)
    Y_tr_video = load_video_targets(vids_tr)
    Y_te = F.normalize(Y_te, dim=-1)
    n_tr, n_te = X_tr.shape[0], X_te_cached.shape[0]
    print(f"[DATA] train={n_tr} test={n_te} device={device}")

    variants = {"cached": X_te_cached}
    if not args.no_crops:
        import laion_clap
        clap_mod = laion_clap.CLAP_Module(enable_fusion=False)
        clap_mod.load_ckpt()
        te_paths = [find_audio_path(v) for v in vids_te]
        print("[CROPS] loading test waveforms ...")
        waves = [load_waveform(p) for p in te_paths]
        for cseed in CROP_SEEDS:
            variants[f"crop_{cseed}"] = encode_crop(waves, cseed, clap_mod, device)
            print(f"  crop {cseed}: encoded {variants[f'crop_{cseed}'].shape}")
        del waves, clap_mod
        torch.cuda.empty_cache()
    print(f"[VARIANTS] {list(variants)}")

    seeded_models = {}
    if not args.skip_train:
        last_train_time = time.time()
        for method, target in METHODS.items():
            for seed in SEEDS:
                model, best_r1 = train_adapter(
                    target, seed, X_tr, Y_tr_text, Y_tr_video,
                    X_te_cached, Y_te, device)
                mpath = os.path.join(SEEDED_MODEL_DIR, f"{method}_seed{seed}.pt")
                torch.save(model.state_dict(), mpath)
                seeded_models.setdefault(method, {})[seed] = {"model": model,
                                                              "best_r1": best_r1,
                                                              "path": mpath}
                print(f"  trained {method} seed={seed} best_a2t_r1={best_r1:.4f} "
                      f"({time.time()-last_train_time:.0f}s)")
                last_train_time = time.time()
    else:
        print("[SKIP-TRAIN] using published seed-42 models only")

    # ---- collect metrics ----
    results = {"config": {"seeds": SEEDS, "crop_seeds": CROP_SEEDS,
                          "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                          "hidden": HIDDEN, "n_train": n_tr, "n_test": n_te,
                          "variants": list(variants)},
               "published_seed42": {},
               "per_seed": {}, "seed_variance_cached": {}, "crop_variance": {}}

    # published seed-42 anchors
    for method, path in PUBLISHED.items():
        sd = torch.load(path, map_location="cpu", weights_only=False)
        m = AudioAdapter(d=512, hidden=HIDDEN).to(device)
        m.load_state_dict(sd)
        results["published_seed42"][method] = {
            v: eval_model_on_features(m, X, Y_te, device)
            for v, X in variants.items()}
        print(f"[ANCHOR] {method} seed42 cached: t2a R@1="
              f"{results['published_seed42'][method]['cached']['t2a']['R@1']:.4f} "
              f"a2t R@1={results['published_seed42'][method]['cached']['a2t']['R@1']:.4f}")

    # per-seed models across all variants
    for method in METHODS:
        results["per_seed"][method] = {}
        for seed, entry in seeded_models.get(method, {}).items():
            results["per_seed"][method][str(seed)] = {
                v: eval_model_on_features(entry["model"], X, Y_te, device)
                for v, X in variants.items()}

    # ---- decision aggregates ----
    crit = 2.776  # t(0.975, df=4) for a 2-method, 3-seed difference
    n_seeds = len(SEEDS)
    for method in METHODS:
        if method not in results["per_seed"] or not results["per_seed"][method]:
            continue
        seed_rows = [results["per_seed"][method][str(s)] for s in SEEDS]
        agg = {}
        for key in ["t2a", "a2t"]:
            for mkey in ["R@1", "R@5", "R@10", "MRR", "median_rank"]:
                mu, sd = mean_std([pick(r["cached"], key, mkey) for r in seed_rows])
                agg[f"{key}_{mkey}_mean"], agg[f"{key}_{mkey}_std"] = mu, sd
        results["seed_variance_cached"][method] = agg
        cv = {}
        for mkey in ["R@1", "MRR"]:
            for key in ["t2a", "a2t"]:
                across = []
                for cseed in CROP_SEEDS:
                    vals = [pick(r[f"crop_{cseed}"], key, mkey) for r in seed_rows]
                    across.append(np.mean(vals))
                mu, sd = mean_std(across)
                cv[f"{key}_{mkey}"], cv[f"{key}_{mkey}_std"] = mu, sd
        results["crop_variance"][method] = cv

    # decision
    if "m4" in results["seed_variance_cached"] and \
       "m6_hybrid" in results["seed_variance_cached"]:
        m4 = results["seed_variance_cached"]["m4"]
        hy = results["seed_variance_cached"]["m6_hybrid"]
        def se(a, b):
            return math.sqrt(a**2 / n_seeds + b**2 / n_seeds)
        t2a_gap = m4["t2a_R@1_mean"] - hy["t2a_R@1_mean"]
        t2a_mrr_gap = m4["t2a_MRR_mean"] - hy["t2a_MRR_mean"]
        a2t_diff = hy["a2t_R@1_mean"] - m4["a2t_R@1_mean"]
        se_t2a = se(m4["t2a_R@1_std"], hy["t2a_R@1_std"])
        se_t2a_mrr = se(m4["t2a_MRR_std"], hy["t2a_MRR_std"])
        se_a2t = se(m4["a2t_R@1_std"], hy["a2t_R@1_std"])
        t2a_within_noise = (t2a_gap <= crit * se_t2a) and (t2a_mrr_gap <= crit * se_t2a_mrr)
        a2t_strict_better = (a2t_diff - crit * se_a2t) > 0.0
        a2t_better = a2t_diff > 0.0
        adopt = a2t_strict_better and t2a_within_noise
        results["decision"] = {
            "adopt_m6_hybrid": bool(adopt),
            "a2t_diff(hy-m4)_R1": a2t_diff, "a2t_diff_95CI_half": crit * se_a2t,
            "a2t_strict_better": a2t_strict_better, "a2t_better_point": a2t_better,
            "t2a_gap(m4-hy)_R1": t2a_gap, "t2a_R1_95CI_half": crit * se_t2a,
            "t2a_MRR_gap(m4-hy)": t2a_mrr_gap, "t2a_MRR_95CI_half": crit * se_t2a_mrr,
            "t2a_within_noise": t2a_within_noise,
            "verdict": "ADOPT M6-hybrid" if adopt else "KEEP M4 as deployable",
        }
    else:
        results["decision"] = {"verdict": "n/a (no seed models)"}

    with open(os.path.join(OUTPUT_DIR, "multi_seed_crop_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["decision"], indent=2))
    print(f"\n[DONE] outputs/alignment/multi_seed_crop_eval.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()