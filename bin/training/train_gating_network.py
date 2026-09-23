"""Train the query-conditioned gating network on z-scored branch similarities.

The gate predicts per-query fusion weights (visual, text, passage, audio) that
are applied to branch similarities z-scored over the gallery — the same fusion
src/routing/query_router.py uses at search time.

Two things keep this honest:

* The audio branch for training videos is **cross-fitted**: the train videos are
  split into K folds and each fold is projected by an adapter fitted on the
  other folds. The deployed adapter was fitted on these videos, so their audio
  similarities are optimistic; training on them teaches the gate to over-trust
  audio, which measurably hurts retrieval on unseen videos.
* Selection and the fixed-weight comparison run on a validation split carved
  from TRAIN (the same split the adapter held out). The test split is untouched.

The run reports the gate against tuned fixed weights and prints the spread of
the predicted weights, so a gate that has collapsed onto one modality is
visible rather than silently shipped.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                        AEMS_WAVLM_FEATURES_PATH, AEMS_GATING_WEIGHTS_PATH,
                        AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
                        AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
                        AEMS_TEXT_CHUNKS_PATH_TEMPLATE,
                        AEMS_FUSION_WEIGHTS, DEVICE, set_seeds)
from src.models.gating_network import GatingNetwork
from src.routing.query_router import BRANCHES, ChunkIndex, zscore
from src.training.audio_adapter_fit import fit_adapter, project
from src.training.query_data import (load_records, questions, validation_split, encode_clip_text,
                                     flatten_questions, stack_embeddings, query_rows, recall_metrics)

parser = argparse.ArgumentParser(description="Train AEMS Gating Network")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--video-embeds", default=AEMS_VID_EMBEDDINGS_PATH)
parser.add_argument("--audio-embeds", default=AEMS_AUDIO_EMBEDDINGS_PATH)
parser.add_argument("--audio-features", default=AEMS_WAVLM_FEATURES_PATH)
parser.add_argument("--text-embeds-train", default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"))
parser.add_argument("--chunk-embeds-train", default=AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split="train"))
parser.add_argument("--output", default=AEMS_GATING_WEIGHTS_PATH)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.2,
                    help="softmax temperature on fused scores (tuned on validation)")
parser.add_argument("--folds", type=int, default=4, help="cross-fitting folds for the audio branch")
parser.add_argument("--adapter-epochs", type=int, default=39)
parser.add_argument("--val-frac", type=float, default=0.15)
parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="train each seed, report mean/std, save the best on VALIDATION")
parser.add_argument("--no-cross-fit", action="store_true",
                    help="train on the deployed adapter's audio instead of out-of-fold "
                         "projections. Biased (the adapter saw these videos); for ablation only.")
args = parser.parse_args()
args.seed = args.seeds[0]
set_seeds(args.seed)

records = load_records(args.manifest, "train")
vid_db = torch.load(args.video_embeds, weights_only=False)
aud_db = torch.load(args.audio_embeds, weights_only=False)
txt_db = torch.load(args.text_embeds_train, weights_only=False)
chunk_db = torch.load(args.chunk_embeds_train, weights_only=False)
feat_db = torch.load(args.audio_features, weights_only=False)
desc_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split="train"), weights_only=False)

usable = [v for v in records
          if v in vid_db and v in aud_db and v in txt_db and v in feat_db and v in desc_db
          and v in chunk_db
          and questions(records[v])]
fit_vids, val_vids = validation_split(usable, args.val_frac)
print(f"[DATA] fit={len(fit_vids)} val={len(val_vids)} (test split untouched)", flush=True)

print("[ENC] Encoding QA questions with CLIP...", flush=True)
texts, rows = flatten_questions(records, fit_vids + val_vids)
q_emb = encode_clip_text(texts, DEVICE)

oof = aud_db
if args.no_cross_fit:
    print("[WARN] --no-cross-fit: training on audio the deployed adapter was fitted on. "
          "These similarities are optimistic and the gate will over-trust audio.", flush=True)
else:
    print(f"[AUDIO] Cross-fitting {args.folds} adapters for out-of-fold train audio...", flush=True)
    oof = {}
    folds = [fit_vids[i::args.folds] for i in range(args.folds)]
    for i, fold in enumerate(folds):
        other = [v for j, f in enumerate(folds) if j != i for v in f]
        targets = [torch.cat([F.normalize(torch.as_tensor(desc_db[v]).float().view(1, -1), dim=1),
                              q_emb[rows[v]]]) for v in other]
        adapter, _, _ = fit_adapter(stack_embeddings(feat_db, other), targets, DEVICE,
                                    epochs=args.adapter_epochs, seed=args.seed)
        projected = project(adapter, stack_embeddings(feat_db, fold), DEVICE)
        for j, v in enumerate(fold):
            oof[v] = projected[j]
        print(f"  fold {i + 1}/{args.folds}: fitted on {len(other)}, projected {len(fold)}", flush=True)


def chunk_index(videos):
    chunk_rows, owner = [], []
    for i, v in enumerate(videos):
        e = F.normalize(torch.as_tensor(chunk_db[v]).float().reshape(-1, 512), dim=1)
        chunk_rows.append(e)
        owner += [i] * e.shape[0]
    return ChunkIndex(torch.cat(chunk_rows).to(DEVICE), torch.tensor(owner, device=DEVICE), len(videos))


def branches(videos, audio_source):
    idx, gt = query_rows(rows, videos, DEVICE)
    q = q_emb[idx].to(DEVICE)
    index = chunk_index(videos)
    sim_chunk = index.max_sim_batch(q)
    sims = [zscore(q @ stack_embeddings(vid_db, videos, DEVICE).T),
            zscore(q @ stack_embeddings(txt_db, videos, DEVICE).T),
            zscore(sim_chunk),
            zscore(q @ stack_embeddings(audio_source, videos, DEVICE).T)]
    return {"q": q, "gt": gt, "sims": sims}


fit = branches(fit_vids, oof)       # out-of-fold audio: honest for training
val = branches(val_vids, aud_db)    # deployed adapter never fitted on val
print(f"[DATA] fit queries={fit['q'].shape[0]} val queries={val['q'].shape[0]}", flush=True)
print(f"[TEXT ] passage-branch R@1: val={recall_metrics(val['sims'][2], val['gt'])['R@1']:.4f}", flush=True)
print(f"[AUDIO] audio-only R@1: fit(out-of-fold)={recall_metrics(fit['sims'][3], fit['gt'])['R@1']:.4f} "
      f"val={recall_metrics(val['sims'][3], val['gt'])['R@1']:.4f}", flush=True)


def fuse(w, sims):
    return sum(w[:, i:i + 1] * sims[i] for i in range(len(sims)))


def fixed_baseline():
    """Best fixed weights on validation, as the bar the gate has to clear."""
    grid = [round(x, 2) for x in np.arange(0, 1.01, 0.1)]
    best, best_w = -1.0, None
    for wv in grid:
        for wt in grid:
            for wa in grid:
                w = torch.tensor([[wv, wt, 1.0, wa]], device=DEVICE)
                r1 = recall_metrics(fuse(w, val["sims"]), val["gt"])["R@1"]
                if r1 > best:
                    best, best_w = r1, (wv, wt, 1.0, wa)
    return best, best_w


fixed_r1, fixed_w = fixed_baseline()
config_w = torch.tensor([[AEMS_FUSION_WEIGHTS[b] for b in BRANCHES]], device=DEVICE)
config_r1 = recall_metrics(fuse(config_w, val["sims"]), val["gt"])["R@1"]
print(f"[BASELINE] val R@1: config weights={config_r1:.4f}  best fixed {fixed_w}={fixed_r1:.4f}", flush=True)

def train_seed(seed):
    """Train one gate; returns (best val R@1, best state, best epoch)."""
    set_seeds(seed)
    gate = GatingNetwork(input_dim=512, hidden_dim=args.hidden_dim,
                         num_modalities=len(BRANCHES)).to(DEVICE)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(seed)
    n = fit["q"].shape[0]
    best_r1, best_state, best_epoch = -1.0, None, -1

    print(f"[TRAIN] seed {seed}: {args.epochs} epochs over {n} queries / "
          f"{len(fit_vids)} candidates", flush=True)
    for epoch in range(args.epochs):
        gate.train()
        perm = rng.permutation(n)
        total, nb = 0.0, 0
        for s_idx in range(0, n, args.batch_size):
            idx = torch.as_tensor(perm[s_idx:s_idx + args.batch_size], device=DEVICE)
            w = gate(fit["q"][idx])
            logits = fuse(w, [sim[idx] for sim in fit["sims"]]) / args.temperature
            loss = F.cross_entropy(logits, fit["gt"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        gate.eval()
        with torch.no_grad():
            r1 = recall_metrics(fuse(gate(val["q"]), val["sims"]), val["gt"])["R@1"]
        print(f"  epoch {epoch + 1:02d}  loss={total / nb:.4f}  val R@1={r1:.4f}", flush=True)
        if r1 > best_r1:
            best_r1, best_epoch = r1, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    return best_r1, best_state, best_epoch


def describe(state):
    gate = GatingNetwork(input_dim=512, hidden_dim=args.hidden_dim,
                         num_modalities=len(BRANCHES)).to(DEVICE)
    gate.load_state_dict(state)
    gate.eval()
    with torch.no_grad():
        w_val = gate(val["q"])
        metrics = recall_metrics(fuse(w_val, val["sims"]), val["gt"])
    return metrics, w_val.mean(0).tolist(), w_val.std(0).tolist()


per_seed = {}
best_overall = (-1.0, None, None)
for seed in args.seeds:
    r1, state, epoch = train_seed(seed)
    metrics, mean_w, std_w = describe(state)
    per_seed[seed] = {"val_R@1": r1, "best_epoch": epoch, "val_metrics": metrics,
                      "weight_mean": mean_w, "weight_std": std_w}
    print(f"[SEED {seed}] val R@1={r1:.4f} (epoch {epoch})  "
          f"w={[round(x, 3) for x in mean_w]}", flush=True)
    if r1 > best_overall[0]:
        best_overall = (r1, state, seed)

best_r1, best_state, best_seed = best_overall
metrics, mean_w, std_w = describe(best_state)
seed_r1s = [per_seed[s]["val_R@1"] for s in args.seeds]

print(f"\n[SEEDS] val R@1 mean={np.mean(seed_r1s):.4f} "
      f"std={np.std(seed_r1s, ddof=1) if len(seed_r1s) > 1 else 0.0:.4f} "
      f"over {len(seed_r1s)} seed(s); keeping seed {best_seed} (best on validation)")
print(f"[BEST] val {metrics}")
print(f"[WEIGHTS] mean {'/'.join(BRANCHES)} = {[round(x, 3) for x in mean_w]}  "
      f"std = {[round(x, 3) for x in std_w]}")
if max(mean_w) > 0.95 or max(std_w) < 0.01:
    print("[WARN] the gate is nearly constant — it has collapsed onto one modality "
          "and is not adapting per query.")
if best_r1 <= fixed_r1:
    print(f"[WARN] the gate ({best_r1:.4f}) does not beat tuned fixed weights "
          f"({fixed_r1:.4f}) on validation; prefer --fusion fixed.")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
torch.save(best_state, args.output)
print(f"[SAVE] gating weights -> {args.output}")

summary = {"val_R@1_gate": best_r1, "val_R@1_best_fixed": fixed_r1, "best_fixed_weights": fixed_w,
           "val_R@1_config_weights": config_r1, "val_metrics": metrics,
           "weight_mean": mean_w, "weight_std": std_w, "best_seed": best_seed,
           "seeds": {str(k): v for k, v in per_seed.items()},
           "val_R@1_seed_mean": float(np.mean(seed_r1s)),
           "val_R@1_seed_std": float(np.std(seed_r1s, ddof=1)) if len(seed_r1s) > 1 else 0.0,
           "temperature": args.temperature, "folds": 0 if args.no_cross_fit else args.folds,
           "cross_fitted": not args.no_cross_fit}
os.makedirs("outputs/aems", exist_ok=True)
summary_path = ("outputs/aems/gating_training_summary_no_crossfit.json" if args.no_cross_fit
                else "outputs/aems/gating_training_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[SAVE] summary -> {summary_path}")
