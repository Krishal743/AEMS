"""Train the per-candidate reranker (stage 2, option 1).

Stage 1 produces a shortlist of K candidates per query; this model rescores that
shortlist by predicting fusion weights for each query-candidate pair. Training
uses cross-entropy over the K candidates, so the model only has to separate the
right video from the shortlist rather than from the whole gallery.

Protocol (docs/PROTOCOL.md): the audio branch for training videos is the
cross-fitted out-of-fold projection, shortlists for training come from fixed
fusion weights (not the gate, which was fitted on these same queries), and
selection is on the validation split.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                        AEMS_WAVLM_FEATURES_PATH, AEMS_AUDIO_OOF_PATH,
                        AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
                        AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE, AEMS_TEXT_CHUNKS_PATH_TEMPLATE,
                        AEMS_PER_CANDIDATE_GATE_PATH, DEVICE, set_seeds)
from src.evaluation.evaluate_retrieval import ground_truth_ranks, metrics_from_ranks
from src.rerank import per_candidate, stage1
from src.routing.query_router import BRANCHES, ChunkIndex, fixed_weights, zscore
from src.training.audio_adapter_fit import out_of_fold_audio
from src.training.query_data import (load_records, questions, validation_split, encode_clip_text,
                                     flatten_questions, stack_embeddings, query_rows)

parser = argparse.ArgumentParser(description="Train the per-candidate reranker")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--output", default=AEMS_PER_CANDIDATE_GATE_PATH)
parser.add_argument("--top-k", type=int, default=50, help="shortlist depth to rerank")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden-dim", type=int, default=64)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--folds", type=int, default=4)
parser.add_argument("--adapter-epochs", type=int, default=39)
parser.add_argument("--val-frac", type=float, default=0.15)
parser.add_argument("--seeds", type=int, nargs="+", default=[42])
args = parser.parse_args()
set_seeds(args.seeds[0])

records = load_records(args.manifest, "train")
vid_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
aud_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
txt_db = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"), weights_only=False)
chunk_db = torch.load(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split="train"), weights_only=False)
feat_db = torch.load(AEMS_WAVLM_FEATURES_PATH, weights_only=False)
desc_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split="train"), weights_only=False)

usable = [v for v, r in records.items()
          if v in vid_db and v in aud_db and v in txt_db and v in chunk_db and v in feat_db
          and v in desc_db and questions(r)]
fit_vids, val_vids = validation_split(usable, args.val_frac)
print(f"[DATA] fit={len(fit_vids)} val={len(val_vids)} (test split untouched)", flush=True)

print("[ENC] Encoding QA questions with CLIP...", flush=True)
texts, rows = flatten_questions(records, fit_vids + val_vids)
q_emb = encode_clip_text(texts, DEVICE)

print("[AUDIO] Out-of-fold audio for the training videos...", flush=True)
oof = out_of_fold_audio(feat_db, desc_db, q_emb, rows, fit_vids, DEVICE, folds=args.folds,
                        epochs=args.adapter_epochs, seed=args.seeds[0], cache_path=AEMS_AUDIO_OOF_PATH)


def branch_sims(vids, audio_source):
    idx, gt = query_rows(rows, vids, DEVICE)
    q = q_emb[idx].to(DEVICE)
    chunk_rows, owner = [], []
    for i, v in enumerate(vids):
        e = F.normalize(torch.as_tensor(chunk_db[v]).float().reshape(-1, 512), dim=1)
        chunk_rows.append(e)
        owner += [i] * e.shape[0]
    chunks = ChunkIndex(torch.cat(chunk_rows).to(DEVICE), torch.tensor(owner, device=DEVICE), len(vids))
    sims = [zscore(q @ stack_embeddings(vid_db, vids, DEVICE).T),
            zscore(q @ stack_embeddings(txt_db, vids, DEVICE).T),
            zscore(chunks.max_sim_batch(q)),
            zscore(q @ stack_embeddings(audio_source, vids, DEVICE).T)]
    return {"q": q, "gt": gt, "sims": sims}


def shortlist(data):
    """Stage-1 scores and candidates from fixed weights (never fitted on these queries)."""
    scores = stage1.fuse(fixed_weights().to(DEVICE), data["sims"])
    candidates = stage1.top_k_candidates(scores, args.top_k)
    return scores, candidates, stage1.gather_branch_scores(data["sims"], candidates)


fit = branch_sims(fit_vids, oof)
val = branch_sims(val_vids, aud_db)
fit_scores, fit_cand, fit_feats = shortlist(fit)
val_scores, val_cand, val_feats = shortlist(val)

fit_pos = stage1.positive_position(fit_cand, fit["gt"])
val_pos = stage1.positive_position(val_cand, val["gt"])
trainable = (fit_pos >= 0).nonzero(as_tuple=True)[0]
print(f"[DATA] shortlist K={args.top_k}: stage-1 recall fit="
      f"{stage1.candidate_recall(fit_cand, fit['gt']):.4f} "
      f"val={stage1.candidate_recall(val_cand, val['gt']):.4f}")
print(f"[DATA] training on {trainable.numel()}/{fit_pos.numel()} queries whose answer is in "
      f"the shortlist (the rest cannot be fixed by reranking)", flush=True)

val_stage1_r1 = metrics_from_ranks(ground_truth_ranks(val_scores, val["gt"]))["R@1"]
print(f"[BASELINE] val R@1 stage-1 (fixed weights) = {val_stage1_r1:.4f}", flush=True)


def evaluate(model):
    model.eval()
    with torch.no_grad():
        scores = per_candidate.score(model, val["q"], val_feats)
        full = stage1.rerank_scores_to_ranking(val_scores, val_cand, scores)
    return metrics_from_ranks(ground_truth_ranks(full, val["gt"]))


def train_seed(seed):
    set_seeds(seed)
    model = per_candidate.build(len(BRANCHES), hidden_dim=args.hidden_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(seed)
    best_r1, best_state, best_epoch = -1.0, None, -1
    idx_pool = trainable.cpu().numpy()

    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(idx_pool)
        total, nb = 0.0, 0
        for s in range(0, len(perm), args.batch_size):
            batch = torch.as_tensor(perm[s:s + args.batch_size], device=DEVICE)
            scores = per_candidate.score(model, fit["q"][batch], fit_feats[batch])
            loss = F.cross_entropy(scores / args.temperature, fit_pos[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        metrics = evaluate(model)
        print(f"  epoch {epoch + 1:02d}  loss={total / nb:.4f}  val R@1={metrics['R@1']:.4f}",
              flush=True)
        if metrics["R@1"] > best_r1:
            best_r1, best_epoch = metrics["R@1"], epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_r1, best_state, best_epoch


per_seed, best_overall = {}, (-1.0, None, None)
for seed in args.seeds:
    r1, state, epoch = train_seed(seed)
    per_seed[seed] = {"val_R@1": r1, "best_epoch": epoch}
    print(f"[SEED {seed}] val R@1={r1:.4f} (epoch {epoch})", flush=True)
    if r1 > best_overall[0]:
        best_overall = (r1, state, seed)

best_r1, best_state, best_seed = best_overall
model = per_candidate.build(len(BRANCHES), hidden_dim=args.hidden_dim).to(DEVICE)
model.load_state_dict(best_state)
metrics = evaluate(model)
seed_r1s = [per_seed[s]["val_R@1"] for s in args.seeds]

print(f"\n[SEEDS] val R@1 mean={np.mean(seed_r1s):.4f} "
      f"std={np.std(seed_r1s, ddof=1) if len(seed_r1s) > 1 else 0.0:.4f}; keeping seed {best_seed}")
print(f"[BEST] val {metrics}  (stage-1 was {val_stage1_r1:.4f})")
if best_r1 <= val_stage1_r1:
    print("[WARN] the per-candidate reranker does not improve on stage-1 alone; "
          "do not deploy it.")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
torch.save(best_state, args.output)
print(f"[SAVE] per-candidate reranker -> {args.output}")

summary = {"top_k": args.top_k, "val_R@1_stage1": val_stage1_r1, "val_R@1_rerank": best_r1,
           "val_metrics": metrics, "best_seed": best_seed,
           "seeds": {str(k): v for k, v in per_seed.items()},
           "val_R@1_seed_mean": float(np.mean(seed_r1s)),
           "val_R@1_seed_std": float(np.std(seed_r1s, ddof=1)) if len(seed_r1s) > 1 else 0.0,
           "shortlist_recall_val": stage1.candidate_recall(val_cand, val["gt"]),
           "temperature": args.temperature}
os.makedirs("outputs/aems", exist_ok=True)
with open("outputs/aems/per_candidate_training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("[SAVE] summary -> outputs/aems/per_candidate_training_summary.json")
