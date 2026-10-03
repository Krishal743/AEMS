"""Fine-tune the dense retriever (E5) on AEMS.

Every other model in the system has been adapted to this corpus — the audio
adapter, the cross-encoder — except the single strongest branch, which is still
off the shelf. Fine-tuning a bi-encoder on in-domain pairs usually gains more
than fine-tuning a reranker did (+1.5 to +2.3 here), and unlike the reranker it
raises stage-1 **recall**, which is the ceiling on everything stage 2 can do.

Training is contrastive: each query is pulled towards its own video's best
passage and pushed away from passages of other videos in its stage-1 shortlist
(hard negatives) plus the rest of the batch (in-batch negatives).

Selection is on validation R@1 of the dense branch alone — the thing being
trained — and the test split is untouched.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (AEMS_MANIFEST_PATH, AEMS_DENSE_CHUNK_TOKENS, AEMS_DENSE_TEXT_MODEL,
                        AEMS_DENSE_PASSAGES_PATH_TEMPLATE, DEVICE, set_seeds)
from src.data.text_chunks import video_chunks
from src.encoders.text_retrieval import MODELS, _pool
from src.evaluation.evaluate_retrieval import ground_truth_ranks, metrics_from_ranks
from src.routing.query_router import chunk_index_from, zscore
from src.training.query_data import (load_records, questions, validation_split,
                                     flatten_questions, query_rows)

parser = argparse.ArgumentParser(description="Fine-tune the dense retrieval encoder on AEMS")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--base-model", default=AEMS_DENSE_TEXT_MODEL)
parser.add_argument("--output", default="models/aems_dense_retriever_v1")
parser.add_argument("--negatives", type=int, default=3, help="hard negatives per query")
parser.add_argument("--queries-per-step", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max-length", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.02)
parser.add_argument("--evals-per-epoch", type=int, default=4)
parser.add_argument("--max-queries", type=int, default=None)
parser.add_argument("--val-frac", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seeds(args.seed)

config = MODELS[args.base_model] if args.base_model in MODELS else None
if config is None:
    raise SystemExit(f"--base-model must be one of {sorted(MODELS)}")
Q_PREFIX, P_PREFIX, POOLING = config["query_prefix"], config["passage_prefix"], config["pooling"]

records = load_records(args.manifest, "train")
usable = [v for v, r in records.items() if questions(r)]
fit_vids, val_vids = validation_split(usable, args.val_frac)
print(f"[DATA] fit={len(fit_vids)} val={len(val_vids)} (test split untouched)", flush=True)

# Passage text per video, chunked exactly as the deployed index is.
passages = {v: video_chunks(records[v], max_chunks=32, budget=AEMS_DENSE_CHUNK_TOKENS)
            for v in fit_vids + val_vids}

from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config["name"])
model = AutoModel.from_pretrained(config["name"]).to(DEVICE)


def embed(texts, prefix, train=False):
    encoded = tokenizer([prefix + t for t in texts], padding=True, truncation=True,
                        max_length=args.max_length, return_tensors="pt").to(DEVICE)
    out = model(**encoded).last_hidden_state
    return F.normalize(_pool(out, encoded["attention_mask"], POOLING), dim=1)


@torch.no_grad()
def embed_eval(texts, prefix, batch_size=128):
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(embed(texts[i:i + batch_size], prefix).float().cpu())
    model.train()
    return torch.cat(out)


val_texts, val_rows = flatten_questions(records, val_vids)
val_idx, val_gt = query_rows(val_rows, val_vids, DEVICE)
val_queries = [val_texts[i] for i in val_idx.tolist()]
val_flat, val_owner = [], []
for i, v in enumerate(val_vids):
    val_flat += passages[v]
    val_owner += [i] * len(passages[v])


def val_r1():
    """R@1 of the dense branch alone on validation — the thing being trained."""
    P = embed_eval(val_flat, P_PREFIX)
    Qe = embed_eval(val_queries, Q_PREFIX)
    index = chunk_index_from({i: P[[j for j, o in enumerate(val_owner) if o == i]]
                              for i in range(len(val_vids))},
                             list(range(len(val_vids))), dim=P.shape[1])
    index = index._replace(rows=index.rows.to(DEVICE), owner=index.owner.to(DEVICE))
    sims = index.max_sim_batch(Qe.to(DEVICE))
    return metrics_from_ranks(ground_truth_ranks(sims, val_gt))["R@1"]


baseline = val_r1()
print(f"[BASELINE] off-the-shelf {config['name']}: val dense-branch R@1={baseline:.4f}", flush=True)

# Training pairs: query -> its own video; hard negatives from other fit videos.
fit_texts, fit_rows = flatten_questions(records, fit_vids)
fit_idx, fit_gt = query_rows(fit_rows, fit_vids, DEVICE)
examples = [(fit_texts[i], int(g)) for i, g in zip(fit_idx.tolist(), fit_gt.tolist())]
if args.max_queries:
    examples = examples[:args.max_queries]
print(f"[DATA] {len(examples)} training queries", flush=True)

optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps = max(1, args.epochs * (len(examples) // args.queries_per_step))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
rng = np.random.default_rng(args.seed)
eval_every = max(1, (len(examples) // args.queries_per_step) // args.evals_per_epoch)
best, best_state, best_tag = baseline, None, "off-the-shelf"
model.train()

for epoch in range(args.epochs):
    order = rng.permutation(len(examples))
    seen, running = 0, 0.0
    for start in range(0, len(order) - args.queries_per_step + 1, args.queries_per_step):
        batch = [examples[i] for i in order[start:start + args.queries_per_step]]
        queries = [q for q, _ in batch]
        docs, labels = [], []
        for slot, (_, video) in enumerate(batch):
            labels.append(len(docs))
            docs.append(rng.choice(passages[fit_vids[video]]))
            others = [v for v in rng.choice(len(fit_vids), args.negatives * 3, replace=False)
                      if v != video][:args.negatives]
            docs += [rng.choice(passages[fit_vids[v]]) for v in others]
        q_emb = embed(queries, Q_PREFIX)
        d_emb = embed(list(docs), P_PREFIX)
        logits = q_emb @ d_emb.T / args.temperature
        loss = F.cross_entropy(logits, torch.tensor(labels, device=DEVICE))
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if seen < steps:
            scheduler.step()
        running += loss.item()
        seen += 1
        if seen % eval_every == 0:
            score = val_r1()
            print(f"  epoch {epoch + 1} step {seen}: loss={running / seen:.4f} "
                  f"val dense R@1={score:.4f}", flush=True)
            if score > best:
                best, best_tag = score, f"epoch {epoch + 1} step {seen}"
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print(f"\n[BEST] {best_tag}: val dense-branch R@1={best:.4f} (off-the-shelf {baseline:.4f})")
if best_state is None:
    print("[WARN] fine-tuning never beat the off-the-shelf encoder on validation; "
          "keeping it and saving nothing.")
else:
    model.load_state_dict(best_state)
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[SAVE] fine-tuned dense retriever -> {args.output}")
    print("       Re-encode passages with bin/embeddings/precompute_dense_passages.py "
          f"--model {args.output} before evaluating the system.")

os.makedirs("outputs/aems", exist_ok=True)
with open("outputs/aems/dense_retriever_training_summary.json", "w") as f:
    json.dump({"base_model": config["name"], "val_R@1_off_the_shelf": baseline,
               "val_R@1_finetuned": best, "best_checkpoint": best_tag,
               "improved": best_state is not None, "negatives": args.negatives,
               "epochs": args.epochs, "lr": args.lr, "seed": args.seed}, f, indent=2)
print("[SAVE] summary -> outputs/aems/dense_retriever_training_summary.json")
