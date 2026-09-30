"""Fine-tune the stage-2 cross-encoder on AEMS.

The deployed reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`, trained on web
search queries against web pages. It has never seen a lecture transcript, yet it
is responsible for the largest single gain in the system (+0.125 R@1). Adapting
it to this corpus is the obvious next lever.

Training is listwise: for each query the model scores one correct passage and N
hard negatives — passages from the *other videos in that query's stage-1
shortlist*, i.e. the videos the deployed system actually confuses — and
cross-entropy pushes the correct one to the top.

Selection is on validation R@1 of the **whole two-stage pipeline**, not on pair
accuracy: a model can rank pairs better while the end-to-end ranking gets worse,
and the pipeline number is what we deploy on. The test split is untouched.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                        AEMS_GATING_WEIGHTS_PATH, AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
                        AEMS_TEXT_CHUNKS_PATH_TEMPLATE, DEVICE, set_seeds)
from src.evaluation.evaluate_retrieval import ground_truth_ranks, metrics_from_ranks
from src.rerank import cross_encoder, finetune, stage1
from src.routing.query_router import ChunkIndex, load_gate, zscore
from src.training.query_data import (load_records, questions, validation_split, encode_clip_text,
                                     flatten_questions, stack_embeddings, query_rows)

parser = argparse.ArgumentParser(description="Fine-tune the cross-encoder reranker on AEMS")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--base-model", default=cross_encoder.MINILM)
parser.add_argument("--output", default="models/aems_cross_encoder_v1")
parser.add_argument("--gate-weights", default=AEMS_GATING_WEIGHTS_PATH)
parser.add_argument("--top-k", type=int, default=50, help="shortlist depth negatives come from")
parser.add_argument("--negatives", type=int, default=8)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--queries-per-step", type=int, default=8, help="queries per optimiser step")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max-length", type=int, default=256)
parser.add_argument("--eval-passages", type=int, default=3)
parser.add_argument("--eval-alpha", type=float, default=0.25)
parser.add_argument("--evals-per-epoch", type=int, default=3)
parser.add_argument("--max-queries", type=int, default=None, help="cap training queries (debug)")
parser.add_argument("--val-frac", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seeds(args.seed)

# ---------------------------------------------------------------- data
records = load_records(args.manifest, "train")
vid_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
aud_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
txt_db = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"), weights_only=False)
chunk_db = torch.load(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split="train"), weights_only=False)

usable = [v for v, r in records.items()
          if v in vid_db and v in aud_db and v in txt_db and v in chunk_db and questions(r)]
fit_vids, val_vids = validation_split(usable, args.val_frac)
print(f"[DATA] fit={len(fit_vids)} val={len(val_vids)} (test split untouched)", flush=True)

gate = load_gate(args.gate_weights, DEVICE)


def build_split(vids):
    texts, rows = flatten_questions(records, vids)
    q_emb = encode_clip_text(texts, DEVICE)
    idx, gt = query_rows(rows, vids, DEVICE)
    q = q_emb[idx].to(DEVICE)
    query_texts = [texts[i] for i in idx.tolist()]

    chunk_rows, owner = [], []
    for i, v in enumerate(vids):
        e = F.normalize(torch.as_tensor(chunk_db[v]).float().reshape(-1, 512), dim=1)
        chunk_rows.append(e)
        owner += [i] * e.shape[0]
    chunks = ChunkIndex(torch.cat(chunk_rows).to(DEVICE), torch.tensor(owner, device=DEVICE), len(vids))

    sims = [zscore(q @ stack_embeddings(vid_db, vids, DEVICE).T),
            zscore(q @ stack_embeddings(txt_db, vids, DEVICE).T),
            zscore(chunks.max_sim_batch(q)),
            zscore(q @ stack_embeddings(aud_db, vids, DEVICE).T)]
    with torch.no_grad():
        w = gate(q)
    scores = stage1.fuse(w, sims)
    store = cross_encoder.PassageStore(records, vids, chunk_db, DEVICE)
    return {"vids": vids, "q": q, "gt": gt, "scores": scores, "texts": query_texts, "store": store}


print("[DATA] Building stage-1 scores...", flush=True)
fit = build_split(fit_vids)
val = build_split(val_vids)

fit_candidates, fit_recall = finetune.shortlist_for_split(fit["scores"], fit["gt"], args.top_k)
val_candidates, val_recall = finetune.shortlist_for_split(val["scores"], val["gt"], args.top_k)
print(f"[DATA] shortlist K={args.top_k}: recall fit={fit_recall:.4f} val={val_recall:.4f}", flush=True)

examples = finetune.build_examples(fit["q"], fit_candidates, fit["gt"], fit["store"],
                                   n_negatives=args.negatives,
                                   generator=torch.Generator().manual_seed(args.seed))
if args.max_queries:
    examples = examples[:args.max_queries]
print(f"[DATA] {len(examples)} training queries "
      f"({len(examples) * (args.negatives + 1)} pairs per epoch); "
      f"{fit['gt'].numel() - len(examples)} dropped as unreachable by reranking", flush=True)

# ---------------------------------------------------------------- model
model, tokenizer = cross_encoder.load_cross_encoder(args.base_model, DEVICE, dtype=torch.float32)
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total_steps = max(1, args.epochs * (len(examples) // args.queries_per_step))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=total_steps,
                                                pct_start=0.1)


def evaluate():
    """Validation R@1 of the full two-stage pipeline with the current weights."""
    model.eval()
    scores = cross_encoder.rerank(model, tokenizer, val["texts"], val["q"], val_candidates,
                                  val["store"], val["vids"], n_passages=args.eval_passages,
                                  max_length=args.max_length, device=DEVICE)
    z = (scores - scores.mean(1, keepdim=True)) / (scores.std(1, keepdim=True) + 1e-6)
    base = val["scores"].gather(1, val_candidates)
    full = stage1.rerank_scores_to_ranking(val["scores"], val_candidates,
                                           base + args.eval_alpha * z)
    model.train()
    return metrics_from_ranks(ground_truth_ranks(full, val["gt"]))


baseline = evaluate()
print(f"[BASELINE] off-the-shelf {args.base_model}: val R@1={baseline['R@1']:.4f}", flush=True)

# ---------------------------------------------------------------- train
rng = np.random.default_rng(args.seed)
best_r1, best_state, best_tag = baseline["R@1"], None, "off-the-shelf"
eval_every = max(1, (len(examples) // args.queries_per_step) // args.evals_per_epoch)
step = 0
model.train()

for epoch in range(args.epochs):
    order = rng.permutation(len(examples))
    running, seen = 0.0, 0
    for start in range(0, len(order) - args.queries_per_step + 1, args.queries_per_step):
        batch = [examples[i] for i in order[start:start + args.queries_per_step]]
        texts_a, texts_b = [], []
        for example in batch:
            a, b = finetune.example_to_pairs(example, fit["texts"], fit["store"])
            texts_a += a
            texts_b += b
        encoded = tokenizer(texts_a, texts_b, padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt").to(DEVICE)
        logits = model(**encoded).logits
        logits = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
        # Each query contributes one group: positive first, then its negatives.
        groups = logits.view(len(batch), args.negatives + 1)
        loss = F.cross_entropy(groups, torch.zeros(len(batch), dtype=torch.long, device=DEVICE))
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step < total_steps:
            scheduler.step()
        running += loss.item()
        seen += 1
        step += 1

        if seen % eval_every == 0:
            metrics = evaluate()
            print(f"  epoch {epoch + 1} step {seen}: loss={running / seen:.4f} "
                  f"val R@1={metrics['R@1']:.4f}", flush=True)
            if metrics["R@1"] > best_r1:
                best_r1, best_tag = metrics["R@1"], f"epoch {epoch + 1} step {seen}"
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print(f"\n[BEST] {best_tag}: val R@1={best_r1:.4f} "
      f"(off-the-shelf was {baseline['R@1']:.4f})")

if best_state is None:
    print("[WARN] fine-tuning never beat the off-the-shelf model on validation. "
          "Keeping the off-the-shelf reranker; nothing saved.")
else:
    model.load_state_dict(best_state)
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[SAVE] fine-tuned cross-encoder -> {args.output}")

summary = {"base_model": args.base_model, "val_R@1_off_the_shelf": baseline["R@1"],
           "val_R@1_finetuned": best_r1, "best_checkpoint": best_tag,
           "improved": best_state is not None, "top_k": args.top_k,
           "negatives": args.negatives, "epochs": args.epochs, "lr": args.lr,
           "n_training_queries": len(examples), "shortlist_recall_fit": fit_recall,
           "shortlist_recall_val": val_recall, "seed": args.seed}
os.makedirs("outputs/aems", exist_ok=True)
with open("outputs/aems/cross_encoder_training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("[SAVE] summary -> outputs/aems/cross_encoder_training_summary.json")
