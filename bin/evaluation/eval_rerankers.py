"""Compare stage-2 rerankers: per-candidate gating vs pretrained cross-encoders.

Stage 1 is the deployed fused search. Each reranker rescores its top K. Every
knob (K, alpha, passages per candidate) is tuned on validation; test is scored
once. Results carry bootstrap CIs, paired deltas against stage 1, and median
per-query latency, because a reranker that wins by a point at 50x the cost is a
different decision from one that wins for free.

The gain is also split by whether a question is answerable from the transcript:
only ~40% are, and a text reranker cannot help the visual ones ("What colour is
the shirt worn by Student 1?"). One blended average would hide that.
"""

import argparse, json, os, re, time
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                        AEMS_GATING_WEIGHTS_PATH, AEMS_PER_CANDIDATE_GATE_PATH,
                        AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, AEMS_TEXT_CHUNKS_PATH_TEMPLATE,
                        DEVICE, set_seeds)
from src.evaluation.evaluate_retrieval import (ground_truth_ranks, hits_at_k, metrics_from_ranks,
                                               bootstrap_ci, paired_bootstrap)
from src.rerank import cross_encoder, per_candidate, stage1
from src.routing.query_router import BRANCHES, ChunkIndex, load_gate, zscore
from src.training.query_data import (load_records, questions, validation_split, encode_clip_text,
                                     flatten_questions, stack_embeddings, query_rows)

parser = argparse.ArgumentParser(description="Compare stage-2 rerankers")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--gate-weights", default=AEMS_GATING_WEIGHTS_PATH)
parser.add_argument("--per-candidate-gate", default=AEMS_PER_CANDIDATE_GATE_PATH)
parser.add_argument("--models", nargs="+", default=["minilm"], choices=["minilm", "bge"])
parser.add_argument("--top-k", type=int, nargs="+", default=[20, 50, 100],
                    help="shortlist depths to try on validation")
parser.add_argument("--passages", type=int, nargs="+", default=[1, 3],
                    help="passages per candidate to try on validation")
parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
parser.add_argument("--bootstrap-iters", type=int, default=1000)
parser.add_argument("--output", default="outputs/aems/reranker_comparison.json")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seeds(args.seed)

CHECKPOINTS = {"minilm": cross_encoder.MINILM, "bge": cross_encoder.BGE}

records = {s: load_records(args.manifest, s) for s in ("train", "test")}
vid_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
aud_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
txt_db = {s: torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split=s), weights_only=False)
          for s in ("train", "test")}
chunk_db = {s: torch.load(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split=s), weights_only=False)
            for s in ("train", "test")}
gate = load_gate(args.gate_weights, DEVICE)


def usable(split):
    return [v for v, r in records[split].items()
            if v in vid_db and v in aud_db and v in txt_db[split] and v in chunk_db[split]
            and questions(r)]


_, val_vids = validation_split(usable("train"))
test_vids = sorted(usable("test"))
splits = {"val": (val_vids, "train"), "test": (test_vids, "test")}


def answerable_from_text(record, question_index):
    """Rough flag: do most of the answer's content words appear in the transcript?"""
    answers = record.get("qa_answers") or []
    if question_index >= len(answers):
        return False
    transcript = (record.get("text_transcript") or "").lower()
    words = [w for w in re.findall(r"\w+", answers[question_index].lower()) if len(w) > 4]
    return bool(words) and sum(w in transcript for w in words) / len(words) > 0.5


def build(split_name):
    vids, split = splits[split_name]
    texts, rows = flatten_questions(records[split], vids)
    q_emb = encode_clip_text(texts, DEVICE)
    idx, gt = query_rows(rows, vids, DEVICE)
    q = q_emb[idx].to(DEVICE)
    query_texts = [texts[i] for i in idx.tolist()]

    chunk_rows, owner = [], []
    for i, v in enumerate(vids):
        e = F.normalize(torch.as_tensor(chunk_db[split][v]).float().reshape(-1, 512), dim=1)
        chunk_rows.append(e)
        owner += [i] * e.shape[0]
    chunks = ChunkIndex(torch.cat(chunk_rows).to(DEVICE), torch.tensor(owner, device=DEVICE), len(vids))

    sims = [zscore(q @ stack_embeddings(vid_db, vids, DEVICE).T),
            zscore(q @ stack_embeddings(txt_db[split], vids, DEVICE).T),
            zscore(chunks.max_sim_batch(q)),
            zscore(q @ stack_embeddings(aud_db, vids, DEVICE).T)]
    with torch.no_grad():
        w = gate(q)
    scores = stage1.fuse(w, sims)

    flags = []
    for v in vids:
        for qi in range(len(questions(records[split][v]))):
            flags.append(answerable_from_text(records[split][v], qi))
    return {"vids": vids, "split": split, "q": q, "gt": gt, "sims": sims, "scores": scores,
            "texts": query_texts, "text_answerable": torch.tensor(flags, device=DEVICE),
            "store": cross_encoder.PassageStore(records[split], vids, chunk_db[split], DEVICE)}


print("[DATA] Building branch scores...", flush=True)
data = {name: build(name) for name in splits}
for name in splits:
    d = data[name]
    print(f"  {name}: {len(d['vids'])} videos, {d['gt'].numel()} queries, "
          f"{d['text_answerable'].float().mean():.1%} answerable from transcript", flush=True)

results = {"stage1": {}}
for name in splits:
    d = data[name]
    results["stage1"][name] = metrics_from_ranks(ground_truth_ranks(d["scores"], d["gt"]))
    results["stage1"][f"{name}_recall_at_k"] = {
        str(k): stage1.candidate_recall(stage1.top_k_candidates(d["scores"], k), d["gt"])
        for k in args.top_k}
print(f"[STAGE 1] val R@1={results['stage1']['val']['R@1']:.4f} "
      f"test R@1={results['stage1']['test']['R@1']:.4f}", flush=True)


def combined(split_name, k, rerank_scores, alpha, candidates):
    """stage1 + alpha * z-scored reranker score, over the shortlist only."""
    d = data[split_name]
    z = (rerank_scores - rerank_scores.mean(1, keepdim=True)) / (rerank_scores.std(1, keepdim=True) + 1e-6)
    base = d["scores"].gather(1, candidates)
    return stage1.rerank_scores_to_ranking(d["scores"], candidates, base + alpha * z)


def single_query_latency(fn, n=25):
    """Median wall-clock ms for one query, which is what a user waits.

    Timing a fully batched pass and dividing by the query count understates a
    reranker by orders of magnitude, so each call here scores one query.
    """
    times = []
    for i in range(n):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(i)
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return float(np.median(times))


def report(tag, split_name, full_scores, extra=None, latency_ms=None):
    d = data[split_name]
    ranks = ground_truth_ranks(full_scores, d["gt"]).cpu()
    entry = dict(metrics_from_ranks(ranks))
    if split_name == "test":
        hits = hits_at_k(ranks, 1)
        base = hits_at_k(ground_truth_ranks(d["scores"], d["gt"]).cpu(), 1)
        entry["R@1_CI95"] = list(bootstrap_ci(hits, iters=args.bootstrap_iters))
        entry["R@1_delta_vs_stage1"] = list(paired_bootstrap(hits, base, iters=args.bootstrap_iters))
        entry["R@5_delta_vs_stage1"] = list(paired_bootstrap(
            hits_at_k(ranks, 5), hits_at_k(ground_truth_ranks(d["scores"], d["gt"]).cpu(), 5),
            iters=args.bootstrap_iters))
        mask = d["text_answerable"].cpu()
        for label, sel in (("text_answerable", mask), ("visual_only", ~mask)):
            if sel.any():
                entry[f"R@1_{label}"] = hits_at_k(ranks[sel], 1).mean().item()
                entry[f"R@1_{label}_delta"] = list(paired_bootstrap(
                    hits_at_k(ranks[sel], 1), base[sel], iters=args.bootstrap_iters))
    if latency_ms is not None:
        entry["latency_ms_per_query"] = latency_ms
    if extra:
        entry.update(extra)
    results.setdefault(tag, {})[split_name] = entry
    return entry


# ---------------------------------------------------------------- option 1
if os.path.exists(args.per_candidate_gate):
    model = per_candidate.load(args.per_candidate_gate, DEVICE, n_branches=len(BRANCHES))
    best = (-1.0, None)
    for k in args.top_k:
        d = data["val"]
        candidates = stage1.top_k_candidates(d["scores"], k)
        feats = stage1.gather_branch_scores(d["sims"], candidates)
        with torch.no_grad():
            scores = per_candidate.score(model, d["q"], feats)
        full = stage1.rerank_scores_to_ranking(d["scores"], candidates, scores)
        r1 = metrics_from_ranks(ground_truth_ranks(full, d["gt"]))["R@1"]
        print(f"  [per-candidate] val K={k}: R@1={r1:.4f}", flush=True)
        if r1 > best[0]:
            best = (r1, k)
    val_r1, k = best
    for split_name in ("val", "test"):
        d = data[split_name]
        candidates = stage1.top_k_candidates(d["scores"], k)
        feats = stage1.gather_branch_scores(d["sims"], candidates)
        with torch.no_grad():
            scores = per_candidate.score(model, d["q"], feats)
        full = stage1.rerank_scores_to_ranking(d["scores"], candidates, scores)

        def one(i, d=d, k=k):
            c = stage1.top_k_candidates(d["scores"][i:i + 1], k)
            f = stage1.gather_branch_scores([s[i:i + 1] for s in d["sims"]], c)
            with torch.no_grad():
                per_candidate.score(model, d["q"][i:i + 1], f)

        latency = single_query_latency(one)
        entry = report("per_candidate", split_name, full, {"top_k": k}, latency)
        print(f"[PER-CANDIDATE] {split_name} R@1={entry['R@1']:.4f} "
              f"({latency:.2f} ms/query)", flush=True)
else:
    print(f"[skip] {args.per_candidate_gate} not found", flush=True)

# ---------------------------------------------------------------- option 2
for key in args.models:
    name = CHECKPOINTS[key]
    print(f"[CROSS-ENCODER] loading {name}...", flush=True)
    ce_model, tokenizer = cross_encoder.load_cross_encoder(name, DEVICE)

    best = (-1.0, None)
    for k in args.top_k:
        for n_pass in args.passages:
            d = data["val"]
            candidates = stage1.top_k_candidates(d["scores"], k)
            scores = cross_encoder.rerank(ce_model, tokenizer, d["texts"], d["q"], candidates,
                                          d["store"], d["vids"], n_passages=n_pass, device=DEVICE)
            for alpha in args.alphas:
                full = combined("val", k, scores, alpha, candidates)
                r1 = metrics_from_ranks(ground_truth_ranks(full, d["gt"]))["R@1"]
                if r1 > best[0]:
                    best = (r1, (k, n_pass, alpha))
            print(f"  [{key}] val K={k} passages={n_pass}: best so far R@1={best[0]:.4f} "
                  f"{best[1]}", flush=True)
    val_r1, (k, n_pass, alpha) = best

    for split_name in ("val", "test"):
        d = data[split_name]
        candidates = stage1.top_k_candidates(d["scores"], k)
        scores = cross_encoder.rerank(ce_model, tokenizer, d["texts"], d["q"], candidates,
                                      d["store"], d["vids"], n_passages=n_pass, device=DEVICE)
        full = combined(split_name, k, scores, alpha, candidates)

        def one(i, d=d, k=k, n_pass=n_pass):
            c = stage1.top_k_candidates(d["scores"][i:i + 1], k)
            cross_encoder.rerank(ce_model, tokenizer, d["texts"][i:i + 1], d["q"][i:i + 1], c,
                                 d["store"], d["vids"], n_passages=n_pass, device=DEVICE)

        latency = single_query_latency(one, n=10)
        entry = report(key, split_name, full,
                       {"top_k": k, "passages": n_pass, "alpha": alpha, "checkpoint": name},
                       latency)
        print(f"[{key.upper()}] {split_name} R@1={entry['R@1']:.4f} ({latency:.1f} ms/query)",
              flush=True)
    del ce_model, tokenizer
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- summary
print("\n" + "=" * 104)
print(f"{'system':<16}{'test R@1 [95% CI]':<28}{'Δ vs stage 1':<24}"
      f"{'R@1 text-ans.':<16}{'R@1 visual':<12}{'ms/query':>9}")
print("=" * 104)
s1 = results["stage1"]["test"]
print(f"{'stage 1':<16}{s1['R@1']:.4f}{'':<22}{'—':<24}"
      f"{'':<16}{'':<12}{'':>9}")
for tag in [t for t in results if t != "stage1"]:
    e = results[tag]["test"]
    ci = e.get("R@1_CI95", [0, 0])
    d = e.get("R@1_delta_vs_stage1", [0, 0, 0])
    sig = "sig" if (d[1] > 0 or d[2] < 0) else "n.s."
    print(f"{tag:<16}{e['R@1']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]      "
          f"{d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] {sig:<4}"
          f"{e.get('R@1_text_answerable', 0):.4f}{'':<10}"
          f"{e.get('R@1_visual_only', 0):.4f}{'':<6}{e.get('latency_ms_per_query', 0):>8.1f}")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVE] {args.output}")
