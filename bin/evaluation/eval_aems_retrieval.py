import sys, json, torch, argparse, os, gc, time, random
import clip
import numpy as np
from src.routing.query_router import (load_gate, zscore, fixed_weights, ChunkIndex,
                                      chunk_index_from, BRANCHES)
from src.encoders.text_retrieval import TextRetrievalEncoder
from src.retrieval.bm25 import BM25PassageIndex
from src.data.text_chunks import lexical_fields
from src.evaluation.evaluate_retrieval import (ground_truth_ranks, hits_at_k,
                                               metrics_from_ranks, bootstrap_ci,
                                               paired_bootstrap)
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_PER_CANDIDATE_GATE_PATH,
                         AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH, AEMS_GATING_WEIGHTS_PATH,
                         AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
                         AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
                         AEMS_TEXT_CHUNKS_PATH_TEMPLATE,
                         AEMS_DENSE_PASSAGES_PATH_TEMPLATE, AEMS_DENSE_TEXT_MODEL,
                         DEVICE, set_seeds)
from src.data.metadata import load_metadata, filter_by_split

set_seeds(42)

OUTPUT_DIR = "outputs/aems"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="AEMS Unified Evaluation")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--visual-variant", default="meanpool", choices=["meanpool", "transformer"])
parser.add_argument("--text-variant", default="fused", choices=["description", "transcript", "fused"])
parser.add_argument("--gate-weights", type=str, default=AEMS_GATING_WEIGHTS_PATH)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap confidence intervals")
parser.add_argument("--bootstrap-iters", type=int, default=1000)
parser.add_argument("--num-queries", type=int, default=None, help="Limit queries for debugging")
parser.add_argument("--rerank", choices=["none", "gate", "cross"], default="none",
                    help="also evaluate the two-stage system with this stage-2 reranker")
parser.add_argument("--rerank-top-k", type=int, default=100, help="shortlist depth to rerank")
parser.add_argument("--rerank-passages", type=int, default=8)
parser.add_argument("--rerank-alpha", type=float, default=0.3)
parser.add_argument("--cross-encoder", default=None,
                    help="defaults to the fine-tuned checkpoint when present")
parser.add_argument("--per-candidate-gate", default=AEMS_PER_CANDIDATE_GATE_PATH)
args = parser.parse_args()

set_seeds(args.seed)
TEXT_PATHS = {
    "description": AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
    "transcript": AEMS_TEXT_EMBEDDINGS_TRANS_PATH_TEMPLATE,
    "fused": AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
}

print("=" * 70)
print("AEMS PHASE A — RETRIEVAL EVALUATION")
print(f"  Visual variant: {args.visual_variant}")
print(f"  Text variant:   {args.text_variant}")
print(f"  Bootstrap CIs:   {args.bootstrap}")
print("=" * 70)

records = filter_by_split(load_metadata(args.manifest), split="test")
print(f"[DATA] Test records: {len(records)}")

queries = []
query_video_ids = []
query_categories = []
for rec in records:
    for q in rec["qa_questions"]:
        queries.append(q)
        query_video_ids.append(rec["video_id"])
        query_categories.append(rec["content_fine_category"])

if args.num_queries:
    queries = queries[:args.num_queries]
    query_video_ids = query_video_ids[:args.num_queries]
    query_categories = query_categories[:args.num_queries]

print(f"[DATA] Queries: {len(queries)}")

print("[EMB] Loading embedding DBs...")
if args.visual_variant == "meanpool":
    video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
    visual_ckpt_id = "aems_video_embeddings_v1.pt"
else:
    video_db = torch.load(AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH, weights_only=False)
    visual_ckpt_id = "aems_video_embeddings_transformer_v1.pt"

audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
chunk_db = torch.load(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split="test"), weights_only=False)
dense_db = torch.load(AEMS_DENSE_PASSAGES_PATH_TEMPLATE.format(split="test"), weights_only=False)
text_db = torch.load(TEXT_PATHS[args.text_variant].format(split="test"), weights_only=False)

print(f"[EMB] Visual DB: {len(video_db)} videos")
print(f"[EMB] Audio DB:  {len(audio_db)} videos")
print(f"[EMB] Text DB:   {len(text_db)} videos")

test_video_ids = set(rec["video_id"] for rec in records)
common_vids = sorted(
    set(video_db.keys()) & set(audio_db.keys()) & set(text_db.keys()) & set(chunk_db.keys())
    & set(dense_db.keys()) & test_video_ids
)
print(f"[DATA] Common test videos: {len(common_vids)}")

missing_visual = test_video_ids - set(video_db.keys())
missing_audio = test_video_ids - set(audio_db.keys())
missing_text = test_video_ids - set(text_db.keys())

if missing_visual:
    print(f"  [WARN] {len(missing_visual)} test videos missing from visual DB (skipped in eval)")
if missing_audio:
    print(f"  [WARN] {len(missing_audio)} test videos missing from audio DB (skipped in eval)")
if missing_text:
    print(f"  [WARN] {len(missing_text)} test videos missing from text DB (skipped in eval)")

query_set = set(query_video_ids)
queries_with_gt = [q for q, v in zip(queries, query_video_ids) if v in common_vids]
query_video_ids_filtered = [v for v in query_video_ids if v in common_vids]
query_categories_filtered = [c for q, v, c in zip(queries, query_video_ids, query_categories) if v in common_vids]

skipped_queries = len(queries) - len(queries_with_gt)
if skipped_queries:
    print(f"  [WARN] {skipped_queries} queries skipped (ground-truth video missing from common set)")

queries = queries_with_gt
query_video_ids = query_video_ids_filtered
query_categories = query_categories_filtered

print(f"[DATA] Effective queries after filtering: {len(queries)}")

print("[MODEL] Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


print("[ENC] Encoding queries via CLIP...")
BATCH_SIZE = 64
clip_text_list = []
for i in range(0, len(queries), BATCH_SIZE):
    batch = queries[i:i + BATCH_SIZE]
    tokens = clip.tokenize(batch, truncate=True).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = normalize(emb)
    clip_text_list.append(emb.cpu())
query_clip = torch.cat(clip_text_list, dim=0)
print(f"  CLIP query shape: {query_clip.shape}")

del clip_model
gc.collect()
torch.cuda.empty_cache()

print("[SIM] Building similarity matrices...")
video_matrix = torch.stack([normalize(video_db[v].float().to(DEVICE)) for v in common_vids]).cpu()
audio_matrix = torch.stack([normalize(audio_db[v].float().to(DEVICE)) for v in common_vids]).cpu()
text_matrix = torch.stack([normalize(text_db[v].float().to(DEVICE)) for v in common_vids]).cpu()

del video_db, audio_db, text_db
gc.collect()

sim_v = query_clip.float() @ video_matrix.T
sim_t = query_clip.float() @ text_matrix.T
sim_a = query_clip.float() @ audio_matrix.T  # audio branch is adapter-projected into CLIP text space

chunk_rows, chunk_owner = [], []
for i, v in enumerate(common_vids):
    e = torch.nn.functional.normalize(torch.as_tensor(chunk_db[v]).float().reshape(-1, 512), dim=1)
    chunk_rows.append(e)
    chunk_owner += [i] * e.shape[0]
chunk_index = ChunkIndex(torch.cat(chunk_rows).to(DEVICE), torch.tensor(chunk_owner, device=DEVICE),
                         len(common_vids))
sim_c = chunk_index.max_sim_batch(query_clip.float().to(DEVICE)).cpu()

print(f"[SIM] Scoring the dense branch ({AEMS_DENSE_TEXT_MODEL})...", flush=True)
_dense_encoder = TextRetrievalEncoder(AEMS_DENSE_TEXT_MODEL, DEVICE)
query_dense = _dense_encoder.encode_queries(queries, batch_size=256)
del _dense_encoder
torch.cuda.empty_cache()
_dense_index = chunk_index_from(dense_db, common_vids,
                                dim=torch.as_tensor(dense_db[common_vids[0]]).shape[-1])
_dense_index = _dense_index._replace(rows=_dense_index.rows.to(DEVICE),
                                     owner=_dense_index.owner.to(DEVICE))
sim_d = _dense_index.max_sim_batch(query_dense.to(DEVICE)).cpu()
del _dense_index
torch.cuda.empty_cache()

print("[SIM] Scoring the BM25 lexical branch...", flush=True)
bm25_records = {r["video_id"]: r for r in load_metadata(args.manifest)}
bm25_index = BM25PassageIndex([lexical_fields(bm25_records[v]) for v in common_vids])
sim_b = bm25_index.score_batch(queries)
del chunk_rows, chunk_index, chunk_db
gc.collect()

print(f"[SIM] sim_v: {sim_v.shape}, sim_t: {sim_t.shape}, sim_a: {sim_a.shape}")

print("[GATE] Loading gating network...")
gate = None
gate_available = os.path.exists(args.gate_weights)
if gate_available:
    gate = load_gate(args.gate_weights, DEVICE)
    print("  Gating weights loaded.")
else:
    print("  No gating weights found — adaptive gating will report zeros.")

sim_gated = None
if gate is not None:
    sim_gated_list = []
    for i in range(0, len(queries), BATCH_SIZE):
        q = query_clip[i:i+BATCH_SIZE].float().to(DEVICE)
        with torch.no_grad():
            w = gate(q).cpu()
        # same fusion as the router: gate weights over per-query z-scored branches
        gated = sum(w[:, b:b + 1] * zscore(sim[i:i+BATCH_SIZE])
                    for b, sim in enumerate((sim_v, sim_t, sim_c, sim_a, sim_b, sim_d)))
        sim_gated_list.append(gated)
    sim_gated = torch.cat(sim_gated_list, dim=0)

systems = {
    "visual_only": sim_v,
    "text_only": sim_t,
    "passage_only": sim_c,
    "audio_only": sim_a,
    "bm25_only": sim_b,
    "dense_only": sim_d,
    "equal_fusion": (sim_v + sim_t + sim_c + sim_a + sim_b + sim_d) / 6,
}
branch_z = [zscore(s) for s in (sim_v, sim_t, sim_c, sim_a, sim_b, sim_d)]
fw = fixed_weights()
systems["fixed_fusion"] = sum(fw[i] * branch_z[i] for i in range(len(branch_z)))
if sim_gated is not None:
    systems["adaptive_gating"] = sim_gated

if args.rerank != "none":
    # Stage 2 rescores only the shortlist of the best stage-1 system available.
    from src.rerank import stage1 as shortlist_utils
    stage1_scores = systems["fixed_fusion"]   # beats the gate on validation
    candidates = shortlist_utils.top_k_candidates(stage1_scores, args.rerank_top_k)
    gt_pos = torch.tensor([common_vids.index(v) for v in query_video_ids])
    print(f"[RERANK] {args.rerank}: shortlist K={args.rerank_top_k}, "
          f"stage-1 recall@K={shortlist_utils.candidate_recall(candidates, gt_pos):.4f}", flush=True)

    if args.rerank == "gate":
        from src.rerank import per_candidate
        model = per_candidate.load(args.per_candidate_gate, DEVICE, n_branches=len(branch_z))
        feats = shortlist_utils.gather_branch_scores(
            [zscore(s) for s in (sim_v, sim_t, sim_c, sim_a, sim_b, sim_d)],
            candidates.to(DEVICE))
        with torch.no_grad():
            rescored = per_candidate.score(model, query_clip.float().to(DEVICE), feats).cpu()
        systems["reranked_gate"] = shortlist_utils.rerank_scores_to_ranking(
            stage1_scores, candidates, rescored)
    else:
        from src.rerank import cross_encoder
        records = {r["video_id"]: r for r in load_metadata(args.manifest)}
        chunk_db_rr = torch.load(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split="test"),
                                 weights_only=False)
        store = cross_encoder.PassageStore(records, common_vids, chunk_db_rr, DEVICE)
        from src.routing.query_router import CROSS_ENCODER_MODEL
        ce_model, tokenizer = cross_encoder.load_cross_encoder(
            args.cross_encoder or CROSS_ENCODER_MODEL, DEVICE)
        rescored = cross_encoder.rerank(ce_model, tokenizer, queries,
                                        query_clip.float().to(DEVICE), candidates.to(DEVICE),
                                        store, common_vids, n_passages=args.rerank_passages,
                                        device=DEVICE).cpu()
        z = (rescored - rescored.mean(1, keepdim=True)) / (rescored.std(1, keepdim=True) + 1e-6)
        base = stage1_scores.gather(1, candidates)
        systems["reranked_cross"] = shortlist_utils.rerank_scores_to_ranking(
            stage1_scores, candidates, base + args.rerank_alpha * z)
        del ce_model, tokenizer, store
        torch.cuda.empty_cache()


REFERENCE_SYSTEM = "text_only"   # what headline gains are quoted against
KS = (1, 5, 10)

gt_index = torch.tensor([common_vids.index(v) for v in query_video_ids])


results = {
    "dataset": "aems",
    "n_queries": len(queries),
    "n_videos": len(common_vids),
    "n_skipped_queries": skipped_queries,
    "visual_variant": args.visual_variant,
    "text_variant": args.text_variant,
    "checkpoint_ids": {
        "visual": visual_ckpt_id,
        "audio": os.path.basename(AEMS_AUDIO_EMBEDDINGS_PATH),
        "text": f"aems_text_embeddings_{args.text_variant}_test.pt",
        "chunks": os.path.basename(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split="test")),
        "bm25": "BM25 over the same passages (no checkpoint)",
        "dense": f"{AEMS_DENSE_TEXT_MODEL} passages: "
                 f"{os.path.basename(AEMS_DENSE_PASSAGES_PATH_TEMPLATE.format(split='test'))}",
        "gating_weights": os.path.basename(args.gate_weights) if gate_available else None,
    },
    "systems": {},
    "category_stratified": {},
}

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

# Rank every system once; every metric below is a reduction over these ranks.
ranks = {name: ground_truth_ranks(sim, gt_index).cpu() for name, sim in systems.items()}

for name, sim in systems.items():
    metrics = metrics_from_ranks(ranks[name], ks=KS)
    sys_entry = dict(metrics)

    if args.bootstrap:
        for k in KS:
            low, high = bootstrap_ci(hits_at_k(ranks[name], k), iters=args.bootstrap_iters)
            sys_entry[f"R@{k}_CI95"] = [low, high]
        if name != REFERENCE_SYSTEM and REFERENCE_SYSTEM in ranks:
            delta, low, high = paired_bootstrap(hits_at_k(ranks[name], 1),
                                                hits_at_k(ranks[REFERENCE_SYSTEM], 1),
                                                iters=args.bootstrap_iters)
            sys_entry[f"R@1_delta_vs_{REFERENCE_SYSTEM}"] = [delta, low, high]

    if name == "adaptive_gating" and gate is not None:
        with torch.no_grad():
            w_all = gate(query_clip.float().to(DEVICE)).cpu()
        sys_entry["w_v_mean"] = float(w_all[:, 0].mean())
        sys_entry["w_v_std"] = float(w_all[:, 0].std())
        sys_entry["w_t_mean"] = float(w_all[:, 1].mean())
        sys_entry["w_t_std"] = float(w_all[:, 1].std())
        for b, name_b in enumerate(BRANCHES):
            sys_entry[f"w_{name_b}_mean"] = float(w_all[:, b].mean())
            sys_entry[f"w_{name_b}_std"] = float(w_all[:, b].std())

    results["systems"][name] = sys_entry

    line = (f"  {name:>20}: R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  "
            f"R@10={metrics['R@10']:.4f}  MRR={metrics['MRR']:.4f}  MdR={metrics['MdR']}")
    if args.bootstrap:
        ci = sys_entry.get("R@1_CI95", [0, 0])
        line += f"  R@1_CI=[{ci[0]:.4f},{ci[1]:.4f}]"
        d = sys_entry.get(f"R@1_delta_vs_{REFERENCE_SYSTEM}")
        if d:
            sig = "sig" if (d[1] > 0 or d[2] < 0) else "n.s."
            line += f"  Δvs{REFERENCE_SYSTEM}={d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] {sig}"
    print(line)

if gate_available and gate is not None:
    with torch.no_grad():
        w_all = gate(query_clip.float().to(DEVICE)).cpu()
    print(f"\n  Gate weights (all queries):")
    for b, name_b in enumerate(BRANCHES):
        print(f"    w_{name_b:<7}: {w_all[:, b].mean():.4f} ± {w_all[:, b].std():.4f}")

print("\n" + "-" * 70)
print("CATEGORY-STRATIFIED R@1")
print("-" * 70)

categories = sorted(set(query_categories))
category_of = torch.tensor([categories.index(c) for c in query_categories])
for ci, cat in enumerate(categories):
    mask = (category_of == ci).nonzero(as_tuple=True)[0]
    cat_row = {"category": cat, "n_queries": int(mask.numel())}
    for name in systems:
        m = metrics_from_ranks(ranks[name][mask], ks=KS)
        for k in KS:
            cat_row[f"{name}_R@{k}"] = m[f"R@{k}"]
    if "adaptive_gating" in systems and "fixed_fusion" in systems:
        cat_row["gate_minus_fixed_R@1"] = (cat_row["adaptive_gating_R@1"]
                                           - cat_row["fixed_fusion_R@1"])
    results["category_stratified"][cat] = cat_row
    best_system = max(systems, key=lambda n: cat_row[f"{n}_R@1"])
    print(f"  {cat:>30} (n={cat_row['n_queries']:>4}): "
          f"best={best_system} {cat_row[f'{best_system}_R@1']:.4f}  "
          f"deployed={cat_row.get('reranked_cross_R@1', cat_row['adaptive_gating_R@1']):.4f}  "
          f"gate-vs-fixed={cat_row.get('gate_minus_fixed_R@1', 0):+.4f}")

output_path = os.path.join(OUTPUT_DIR, f"eval_results_{args.visual_variant}_{args.text_variant}_v1.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {output_path}")

summary_path = os.path.join(OUTPUT_DIR,
                            f"summary_table_{args.visual_variant}_{args.text_variant}_v1.md")
with open(summary_path, "w") as f:
    f.write(f"# AEMS Phase A — Retrieval Results\n\n")
    f.write(f"**Configuration**: visual={args.visual_variant}, text={args.text_variant}, "
            f"n_queries={len(queries)}, n_videos={len(common_vids)}\n\n")
    f.write(f"**Checkpoints**:\n")
    for k, v in results["checkpoint_ids"].items():
        f.write(f"- {k}: {v}\n")
    f.write(f"\n| System | R@1 | R@5 | R@10 | MRR | Δ R@1 vs {REFERENCE_SYSTEM} |\n")
    f.write(f"|---|---|---|---|---|---|\n")
    for name in systems:
        r = results["systems"][name]
        ci_str = ""
        if args.bootstrap and "R@1_CI95" in r:
            ci_str = f" [{r['R@1_CI95'][0]:.4f}, {r['R@1_CI95'][1]:.4f}]"
        d = r.get(f"R@1_delta_vs_{REFERENCE_SYSTEM}")
        delta_str = "—" if not d else (f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"
                                       + ("" if (d[1] > 0 or d[2] < 0) else " n.s."))
        f.write(f"| {name} | {r['R@1']:.4f}{ci_str} | {r['R@5']:.4f} | {r['R@10']:.4f} "
                f"| {r['MRR']:.4f} | {delta_str} |\n")
    ag = results["systems"].get("adaptive_gating", {})
    if f"w_{BRANCHES[0]}_mean" in ag:
        weights_text = ", ".join(f"{b}={ag[f'w_{b}_mean']:.4f}±{ag[f'w_{b}_std']:.4f}"
                                 for b in BRANCHES)
        f.write(f"\n**Gating weights**: {weights_text}\n")
    f.write(f"\n## Category-stratified R@1\n")
    f.write(f"| Category | n_q |")
    for name in systems:
        f.write(f" {name}_R@1 |")
    f.write(f"\n|---|----|")
    for _ in systems:
        f.write(f"---|")
    f.write(f"\n")
    for cat_row in results["category_stratified"].values():
        f.write(f"| {cat_row['category']} | {cat_row['n_queries']} |")
        for name in systems:
            f.write(f" {cat_row[f'{name}_R@1']:.4f} |")
        f.write(f"\n")
print(f"Summary table: {summary_path}")

if gate_available and gate is not None:
    weight_summary = {
        **{f"w_{n}_{stat}": float(getattr(w_all[:, i], stat)())
           for i, n in enumerate(BRANCHES) for stat in ("mean", "std")},
    }
    weight_path = os.path.join(OUTPUT_DIR, "gating_weights_summary.json")
    with open(weight_path, "w") as f:
        json.dump(weight_summary, f, indent=2)
    print(f"Gating weights summary: {weight_path}")

print("\n[DONE] Evaluation complete.")
