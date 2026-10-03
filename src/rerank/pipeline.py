"""Stage 2 applied to a batch of queries, in one place.

The chain (per-candidate gate, then cross-encoder) was written out twice — once
batched in `bin/evaluation/eval_aems_retrieval.py` and once per-query in
`query_router.apply_rerank` — and a third copy was about to appear in the
subset evaluator. The blend arithmetic is fiddly enough (z-score each stage
per query, add to the stage-1 base, keep non-shortlisted videos below every
shortlisted one) that three copies would drift.
"""

import torch

from src.rerank import stage1


def _z(x):
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)


def rerank_batch(mode, stage1_scores, sims, clip_queries, query_texts, video_ids,
                 records=None, chunk_db=None, gate_path=None, cross_encoder_name=None,
                 top_k=100, n_passages=8, alpha=0.5, beta=0.5, device="cuda",
                 n_branches=None):
    """Rescore each query's top-`top_k` shortlist.

    Returns (scores, candidates). `mode` is none/gate/cross/chain; `sims` are the
    z-scored per-branch matrices the gate reads, in BRANCHES order.
    """
    if mode == "none":
        return stage1_scores, None

    # Callers pass stage-1 scores on either device; everything below is blended
    # onto whichever one they used rather than assuming CPU.
    home = stage1_scores.device
    candidates = stage1.top_k_candidates(stage1_scores, top_k)
    gated_z = None

    if mode in ("gate", "chain"):
        from src.rerank import per_candidate
        model = per_candidate.load(gate_path, device,
                                   n_branches=n_branches or len(sims))
        feats = stage1.gather_branch_scores([s.to(device) for s in sims],
                                            candidates.to(device))
        with torch.no_grad():
            gated = per_candidate.score(model, clip_queries.float().to(device), feats).cpu()
        if mode == "gate":
            return (stage1.rerank_scores_to_ranking(stage1_scores, candidates.to(home),
                                                    gated.to(home)), candidates)
        gated_z = _z(gated).to(home)

    from src.rerank import cross_encoder
    store = cross_encoder.PassageStore(records, video_ids, chunk_db, device)
    model, tokenizer = cross_encoder.load_cross_encoder(cross_encoder_name, device)
    rescored = cross_encoder.rerank(model, tokenizer, query_texts,
                                    clip_queries.float().to(device), candidates.to(device),
                                    store, video_ids, n_passages=n_passages,
                                    device=device).cpu()
    base = stage1_scores.gather(1, candidates.to(home))
    if gated_z is not None:
        base = base + beta * gated_z
    out = stage1.rerank_scores_to_ranking(stage1_scores, candidates.to(home),
                                          base + alpha * _z(rescored).to(home))
    del model, tokenizer, store
    if device == "cuda":
        torch.cuda.empty_cache()
    return out, candidates
