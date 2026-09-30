"""Training data for fine-tuning the cross-encoder on AEMS.

The deployed reranker is trained on MS-MARCO web search and has never seen a
lecture transcript. Fine-tuning needs (query, positive passage, hard negatives)
triples, where the negatives are the passages the *deployed system actually
confuses* — passages from other videos in the query's stage-1 shortlist. Random
negatives teach almost nothing, because the model already separates a lecture on
thermodynamics from one on Renaissance art; the hard cases are the near misses.
"""

import torch

from src.rerank import stage1


def build_examples(query_emb, candidates, gt, store, n_negatives=8, generator=None):
    """One training example per query: a positive passage and hard negatives.

    query_emb:  (n_queries, 512) CLIP query embeddings
    candidates: (n_queries, K) stage-1 shortlist, as video indices
    gt:         (n_queries,) index of the correct video for each query
    store:      PassageStore for this split

    Queries whose correct video is absent from the shortlist are dropped: there
    is no positive to learn from, and those are exactly the queries reranking
    cannot fix. Each example records the query row, the positive
    (video, passage) pair, and the negative pairs.
    """
    generator = generator or torch.Generator().manual_seed(0)
    gt = torch.as_tensor(gt, device=candidates.device).view(-1)
    in_shortlist = (candidates == gt.view(-1, 1))
    keep = in_shortlist.any(dim=1).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return []

    # The best passage of every shortlisted video, by CLIP similarity.
    best = store.top_passages(query_emb, candidates, 1)

    examples = []
    for row in keep.tolist():
        positive_slot = int(in_shortlist[row].float().argmax())
        negative_slots = [c for c in range(candidates.shape[1]) if c != positive_slot]
        if not negative_slots:
            continue
        picked = torch.randperm(len(negative_slots), generator=generator)[:n_negatives]
        negatives = [(int(candidates[row, negative_slots[j]]),
                      int(best[row, negative_slots[j], 0])) for j in picked.tolist()]
        if not negatives:
            continue
        examples.append({"query_row": row,
                         "positive": (int(gt[row]), int(best[row, positive_slot, 0])),
                         "negatives": negatives})
    return examples


def example_to_pairs(example, queries, store):
    """(texts_a, texts_b) for one example: the positive first, then negatives."""
    query = queries[example["query_row"]]
    pairs = [example["positive"]] + example["negatives"]
    texts_b = [store.texts[video][passage] if passage < len(store.texts[video])
               else "" for video, passage in pairs]
    return [query] * len(pairs), texts_b


def shortlist_for_split(scores, gt, k):
    """Stage-1 shortlist and the fraction of queries whose answer it contains."""
    candidates = stage1.top_k_candidates(scores, k)
    return candidates, stage1.candidate_recall(candidates, gt)
