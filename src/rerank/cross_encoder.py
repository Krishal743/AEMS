"""Option 2: a pretrained cross-encoder reranker.

Every branch in stage 1 is a bi-encoder: query and video are embedded
independently and compared with a dot product, so the query never "reads" the
transcript. A cross-encoder puts the query and a passage through one model
together, letting every query token attend to every passage token. That is far
more accurate and far too slow to run over the whole gallery, which is why it
only ever sees stage 1's shortlist.

Scoring is per passage, not per video: a candidate's score is its best-matching
passage, mirroring the max-sim passage branch. Passages are the same chunks the
passage branch indexes (`src.data.text_chunks.video_chunks`), and only the top
few per candidate by CLIP similarity are sent to the cross-encoder, since
scoring all ~30 would cost 10x for passages stage 1 already ranked last.
"""

import torch
import torch.nn.functional as F

MINILM = "cross-encoder/ms-marco-MiniLM-L6-v2"
BGE = "BAAI/bge-reranker-v2-m3"


def load_cross_encoder(model_name=MINILM, device="cuda", dtype=torch.float16):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)
    model.to(device).eval()
    return model, tokenizer


class PassageStore:
    """Chunk texts and embeddings per video, aligned with the passage branch."""

    def __init__(self, records, video_ids, chunk_db, device="cuda"):
        from src.data.text_chunks import video_chunks
        self.device = device
        self.texts = [video_chunks(records[v]) for v in video_ids]
        lengths = [torch.as_tensor(chunk_db[v]).reshape(-1, 512).shape[0] for v in video_ids]
        self.max_chunks = max(lengths)
        padded = torch.zeros(len(video_ids), self.max_chunks, 512)
        mask = torch.zeros(len(video_ids), self.max_chunks, dtype=torch.bool)
        for i, v in enumerate(video_ids):
            e = F.normalize(torch.as_tensor(chunk_db[v]).float().reshape(-1, 512), dim=1)
            padded[i, :e.shape[0]] = e
            mask[i, :e.shape[0]] = True
            # A video's stored embeddings and its recomputed texts must line up,
            # or the cross-encoder would score the wrong passage.
            if len(self.texts[i]) != e.shape[0]:
                raise ValueError(f"{v}: {len(self.texts[i])} chunk texts but {e.shape[0]} embeddings; "
                                 f"re-run bin/embeddings/precompute_text_chunks.py")
        self.embeddings = padded.to(device)
        self.mask = mask.to(device)

    def top_passages(self, query_emb, candidates, n_passages):
        """(n_queries, k, n_passages) chunk indices, best CLIP match first."""
        emb = self.embeddings[candidates]                       # (b, k, max_chunks, 512)
        mask = self.mask[candidates]                            # (b, k, max_chunks)
        sims = torch.einsum("bd,bkcd->bkc", query_emb.to(emb.dtype), emb)
        sims = sims.masked_fill(~mask, -1e4)
        return sims.topk(min(n_passages, sims.shape[-1]), dim=-1).indices


@torch.no_grad()
def rerank(model, tokenizer, queries, query_emb, candidates, store, video_ids,
           n_passages=3, batch_size=256, max_length=256, query_batch=32, device="cuda"):
    """(n_queries, k) cross-encoder scores: each candidate's best passage.

    `queries` are the raw query strings, `candidates` the (n_queries, k) indices
    from stage 1, and `video_ids` maps those indices back to videos.
    """
    n_queries, k = candidates.shape
    out = torch.full((n_queries, k), -1e4, device=device)

    for start in range(0, n_queries, query_batch):
        stop = min(start + query_batch, n_queries)
        chunk_ids = store.top_passages(query_emb[start:stop], candidates[start:stop], n_passages)
        pairs, slots = [], []
        for qi in range(stop - start):
            for ci in range(k):
                video = candidates[start + qi, ci].item()
                for pi in chunk_ids[qi, ci].tolist():
                    if pi < len(store.texts[video]):
                        pairs.append((queries[start + qi], store.texts[video][pi]))
                        slots.append((qi, ci))
        if not pairs:
            continue
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            encoded = tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                padding=True, truncation=True, max_length=max_length,
                                return_tensors="pt").to(device)
            logits = model(**encoded).logits
            scores.append(logits[:, 0].float() if logits.shape[-1] == 1 else logits[:, -1].float())
        scores = torch.cat(scores)
        slot_index = torch.tensor([qi * k + ci for qi, ci in slots], device=device)
        pooled = torch.full(((stop - start) * k,), -1e4, device=device)
        pooled = pooled.index_reduce_(0, slot_index, scores, "amax", include_self=True)
        out[start:stop] = pooled.view(stop - start, k)
    return out
