"""One place that builds branch similarity scores.

Every trainer and evaluator used to assemble its own list of per-branch score
matrices. That duplication caused three separate defects while the branch set
grew: two loud failures ("6 weights but 5 branch score matrices") and one silent
one, where the query CLIs ran on four of six branches for an unknown period
because the index was loaded without the dense, BM25 and frame sources.

Anything that needs branch scores should call `BranchSources.load(...)` and
`build_sims(...)` here, so adding a branch means editing this file and
`BRANCHES` — not five call sites.
"""

import torch
import torch.nn.functional as F

from src.config import (AEMS_AUDIO_EMBEDDINGS_PATH, AEMS_DENSE_PASSAGES_PATH_TEMPLATE,
                        AEMS_DENSE_TEXT_MODEL, AEMS_FRAME_EMBEDDINGS_PATH,
                        AEMS_TEXT_CHUNKS_PATH_TEMPLATE, AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE)
from src.data.text_chunks import lexical_fields
from src.retrieval.bm25 import BM25PassageIndex
from src.routing.query_router import BRANCHES, chunk_index_from, zscore
from src.training.query_data import stack_embeddings


class BranchSources:
    """The per-split databases every branch is built from."""

    def __init__(self, records, frames, caption, chunks, audio, dense):
        self.records, self.frames, self.caption = records, frames, caption
        self.chunks, self.audio, self.dense = chunks, audio, dense

    @classmethod
    def load(cls, records, split):
        return cls(
            records=records,
            frames=torch.load(AEMS_FRAME_EMBEDDINGS_PATH, weights_only=False),
            caption=torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split=split),
                               weights_only=False),
            chunks=torch.load(AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split=split),
                              weights_only=False),
            audio=torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False),
            dense=torch.load(AEMS_DENSE_PASSAGES_PATH_TEMPLATE.format(split=split),
                             weights_only=False))

    def usable(self, video_ids):
        """Videos present in every source, so no branch silently drops rows."""
        return [v for v in video_ids
                if all(v in d for d in (self.frames, self.caption, self.chunks,
                                        self.audio, self.dense))]


def _on(index, device):
    return index._replace(rows=index.rows.to(device), owner=index.owner.to(device))


def build_sims(sources, video_ids, clip_queries, query_texts, dense_queries, device,
               audio_db=None):
    """Per-branch z-scored similarities, in BRANCHES order.

    `audio_db` overrides the audio source — training passes cross-fitted
    out-of-fold projections here, since the deployed adapter saw these videos.
    """
    audio = audio_db if audio_db is not None else sources.audio
    dense_dim = torch.as_tensor(sources.dense[video_ids[0]]).shape[-1]

    sims = {
        "visual": _on(chunk_index_from(sources.frames, video_ids), device).max_sim_batch(clip_queries),
        "text": clip_queries @ stack_embeddings(sources.caption, video_ids, device).T,
        "chunk": _on(chunk_index_from(sources.chunks, video_ids), device).max_sim_batch(clip_queries),
        "audio": clip_queries @ stack_embeddings(audio, video_ids, device).T,
        "bm25": BM25PassageIndex(
            [lexical_fields(sources.records[v]) for v in video_ids]
        ).score_batch(query_texts).to(device),
        "dense": _on(chunk_index_from(sources.dense, video_ids, dim=dense_dim),
                     device).max_sim_batch(dense_queries),
    }
    missing = set(BRANCHES) - set(sims)
    if missing:
        raise ValueError(f"build_sims does not produce {sorted(missing)}; "
                         f"add it here when adding a branch to BRANCHES")
    return [zscore(sims[b]) for b in BRANCHES]


def encode_queries(texts, device, dense_model=None):
    """(clip, dense) query matrices — the two spaces the branches are scored in."""
    from src.training.query_data import encode_clip_text

    from src.encoders.text_retrieval import load_dense_encoder

    clip = encode_clip_text(texts, device)
    encoder = load_dense_encoder(device, dense_model)
    dense = encoder.encode_queries(texts, batch_size=256)
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    return clip, dense
