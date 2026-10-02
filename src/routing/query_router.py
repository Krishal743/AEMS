"""Route queries by type and compute multimodal similarities.

There are six branches. The visual branch scores a query against each video's
best-matching *frame* rather than the average of its 16 frames: mean-pooling
cost the text side 0.053 R@1 before the passage branch fixed it, and it cost the
visual side similarly (0.225 -> 0.297 alone on validation). Caption embeddings
are CLIP; the passage
branch keeps each video's text as separate CLIP-encoded chunks and scores a
query against its best-matching passage (late interaction), which recovers the
detail the mean-pooled caption vector averages away; audio embeddings are WavLM
features projected into CLIP text space by the audio adapter; and the BM25
branch matches the query's literal words against those same passages, which
catches rare exact terms ("the Bellman equation") that dense embeddings blur;
and the dense branch encodes the same passages with a retrieval-trained text
model (E5), which CLIP's caption-trained text tower is no substitute for.

A query supplies up to three things: a CLIP query scored against the
visual/caption matrices, an audio query scored against the audio matrix (for
text queries this is the same CLIP text vector; for an audio clip it is the
adapter-projected WavLM embedding), and the raw query text for BM25. Branches a
query type cannot reach are left out and get zero weight.

Fusion z-scores each branch's similarities per query over the gallery, so the
branches are on a common scale, then takes a weighted sum. Weights are fixed
(AEMS_FUSION_WEIGHTS) by default, or predicted per query by the gating network.
"""

import os
from typing import NamedTuple

import torch
import torch.nn.functional as F
import clip

from src.config import (
    AEMS_MANIFEST_PATH,
    AEMS_VID_EMBEDDINGS_PATH,
    AEMS_FRAME_EMBEDDINGS_PATH,
    AEMS_AUDIO_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
    AEMS_TEXT_CHUNKS_PATH_TEMPLATE,
    AEMS_DENSE_PASSAGES_PATH_TEMPLATE,
    AEMS_DENSE_TEXT_MODEL,
    AEMS_GATING_WEIGHTS_PATH,
    AEMS_AUDIO_ADAPTER_PATH,
    AEMS_PER_CANDIDATE_GATE_PATH,
    AEMS_FUSION_WEIGHTS,
)
from src.models.gating_network import GatingNetwork
from src.models.audio_adapter import load_audio_adapter

# Fine-tuned on AEMS (bin/training/train_cross_encoder.py); falls back to the
# off-the-shelf checkpoint when that has not been trained yet.
FINETUNED_CROSS_ENCODER = "models/aems_cross_encoder_v1"
CROSS_ENCODER_MODEL = (FINETUNED_CROSS_ENCODER if os.path.isdir(FINETUNED_CROSS_ENCODER)
                       else "cross-encoder/ms-marco-MiniLM-L6-v2")

BRANCHES = ("visual", "text", "chunk", "audio", "bm25", "dense")


def add_index_args(parser):
    parser.add_argument("--video-embeds", default=AEMS_VID_EMBEDDINGS_PATH)
    parser.add_argument("--frame-embeds", default=AEMS_FRAME_EMBEDDINGS_PATH)
    parser.add_argument("--audio-embeds", default=AEMS_AUDIO_EMBEDDINGS_PATH)
    parser.add_argument("--caption-embeds",
                        default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
    parser.add_argument("--chunk-embeds",
                        default=AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split="test"))
    parser.add_argument("--dense-embeds",
                        default=AEMS_DENSE_PASSAGES_PATH_TEMPLATE.format(split="test"))
    parser.add_argument("--dense-model", default=AEMS_DENSE_TEXT_MODEL)
    parser.add_argument("--fusion", choices=["gate", "fixed"], default="gate",
                        help="gate: the gating network (default). It is initialised from "
                             "AEMS_FUSION_WEIGHTS and keeps that prior unless training "
                             "improves on it, so it is never worse than fixed weights — "
                             "on current data it reproduces them exactly. "
                             "fixed: AEMS_FUSION_WEIGHTS directly")
    parser.add_argument("--gate-weights", default=AEMS_GATING_WEIGHTS_PATH)
    parser.add_argument("--rerank", choices=["none", "gate", "cross"], default="none",
                        help="stage-2 reranking of the shortlist: none (default, fastest); "
                             "gate: per-candidate gating network; cross: cross-encoder "
                             "(reads query and passage together — much better, much slower)")
    parser.add_argument("--rerank-top-k", type=int, default=100, help="shortlist depth to rerank")
    parser.add_argument("--per-candidate-gate", default=AEMS_PER_CANDIDATE_GATE_PATH)
    parser.add_argument("--cross-encoder", default=CROSS_ENCODER_MODEL)
    parser.add_argument("--rerank-passages", type=int, default=8,
                        help="passages per candidate scored by the cross-encoder; reading more "
                             "of each candidate beat shortlisting more of them on validation")
    parser.add_argument("--rerank-alpha", type=float, default=0.3,
                        help="weight of the cross-encoder score against the stage-1 score")
    parser.add_argument("--top-k", type=int, default=5)


def load_fusion_gate(args, device):
    """The gating network when --fusion gate, else None (fixed weights).

    Falls back to fixed weights, with a warning, when the checkpoint is missing
    so a fresh checkout can still search before the gate has been trained.
    """
    if args.fusion != "gate":
        return None
    if not os.path.exists(args.gate_weights):
        print(f"[WARN] {args.gate_weights} not found; falling back to fixed fusion weights")
        return None
    return load_gate(args.gate_weights, device)


def load_clip(device):
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model


def encode_text_query(clip_model, text, device):
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return F.normalize(emb.float(), dim=1)


def encode_image_query(clip_model, image_tensor, device):
    with torch.no_grad():
        emb = clip_model.encode_image(image_tensor.to(device))
    return F.normalize(emb.float(), dim=1)


def encode_audio_query(wavlm_encoder, adapter, audio_path, device):
    """Audio clip -> WavLM features -> adapter -> (1, 512) CLIP-text-space vector."""
    feats = wavlm_encoder.encode_file(audio_path).to(device).view(1, -1)
    with torch.no_grad():
        return adapter(feats).float()


def load_adapter(path=AEMS_AUDIO_ADAPTER_PATH, device="cpu"):
    return load_audio_adapter(path, device)


def encode_video_query(clip_model, frame_tensor, device):
    with torch.no_grad():
        embs = F.normalize(clip_model.encode_image(frame_tensor.to(device)).float(), dim=1)
    return F.normalize(embs.mean(dim=0, keepdim=True), dim=1)


class ChunkIndex(NamedTuple):
    """All videos' passage embeddings stacked, with the video each row belongs to."""
    rows: torch.Tensor   # (total_chunks, 512), L2-normalized
    owner: torch.Tensor  # (total_chunks,) index into video_ids
    n_videos: int

    def max_sim(self, query):
        """(1, 512) query -> (n_videos,) best-matching-passage score per video."""
        return self.max_sim_batch(query).squeeze(0)

    def max_sim_batch(self, queries, batch_size=256):
        """(n, 512) queries -> (n, n_videos) best-matching-passage scores."""
        queries = queries.to(self.rows.device, self.rows.dtype)
        owner = self.owner.to(self.rows.device)
        out = []
        for i in range(0, queries.shape[0], batch_size):
            scores = queries[i:i + batch_size] @ self.rows.T
            pooled = torch.full((scores.shape[0], self.n_videos), -1e4,
                                device=scores.device, dtype=scores.dtype)
            out.append(pooled.index_reduce_(1, owner, scores, "amax", include_self=True))
        return torch.cat(out)


class SearchIndex(NamedTuple):
    video_ids: list
    visual: object               # ChunkIndex over frames, or a matrix for a legacy index
    text: torch.Tensor
    chunk: ChunkIndex
    audio: torch.Tensor
    bm25: object = None          # BM25PassageIndex, or None when not built
    dense: ChunkIndex = None     # retrieval-encoder passages, own embedding space


def chunk_index_from(db, video_ids, dim=512):
    """Stack per-video matrices into one ChunkIndex (passages, frames, anything)."""
    rows, owner = [], []
    for i, v in enumerate(video_ids):
        e = torch.as_tensor(db[v]).float().reshape(-1, dim)
        rows.append(F.normalize(e, dim=1))
        owner += [i] * e.shape[0]
    return ChunkIndex(torch.cat(rows), torch.tensor(owner), len(video_ids))


def load_search_index(video_path, audio_path, caption_path, chunk_path=None,
                      manifest_path=None, dense_path=None, frame_path=None):
    """Load the branches; matrices are L2-normalized row-wise.

    chunk_path and manifest_path may be None, in which case the passage and
    BM25 branches are absent and get zero weight — older indexes stay usable.
    BM25 needs the manifest because it matches raw passage text, not vectors.
    """
    video_db = torch.load(video_path, weights_only=False)
    audio_db = torch.load(audio_path, weights_only=False)
    caption_db = torch.load(caption_path, weights_only=False)
    chunk_db = torch.load(chunk_path, weights_only=False) if chunk_path else {}
    frame_db = torch.load(frame_path, weights_only=False) if frame_path else {}
    dense_db = torch.load(dense_path, weights_only=False) if dense_path else {}
    video_ids = sorted(v for v in video_db if v in audio_db and v in caption_db)

    def matrix(db):
        rows = []
        for v in video_ids:
            e = torch.as_tensor(db[v]).float()
            if e.dim() == 2:  # legacy MSR-VTT: one row per caption, max-pooled
                e = e.max(dim=0).values
            rows.append(e)
        return F.normalize(torch.stack(rows), dim=1)

    chunks = chunk_index_from(chunk_db, video_ids) if chunk_db else None
    dense = None
    if dense_db:
        dim = torch.as_tensor(dense_db[video_ids[0]]).shape[-1]
        dense = chunk_index_from(dense_db, video_ids, dim=dim)

    bm25 = None
    if manifest_path:
        from src.data.metadata import load_metadata
        from src.data.text_chunks import lexical_fields
        from src.retrieval.bm25 import BM25PassageIndex
        records = {r["video_id"]: r for r in load_metadata(manifest_path)}
        bm25 = BM25PassageIndex([lexical_fields(records[v]) for v in video_ids])

    # Frames when available (best-frame scoring), else the mean-pooled matrix.
    visual = chunk_index_from(frame_db, video_ids) if frame_db else matrix(video_db)
    return SearchIndex(video_ids, visual, matrix(caption_db), chunks,
                       matrix(audio_db), bm25, dense)


def load_gate(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    for key in ("fc1.weight", "fc3.weight"):
        if key not in state:
            raise RuntimeError(f"{path} is not a GatingNetwork checkpoint: {key} is missing")
    n_out = state["fc3.weight"].shape[0]
    if n_out != len(BRANCHES):
        raise RuntimeError(f"{path} predicts {n_out} weights but the router has "
                           f"{len(BRANCHES)} branches {BRANCHES}; retrain the gate "
                           f"with bin/training/train_gating_network.py")
    gate = GatingNetwork(input_dim=512, hidden_dim=state["fc1.weight"].shape[0],
                         num_modalities=n_out).to(device)
    missing, unexpected = gate.load_state_dict(state, strict=False)
    # strict=False tolerates the temperature/scale buffers older checkpoints lack
    # (forward() never reads them), but a missing learnable weight would silently
    # leave part of the gate randomly initialised.
    missing_params = sorted(set(missing) & {n for n, _ in gate.named_parameters()})
    if missing_params or unexpected:
        raise RuntimeError(f"{path} does not match GatingNetwork: "
                           f"missing={missing_params} unexpected={sorted(unexpected)}")
    gate.eval()
    return gate


def compute_modal_similarities(video_matrix, caption_matrix, audio_matrix,
                               clip_query=None, audio_query=None, chunk_index=None,
                               bm25_index=None, query_text=None, dense_index=None,
                               dense_query=None):
    """Return one similarity vector per branch; None where the query can't reach it."""
    def score(q, m):
        if q is None:
            return None
        if isinstance(m, ChunkIndex):          # frames: best-matching one wins
            return m.max_sim(q)
        return (q.to(m.device, m.dtype) @ m.T).squeeze(0)
    sim_c = None if (clip_query is None or chunk_index is None) else chunk_index.max_sim(clip_query)
    sim_b = None
    if bm25_index is not None and query_text:
        sim_b = torch.tensor(bm25_index.score(query_text), dtype=torch.float32)
    sim_d = None
    if dense_index is not None and dense_query is not None:
        sim_d = dense_index.max_sim(dense_query)
    return (score(clip_query, video_matrix), score(clip_query, caption_matrix),
            sim_c, score(audio_query, audio_matrix), sim_b, sim_d)


def zscore(sim):
    """Standardize similarities over the gallery (last dim) per query."""
    return (sim - sim.mean(-1, keepdim=True)) / (sim.std(-1, keepdim=True) + 1e-6)


def fixed_weights(weights=None):
    weights = AEMS_FUSION_WEIGHTS if weights is None else weights
    return torch.tensor([float(weights[b]) for b in BRANCHES])


def search(index, clip_query=None, audio_query=None, gate=None, weights=None,
           query_text=None, dense_query=None):
    """Score a query against the index.

    Returns (weights, then one similarity vector per branch) on CPU, where sims are
    z-scored per branch and the fused score is sum(weights[i] * sim_i).
    Unavailable branches get all-zero similarities and zero weight, and the
    remaining weights are renormalized to sum to 1. With a gate and a CLIP
    query, weights come from the gate; otherwise from `weights` (a dict keyed
    by branch, default AEMS_FUSION_WEIGHTS).
    """
    video_ids = index.video_ids
    sims = compute_modal_similarities(index.visual, index.text, index.audio,
                                      clip_query, audio_query, index.chunk,
                                      index.bm25, query_text, index.dense, dense_query)
    mask = torch.tensor([s is not None for s in sims], dtype=torch.float32)
    if not mask.any():
        raise ValueError("query produced no embedding for any branch")

    if gate is not None and clip_query is not None:
        device = next(gate.parameters()).device
        with torch.no_grad():
            w = gate(clip_query.to(device).float()).cpu().squeeze(0)
    else:
        w = fixed_weights(weights)
    w = w * mask
    if w.sum() <= 0:  # e.g. an audio-only query while the audio weight is 0
        w = mask
    w = w / w.sum()

    zeros = torch.zeros(len(video_ids))
    return (w,) + tuple(zeros if s is None else zscore(s.float().cpu()) for s in sims)


def apply_rerank(args, weights, sims, index, clip_query=None, query_text=None,
                 records=None, chunk_db=None, device="cpu"):
    """Fuse the branches, then optionally rescore the shortlist with stage 2.

    Returns stage-1 fused scores unchanged when --rerank none, so the fast path
    costs nothing. A reranker that cannot run (missing checkpoint, or no query
    text for the cross-encoder) warns and falls back to stage 1 rather than
    failing the search.
    """
    fused = sum(float(w) * s for w, s in zip(weights, sims))
    if args.rerank == "none" or clip_query is None:
        return fused

    from src.rerank import stage1 as shortlist
    scores = fused.unsqueeze(0)
    candidates = shortlist.top_k_candidates(scores, args.rerank_top_k)

    if args.rerank == "gate":
        from src.rerank import per_candidate
        if not os.path.exists(args.per_candidate_gate):
            print(f"[WARN] {args.per_candidate_gate} not found; skipping reranking")
            return fused
        model = per_candidate.load(args.per_candidate_gate, device, n_branches=len(BRANCHES))
        feats = shortlist.gather_branch_scores([s.unsqueeze(0) for s in sims], candidates)
        with torch.no_grad():
            rescored = per_candidate.score(model, clip_query.to(device), feats.to(device)).cpu()
        return shortlist.rerank_scores_to_ranking(scores, candidates, rescored).squeeze(0)

    from src.rerank import cross_encoder
    if query_text is None or records is None or chunk_db is None:
        print("[WARN] cross-encoder reranking needs the query text, the manifest records and "
              "the chunk embeddings; falling back to stage 1")
        return fused
    model, tokenizer = cross_encoder.load_cross_encoder(args.cross_encoder, device)
    store = cross_encoder.PassageStore(records, index.video_ids, chunk_db, device)
    rescored = cross_encoder.rerank(model, tokenizer, [query_text], clip_query.to(device),
                                    candidates.to(device), store, index.video_ids,
                                    n_passages=args.rerank_passages, device=device).cpu()
    z = (rescored - rescored.mean()) / (rescored.std() + 1e-6)
    base = scores.gather(1, candidates)
    return shortlist.rerank_scores_to_ranking(scores, candidates,
                                              base + args.rerank_alpha * z).squeeze(0)


def check_rerank_supported(parser, args, query_text):
    """Reject --rerank cross when there is no query text to read passages against.

    The cross-encoder scores (query text, passage text) pairs, so an image- or
    video-only query cannot use it. Failing loudly beats silently ignoring the
    flag and reporting unreranked results as if they were reranked.
    """
    if args.rerank == "cross" and not query_text:
        parser.error("--rerank cross needs query text: the cross-encoder reads the query "
                     "against transcript passages. Use --rerank gate for image/video queries.")


def rerank_and_report(args, weights, sims, index, clip_query, query_text=None,
                      device="cpu", manifest_path=None):
    """Run stage 2 if requested and print the reranked ordering."""
    if args.rerank == "none":
        return None

    records, chunk_db = None, None
    if args.rerank == "cross":
        from src.data.metadata import load_metadata
        records = {r["video_id"]: r for r in load_metadata(manifest_path or AEMS_MANIFEST_PATH)}
        chunk_db = torch.load(args.chunk_embeds, weights_only=False)

    ranking = apply_rerank(args, weights, sims, index, clip_query=clip_query.cpu(),
                           query_text=query_text, records=records, chunk_db=chunk_db,
                           device=device)
    order = torch.argsort(ranking, descending=True)[:args.top_k]
    print(f"\nReranked top-{args.top_k} (--rerank {args.rerank}, "
          f"shortlist {args.rerank_top_k}):")
    for rank, idx in enumerate(order.tolist(), 1):
        print(f"  {rank}. {index.video_ids[idx]}  score={ranking[idx]:.4f}")
    return ranking
