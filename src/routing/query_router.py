"""Route queries by type and compute multimodal similarities.

There are four branches. Visual and caption embeddings are CLIP; the passage
branch keeps each video's text as separate CLIP-encoded chunks and scores a
query against its best-matching passage (late interaction), which recovers the
detail the mean-pooled caption vector averages away; audio embeddings are WavLM
features
projected into CLIP text space by the audio adapter. A query supplies up to two
vectors: a CLIP query scored against the visual/caption matrices, and an audio
query scored against the audio matrix (for text queries this is the same CLIP
text vector; for an audio clip it is the adapter-projected WavLM embedding).
Branches a query type cannot reach are left out and get zero weight.

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
    AEMS_VID_EMBEDDINGS_PATH,
    AEMS_AUDIO_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE,
    AEMS_TEXT_CHUNKS_PATH_TEMPLATE,
    AEMS_GATING_WEIGHTS_PATH,
    AEMS_AUDIO_ADAPTER_PATH,
    AEMS_PER_CANDIDATE_GATE_PATH,
    AEMS_FUSION_WEIGHTS,
)
from src.models.gating_network import GatingNetwork
from src.models.audio_adapter import load_audio_adapter

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

BRANCHES = ("visual", "text", "chunk", "audio")


def add_index_args(parser):
    parser.add_argument("--video-embeds", default=AEMS_VID_EMBEDDINGS_PATH)
    parser.add_argument("--audio-embeds", default=AEMS_AUDIO_EMBEDDINGS_PATH)
    parser.add_argument("--caption-embeds",
                        default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
    parser.add_argument("--chunk-embeds",
                        default=AEMS_TEXT_CHUNKS_PATH_TEMPLATE.format(split="test"))
    parser.add_argument("--fusion", choices=["gate", "fixed"], default="gate",
                        help="gate: per-query gating network (default); fixed: AEMS_FUSION_WEIGHTS")
    parser.add_argument("--gate-weights", default=AEMS_GATING_WEIGHTS_PATH)
    parser.add_argument("--rerank", choices=["none", "gate", "cross"], default="none",
                        help="stage-2 reranking of the shortlist: none (default, fastest); "
                             "gate: per-candidate gating network; cross: cross-encoder "
                             "(reads query and passage together — much better, much slower)")
    parser.add_argument("--rerank-top-k", type=int, default=20, help="shortlist depth to rerank")
    parser.add_argument("--per-candidate-gate", default=AEMS_PER_CANDIDATE_GATE_PATH)
    parser.add_argument("--cross-encoder", default=CROSS_ENCODER_MODEL)
    parser.add_argument("--rerank-passages", type=int, default=3,
                        help="passages per candidate scored by the cross-encoder")
    parser.add_argument("--rerank-alpha", type=float, default=0.5,
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
    visual: torch.Tensor
    text: torch.Tensor
    chunk: ChunkIndex
    audio: torch.Tensor


def load_search_index(video_path, audio_path, caption_path, chunk_path=None):
    """Load the four branches; matrices are L2-normalized row-wise.

    chunk_path may be None (the passage branch is then absent and gets zero
    weight), which keeps older indexes usable.
    """
    video_db = torch.load(video_path, weights_only=False)
    audio_db = torch.load(audio_path, weights_only=False)
    caption_db = torch.load(caption_path, weights_only=False)
    chunk_db = torch.load(chunk_path, weights_only=False) if chunk_path else {}
    video_ids = sorted(v for v in video_db if v in audio_db and v in caption_db)

    def matrix(db):
        rows = []
        for v in video_ids:
            e = torch.as_tensor(db[v]).float()
            if e.dim() == 2:  # legacy MSR-VTT: one row per caption, max-pooled
                e = e.max(dim=0).values
            rows.append(e)
        return F.normalize(torch.stack(rows), dim=1)

    chunks = None
    if chunk_db:
        rows, owner = [], []
        for i, v in enumerate(video_ids):
            e = torch.as_tensor(chunk_db[v]).float().reshape(-1, 512)
            rows.append(F.normalize(e, dim=1))
            owner += [i] * e.shape[0]
        chunks = ChunkIndex(torch.cat(rows), torch.tensor(owner), len(video_ids))

    return SearchIndex(video_ids, matrix(video_db), matrix(caption_db), chunks, matrix(audio_db))


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
                               clip_query=None, audio_query=None, chunk_index=None):
    """Return (sim_v, sim_t, sim_c, sim_a); a branch is None when unreachable."""
    def score(q, m):
        return None if q is None else (q.to(m.device, m.dtype) @ m.T).squeeze(0)
    sim_c = None if (clip_query is None or chunk_index is None) else chunk_index.max_sim(clip_query)
    return (score(clip_query, video_matrix), score(clip_query, caption_matrix),
            sim_c, score(audio_query, audio_matrix))


def zscore(sim):
    """Standardize similarities over the gallery (last dim) per query."""
    return (sim - sim.mean(-1, keepdim=True)) / (sim.std(-1, keepdim=True) + 1e-6)


def fixed_weights(weights=None):
    weights = AEMS_FUSION_WEIGHTS if weights is None else weights
    return torch.tensor([float(weights[b]) for b in BRANCHES])


def search(index, clip_query=None, audio_query=None, gate=None, weights=None):
    """Score a query against the index.

    Returns (weights, sim_v, sim_t, sim_c, sim_a) on CPU, where sims are
    z-scored per branch and the fused score is sum(weights[i] * sim_i).
    Unavailable branches get all-zero similarities and zero weight, and the
    remaining weights are renormalized to sum to 1. With a gate and a CLIP
    query, weights come from the gate; otherwise from `weights` (a dict keyed
    by branch, default AEMS_FUSION_WEIGHTS).
    """
    video_ids = index.video_ids
    sims = compute_modal_similarities(index.visual, index.text, index.audio,
                                      clip_query, audio_query, index.chunk)
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
