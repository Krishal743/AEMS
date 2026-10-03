"""Gallery and queries for the multimodal (non-text) query benchmark.

Shared by `bin/evaluation/eval_multimodal_queries.py` and
`bin/training/tune_modality_weights.py`. They must agree exactly on how the
gallery is built and which branches a query can reach: the tuner selects
weights per modality and the evaluator applies them, so any disagreement
silently invalidates both.

Hold-out rules live in `src/evaluation/holdout.py`.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (AEMS_AUDIO_ADAPTER_PATH, AEMS_AUDIO_DIR, AEMS_AUDIO_CLIP_SEC,
                        AEMS_WAVLM_SR)
from src.data.text_chunks import lexical_fields
from src.evaluation.holdout import TEMPORAL, frame_holdout, audio_segment_is_disjoint
from src.models.audio_adapter import load_audio_adapter
from src.retrieval.bm25 import BM25PassageIndex
from src.retrieval.branches import encode_queries
from src.routing.query_router import BRANCHES, chunk_index_from, zscore
from src.training.query_data import questions, stack_embeddings

# Which branches each query modality can reach at all. A CLIP-space query has no
# text, so BM25 and the dense retriever are unreachable by construction — not a
# tuning choice, a property of the query.
MODALITIES = ("text", "image", "video", "audio", "audio_clip", "image+text")


class Gallery:
    """The indexed side, with frames held out so queries cannot match themselves."""

    def __init__(self, records, sources, video_ids, device, frame_split=TEMPORAL):
        self.records, self.video_ids, self.device = records, video_ids, device
        self.n = len(video_ids)
        frames = {v: torch.as_tensor(sources.frames[v]).float().reshape(-1, 512)
                  for v in video_ids}
        counts = {f.shape[0] for f in frames.values()}
        if len(counts) != 1:
            raise ValueError(f"frames per video must be uniform, found {sorted(counts)}")
        held = {v: frame_holdout(f, frame_split) for v, f in frames.items()}
        self.index_frames = {v: i for v, (i, _) in held.items()}
        self.query_frames = {v: q for v, (_, q) in held.items()}
        self.frames = frames
        self.frame_split = frame_split

        self.visual = self._on(chunk_index_from(self.index_frames, video_ids))
        self.chunk = self._on(chunk_index_from(sources.chunks, video_ids))
        dim = torch.as_tensor(sources.dense[video_ids[0]]).shape[-1]
        self.dense = self._on(chunk_index_from(sources.dense, video_ids, dim=dim))
        self.caption = stack_embeddings(sources.caption, video_ids, device)
        self.audio = stack_embeddings(sources.audio, video_ids, device)
        self.bm25 = BM25PassageIndex([lexical_fields(records[v]) for v in video_ids])

    def _on(self, index):
        return index._replace(rows=index.rows.to(self.device),
                              owner=index.owner.to(self.device))

    def branch_sims(self, clip_q=None, dense_q=None, texts=None, audio_q=None):
        """Z-scored similarity per branch, None where the query cannot reach it."""
        sims = {b: None for b in BRANCHES}
        if clip_q is not None:
            q = F.normalize(clip_q.float().to(self.device), dim=1)
            sims["visual"] = self.visual.max_sim_batch(q)
            sims["text"] = q @ self.caption.T
            sims["chunk"] = self.chunk.max_sim_batch(q)
        if audio_q is not None:
            a = F.normalize(audio_q.float().to(self.device), dim=1)
            sims["audio"] = a @ self.audio.T
        if texts is not None:
            sims["bm25"] = self.bm25.score_batch(texts).to(self.device)
        if dense_q is not None:
            sims["dense"] = self.dense.max_sim_batch(dense_q.float().to(self.device))
        return [None if sims[b] is None else zscore(sims[b]) for b in BRANCHES]


def fuse(sims, weights, device):
    """Weighted sum over reachable branches, weights renormalized over them.

    `weights` is a dict keyed by branch. Unreachable branches contribute nothing
    and their weight is redistributed, which is what the deployed router does.
    """
    present = [i for i, s in enumerate(sims) if s is not None]
    if not present:
        raise ValueError("query reached no branch")
    w = torch.tensor([float(weights[b]) for b in BRANCHES], device=device)
    mask = torch.zeros(len(BRANCHES), device=device)
    mask[present] = 1.0
    w = w * mask
    if w.sum() <= 0:
        w = mask
    w = w / w.sum()
    return sum(w[i] * sims[i] for i in present)


def text_queries(records, video_ids, device):
    """One question per video, so every modality is scored on the same footing."""
    texts = [questions(records[v])[0] for v in video_ids]
    clip_q, dense_q = encode_queries(texts, device)
    return clip_q, dense_q, texts


def frame_query(gallery, video_ids, image_frame):
    return torch.stack([gallery.query_frames[v][image_frame] for v in video_ids])


def video_query(gallery, video_ids):
    return torch.stack([F.normalize(gallery.query_frames[v].mean(0), dim=0)
                        for v in video_ids])


def audio_queries(records, video_ids, device, position=0.25, min_duration=60.0,
                  clip_sec=AEMS_AUDIO_CLIP_SEC, verbose=True):
    """Held-out audio segments -> (embeddings, gallery indices they belong to).

    The indexed embedding averages fixed segments at the start, middle and end,
    so the query segment has to clear all three to be a genuine hold-out.
    """
    import librosa
    from src.encoders.wavlm_encode import WavLMEncoder

    eligible = [(i, v) for i, v in enumerate(video_ids)
                if float(records[v].get("duration_seconds", 0.0)) >= min_duration
                and audio_segment_is_disjoint(float(records[v]["duration_seconds"]),
                                              position, clip_sec)]
    if verbose:
        print(f"[AUDIO] {len(eligible)}/{len(video_ids)} videos admit a {clip_sec:g}s "
              f"segment at the {position:.0%} mark disjoint from all indexed segments",
              flush=True)

    seg_len = AEMS_WAVLM_SR * int(clip_sec)
    waves, keep = [], []
    for i, v in eligible:
        path = records[v].get("audio_path") or os.path.join(AEMS_AUDIO_DIR, f"{v}.wav")
        try:
            wav, _ = librosa.load(path, sr=AEMS_WAVLM_SR, mono=True)
        except Exception as exc:
            print(f"  [SKIP] {v}: {exc}", flush=True)
            continue
        start = int(max(0, len(wav) * position - seg_len / 2))
        seg = wav[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        waves.append(seg.astype("float32"))
        keep.append(i)

    encoder = WavLMEncoder(device)
    adapter = load_audio_adapter(AEMS_AUDIO_ADAPTER_PATH, device)
    out = []
    for s in range(0, len(waves), 32):
        with torch.no_grad():
            out.append(adapter(encoder.encode_segments(waves[s:s + 32]).to(device))
                       .float().cpu())
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out), torch.tensor(keep, device=device)


def build_query(modality, gallery, records, video_ids, device, image_frame=4,
                audio=None, text=None):
    """(sims, gt) for one modality. `audio`/`text` are reused across modalities."""
    gt_all = torch.arange(len(video_ids), device=device)
    if modality == "text":
        clip_q, dense_q, texts = text
        return gallery.branch_sims(clip_q=clip_q, dense_q=dense_q, texts=texts), gt_all
    if modality == "image":
        return gallery.branch_sims(clip_q=frame_query(gallery, video_ids, image_frame)), gt_all
    if modality == "video":
        return gallery.branch_sims(clip_q=video_query(gallery, video_ids)), gt_all
    if modality == "image+text":
        clip_q, dense_q, texts = text
        img = F.normalize(frame_query(gallery, video_ids, image_frame).to(device), dim=1)
        mixed = F.normalize(img + clip_q.to(device), dim=1)
        return gallery.branch_sims(clip_q=mixed, dense_q=dense_q, texts=texts), gt_all
    if modality == "audio":
        emb, gt = audio
        return gallery.branch_sims(audio_q=emb), gt
    if modality == "audio_clip":
        emb, gt = audio
        # Same vector, scored against more branches: the adapter emits a CLIP-space
        # vector, so it can be read as a CLIP query as well as an audio one.
        return gallery.branch_sims(clip_q=emb, audio_q=emb), gt
    raise ValueError(f"unknown modality {modality!r}; choose from {MODALITIES}")
