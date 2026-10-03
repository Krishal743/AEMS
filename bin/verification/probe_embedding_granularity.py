"""What do our embeddings actually encode: who is speaking, or what is said?

The multimodal benchmark produced a contradiction worth resolving. The audio
branch retrieves its own video from a held-out 10 s segment with R@1 0.9131,
yet the same embeddings queried by *text* score 0.0532 — the weakest branch in
the system. A representation cannot be both excellent and useless at describing
content. The hypothesis is that it encodes speaker and recording identity
rather than meaning.

That is testable. For each video take two disjoint segments and compare:

* **same video** — same speaker, same room, *different words*.
* **content twin** — the different video whose transcript is most similar by the
  dense retriever: different speaker, different room, *most similar words*.
* **random other** — the floor.

If same-video similarity far exceeds the content twin, the embedding is keyed
to identity, not content: two speakers saying the same thing land far apart, and
one speaker saying different things lands close together. The same three
comparisons are run on the dense text branch, which should behave the opposite
way, and on the visual branch.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (AEMS_MANIFEST_PATH, AEMS_AUDIO_ADAPTER_PATH, AEMS_AUDIO_DIR,
                        AEMS_AUDIO_CLIP_SEC, AEMS_WAVLM_SR, DEVICE, set_seeds)
from src.evaluation.holdout import audio_segment_is_disjoint
from src.models.audio_adapter import load_audio_adapter
from src.retrieval.branches import BranchSources
from src.routing.query_router import chunk_index_from
from src.training.query_data import load_records, questions, stack_embeddings

parser = argparse.ArgumentParser(description="Probe what each embedding encodes")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--split", default="test")
parser.add_argument("--n", type=int, default=300, help="videos to probe")
parser.add_argument("--min-duration", type=float, default=120.0,
                    help="two disjoint query segments plus the three indexed ones "
                         "need room; 120s keeps 25%% and 75%% well apart")
parser.add_argument("--output", default="outputs/aems/embedding_granularity.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
set_seeds(args.seed)

records = load_records(args.manifest, args.split)
sources = BranchSources.load(records, args.split)
video_ids = sources.usable([v for v, r in records.items() if questions(r)])
eligible = [v for v in video_ids
            if float(records[v].get("duration_seconds", 0.0)) >= args.min_duration
            and all(audio_segment_is_disjoint(float(records[v]["duration_seconds"]),
                                              p, AEMS_AUDIO_CLIP_SEC) for p in (0.25, 0.75))]
rng = np.random.default_rng(args.seed)
probe = [eligible[i] for i in rng.permutation(len(eligible))[:args.n]]
print(f"[DATA] probing {len(probe)} of {len(eligible)} eligible videos", flush=True)

# ---- content twins: the most textually similar *other* video, by the dense branch
dim = torch.as_tensor(sources.dense[video_ids[0]]).shape[-1]
dense_index = chunk_index_from(sources.dense, video_ids, dim=dim)
dense_index = dense_index._replace(rows=dense_index.rows.to(DEVICE),
                                   owner=dense_index.owner.to(DEVICE))
pos = {v: i for i, v in enumerate(video_ids)}
# A video's own passages represent it; its best-matching other video is its twin.
twin, twin_sim = {}, {}
for v in probe:
    q = F.normalize(torch.as_tensor(sources.dense[v]).float().reshape(-1, dim), dim=1)
    sims = dense_index.max_sim_batch(q.to(DEVICE)).mean(0)
    sims[pos[v]] = -1e4
    j = int(sims.argmax())
    twin[v], twin_sim[v] = video_ids[j], float(sims[j])
print(f"[TWIN] mean dense similarity to the content twin = "
      f"{np.mean(list(twin_sim.values())):.4f}", flush=True)

# ---- audio: two disjoint segments per video
import librosa
from src.encoders.wavlm_encode import WavLMEncoder

seg_len = AEMS_WAVLM_SR * int(AEMS_AUDIO_CLIP_SEC)
needed = sorted({v for v in probe} | {twin[v] for v in probe})


def segment(video, position):
    path = records[video].get("audio_path") or os.path.join(AEMS_AUDIO_DIR, f"{video}.wav")
    wav, _ = librosa.load(path, sr=AEMS_WAVLM_SR, mono=True)
    start = int(max(0, len(wav) * position - seg_len / 2))
    seg = wav[start:start + seg_len]
    if len(seg) < seg_len:
        seg = np.pad(seg, (0, seg_len - len(seg)))
    return seg.astype("float32")


encoder = WavLMEncoder(DEVICE)
adapter = load_audio_adapter(AEMS_AUDIO_ADAPTER_PATH, DEVICE)


def encode(videos, position):
    out = {}
    batch, keys = [], []
    for v in videos:
        try:
            batch.append(segment(v, position))
            keys.append(v)
        except Exception as exc:
            print(f"  [SKIP] {v}: {exc}", flush=True)
        if len(batch) == 32:
            with torch.no_grad():
                e = adapter(encoder.encode_segments(batch).to(DEVICE)).float().cpu()
            out.update(dict(zip(keys, F.normalize(e, dim=1))))
            batch, keys = [], []
    if batch:
        with torch.no_grad():
            e = adapter(encoder.encode_segments(batch).to(DEVICE)).float().cpu()
        out.update(dict(zip(keys, F.normalize(e, dim=1))))
    return out


print("[AUDIO] encoding segments at 25% and 75%...", flush=True)
a25, a75 = encode(needed, 0.25), encode(needed, 0.75)
del encoder
torch.cuda.empty_cache()

# ---- the three comparisons, for each branch
def cos(a, b):
    return float(torch.dot(a, b))


visual = {v: F.normalize(torch.as_tensor(sources.frames[v]).float().reshape(-1, 512), dim=1)
          for v in needed}
dense_vec = {v: F.normalize(torch.as_tensor(sources.dense[v]).float().reshape(-1, dim).mean(0),
                            dim=0) for v in needed}

rows = {b: {"same_video": [], "content_twin": [], "random_other": []}
        for b in ("audio", "dense_text", "visual")}
others = [v for v in probe]
for v in probe:
    t = twin[v]
    r = others[int(rng.integers(len(others)))]
    while r == v:
        r = others[int(rng.integers(len(others)))]
    if v in a25 and v in a75:
        rows["audio"]["same_video"].append(cos(a25[v], a75[v]))
    if v in a25 and t in a25:
        rows["audio"]["content_twin"].append(cos(a25[v], a25[t]))
    if v in a25 and r in a25:
        rows["audio"]["random_other"].append(cos(a25[v], a25[r]))
    # text: first half of the passages vs second half, so "same video" is also
    # two different stretches of speech rather than the identical vector.
    pv = F.normalize(torch.as_tensor(sources.dense[v]).float().reshape(-1, dim), dim=1)
    if pv.shape[0] >= 2:
        h = pv.shape[0] // 2
        rows["dense_text"]["same_video"].append(
            cos(F.normalize(pv[:h].mean(0), dim=0), F.normalize(pv[h:].mean(0), dim=0)))
    rows["dense_text"]["content_twin"].append(cos(dense_vec[v], dense_vec[t]))
    rows["dense_text"]["random_other"].append(cos(dense_vec[v], dense_vec[r]))
    fv = visual[v]
    h = fv.shape[0] // 2
    rows["visual"]["same_video"].append(
        cos(F.normalize(fv[:h].mean(0), dim=0), F.normalize(fv[h:].mean(0), dim=0)))
    rows["visual"]["content_twin"].append(
        cos(F.normalize(fv.mean(0), dim=0), F.normalize(visual[t].mean(0), dim=0)))
    rows["visual"]["random_other"].append(
        cos(F.normalize(fv.mean(0), dim=0), F.normalize(visual[r].mean(0), dim=0)))

print(f"\n{'branch':<12} {'same video':>12} {'content twin':>14} {'random other':>14} "
      f"{'identity gap':>14}")
print("-" * 70)
out = {}
for b, d in rows.items():
    m = {k: float(np.mean(v)) for k, v in d.items()}
    # How much more does sharing a recording matter than sharing a topic?
    m["identity_gap"] = m["same_video"] - m["content_twin"]
    out[b] = m
    print(f"{b:<12} {m['same_video']:>12.4f} {m['content_twin']:>14.4f} "
          f"{m['random_other']:>14.4f} {m['identity_gap']:>+14.4f}")

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"n_probed": len(probe), "mean_twin_dense_sim": float(np.mean(list(twin_sim.values()))),
               "branches": out}, f, indent=2)
print(f"\n[SAVE] {args.output}")
