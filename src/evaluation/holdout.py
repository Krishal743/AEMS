"""Hold-out splits for query-by-example benchmarks.

Benchmarking an image, video or audio query means querying with material taken
from the target video itself. If that exact material is also in the index, the
query scores a perfect match and the benchmark measures nothing, so the two
sides have to be disjoint.

Disjoint is not enough on its own. Frames sampled uniformly from one video are
near-duplicates of their temporal neighbours, so an even/odd split is formally
disjoint while still handing the query an almost identical frame to match
against. Splitting by time instead puts half the runtime between a query frame
and the nearest indexed one. Both are provided because the gap between them is
itself the measurement of how much near-duplicate matching is going on.
"""

INTERLEAVE = "interleave"
TEMPORAL = "temporal"
SPLITS = (TEMPORAL, INTERLEAVE)


def frame_holdout(frames, split=TEMPORAL):
    """(index_frames, query_frames) for one video's (n_frames, dim) matrix.

    `temporal` indexes the first half of the runtime and queries the second.
    `interleave` indexes even positions and queries odd ones — adjacent in time,
    so the query side is near-duplicate to the index side.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    n = frames.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 frames to hold any out, got {n}")
    if split == INTERLEAVE:
        return frames[0::2], frames[1::2]
    return frames[: n // 2], frames[n // 2:]


def audio_segment_is_disjoint(duration, position, clip_sec):
    """Does a `clip_sec` segment at `position` miss all three indexed segments?

    The indexed embedding averages fixed segments at the start, middle and end
    (see src/encoders/wavlm_encode.py), so a query segment has to clear all
    three to be a genuine hold-out.
    """
    if duration <= clip_sec:
        return False
    start = max(0.0, duration * position - clip_sec / 2)
    end = start + clip_sec
    mid = (duration - clip_sec) / 2
    indexed = [(0.0, clip_sec), (mid, mid + clip_sec), (duration - clip_sec, duration)]
    return all(end <= a or start >= b for a, b in indexed)
