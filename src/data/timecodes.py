"""Timecoded transcripts, and deriving moment labels from them.

AEMS has no temporal annotations, but 6,769 of 6,770 videos carry a timecoded
transcript that the project has never used. Moment labels can be derived from
it, and the construction matters more than the code:

**The label is found using the answer; a method only ever sees the question.**
For each QA pair the span is the run of consecutive transcript segments that
best covers the *answer*'s content words. The answer is privileged information,
available when building the benchmark and never at inference. Without that
split the task would be circular — labels defined by matching text, solved by
matching the same text.

The residual confound is that questions share some wording with their answers,
so a question partly points at its own span. Measured on a sample, the median
question overlap with its ground-truth window is 0.22 against 0.39 for the
answer, so the signal is real but far from giving the span away. Benchmarks
built here should report results stratified by that overlap, as elsewhere in
this project.

**What this cannot label:** a span is only found when the answer is *spoken*.
Questions whose answers are only shown — the visually-grounded tier — get no
label, so this measures grounding of spoken content and is structurally biased
toward transcript-based methods. That bias has to be stated wherever the
numbers are.
"""

from src.data.query_subsets import content_words

# A moment is at most this many consecutive transcript segments. Beyond ~5 the
# window stops being a moment and starts being a section of the video.
MAX_SEGMENTS = 5
# Charged per extra segment, so a long window must earn its length in coverage
# rather than winning by containing more words.
LENGTH_PENALTY = 0.02


def parse_timecode(stamp):
    """'00:01:02.760' -> 62.76 seconds."""
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def segments(record):
    """[(start, end, text)] in seconds, in order, from a manifest record."""
    out = []
    for entry in record.get("timecoded_text_to_speech") or []:
        try:
            start, end = parse_timecode(entry["start"]), parse_timecode(entry["end"])
        except (KeyError, ValueError):
            continue
        if end >= start:
            out.append((start, end, entry.get("text", "")))
    return out


def locate(segs, target, max_segments=MAX_SEGMENTS, length_penalty=LENGTH_PENALTY):
    """Best-covering window for `target` text: (score, start_s, end_s, i, j).

    Score is the fraction of the target's content words appearing in the window,
    less the length penalty. Returns None when the target has no content words
    or there are no segments.
    """
    words = content_words(target)
    if not words or not segs:
        return None
    best = None
    for width in range(1, max_segments + 1):
        for i in range(len(segs) - width + 1):
            found = set()
            for _, _, text in segs[i:i + width]:
                found |= content_words(text)
            score = len(found & words) / len(words) - length_penalty * (width - 1)
            if best is None or score > best[0]:
                best = (score, segs[i][0], segs[i + width - 1][1], i, i + width)
    return best


def iou(a, b):
    """Temporal intersection-over-union of two (start, end) spans."""
    (a0, a1), (b0, b1) = a, b
    if a1 <= a0 or b1 <= b0:
        return 0.0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def frames_inside(span, duration, n_frames=16):
    """How many uniformly sampled frames land inside `span`.

    Uniform sampling at 16 frames puts ~20 s between frames on a median AEMS
    video, while a derived moment is ~8 s, so a frame-based localiser is often
    looking at nothing from the right moment. Counting this is the honest
    ceiling on any such method.
    """
    if duration <= 0:
        return 0
    times = [(k + 0.5) * duration / n_frames for k in range(n_frames)]
    return sum(1 for t in times if span[0] <= t <= span[1])
