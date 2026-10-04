"""Deciding which queries actually require watching the video.

The AEMS questions were generated from each video's own transcript and
description and reuse their wording, so the aggregate benchmark largely
measures matching a question back to its source text: R@1 is 0.965 where a
question shares most of its words with that text and 0.034 where it shares none.

Two measurable signals separate the queries where that shortcut is unavailable,
and both are needed because they fail differently:

* **answer overlap** — the manifest carries a written answer per question. If
  the answer's content words are absent from the transcript and description,
  the answer is not in the text. This is the stronger signal, but on its own it
  admits questions whose *wording* still gives the video away.
* **question overlap** — how much of the question's own wording appears in its
  source text. Low overlap removes the retrieval-time shortcut, but on its own
  it admits questions that are simply phrased differently from a text that
  still contains the answer.

`VISUAL_MARKERS` is a third, deliberately crude signal used only to label a
high-precision tier. It is a keyword list and will both over- and under-fire;
it is never used on its own, and the subset it defines is meant to be spot
checked by hand.
"""

import re
from functools import lru_cache

# Words too common to carry retrieval signal. "video" is included because it
# appears in nearly every generated question and nearly every description.
STOPWORDS = set(
    "what is the a an of in on at to for and or by with how why who when which that this "
    "does do are was were his her its their from as be been it you we they there here "
    "about into over under after before during video and also such more most very can "
    "will would should could has have had not no yes".split())

VISUAL_MARKERS = re.compile(
    r"\b(colou?r|wearing|wears|worn|shirt|written|writes|on the board|screen|how many|"
    r"count|number of|background|gesture|holding|holds|appears|visual|shown|shows|"
    r"displayed|logo|sign|label|behind|in front|clothing|hair|object|image|scene|frame|"
    r"camera|angle|lighting|facial|expression)\b", re.I)

TEXT_BLIND = "text_blind"
ANSWER_GROUNDED = "answer_grounded"
VISUALLY_GROUNDED = "visually_grounded"


@lru_cache(maxsize=200_000)
def content_words(text):
    """Lowercased content words: the unit both overlap measures are counted in.

    Cached and frozen: moment labelling slides overlapping windows over the same
    transcript segments thousands of times per video, so the same text is
    tokenised repeatedly. Returning a frozenset keeps the cached value safe to
    hand out; it still compares equal to a set and still supports `|`.
    """
    return frozenset(w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
                     if w not in STOPWORDS and len(w) > 2)


def overlap(text, pool):
    """Fraction of `text`'s content words present in `pool`, or None if it has none."""
    words = content_words(text)
    return None if not words else len(words & pool) / len(words)


def source_pool(record):
    """The text a query could be answered from without watching the video."""
    return (content_words(record.get("text_description"))
            | content_words(record.get("text_transcript")))


def tiers_for(question_overlap, answer_overlap, has_visual_marker,
              max_question_overlap=0.3, max_answer_overlap=0.3):
    """Which subsets this query belongs to.

    `visually_grounded` requires both overlap conditions *and* a visual marker,
    so it is a subset of the other two rather than a third independent slice.
    """
    tiers = []
    if question_overlap <= max_question_overlap:
        tiers.append(TEXT_BLIND)
    if answer_overlap <= max_answer_overlap:
        tiers.append(ANSWER_GROUNDED)
    if len(tiers) == 2 and has_visual_marker:
        tiers.append(VISUALLY_GROUNDED)
    return tiers
