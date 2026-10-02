"""Split a video's text into CLIP-sized chunks.

CLIP's text encoder takes 77 tokens, but AEMS transcripts run to a median of
677 words. The mean-pooled text branch averages all chunks into one vector,
which captures what a video is broadly about but dilutes any single moment.
The chunk branch keeps the chunks separate so a query can match the one
passage that answers it (max-sim / late interaction).
"""

from src.config import CHUNK_TOKEN_BUDGET

MAX_CHUNKS = 64


def split_into_chunks(text, budget=CHUNK_TOKEN_BUDGET):
    """Greedy word packing into ~budget-token chunks (same estimate as the
    mean-pooled text branch, so both views chunk identically)."""
    if not text or not text.strip():
        return []
    chunks, current, current_tokens = [], [], 0
    for word in text.split():
        word_len = len(word) // 4 + 1
        if current_tokens + word_len + 1 > budget and current:
            chunks.append(" ".join(current))
            current, current_tokens = [word], word_len
        else:
            current.append(word)
            current_tokens += word_len + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def video_chunks(record, max_chunks=MAX_CHUNKS, budget=CHUNK_TOKEN_BUDGET):
    """Description first (it summarises), then transcript passages.

    `budget` is the chunk size in tokens: 72 for CLIP's 77-token window, far
    larger for a retrieval encoder that accepts 512.
    """
    parts = []
    description = (record.get("text_description") or "").strip()
    if description:
        parts.append(description)
    parts.extend(split_into_chunks(record.get("text_transcript") or "", budget=budget))
    if not parts:
        title = (record.get("youtube_title") or "").strip()
        parts = [title] if title else []
    return parts[:max_chunks]


def lexical_fields(record, max_chunks=96):
    """Text for the BM25 branch: the passages plus the metadata fields.

    The dense branches only see description + transcript, because CLIP's 77-token
    window makes extra text expensive. BM25 has no such limit, and titles, YouTube
    descriptions and tags carry exactly the rare proper nouns lexical matching is
    good at. Measured on validation: R@1 0.561 -> 0.569 over passages alone.
    """
    parts = video_chunks(record, max_chunks=max_chunks)
    for key in ("youtube_title", "youtube_description"):
        value = (record.get(key) or "").strip()
        if value:
            parts += split_into_chunks(value)
    tags = record.get("youtube_tags") or []
    if tags:
        parts.append(" ".join(str(t) for t in tags))
    return parts[:max_chunks]
