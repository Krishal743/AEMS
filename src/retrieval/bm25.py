"""BM25 lexical retrieval over transcript passages.

Every existing branch is dense: it matches meaning, which is what makes it
robust to paraphrase, but also what makes it blur rare exact terms. A question
like "what is the Bellman equation" shares an unusual token with exactly one
lecture, and lexical matching finds that immediately where an embedding may
not. BM25 is the standard complement to dense retrieval for this reason, and at
6,770 documents it costs nothing.

Scoring mirrors the passage branch: each passage is a document, and a video's
score is its best-matching passage, so the two are directly comparable.
"""

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
# Tuned on the validation split (k1 in {0.9, 1.5, 2.0} x b in {0.4, 0.75});
# k1=0.9 b=0.75 was best, though the spread across the grid was under a point.
K1 = 0.9
B = 0.75


def tokenize(text):
    return TOKEN.findall(text.lower())


class BM25PassageIndex:
    """BM25 over every video's passages, pooled to a per-video score by max."""

    def __init__(self, passages_per_video, k1=K1, b=B):
        """passages_per_video: list (one entry per video) of lists of passage strings."""
        self.k1, self.b = k1, b
        self.owner = []          # passage -> video index
        self.tokenized = []
        for video, passages in enumerate(passages_per_video):
            for text in passages:
                self.tokenized.append(Counter(tokenize(text)))
                self.owner.append(video)
        self.n_videos = len(passages_per_video)
        self.n_passages = len(self.tokenized)
        self.lengths = [sum(c.values()) for c in self.tokenized]
        self.avg_length = (sum(self.lengths) / self.n_passages) if self.n_passages else 0.0

        # Inverted index: term -> [(passage index, term frequency), ...]
        self.postings = {}
        for i, counts in enumerate(self.tokenized):
            for term, tf in counts.items():
                self.postings.setdefault(term, []).append((i, tf))

        # Robertson/Sparck-Jones idf with the +1 smoothing that keeps it positive
        # for terms appearing in more than half the collection.
        self.idf = {}
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log(1 + (self.n_passages - df + 0.5) / (df + 0.5))

    def score(self, query):
        """(n_videos,) best-passage BM25 score for one query string."""
        scores = [0.0] * self.n_videos
        if not self.n_passages:
            return scores
        passage_scores = {}
        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for passage, tf in posting:
                norm = 1 - self.b + self.b * self.lengths[passage] / (self.avg_length or 1.0)
                contribution = idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
                passage_scores[passage] = passage_scores.get(passage, 0.0) + contribution
        for passage, value in passage_scores.items():
            video = self.owner[passage]
            if value > scores[video]:
                scores[video] = value
        return scores

    def _matrix(self):
        """Sparse passage x term matrix of precomputed BM25 weights.

        Every term's contribution to a passage depends only on that passage, so
        the whole index can be precomputed once and a batch of queries scored as
        one sparse product instead of a Python loop per query. Built lazily
        because single-query search does not need it.
        """
        if getattr(self, "_weights", None) is not None:
            return self._weights, self._vocab
        from scipy import sparse

        vocab = {term: i for i, term in enumerate(self.postings)}
        rows, cols, values = [], [], []
        for term, posting in self.postings.items():
            idf, col = self.idf[term], vocab[term]
            for passage, tf in posting:
                norm = 1 - self.b + self.b * self.lengths[passage] / (self.avg_length or 1.0)
                rows.append(passage)
                cols.append(col)
                values.append(idf * tf * (self.k1 + 1) / (tf + self.k1 * norm))
        self._weights = sparse.csr_matrix((values, (rows, cols)),
                                          shape=(self.n_passages, len(vocab)))
        self._vocab = vocab
        return self._weights, self._vocab

    def score_batch(self, queries, batch_size=512):
        """(n_queries, n_videos) float32 tensor of best-passage scores."""
        import numpy as np
        import torch
        from scipy import sparse

        if not self.n_passages:
            return torch.zeros(len(queries), self.n_videos)
        weights, vocab = self._matrix()
        owner = np.asarray(self.owner)
        out = torch.empty(len(queries), self.n_videos, dtype=torch.float32)

        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            rows, cols = [], []
            for i, text in enumerate(batch):
                for term in tokenize(text):
                    col = vocab.get(term)
                    if col is not None:
                        rows.append(i)
                        cols.append(col)
            query_matrix = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(len(batch), weights.shape[1]))
            passage_scores = np.asarray((query_matrix @ weights.T).todense())
            # pool passages to videos by max, as score() does
            pooled = np.zeros((len(batch), self.n_videos), dtype=np.float32)
            np.maximum.at(pooled.T, owner, passage_scores.T)
            out[start:start + len(batch)] = torch.from_numpy(pooled)
        return out
