"""Retrieval-trained text encoders (BGE / E5 / GTE).

The caption and passage branches are encoded by CLIP's text tower, which was
trained to match *image captions* through a 77-token window. We are asking it to
do document retrieval instead, and it shows: BM25 scores 0.550 R@1 while CLIP
caption scores 0.389 and CLIP passage 0.365. A bag-of-words model beating both
dense text branches is the symptom of an encoder mismatch, not of dense
retrieval being weak.

These models are trained for exactly this task, take 512 tokens, and need no new
dependency — `transformers` AutoModel plus pooling. Each family has its own
conventions, and getting them wrong quietly costs accuracy:

* E5 requires "query: " / "passage: " prefixes and uses mean pooling.
* BGE uses CLS pooling and a query instruction (no prefix on passages).
* GTE uses mean pooling and no prefixes.
"""

import torch
import torch.nn.functional as F

MODELS = {
    "bge": {"name": "BAAI/bge-base-en-v1.5", "pooling": "cls",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "passage_prefix": ""},
    "e5": {"name": "intfloat/e5-base-v2", "pooling": "mean",
           "query_prefix": "query: ", "passage_prefix": "passage: "},
    "gte": {"name": "thenlper/gte-base", "pooling": "mean",
            "query_prefix": "", "passage_prefix": ""},
}


def load_dense_encoder(device, key=None, name=None):
    """The dense-branch encoder, preferring AEMS fine-tuned weights when present.

    Every caller must go through this: queries have to be encoded by the same
    weights the passages were, and seven call sites each building their own
    encoder is how the branch set silently drifted out of sync before.
    """
    import os
    from src.config import AEMS_DENSE_MODEL_PATH, AEMS_DENSE_TEXT_MODEL

    if name is None and os.path.isdir(AEMS_DENSE_MODEL_PATH):
        name = AEMS_DENSE_MODEL_PATH
    return TextRetrievalEncoder(key or AEMS_DENSE_TEXT_MODEL, device, name=name)


def _pool(hidden, attention_mask, how):
    if how == "cls":
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class TextRetrievalEncoder:
    """Encodes queries and passages into one L2-normalized retrieval space."""

    def __init__(self, key="bge", device="cuda", dtype=torch.float16, max_length=512,
                 name=None):
        """`name` loads weights from elsewhere (e.g. a fine-tuned checkpoint) while
        keeping `key`'s prefix and pooling conventions, which must match what the
        passages were encoded with."""
        from transformers import AutoModel, AutoTokenizer

        if key not in MODELS:
            raise ValueError(f"unknown text encoder {key!r}; choose from {sorted(MODELS)}")
        self.config = dict(MODELS[key])
        if name:
            self.config["name"] = name
        self.key = key
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["name"])
        self.model = AutoModel.from_pretrained(self.config["name"], dtype=dtype)
        self.model.to(device).eval()

    @property
    def dim(self):
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, is_query, batch_size=64):
        """(n, dim) L2-normalized embeddings on CPU.

        `is_query` picks the prefix: these models are asymmetric, and encoding a
        passage as if it were a query measurably degrades retrieval.
        """
        prefix = self.config["query_prefix" if is_query else "passage_prefix"]
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            pooled = _pool(hidden, encoded["attention_mask"], self.config["pooling"])
            out.append(F.normalize(pooled.float(), dim=1).cpu())
        return torch.cat(out)

    def encode_queries(self, texts, **kwargs):
        return self.encode(texts, is_query=True, **kwargs)

    def encode_passages(self, texts, **kwargs):
        return self.encode(texts, is_query=False, **kwargs)
