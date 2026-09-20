#!/usr/bin/env python3
"""
Shared helpers & evaluation harness for AEMS audio alignment methods.
=====================================================================
All methods map CLAP-audio embeddings into CLIP-text space. This module
provides:
  - loading of disjoint train/test anchor pairs (no leakage)
  - linear-map building blocks (ridge, Procrustes, CCA, whitening, local scaling)
  - a single evaluation harness reporting retrieval + geometry metrics on the
    held-out TEST set, in BOTH directions (CLIP-text->audio, audio->CLIP-text)
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    AEMS_MANIFEST_PATH,
    AEMS_AUDIO_EMBEDDINGS_PATH,
    AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
)
from src.data.metadata import load_metadata, filter_by_split

OUTPUT_DIR = "outputs/alignment"
ALIGNED_DIR = "embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ALIGNED_DIR, exist_ok=True)


def save_db(audio_proj, tag, full_vids):
    """Save aligned audio as {video_id: tensor} for a list of videos."""
    out = {}
    for i, v in enumerate(full_vids):
        out[v] = F.normalize(audio_proj[i], dim=-1).detach().cpu()
    path = os.path.join(ALIGNED_DIR, f"aems_audio_aligned_{tag}.pt")
    torch.save(out, path)
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_anchor_pairs(split):
    """Return (audio_mat, text_mat, video_ids) for the given split.
    Matrices are L2-normalized float32, keyed by disjoint train/test videos."""
    recs = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split=split)
    ids = set(r["video_id"] for r in recs)
    audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, map_location="cpu",
                          weights_only=False)
    text_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split=split),
                         map_location="cpu", weights_only=False)
    vids = sorted(set(audio_db.keys()) & set(text_db.keys()) & ids)
    X = F.normalize(torch.stack([audio_db[v].float() for v in vids]), dim=-1)
    Y = F.normalize(torch.stack([text_db[v].float() for v in vids]), dim=-1)
    return X, Y, vids


# ---------------------------------------------------------------------------
# Linear-map building blocks
# ---------------------------------------------------------------------------
def ridge_fit(X, Y, lam=1e-3):
    Xt, Yt = X.T, Y.T
    A = Xt @ Xt.T + lam * torch.eye(X.shape[1])
    W = torch.linalg.solve(A, Xt @ Yt.T)
    return W


def procrustes_map(X, Y):
    """Orthogonal Procrustes on centered inputs. Returns (W, b)."""
    mx = X.mean(0, keepdim=True)
    my = Y.mean(0, keepdim=True)
    Xc, Yc = X - mx, Y - my
    U, _, Vt = torch.linalg.svd(Xc.T @ Yc)
    W = U @ Vt
    b = (my - mx @ W.T).squeeze(0)
    return W, b


def whitening_stats(X, eps=1e-4):
    """ZCA whitening transform. Returns (W_whiten, mean) such that
    (X - mean) @ W_whiten is decorrelated & unit-variance (not L2 normalized)."""
    mean = X.mean(0, keepdim=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / Xc.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp(min=0)
    D_inv = torch.diag(1.0 / (evals + eps).sqrt())
    W_whiten = evecs @ D_inv @ evecs.T
    return W_whiten, mean.squeeze(0)


def apply_whiten(X, W_whiten, mean):
    return (X - mean) @ W_whiten


def cca_directions(X, Y, rank, eps=1e-4):
    """Return projection Wx (audio->shared) that maximizes correlation.
    Returns (Wx, Wx_project) where Wx [d, rank]."""
    n = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    Cxx = (Xc.T @ Xc) / (n - 1) + eps * torch.eye(X.shape[1])
    Cyy = (Yc.T @ Yc) / (n - 1) + eps * torch.eye(Y.shape[1])
    Cxy = (Xc.T @ Yc) / (n - 1)
    Cyy_inv = torch.linalg.inv(Cyy)
    # M eigenproblem for audio view
    M = torch.linalg.inv(Cxx) @ Cxy @ Cyy_inv @ Cxy.T
    evals, evecs = torch.linalg.eigh(M)
    order = torch.argsort(evals, descending=True)
    Wx = evecs[:, order[:rank]]  # [d, rank]
    return Wx


def local_scaling(emb, k=5):
    """Hubness reduction: divide each embedding by its k-th-neighbor distance."""
    n = emb.shape[0]
    sim = emb @ emb.T
    knn_vals, _ = torch.topk(sim, k + 1, dim=1)
    sigma = knn_vals[:, k].clamp(min=1e-3).unsqueeze(1)
    return F.normalize(emb / sigma, dim=-1)


# ---------------------------------------------------------------------------
# Evaluation harness (held-out TEST, both directions)
# ---------------------------------------------------------------------------
def hubness_skew(gallery, queries, K=5):
    sim = queries @ gallery.T
    idx = torch.argsort(sim, descending=True)[:, :K]
    n = gallery.shape[0]
    occ = torch.zeros(n)
    occ.index_add_(0, idx.flatten(), torch.ones(idx.numel()))
    occ = occ.detach().cpu().numpy()
    return float((((occ - occ.mean()) / (occ.std() + 1e-8)) ** 3).mean())


def eval_direction(query_mat, gallery_mat, ks=(1, 5, 10)):
    """query_mat [n,d], gallery_mat [n,d], matched rows. Returns metrics dict."""
    n = query_mat.shape[0]
    sim = query_mat @ gallery_mat.T
    ranks = torch.argsort(sim, descending=True)
    gt = torch.arange(n)
    out = {}
    for k in ks:
        topk = ranks[:, :k]
        out[f"R@{k}"] = float((topk == gt.view(-1, 1)).any(dim=1).float().mean())
    # MRR
    rank_of_gt = (ranks == gt.view(-1, 1)).nonzero()[:, 1].float()
    out["MRR"] = float((1.0 / (rank_of_gt + 1)).mean())
    out["mean_rank"] = float(rank_of_gt.mean())
    out["median_rank"] = float(rank_of_gt.median())
    return out


def evaluate_method(audio_proj, text_mat, tag, cat_by_vid=None, vids=None):
    """Produce the full metric block for a projected audio gallery.
    audio_proj [n,d] is the aligned audio; text_mat [n,d] is the CLIP-text
    description (test), rows matched. Returns a dict."""
    audio_proj = F.normalize(audio_proj, dim=-1)
    res = {
        "cliptext_to_audio": eval_direction(text_mat, audio_proj),
        "audio_to_cliptext": eval_direction(audio_proj, text_mat),
    }
    # Alignment (positive-pair cosine)
    pos = (audio_proj * text_mat).sum(dim=1)
    res["alignment_cosine"] = float(pos.mean())
    # Modality gap (centroid cosine distance to text)
    ac = F.normalize(audio_proj.mean(0, keepdim=True), dim=-1)
    tc = F.normalize(text_mat.mean(0, keepdim=True), dim=-1)
    res["modality_gap"] = float(1.0 - (ac * tc).sum())
    # Hubness skew (text->audio)
    res["hubness_skew_t2a"] = hubness_skew(audio_proj, text_mat)
    # Anisotropy (mean pairwise cosine of gallery)
    sub = audio_proj[torch.randperm(audio_proj.shape[0])[:1500]]
    sims = sub @ sub.T
    mask = ~torch.eye(sub.shape[0], dtype=torch.bool)
    res["gallery_anisotropy"] = float(sims[mask].mean())
    # Category-stratified R@1 (text->audio)
    if cat_by_vid is not None and vids is not None:
        cats = {}
        n = audio_proj.shape[0]
        sim = text_mat @ audio_proj.T
        top1 = torch.argsort(sim, descending=True)[:, 0]
        for i in range(n):
            c = cat_by_vid.get(vids[i], "?")
            cats.setdefault(c, {"n": 0, "hits": 0})
            cats[c]["n"] += 1
            if top1[i].item() == i:
                cats[c]["hits"] += 1
        res["category_r1"] = {c: d["hits"] / max(d["n"], 1)
                              for c, d in cats.items()}
    return res


def collect_text_embeddings(vids):
    """Load CLIP-text (description) for arbitrary video list (test)."""
    text_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split="test"),
                         map_location="cpu", weights_only=False)
    return torch.stack([text_db[v].float() for v in vids])
