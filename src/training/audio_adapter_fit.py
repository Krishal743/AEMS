"""Shared fitting routine for the WavLM -> CLIP-text audio adapter.

Used by bin/training/train_audio_adapter.py to fit the deployed adapter, and by
bin/training/train_gating_network.py to cross-fit per-fold adapters so the gate
is trained on out-of-fold (honest) audio similarities.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.models.audio_adapter import AudioAdapter

DEFAULT_EPOCHS = 40
DEFAULT_BATCH = 256
DEFAULT_LR = 1e-3


def fit_adapter(features, targets, device, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH,
                lr=DEFAULT_LR, seed=42, eval_fn=None, log=None):
    """Fit an AudioAdapter with symmetric InfoNCE.

    features: (n, d) L2-normalized audio features.
    targets:  list of n tensors, each (k_i, 512) CLIP-text positives for that
              item; one positive is sampled per item per step.
    eval_fn:  optional callable(model) -> float score; when given, the returned
              model is the best-scoring epoch's state. Callers must score on a
              split the adapter is not being fitted on.

    Returns (model, best_score, best_epoch); best_score is None without eval_fn.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = features.to(device)
    model = AudioAdapter(input_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_score, best_state, best_epoch = None, None, epochs
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        total, nb = 0.0, 0
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            y = torch.stack([targets[i][rng.integers(len(targets[i]))] for i in idx]).to(device)
            logits = model.logit_scale.exp().clamp(max=100) * model(X[idx]) @ y.T
            labels = torch.arange(len(idx), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        if eval_fn is not None:
            model.eval()
            score = eval_fn(model)
            if best_score is None or score > best_score:
                best_score, best_epoch = score, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if log:
                log(f"  epoch {ep + 1:02d}  loss={total / nb:.4f}  {score:.4f}")
        elif log:
            log(f"  epoch {ep + 1:02d}  loss={total / nb:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_score, best_epoch


@torch.no_grad()
def project(model, features, device):
    """(n, d) features -> (n, 512) CLIP-space embeddings on CPU."""
    return model(features.to(device)).cpu()


def out_of_fold_audio(feat_db, desc_db, query_emb, rows, fit_vids, device, folds=4,
                      epochs=39, seed=42, cache_path=None, log=print):
    """Project each fold of `fit_vids` with an adapter fitted on the other folds.

    The deployed adapter is fitted on these videos, so its similarities for them
    are optimistic; a downstream model trained on those scores learns to
    over-trust audio. Cross-fitting removes that bias — see docs/PROTOCOL.md.

    Cached to `cache_path` when given, since every downstream trainer needs the
    same projections.
    """
    import os
    import torch
    import torch.nn.functional as F

    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        if set(cached) >= set(fit_vids):
            log(f"[AUDIO] out-of-fold projections loaded from {cache_path}")
            return cached

    def matrix(vids):
        return F.normalize(torch.stack([torch.as_tensor(feat_db[v]).float() for v in vids]), dim=1)

    oof = {}
    chunks = [fit_vids[i::folds] for i in range(folds)]
    for i, fold in enumerate(chunks):
        other = [v for j, f in enumerate(chunks) if j != i for v in f]
        targets = [torch.cat([F.normalize(torch.as_tensor(desc_db[v]).float().view(1, -1), dim=1),
                              query_emb[rows[v]]]) for v in other]
        adapter, _, _ = fit_adapter(matrix(other), targets, device, epochs=epochs, seed=seed)
        projected = project(adapter, matrix(fold), device)
        for j, v in enumerate(fold):
            oof[v] = projected[j]
        log(f"  fold {i + 1}/{folds}: fitted on {len(other)}, projected {len(fold)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(oof, cache_path)
        log(f"[AUDIO] cached out-of-fold projections -> {cache_path}")
    return oof
