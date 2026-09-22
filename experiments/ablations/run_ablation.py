"""
Comprehensive Gating Network Ablation Runner.
Supports all loss functions, architectures, negative sampling strategies, and temperature configs.
Usage: python run_ablation.py --method pairwise_margin --margin 0.2 --negatives 10
"""
import argparse, gc, json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from src.encoders.clap_encode import CLAPEncoder
from src.models.gating_network import GatingNetwork, GatingNetworkPerCandidate
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_CLAP_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, DEVICE, set_seeds)
from src.data.metadata import load_metadata, filter_by_split

# =====================================================================
# Similarity
# =====================================================================
def angular_similarity(q, v, temperature=1.0):
    cs = q @ v.T
    a = 1 - torch.acos(torch.clamp(cs, -1, 1)) / np.pi
    return a / temperature if temperature != 1.0 else a

# =====================================================================
# Loss Functions
# =====================================================================
def loss_ce(sim_gated, correct_idx):
    log_p = F.log_softmax(sim_gated, dim=1)
    return F.nll_loss(log_p, correct_idx)

def loss_pairwise_margin(sim_gated, correct_idx, margin=0.2, n_neg=10, hard=False):
    N = sim_gated.shape[1]
    correct_scores = sim_gated[torch.arange(len(correct_idx)), correct_idx]
    if hard:
        neg_scores_all = sim_gated.clone()
        neg_scores_all[torch.arange(len(correct_idx)), correct_idx] = -float('inf')
        neg_scores, _ = torch.topk(neg_scores_all, min(n_neg, N-1), dim=1)
    else:
        neg_mask = torch.ones_like(sim_gated, dtype=torch.bool)
        neg_mask[torch.arange(len(correct_idx)), correct_idx] = False
        neg_indices = neg_mask.nonzero()
        perm = neg_indices[torch.randperm(len(neg_indices))][:len(correct_idx)*n_neg]
        neg_scores = sim_gated[perm[:, 0], perm[:, 1]].view(len(correct_idx), n_neg)
    target = torch.ones(len(correct_idx), n_neg, device=sim_gated.device)
    return F.margin_ranking_loss(correct_scores.unsqueeze(1).expand_as(neg_scores), neg_scores, target, margin, reduction='mean')

def loss_mnrl(sim_gated, correct_idx, temperature=0.07):
    sim_correct = sim_gated[torch.arange(len(correct_idx)), correct_idx]
    log_denom = torch.logsumexp(sim_gated / temperature, dim=1)
    return (-sim_correct / temperature + log_denom).mean()

def loss_info_nce(sim_gated, correct_idx, temperature=0.07):
    return loss_mnrl(sim_gated, correct_idx, temperature)

def loss_soft_label(sim_v, sim_t, sim_a, weights, correct_idx, temperature=0.5):
    sim_correct_v = sim_v[torch.arange(len(correct_idx)), correct_idx]
    sim_correct_t = sim_t[torch.arange(len(correct_idx)), correct_idx]
    sim_correct_a = sim_a[torch.arange(len(correct_idx)), correct_idx]
    target = F.softmax(torch.stack([sim_correct_v, sim_correct_t, sim_correct_a], dim=1) / temperature, dim=1)
    return F.kl_div(F.log_softmax(weights, dim=1), target, reduction='batchmean')

# =====================================================================
# Differentiable Ranking Losses (Phase A)
# =====================================================================
def loss_listnet(sim_gated, correct_idx, temperature=0.1):
    """ListNet: optimizes probability that correct item is at top.
    Uses cross-entropy between predicted top-1 distribution and target (one-hot for correct)."""
    log_p = F.log_softmax(sim_gated / temperature, dim=1)
    target = torch.zeros_like(sim_gated)
    target[torch.arange(len(correct_idx)), correct_idx] = 1.0
    return F.kl_div(log_p, target, reduction='batchmean')

def loss_listmle(sim_gated, correct_idx, temperature=0.1, topk=10):
    """ListMLE top-1 (Plackett-Luce): probability the correct item is ranked #1.
    Computed as softmax-CE of the correct item vs all others — direct R@1 signal.
    Vectorized and O(batch * N), feasible for the 5748-candidate gallery."""
    batch_size = sim_gated.shape[0]
    idx_arange = torch.arange(batch_size, device=sim_gated.device)
    scores_correct = sim_gated[idx_arange, correct_idx] / temperature
    log_denom = torch.logsumexp(sim_gated / temperature, dim=1)
    return (-scores_correct + log_denom).mean()

def loss_approx_ndcg(sim_gated, correct_idx, temperature=0.1):
    """Approximate NDCG using pairwise sigmoid approximation.
    Optimizes a smooth surrogate for NDCG@k."""
    batch_size = sim_gated.shape[0]
    N = sim_gated.shape[1]
    total_loss = 0.0
    for i in range(batch_size):
        scores = sim_gated[i] / temperature
        target = correct_idx[i]
        correct_score = scores[target]
        neg_scores = scores[torch.arange(N, device=scores.device) != target]
        diffs = correct_score - neg_scores
        weights = 1.0 / torch.log2(torch.arange(2, len(diffs)+2, device=scores.device).float())
        pairwise = torch.sigmoid(-diffs)
        loss_i = (weights * pairwise).sum() / weights.sum()
        total_loss += loss_i
    return total_loss / batch_size

def loss_listnet_hybrid(sim_gated, correct_idx, temperature=0.1, alpha=0.5):
    """Hybrid: alpha * ListNet + (1-alpha) * Pairwise Margin."""
    l_listnet = loss_listnet(sim_gated, correct_idx, temperature)
    l_pairwise = loss_pairwise_margin(sim_gated, correct_idx, margin=0.1, n_neg=10, hard=True)
    return alpha * l_listnet + (1 - alpha) * l_pairwise

LOSS_FUNCTIONS = {
    'ce': lambda s, c, **kw: loss_ce(s, c),
    'pairwise_margin': lambda s, c, **kw: loss_pairwise_margin(s, c, kw.get('margin', 0.2), kw.get('n_neg', 10), kw.get('hard', False)),
    'mnrl': lambda s, c, **kw: loss_mnrl(s, c, kw.get('temperature', 0.07)),
    'info_nce': lambda s, c, **kw: loss_info_nce(s, c, kw.get('temperature', 0.07)),
    'listnet': lambda s, c, **kw: loss_listnet(s, c, kw.get('temperature', 0.1)),
    'listmle': lambda s, c, **kw: loss_listmle(s, c, kw.get('temperature', 0.1)),
    'approx_ndcg': lambda s, c, **kw: loss_approx_ndcg(s, c, kw.get('temperature', 0.1)),
    'listnet_hybrid': lambda s, c, **kw: loss_listnet_hybrid(s, c, kw.get('temperature', 0.1), kw.get('alpha', 0.5)),
}

# =====================================================================
# Architecture variants
# =====================================================================
def make_gating_net(arch='default', dropout=0.3, learnable_temp=False, learnable_scale=False, weight_reg=0.0, per_candidate=False, constant_weights=False, init_weights=None):
    if constant_weights:
        return GatingNetwork(input_dim=512, num_modalities=3, constant_weights=True,
                             init_weights=init_weights, weight_reg=weight_reg)
    if per_candidate:
        hidden = {'tiny': 32, 'small': 64, 'default': 128, 'large': 256}.get(arch, 32)
        return GatingNetworkPerCandidate(input_dim=512, hidden_dim=hidden, num_modalities=3,
                                         dropout=dropout, learnable_temp=learnable_temp,
                                         learnable_scale=learnable_scale)
    else:
        configs = {
            'default': dict(input_dim=512, hidden_dim=128, num_modalities=3, dropout=dropout),
            'tiny': dict(input_dim=512, hidden_dim=32, num_modalities=3, dropout=dropout),
            'small': dict(input_dim=512, hidden_dim=64, num_modalities=3, dropout=dropout),
            'large': dict(input_dim=512, hidden_dim=256, num_modalities=3, dropout=dropout),
            'deep4': dict(input_dim=512, hidden_dim=128, num_modalities=3, dropout=dropout),
        }
        cfg = configs.get(arch, configs['default'])
        return GatingNetwork(**cfg, learnable_temp=learnable_temp, learnable_scale=learnable_scale, weight_reg=weight_reg)

# =====================================================================
# Data loading
# =====================================================================
def load_data(args):
    set_seeds(args.seed)
    metadata = load_metadata(args.manifest)
    train_items = filter_by_split(metadata, "train")
    test_items = filter_by_split(metadata, "test")

    video_db = torch.load(args.video_embeds, weights_only=False)
    audio_db = torch.load(args.audio_embeds, weights_only=False)
    text_db_train = torch.load(args.text_embeds_train, weights_only=False)
    text_db_test = torch.load(args.text_embeds_test, weights_only=False)

    all_vids = set()
    for item in train_items + test_items:
        all_vids.add(item["video_id"])

    common_all = sorted(set(video_db.keys()) & set(audio_db.keys()) & set(text_db_train.keys()) & all_vids)
    common_test = sorted(set(video_db.keys()) & set(audio_db.keys()) & set(text_db_test.keys()) &
                         set(item["video_id"] for item in test_items))

    train_queries, train_vids = [], []
    for item in train_items:
        vid = item["video_id"]
        if vid not in common_all: continue
        for q in item["qa_questions"]:
            train_queries.append(q)
            train_vids.append(vid)

    test_queries, test_vids = [], []
    for item in test_items:
        vid = item["video_id"]
        if vid not in common_test: continue
        for q in item["qa_questions"]:
            test_queries.append(q)
            test_vids.append(vid)

    vid_mat = torch.stack([F.normalize(video_db[v].float(), dim=0) for v in common_all])
    aud_mat = torch.stack([F.normalize(audio_db[v].float(), dim=0) for v in common_all])
    txt_mat = torch.stack([F.normalize(text_db_train[v].float(), dim=0) for v in common_all])
    vid_mat_test = torch.stack([F.normalize(video_db[v].float(), dim=0) for v in common_test])
    aud_mat_test = torch.stack([F.normalize(audio_db[v].float(), dim=0) for v in common_test])
    txt_mat_test = torch.stack([F.normalize(text_db_test[v].float(), dim=0) for v in common_test])

    print("[ENC] Encoding queries...", flush=True)
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clap_encoder = CLAPEncoder(device=DEVICE)

    def enc_clip(texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            toks = clip.tokenize(texts[i:i+bs], truncate=True).to(DEVICE)
            with torch.no_grad():
                e = F.normalize(clip_model.encode_text(toks), dim=-1)
            out.append(e.cpu())
            del e, toks; gc.collect(); torch.cuda.empty_cache()
        return torch.cat(out)

    def enc_clap(texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            with torch.no_grad():
                e = clap_encoder.encode_text(texts[i:i+bs])
            if isinstance(e, np.ndarray): e = torch.from_numpy(e).float()
            e = F.normalize(e, dim=-1)
            out.append(e.cpu())
            del e; gc.collect(); torch.cuda.empty_cache()
        return torch.cat(out)

    q_clip_train = enc_clip(train_queries)
    q_clap_train = enc_clap(train_queries)
    q_clip_test = enc_clip(test_queries)
    q_clap_test = enc_clap(test_queries)
    del clip_model, clap_encoder; gc.collect(); torch.cuda.empty_cache()

    return {
        'train_queries': train_queries, 'train_vids': train_vids,
        'test_queries': test_queries, 'test_vids': test_vids,
        'common_all': common_all, 'common_test': common_test,
        'vid_mat': vid_mat, 'aud_mat': aud_mat, 'txt_mat': txt_mat,
        'vid_mat_test': vid_mat_test, 'aud_mat_test': aud_mat_test, 'txt_mat_test': txt_mat_test,
        'q_clip_train': q_clip_train, 'q_clap_train': q_clap_train,
        'q_clip_test': q_clip_test, 'q_clap_test': q_clap_test,
        'vid_to_idx': {v: i for i, v in enumerate(common_all)},
    }

# =====================================================================
# Training loop
# =====================================================================
def train_and_eval(args, data, config):
    set_seeds(args.seed)
    loss_name = config.get('loss', 'ce')
    arch = config.get('arch', 'default')
    n_neg = config.get('n_neg', 10)
    margin = config.get('margin', 0.2)
    hard_neg = config.get('hard_neg', False)
    temperature = config.get('temperature', 0.07)
    lr = config.get('lr', 1e-3)
    epochs = config.get('epochs', 15)
    tau_audio = config.get('tau_audio', 0.5)
    tau_text = config.get('tau_text', 1.0)
    tau_visual = config.get('tau_visual', 1.0)
    use_soft_label = config.get('soft_label', False)
    soft_tau = config.get('soft_tau', 0.5)
    learnable_temp = config.get('learnable_temp', False)
    learnable_scale = config.get('learnable_scale', False)
    weight_reg = config.get('weight_reg', 0.0)
    per_candidate = config.get('per_candidate', False)
    distill = config.get('distill', False)
    distill_type = config.get('distill_type', 'kl')
    distill_alpha = config.get('distill_alpha', 0.5)
    scale_audio = config.get('scale_audio', 0.8)
    constant_weights = config.get('constant_weights', False)
    init_weights = config.get('init_weights', None)
    target_weights = config.get('target_weights', None)

    gating_net = make_gating_net(arch, dropout=config.get('dropout', 0.3),
                                 learnable_temp=learnable_temp, learnable_scale=learnable_scale,
                                 weight_reg=weight_reg, per_candidate=per_candidate,
                                 constant_weights=constant_weights, init_weights=init_weights).to(DEVICE)
    if target_weights is not None:
        gating_net.target_weights = torch.tensor(target_weights).float().to(DEVICE)
    optimizer = torch.optim.AdamW(gating_net.parameters(), lr=lr, weight_decay=1e-5)

    num_train = len(data['train_queries'])
    num_candidates = len(data['common_all'])
    gt_indices = torch.tensor([data['vid_to_idx'][data['train_vids'][i]] for i in range(num_train)], device=DEVICE)

    # Oracle distillation targets (precomputed, shape (num_train, 3))
    oracle_weights = data.get('oracle_weights_train', None)
    if distill and oracle_weights is None:
        # Compute per-query soft oracle on the fly: softmax of sim to GT video across modalities
        print("  [ORACLE] Computing per-query soft oracle targets...", flush=True)
        with torch.no_grad():
            q_clip_all = data['q_clip_train'].float().to(DEVICE)
            q_clap_all = data['q_clap_train'].float().to(DEVICE)
            gt_indices_all = gt_indices
            # Compute sim to GT per modality (using default taus)
            sim_gt_v = angular_similarity(q_clip_all, data['vid_mat'][gt_indices_all].float().to(DEVICE), tau_visual)
            sim_gt_t = angular_similarity(q_clip_all, data['txt_mat'][gt_indices_all].float().to(DEVICE), tau_text)
            sim_gt_a = angular_similarity(q_clap_all, data['aud_mat'][gt_indices_all].float().to(DEVICE), tau_audio)
            # Soft oracle: which modality helps ranking? Use softmax over (sim_gt - mean_candidate_sim)
            oracle_weights = F.softmax(torch.stack([sim_gt_v, sim_gt_t, sim_gt_a], dim=1), dim=1).cpu()
        data['oracle_weights_train'] = oracle_weights
        del q_clip_all, q_clap_all

    print(f"\n{'='*60}", flush=True)
    print(f"CONFIG: loss={loss_name} arch={arch} n_neg={n_neg} margin={margin} hard={hard_neg} temp={temperature} lr={lr} epochs={epochs}", flush=True)
    print(f"  τ_audio={tau_audio} τ_text={tau_text} τ_visual={tau_visual} scale_a={scale_audio} lr={lr}", flush=True)
    print(f"  learnable_temp={learnable_temp} learnable_scale={learnable_scale} weight_reg={weight_reg} per_candidate={per_candidate}", flush=True)
    print(f"  distill={distill} {'('+distill_type+' a='+str(distill_alpha)+')' if distill else ''}", flush=True)
    print(f"  Architecture: {gating_net}", flush=True)
    print(f"  Train queries: {num_train}, Candidates: {num_candidates}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        gating_net.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        num_batches = 0
        indices = torch.randperm(num_train)

        for q_start in range(0, num_train, 32):
            q_end = min(q_start + 32, num_train)
            batch_len = q_end - q_start
            if batch_len < 16: continue

            batch_idx = indices[q_start:q_end]
            q_clip = data['q_clip_train'][batch_idx].float().to(DEVICE)
            q_clap = data['q_clap_train'][batch_idx].float().to(DEVICE)

            # Resolve temperatures / scales (fixed or learnable)
            if learnable_temp:
                taus = gating_net.get_temperatures()
                t_v, t_t, t_a = taus['visual'], taus['text'], taus['audio']
            else:
                t_v, t_t, t_a = tau_visual, tau_text, tau_audio
            if learnable_scale:
                scales = gating_net.get_scales()
                s_a = scales['audio']
            else:
                s_a = scale_audio

            all_sim_v = torch.zeros(batch_len, num_candidates, device=DEVICE)
            all_sim_a = torch.zeros(batch_len, num_candidates, device=DEVICE)
            all_sim_t = torch.zeros(batch_len, num_candidates, device=DEVICE)

            for vs in range(0, num_candidates, 500):
                ve = min(vs + 500, num_candidates)
                vb_v = data['vid_mat'][vs:ve].float().to(DEVICE)
                vb_a = data['aud_mat'][vs:ve].float().to(DEVICE)
                vb_t = data['txt_mat'][vs:ve].float().to(DEVICE)
                all_sim_v[:, vs:ve] = angular_similarity(q_clip, vb_v, t_v)
                all_sim_a[:, vs:ve] = angular_similarity(q_clap, vb_a, t_a)
                all_sim_t[:, vs:ve] = angular_similarity(q_clip, vb_t, t_t)
                del vb_v, vb_a, vb_t

            if per_candidate:
                # weights: (batch, n_candidates, 3)
                weights = gating_net(q_clip, all_sim_v, all_sim_t, all_sim_a)
                sim_gated = (weights[:, :, 0] * all_sim_v +
                             weights[:, :, 1] * all_sim_t +
                             weights[:, :, 2] * all_sim_a * s_a)
            else:
                weights = gating_net(q_clip)  # (batch, 3)
                sim_gated = (weights[:, 0:1] * all_sim_v +
                             weights[:, 1:2] * all_sim_t +
                             weights[:, 2:3] * all_sim_a * s_a)

            batch_gt = gt_indices[batch_idx]

            if use_soft_label:
                loss_main = loss_soft_label(all_sim_v, all_sim_t, all_sim_a, weights.mean(dim=1) if per_candidate else weights, batch_gt, soft_tau)
            elif loss_name in LOSS_FUNCTIONS:
                loss_main = LOSS_FUNCTIONS[loss_name](sim_gated, batch_gt,
                    margin=margin, n_neg=n_neg, hard=hard_neg, temperature=temperature)
            else:
                loss_main = loss_ce(sim_gated, batch_gt)

            loss = loss_main

            # Oracle distillation
            if distill and oracle_weights is not None:
                w_pred = weights.mean(dim=1) if per_candidate else weights
                batch_oracle = oracle_weights[batch_idx].to(DEVICE)
                if distill_type == 'kl':
                    loss_d = F.kl_div(F.log_softmax(w_pred, dim=-1), batch_oracle.clamp(min=1e-7), reduction='batchmean')
                elif distill_type == 'mse':
                    loss_d = F.mse_loss(w_pred, batch_oracle)
                elif distill_type == 'ce':
                    loss_d = F.cross_entropy(w_pred, batch_oracle.argmax(dim=-1))
                else:
                    loss_d = F.mse_loss(w_pred, batch_oracle)
                loss = distill_alpha * loss_d + (1 - distill_alpha) * loss_main

            # Weight regularization toward target (e.g. InverseRank)
            loss = loss + gating_net.regularization_loss()
            if (not constant_weights) and weight_reg > 0 and target_weights is not None and not per_candidate:
                mean_w = weights.detach().mean(dim=0)
                loss = loss + weight_reg * F.mse_loss(weights.mean(dim=0), gating_net.target_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gating_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss
            num_batches += 1
            del q_clip, q_clap, all_sim_v, all_sim_a, all_sim_t, weights, sim_gated, loss

        avg_loss = (total_loss / max(num_batches, 1)).item()
        if avg_loss < best_loss: best_loss = avg_loss
        elapsed = time.time() - t0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            if learnable_temp:
                taus = gating_net.get_temperatures()
                print(f"  Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  Best: {best_loss:.4f}  τ:[{taus['visual'].item():.2f},{taus['text'].item():.2f},{taus['audio'].item():.2f}]  Time: {elapsed:.0f}s", flush=True)
            else:
                print(f"  Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  Best: {best_loss:.4f}  Time: {elapsed:.0f}s", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    # Eval
    gating_net.eval()
    if learnable_temp:
        taus = gating_net.get_temperatures()
        t_v, t_t, t_a = taus['visual'].item(), taus['text'].item(), taus['audio'].item()
    else:
        t_v, t_t, t_a = tau_visual, tau_text, tau_audio
    if learnable_scale:
        s_a = gating_net.get_scales()['audio'].item()
    else:
        s_a = scale_audio
    sim_v = angular_similarity(data['q_clip_test'].float().to(DEVICE), data['vid_mat_test'].float().to(DEVICE), t_v).to(DEVICE)
    sim_t = angular_similarity(data['q_clip_test'].float().to(DEVICE), data['txt_mat_test'].float().to(DEVICE), t_t).to(DEVICE)
    sim_a = angular_similarity(data['q_clap_test'].float().to(DEVICE), data['aud_mat_test'].float().to(DEVICE), t_a).to(DEVICE)

    if per_candidate:
        with torch.no_grad():
            gate_w = gating_net(data['q_clip_test'].float().to(DEVICE), sim_v, sim_t, sim_a)
        sim_gated = (gate_w[:, :, 0] * sim_v + gate_w[:, :, 1] * sim_t + gate_w[:, :, 2] * sim_a * s_a)
        avg_w = gate_w.mean(dim=(0, 1)).cpu()
    else:
        with torch.no_grad():
            gate_w = gating_net(data['q_clip_test'].float().to(DEVICE))
        sim_gated = (gate_w[:, 0:1] * sim_v + gate_w[:, 1:2] * sim_t + gate_w[:, 2:3] * sim_a * s_a)
        avg_w = gate_w.mean(dim=0).cpu()

    metrics = evaluate_retrieval(sim_gated, data['test_vids'], data['common_test'], ks=[1, 5, 10])
    text_metrics = evaluate_retrieval(sim_t, data['test_vids'], data['common_test'], ks=[1, 5, 10])
    vis_metrics = evaluate_retrieval(sim_v, data['test_vids'], data['common_test'], ks=[1, 5, 10])
    aud_metrics = evaluate_retrieval(sim_a, data['test_vids'], data['common_test'], ks=[1, 5, 10])

    total_time = time.time() - t0

    print(f"\n  RESULTS:", flush=True)
    print(f"    Visual only:  R@1={vis_metrics['R@1']:.4f}  R@5={vis_metrics['R@5']:.4f}  R@10={vis_metrics['R@10']:.4f}", flush=True)
    print(f"    Text only:    R@1={text_metrics['R@1']:.4f}  R@5={text_metrics['R@5']:.4f}  R@10={text_metrics['R@10']:.4f}", flush=True)
    print(f"    Audio only:   R@1={aud_metrics['R@1']:.4f}  R@5={aud_metrics['R@5']:.4f}  R@10={aud_metrics['R@10']:.4f}", flush=True)
    print(f"    Gating:       R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}", flush=True)
    print(f"    Weights:      w_v={avg_w[0]:.4f}  w_t={avg_w[1]:.4f}  w_a={avg_w[2]:.4f}  (τ:[{t_v:.2f},{t_t:.2f},{t_a:.2f}] s_a={s_a:.2f})", flush=True)
    print(f"    Time: {total_time:.0f}s", flush=True)

    return {
        'config': config,
        'metrics': metrics,
        'text_metrics': text_metrics,
        'vis_metrics': vis_metrics,
        'aud_metrics': aud_metrics,
        'avg_weights': [float(avg_w[0]), float(avg_w[1]), float(avg_w[2])],
        'tau_audio': t_a, 'tau_text': t_t, 'tau_visual': t_v,
        'scale_audio': s_a,
        'best_loss': best_loss,
        'time_s': total_time,
    }

# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
    parser.add_argument("--video-embeds", type=str, default=AEMS_VID_EMBEDDINGS_PATH)
    parser.add_argument("--audio-embeds", type=str, default=AEMS_CLAP_AUDIO_EMBEDDINGS_PATH)
    parser.add_argument("--text-embeds-train", type=str, default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"))
    parser.add_argument("--text-embeds-test", type=str, default=AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"))
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "ce", "pairwise_margin", "mnrl", "info_nce", "soft_label", "arch", "neg_sampling", "temperature", 
                                "listnet", "listmle", "approx_ndcg", "listnet_hybrid", "optimize_const", "reg_weights", "learnable_temp", "learnable_temp2", "per_candidate"])
    parser.add_argument("--config", type=str, default=None, help="JSON config string")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output", type=str, default="outputs/ablations")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.config:
        config = json.loads(args.config)
        data = load_data(args)
        result = train_and_eval(args, data, config)
        with open(os.path.join(args.output, f"result_{config.get('name', 'custom')}.json"), "w") as f:
            json.dump(result, f, indent=2)
        return

    if args.method == "all":
        data = load_data(args)
        all_configs = get_all_configs(args.epochs)
        all_results = []
        for i, cfg in enumerate(all_configs):
            print(f"\n>>> Running config {i+1}/{len(all_configs)}: {cfg.get('name', 'unnamed')}", flush=True)
            result = train_and_eval(args, data, cfg)
            all_results.append(result)
            with open(os.path.join(args.output, "all_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)
        print_summary(all_results)
    else:
        data = load_data(args)
        configs = get_method_configs(args.method, args.epochs)
        for cfg in configs:
            result = train_and_eval(args, data, cfg)
            with open(os.path.join(args.output, f"result_{cfg.get('name', 'unknown')}.json"), "w") as f:
                json.dump(result, f, indent=2)

def get_all_configs(epochs=15):
    configs = []
    configs.extend(get_method_configs('loss_functions', epochs))
    configs.extend(get_method_configs('architectures', epochs))
    configs.extend(get_method_configs('neg_sampling', epochs))
    configs.extend(get_method_configs('temperature', epochs))
    return configs

def get_method_configs(method, epochs=15):
    configs = []

    if method in ['loss_functions', 'ce']:
        configs.append({'name': 'P1_ce_baseline', 'loss': 'ce', 'epochs': epochs})

    if method in ['loss_functions', 'pairwise_margin']:
        for hard in [False, True]:
            for margin in [0.1, 0.2, 0.5]:
                for n_neg in [10, 100]:
                    configs.append({
                        'name': f"P2_pm_m{margin}_n{n_neg}_{'hard' if hard else 'rand'}",
                        'loss': 'pairwise_margin', 'margin': margin, 'n_neg': n_neg,
                        'hard_neg': hard, 'epochs': epochs
                    })

    if method in ['loss_functions', 'mnrl', 'info_nce']:
        for temp in [0.01, 0.07, 0.1, 0.5]:
            for n_neg in [100, 500]:
                configs.append({
                    'name': f"P3_mnrl_t{temp}_n{n_neg}",
                    'loss': 'mnrl', 'temperature': temp, 'n_neg': n_neg, 'epochs': epochs
                })

    if method in ['loss_functions', 'soft_label']:
        for st in [0.25, 0.5, 1.0, 2.0]:
            configs.append({
                'name': f"P4_softlabel_t{st}",
                'soft_label': True, 'soft_tau': st, 'epochs': epochs
            })

    if method in ['architectures', 'arch']:
        for arch in ['tiny', 'small', 'default', 'large']:
            configs.append({
                'name': f"A1_arch_{arch}",
                'loss': 'ce', 'arch': arch, 'epochs': epochs
            })

    if method in ['neg_sampling']:
        for n_neg in [10, 50, 100, 500]:
            configs.append({
                'name': f"N1_mnrl_n{n_neg}",
                'loss': 'mnrl', 'n_neg': n_neg, 'temperature': 0.07, 'epochs': epochs
            })
        for hard in [False, True]:
            configs.append({
                'name': f"N2_pm_n10_{'hard' if hard else 'rand'}",
                'loss': 'pairwise_margin', 'margin': 0.2, 'n_neg': 10, 'hard_neg': hard, 'epochs': epochs
            })

    if method in ['temperature']:
        for tau_a in [0.1, 0.3, 0.5, 1.0]:
            for tau_t in [0.5, 1.0, 2.0]:
                configs.append({
                    'name': f"T1_tau_a{tau_a}_t{tau_t}",
                    'loss': 'mnrl', 'temperature': 0.07, 'tau_audio': tau_a, 'tau_text': tau_t,
                    'tau_visual': 1.0, 'epochs': epochs
                })

    # Phase A: ListNet, ListMLE, Approx NDCG, Hybrid
    if method in ['loss_functions', 'listnet']:
        for temp in [0.05, 0.1, 0.2]:
            configs.append({
                'name': f"L1_listnet_t{temp}",
                'loss': 'listnet', 'temperature': temp, 'epochs': epochs
            })
    if method in ['loss_functions', 'listmle']:
        for temp in [0.05, 0.1, 0.2]:
            configs.append({
                'name': f"L2_listmle_t{temp}",
                'loss': 'listmle', 'temperature': temp, 'epochs': epochs
            })
    if method in ['loss_functions', 'approx_ndcg']:
        for temp in [0.05, 0.1, 0.2]:
            configs.append({
                'name': f"L3_ndcg_t{temp}",
                'loss': 'approx_ndcg', 'temperature': temp, 'epochs': epochs
            })
    if method in ['loss_functions', 'listnet_hybrid']:
        for alpha in [0.3, 0.5, 0.7]:
            configs.append({
                'name': f"L4_hybrid_a{alpha}",
                'loss': 'listnet_hybrid', 'alpha': alpha, 'epochs': epochs
            })

    # Phase B: Oracle Distillation
    if method in ['oracle_distill']:
        # Will be handled specially - needs oracle weights computed first
        pass

    # Phase C2: Constant weight optimization (learn global weights directly)
    if method in ['optimize_const']:
        sweep_best = [0.359, 0.639, 0.002]
        configs.append({'name': 'C2_data_sweepbest_pm', 'loss': 'pairwise_margin', 'margin': 0.1,
                        'n_neg': 10, 'hard_neg': True, 'constant_weights': True,
                        'init_weights': sweep_best, 'epochs': epochs})
        for init_sel in ['uniform', 'inverserank']:
            for temp in [0.07]:
                configs.append({
                    'name': f"C2_const_{init_sel}_t{temp}",
                    'loss': 'mnrl', 'temperature': temp, 'constant_weights': True,
                    'init_weights': [0.333, 0.333, 0.333] if init_sel == 'uniform' else [0.359, 0.639, 0.002],
                    'epochs': epochs
                })
        for init_sel in ['uniform', 'inverserank']:
            configs.append({
                'name': f"C2b_const_{init_sel}_pm",
                'loss': 'pairwise_margin', 'margin': 0.2, 'n_neg': 10, 'hard_neg': True,
                'constant_weights': True,
                'init_weights': [0.333, 0.333, 0.333] if init_sel == 'uniform' else [0.359, 0.639, 0.002],
                'epochs': epochs
            })
        # C5: constant net tethered toward train-selected optimum via weight_reg
        for reg in [0.01, 0.1, 1.0]:
            configs.append({
                'name': f"C5_const_reg{reg}_pm",
                'loss': 'pairwise_margin', 'margin': 0.2, 'n_neg': 10, 'hard_neg': True,
                'constant_weights': True, 'weight_reg': reg,
                'init_weights': [0.30, 0.69, 0.01], 'target_weights': [0.30, 0.69, 0.01],
                'epochs': epochs
            })
        # C6: constant net with approx-NDCG (direct R@k surrogate)
        for temp in [0.05, 0.1]:
            configs.append({
                'name': f"C6_const_ndcg_t{temp}",
                'loss': 'approx_ndcg', 'temperature': temp,
                'constant_weights': True, 'init_weights': [0.333, 0.333, 0.333],
                'epochs': epochs
            })

    # Phase C3: Per-query with regularization toward InverseRank target
    if method in ['reg_weights']:
        for reg in [0.001, 0.01, 0.1]:
            configs.append({
                'name': f"C3_reg{reg}_pm",
                'loss': 'pairwise_margin', 'margin': 0.1, 'n_neg': 10, 'hard_neg': True,
                'weight_reg': reg, 'target_weights': [0.359, 0.639, 0.002],
                'epochs': epochs
            })
        for reg in [0.001, 0.01, 0.1]:
            configs.append({
                'name': f"C3b_reg{reg}_mnrl",
                'loss': 'mnrl', 'temperature': 0.01,
                'weight_reg': reg, 'target_weights': [0.359, 0.639, 0.002],
                'epochs': epochs
            })

    # Phase C4: Learnable temperature/scale on best config (handled under reg_weights/learnable_temp2)

    if method in ['learnable_temp2']:
        for lr in [1e-3, 5e-4]:
            configs.append({
                'name': f"LT1_learnable_lr{lr}",
                'loss': 'pairwise_margin', 'margin': 0.1, 'n_neg': 10, 'hard_neg': True,
                'learnable_temp': True, 'learnable_scale': True,
                'lr': lr, 'epochs': epochs
            })
        configs.append({
            'name': f"LT2_learnable_wd",
            'loss': 'pairwise_margin', 'margin': 0.1, 'n_neg': 10, 'hard_neg': True,
            'learnable_temp': True, 'learnable_scale': True,
            'weight_reg': 0.01, 'target_weights': [0.324, 0.668, 0.009], 'lr': 1e-3, 'epochs': epochs
        })

    # Phase D: Per-Candidate Gating
    if method in ['per_candidate']:
        configs.append({
            'name': f"PC1_percand_pm",
            'loss': 'pairwise_margin', 'margin': 0.1, 'n_neg': 10, 'hard_neg': True,
            'per_candidate': True, 'arch': 'tiny', 'epochs': epochs
        })
        configs.append({
            'name': f"PC2_percand_listnet",
            'loss': 'listnet', 'temperature': 0.1,
            'per_candidate': True, 'arch': 'tiny', 'epochs': epochs
        })

    return configs

def print_summary(results):
    print("\n" + "=" * 90)
    print("ABLATION SUMMARY — sorted by R@1")
    print("=" * 90)
    sorted_r = sorted(results, key=lambda x: -x['metrics']['R@1'])
    text_r1 = sorted_r[0]['text_metrics']['R@1'] if sorted_r else 0
    print(f"{'Name':>50} | {'R@1':>6} {'R@5':>6} {'R@10':>6} | {'ΔText':>6} | {'w_v':>5} {'w_t':>5} {'w_a':>5} | Time")
    print("-" * 90)
    for r in sorted_r:
        m = r['metrics']
        w = r['avg_weights']
        delta = m['R@1'] - r['text_metrics']['R@1']
        marker = " ***" if delta > 0.001 else (" ~" if abs(delta) < 0.001 else "")
        print(f"{r['config']['name']:>50} | {m['R@1']:.4f} {m['R@5']:.4f} {m['R@10']:.4f} | {delta:+.4f} | {w[0]:.3f} {w[1]:.3f} {w[2]:.3f} | {r['time_s']:.0f}s{marker}")

if __name__ == "__main__":
    main()
