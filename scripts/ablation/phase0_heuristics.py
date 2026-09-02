"""
Phase 0: Heuristic Baselines for Gating Network Ablation Study.
No training needed — evaluates fixed/adaptive weighting schemes directly.
"""
import gc, json, torch, torch.nn.functional as F
import clip
import numpy as np
from src.encoders.clap_encode import CLAPEncoder
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.config import (AEMS_MANIFEST_PATH, AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                         AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE, DEVICE, set_seeds)
from src.data.metadata import load_metadata, filter_by_split

def angular_similarity(q, v, temperature=1.0):
    device = q.device if q.device.type == 'cuda' else v.device
    qd = q.to(device)
    vd = v.to(device)
    cs = qd @ vd.T
    a = 1 - torch.acos(torch.clamp(cs, -1, 1)) / np.pi
    return a / temperature if temperature != 1.0 else a

set_seeds(42)

QUERY_BATCH = 100

print("[INIT] Loading data...", flush=True)
metadata = load_metadata(AEMS_MANIFEST_PATH)
train_items = filter_by_split(metadata, "train")
test_items = filter_by_split(metadata, "test")

video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)
text_db_train = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="train"), weights_only=False)
text_db_test = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"), weights_only=False)

all_vids = set()
for item in train_items + test_items:
    all_vids.add(item["video_id"])

common_all = sorted(set(video_db.keys()) & set(audio_db.keys()) & set(text_db_train.keys()) & all_vids)
common_test = sorted(set(video_db.keys()) & set(audio_db.keys()) & set(text_db_test.keys()) &
                     set(item["video_id"] for item in test_items))

train_queries, train_vids = [], []
for item in train_items:
    vid = item["video_id"]
    if vid not in common_all:
        continue
    for q in item["qa_questions"]:
        train_queries.append(q)
        train_vids.append(vid)

test_queries, test_vids = [], []
for item in test_items:
    vid = item["video_id"]
    if vid not in common_test:
        continue
    for q in item["qa_questions"]:
        test_queries.append(q)
        test_vids.append(vid)

print(f"  Train queries: {len(train_queries)}, Test queries: {len(test_queries)}", flush=True)
print(f"  All candidates: {len(common_all)}, Test candidates: {len(common_test)}", flush=True)

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
        if isinstance(e, np.ndarray):
            e = torch.from_numpy(e).float()
        e = F.normalize(e, dim=-1)
        out.append(e.cpu())
        del e; gc.collect(); torch.cuda.empty_cache()
    return torch.cat(out)

q_clip_train = enc_clip(train_queries)
q_clap_train = enc_clap(train_queries)
q_clip_test = enc_clip(test_queries)
q_clap_test = enc_clap(test_queries)

del clip_model, clap_encoder
gc.collect(); torch.cuda.empty_cache()

print("[SIM] Computing similarity matrices...", flush=True)
# Test set similarities
sim_v_test = angular_similarity(q_clip_test.float(), vid_mat_test).to(DEVICE)
sim_t_test = angular_similarity(q_clip_test.float(), txt_mat_test).to(DEVICE)
sim_a_test = angular_similarity(q_clap_test.float(), aud_mat_test, temperature=0.5).to(DEVICE)

# Train set similarities (for oracle/heuristic computation)
N_TRAIN_CANDS = min(500, len(common_all))
vid_mat_sub = vid_mat[:N_TRAIN_CANDS].float().to(DEVICE)
aud_mat_sub = aud_mat[:N_TRAIN_CANDS].float().to(DEVICE)
txt_mat_sub = txt_mat[:N_TRAIN_CANDS].float().to(DEVICE)

sim_v_train = angular_similarity(q_clip_train.float(), vid_mat_sub).to(DEVICE)
sim_t_train = angular_similarity(q_clip_train.float(), txt_mat_sub).to(DEVICE)
sim_a_train = angular_similarity(q_clap_train.float(), aud_mat_sub, temperature=0.5).to(DEVICE)

vid_to_idx_all = {v: i for i, v in enumerate(common_all)}
train_gt = torch.tensor([vid_to_idx_all[v] for v in train_vids], device=DEVICE)

# Also compute full train set similarities for oracle (slow but needed)
print("[SIM] Computing full train similarities for oracle (this takes a while)...", flush=True)
sim_v_train_full = torch.zeros(len(train_queries), len(common_all))
sim_t_train_full = torch.zeros(len(train_queries), len(common_all))
sim_a_train_full = torch.zeros(len(train_queries), len(common_all))

for vs in range(0, len(common_all), 500):
    ve = min(vs + 500, len(common_all))
    vb_v = vid_mat[vs:ve].float().to(DEVICE)
    vb_t = txt_mat[vs:ve].float().to(DEVICE)
    vb_a = aud_mat[vs:ve].float().to(DEVICE)
    sim_v_train_full[:, vs:ve] = angular_similarity(q_clip_train.float(), vb_v).cpu()
    sim_t_train_full[:, vs:ve] = angular_similarity(q_clip_train.float(), vb_t).cpu()
    sim_a_train_full[:, vs:ve] = angular_similarity(q_clap_train.float(), vb_a, temperature=0.5).cpu()
    del vb_v, vb_t, vb_a; gc.collect(); torch.cuda.empty_cache()

print("  Done.", flush=True)

def eval_system(name, sim, vid_ids=common_test):
    m = evaluate_retrieval(sim, test_vids, vid_ids, ks=[1, 5, 10])
    print(f"  {name:>30}: R@1={m['R@1']:.4f}  R@5={m['R@5']:.4f}  R@10={m['R@10']:.4f}")
    return m

results = {}

print("\n" + "=" * 70)
print("PHASE 0: HEURISTIC BASELINES")
print("=" * 70)

# Single modality baselines
results["visual_only"] = eval_system("Visual only", sim_v_test)
results["text_only"] = eval_system("Text only", sim_t_test)
results["audio_only"] = eval_system("Audio only", sim_a_test)

# Fixed weighting
for vt, tt, at, label in [
    (0.2, 0.7, 0.1, "Fixed w=[0.2, 0.7, 0.1]"),
    (0.15, 0.8, 0.05, "Fixed w=[0.15, 0.8, 0.05]"),
    (0.3, 0.6, 0.1, "Fixed w=[0.3, 0.6, 0.1]"),
    (0.0, 1.0, 0.0, "Fixed w=[0, 1, 0] = Text only"),
    (1/3, 1/3, 1/3, "Fixed w=[1/3, 1/3, 1/3] = Equal"),
]:
    fused = vt * sim_v_test * 1.0 + tt * sim_t_test * 1.0 + at * sim_a_test * 0.8
    results[f"fixed_{label}"] = eval_system(label, fused)

# Max-confidence per query: weight by max similarity of each modality
with torch.no_grad():
    mv = sim_v_test.max(dim=1).values
    mt = sim_t_test.max(dim=1).values
    ma = sim_a_test.max(dim=1).values
    for tau in [0.1, 0.5, 1.0, 2.0]:
        w = F.softmax(torch.stack([mv/tau, mt/tau, ma/tau], dim=1), dim=1)
        fused = w[:, 0:1] * sim_v_test + w[:, 1:2] * sim_t_test + w[:, 2:3] * sim_a_test * 0.8
        results[f"maxconf_tau{tau}"] = eval_system(f"MaxConf tau={tau}", fused)

# Mean-confidence per query: weight by mean similarity of each modality
with torch.no_grad():
    mv = sim_v_test.mean(dim=1)
    mt = sim_t_test.mean(dim=1)
    ma = sim_a_test.mean(dim=1)
    for tau in [0.1, 0.5, 1.0, 2.0]:
        w = F.softmax(torch.stack([mv/tau, mt/tau, ma/tau], dim=1), dim=1)
        fused = w[:, 0:1] * sim_v_test + w[:, 1:2] * sim_t_test + w[:, 2:3] * sim_a_test * 0.8
        results[f"meanconf_tau{tau}"] = eval_system(f"MeanConf tau={tau}", fused)

# Rank-based: weight by how well each modality ranks the correct video (on train set)
print("\n[RANK] Computing rank-based weights from train set...", flush=True)
with torch.no_grad():
    nq = len(train_queries)
    rank_v_scores = torch.zeros(nq)
    rank_t_scores = torch.zeros(nq)
    rank_a_scores = torch.zeros(nq)
    for i in range(nq):
        rv = torch.argsort(sim_v_train_full[i], descending=True)
        rt = torch.argsort(sim_t_train_full[i], descending=True)
        ra = torch.argsort(sim_a_train_full[i], descending=True)
        gt = train_gt[i].item()
        rank_v_scores[i] = 1.0 / (rv.tolist().index(gt) + 1)
        rank_t_scores[i] = 1.0 / (rt.tolist().index(gt) + 1)
        rank_a_scores[i] = 1.0 / (ra.tolist().index(gt) + 1)

    avg_inv_rank_v = rank_v_scores.mean().item()
    avg_inv_rank_t = rank_t_scores.mean().item()
    avg_inv_rank_a = rank_a_scores.mean().item()
    print(f"  Avg inverse rank: v={avg_inv_rank_v:.4f}, t={avg_inv_rank_t:.4f}, a={avg_inv_rank_a:.4f}")

    # Use inverse rank as fixed weights
    total = avg_inv_rank_v + avg_inv_rank_t + avg_inv_rank_a
    wv, wt, wa = avg_inv_rank_v/total, avg_inv_rank_t/total, avg_inv_rank_a/total
    fused = wv * sim_v_test + wt * sim_t_test + wa * sim_a_test * 0.8
    results["invrank_fixed"] = eval_system(f"InverseRank fixed [{wv:.3f},{wt:.3f},{wa:.3f}]", fused)

# Oracle: per-query pick the modality that ranks correct video highest (on train)
print("\n[ORACLE] Computing per-query oracle weights from train set...", flush=True)
with torch.no_grad():
    oracle_weights = torch.zeros(nq, 3)
    for i in range(nq):
        gt = train_gt[i].item()
        scores = torch.tensor([
            sim_v_train_full[i, gt].item(),
            sim_t_train_full[i, gt].item(),
            sim_a_train_full[i, gt].item()
        ])
        best = scores.argmax()
        oracle_weights[i, best] = 1.0

    avg_oracle_w = oracle_weights.mean(dim=0)
    print(f"  Oracle avg weights: v={avg_oracle_w[0]:.3f}, t={avg_oracle_w[1]:.3f}, a={avg_oracle_w[2]:.3f}")

    # Evaluate oracle on train (as an approximation of ceiling)
    oracle_fused_train = (oracle_weights[:, 0:1].to(DEVICE) * sim_v_train_full.to(DEVICE) +
                          oracle_weights[:, 1:2].to(DEVICE) * sim_t_train_full.to(DEVICE) +
                          oracle_weights[:, 2:3].to(DEVICE) * sim_a_train_full.to(DEVICE))
    oracle_train_metrics = evaluate_retrieval(oracle_fused_train, train_vids, common_all, ks=[1, 5, 10])
    print(f"  Oracle (train set): R@1={oracle_train_metrics['R@1']:.4f}  R@5={oracle_train_metrics['R@5']:.4f}  R@10={oracle_train_metrics['R@10']:.4f}")

# Soft oracle: weight proportional to each modality's similarity to correct video
print("\n[SOFT-ORACLE] Computing soft oracle weights...", flush=True)
with torch.no_grad():
    soft_oracle_w = torch.zeros(nq, 3)
    for i in range(nq):
        gt = train_gt[i].item()
        scores = torch.tensor([
            sim_v_train_full[i, gt].item(),
            sim_t_train_full[i, gt].item(),
            sim_a_train_full[i, gt].item()
        ])
        soft_oracle_w[i] = F.softmax(scores / 0.5, dim=0)

    avg_soft_w = soft_oracle_w.mean(dim=0)
    print(f"  Soft oracle avg weights: v={avg_soft_w[0]:.3f}, t={avg_soft_w[1]:.3f}, a={avg_soft_w[2]:.3f}")

    soft_fused_train = (soft_oracle_w[:, 0:1].to(DEVICE) * sim_v_train_full.to(DEVICE) +
                        soft_oracle_w[:, 1:2].to(DEVICE) * sim_t_train_full.to(DEVICE) +
                        soft_oracle_w[:, 2:3].to(DEVICE) * sim_a_train_full.to(DEVICE))
    soft_metrics = evaluate_retrieval(soft_fused_train, train_vids, common_all, ks=[1, 5, 10])
    print(f"  Soft oracle (train set): R@1={soft_metrics['R@1']:.4f}  R@5={soft_metrics['R@5']:.4f}  R@10={soft_metrics['R@10']:.4f}")

# Linear model: w = softmax(W @ [mean_sim_v, mean_sim_t, mean_sim_a] + b)
print("\n[LINEAR] Training linear gating model...", flush=True)
with torch.no_grad():
    train_features = torch.stack([
        sim_v_train_full.mean(dim=1),
        sim_t_train_full.mean(dim=1),
        sim_a_train_full.mean(dim=1)
    ], dim=1)  # (nq, 3)

    W = torch.randn(3, 3, device=DEVICE) * 0.01
    b = torch.zeros(3, device=DEVICE)
    lr = 0.01
    for ep in range(100):
        w = F.softmax(train_features.float().to(DEVICE) @ W + b, dim=1)
        fused = (w[:, 0:1] * sim_v_train_full.to(DEVICE) +
                 w[:, 1:2] * sim_t_train_full.to(DEVICE) +
                 w[:, 2:3] * sim_a_train_full.to(DEVICE))
        log_p = F.log_softmax(fused, dim=1)
        loss = F.nll_loss(log_p, train_gt)
        loss.backward()
        W.data -= lr * W.grad
        b.data -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()

    w_test = F.softmax(train_features.float().to(DEVICE) @ W + b, dim=1).cpu()
    fused = (w_test[:, 0:1].to(DEVICE) * sim_v_test +
             w_test[:, 1:2].to(DEVICE) * sim_t_test +
             w_test[:, 2:3].to(DEVICE) * sim_a_test * 0.8)
    avg_w = w_test.mean(dim=0)
    print(f"  Linear model avg weights: v={avg_w[0]:.3f}, t={avg_w[1]:.3f}, a={avg_w[2]:.3f}")
    results["linear_model"] = eval_system("Linear model", fused)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, m in sorted(results.items(), key=lambda x: -x[1]['R@1']):
    print(f"  {name:>40}: R@1={m['R@1']:.4f}  R@5={m['R@5']:.4f}  R@10={m['R@10']:.4f}")

with open("outputs/ablations/phase0_heuristics.json", "w") as f:
    json.dump({k: v for k, v in results.items()}, f, indent=2)
print(f"\n[SAVE] Results saved to outputs/ablations/phase0_heuristics.json")
