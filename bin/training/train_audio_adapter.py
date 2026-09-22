"""Train the WavLM -> CLIP-text audio adapter and export the audio branch.

Positives per train video: its CLIP description embedding plus its CLIP-encoded
QA questions (one sampled per step), with symmetric InfoNCE. The checkpoint is
selected by text->audio MRR on a validation split carved from TRAIN videos, so
the test split is never seen. Finally every video's WavLM features are
projected and saved as the searchable audio branch.
"""

import argparse, json, os, random
import numpy as np
import torch
import torch.nn.functional as F
import clip
from src.config import (AEMS_MANIFEST_PATH, AEMS_WAVLM_FEATURES_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                        AEMS_AUDIO_ADAPTER_PATH, AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
                        DEVICE, set_seeds)
from src.models.audio_adapter import AudioAdapter

parser = argparse.ArgumentParser(description="Train WavLM audio adapter (WavLM -> CLIP text)")
parser.add_argument("--manifest", default=AEMS_MANIFEST_PATH)
parser.add_argument("--features", default=AEMS_WAVLM_FEATURES_PATH)
parser.add_argument("--adapter-out", default=AEMS_AUDIO_ADAPTER_PATH)
parser.add_argument("--embeddings-out", default=AEMS_AUDIO_EMBEDDINGS_PATH)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--val-frac", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seeds(args.seed)

manifest = json.load(open(args.manifest))
features = torch.load(args.features, weights_only=False)
desc_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split="train"), weights_only=False)
questions = {r["video_id"]: [q for q in r["qa_questions"] if isinstance(q, str) and q.strip()]
             for r in manifest if r["split"] == "train"}

train_all = sorted(v for v in questions if v in features and v in desc_db and questions[v])
shuffled = train_all[:]
random.Random(0).shuffle(shuffled)
n_val = int(args.val_frac * len(shuffled))
val_vids, tr_vids = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])
print(f"[DATA] train={len(tr_vids)} val={len(val_vids)} (test split untouched)")

print("[ENC] Encoding QA questions with CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
flat = [(v, q) for v in tr_vids + val_vids for q in questions[v]]
q_emb = []
with torch.no_grad():
    for i in range(0, len(flat), 512):
        tokens = clip.tokenize([q for _, q in flat[i:i + 512]], truncate=True).to(DEVICE)
        q_emb.append(F.normalize(clip_model.encode_text(tokens).float(), dim=1).cpu())
q_emb = torch.cat(q_emb)
del clip_model
torch.cuda.empty_cache()

q_rows = {}
for i, (v, _) in enumerate(flat):
    q_rows.setdefault(v, []).append(i)

def feature_matrix(vids):
    return F.normalize(torch.stack([torch.as_tensor(features[v]).float() for v in vids]), dim=1)

X_tr, X_val = feature_matrix(tr_vids).to(DEVICE), feature_matrix(val_vids).to(DEVICE)
targets = [torch.cat([F.normalize(torch.as_tensor(desc_db[v]).float().view(1, -1), dim=1),
                      q_emb[q_rows[v]]]) for v in tr_vids]
val_rows = torch.tensor([i for v in val_vids for i in q_rows[v]])
val_gt = torch.tensor([j for j, v in enumerate(val_vids) for _ in q_rows[v]])

def val_mrr(model):
    model.eval()
    with torch.no_grad():
        sim = q_emb[val_rows] @ model(X_val).cpu().T
    rank = (sim > sim.gather(1, val_gt[:, None])).sum(1).float()
    return (1 / (rank + 1)).mean().item(), (rank < 1).float().mean().item()

model = AudioAdapter(input_dim=X_tr.shape[1]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
rng = np.random.default_rng(args.seed)
best_mrr, best_state, best_ep = -1.0, None, -1

for ep in range(args.epochs):
    model.train()
    perm = rng.permutation(len(tr_vids))
    total, n = 0.0, 0
    for s in range(0, len(perm), args.batch_size):
        idx = perm[s:s + args.batch_size]
        y = torch.stack([targets[i][rng.integers(len(targets[i]))] for i in idx]).to(DEVICE)
        logits = model.logit_scale.exp().clamp(max=100) * model(X_tr[idx]) @ y.T
        labels = torch.arange(len(idx), device=DEVICE)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item(); n += 1
    sched.step()
    mrr, r1 = val_mrr(model)
    print(f"  epoch {ep + 1:02d}  loss={total / n:.4f}  val t2a MRR={mrr:.4f} R@1={r1:.4f}")
    if mrr > best_mrr:
        best_mrr, best_ep = mrr, ep + 1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print(f"[BEST] epoch {best_ep}  val t2a MRR={best_mrr:.4f}")
os.makedirs(os.path.dirname(args.adapter_out), exist_ok=True)
torch.save(best_state, args.adapter_out)
print(f"[SAVE] adapter -> {args.adapter_out}")

model.load_state_dict(best_state)
model.eval()
all_vids = sorted(features)
with torch.no_grad():
    projected = model(feature_matrix(all_vids).to(DEVICE)).cpu()
torch.save({v: projected[i] for i, v in enumerate(all_vids)}, args.embeddings_out)
print(f"[SAVE] {len(all_vids)} CLIP-space audio embeddings -> {args.embeddings_out}")
