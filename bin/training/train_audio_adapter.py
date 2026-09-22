"""Train the WavLM -> CLIP-text audio adapter and export the audio branch.

Positives per train video: its CLIP description embedding plus its CLIP-encoded
QA questions (one sampled per step), with symmetric InfoNCE. The checkpoint is
selected by text->audio MRR on a validation split carved from TRAIN videos, so
the test split is never seen. Finally every video's WavLM features are
projected and saved as the searchable audio branch.
"""

import argparse, os
import torch
import torch.nn.functional as F
from src.config import (AEMS_MANIFEST_PATH, AEMS_WAVLM_FEATURES_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                        AEMS_AUDIO_ADAPTER_PATH, AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE,
                        DEVICE, set_seeds)
from src.training.audio_adapter_fit import fit_adapter, project
from src.training.query_data import (load_records, questions, validation_split, encode_clip_text,
                                     flatten_questions, stack_embeddings, query_rows, recall_metrics)

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

records = load_records(args.manifest, "train")
features = torch.load(args.features, weights_only=False)
desc_db = torch.load(AEMS_TEXT_EMBEDDINGS_DESC_PATH_TEMPLATE.format(split="train"), weights_only=False)

usable = [v for v in records if v in features and v in desc_db and questions(records[v])]
fit_vids, val_vids = validation_split(usable, args.val_frac)
print(f"[DATA] fit={len(fit_vids)} val={len(val_vids)} (test split untouched)")

print("[ENC] Encoding QA questions with CLIP...")
texts, rows = flatten_questions(records, fit_vids + val_vids)
q_emb = encode_clip_text(texts, DEVICE)

X_fit = stack_embeddings(features, fit_vids)
X_val = stack_embeddings(features, val_vids, DEVICE)
targets = [torch.cat([F.normalize(torch.as_tensor(desc_db[v]).float().view(1, -1), dim=1),
                      q_emb[rows[v]]]) for v in fit_vids]
val_idx, val_gt = query_rows(rows, val_vids)
val_queries = q_emb[val_idx]


def val_mrr(model):
    with torch.no_grad():
        sim = val_queries @ model(X_val).cpu().T
    return recall_metrics(sim, val_gt)["MRR"]


model, best_mrr, best_epoch = fit_adapter(
    X_fit, targets, DEVICE, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    seed=args.seed, eval_fn=val_mrr, log=lambda m: print(m.replace("  0.", "  val t2a MRR=0.")))
print(f"[BEST] epoch {best_epoch}  val t2a MRR={best_mrr:.4f}")

os.makedirs(os.path.dirname(args.adapter_out), exist_ok=True)
torch.save(model.state_dict(), args.adapter_out)
print(f"[SAVE] adapter -> {args.adapter_out}")

all_vids = sorted(features)
projected = project(model, stack_embeddings(features, all_vids), DEVICE)
torch.save({v: projected[i] for i, v in enumerate(all_vids)}, args.embeddings_out)
print(f"[SAVE] {len(all_vids)} CLIP-space audio embeddings -> {args.embeddings_out}")
