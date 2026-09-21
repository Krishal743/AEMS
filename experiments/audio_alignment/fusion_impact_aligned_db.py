#!/usr/bin/env python3
"""
fusion_impact_aligned_db.py -- End-to-end fusion impact of any aligned audio DB.
================================================================================
Standalone, parameterized version of fusion_impact_m4.py (which is untouched).
For each given aligned-audio DB, evaluates the live AEMS retrieval stack
(audio-only CLIP-text branch, equal fusion, adaptive gating) on the held-out
TEST set against the RAW CLAP-audio reference, and writes one JSON.

Run:
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  python experiments/audio_alignment/fusion_impact_aligned_db.py \
      --audio-dbs aems_audio_aligned_m4_adapter_ct.pt,aems_audio_aligned_m6_hybrid.pt \
      --labels m4,m6_hybrid
"""

import json
import os
import argparse

import numpy as np
import torch
import clip
from src.models.gating_network import GatingNetwork
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.config import (DEVICE, set_seeds, AEMS_MANIFEST_PATH,
                        AEMS_VID_EMBEDDINGS_PATH, AEMS_AUDIO_EMBEDDINGS_PATH,
                        AEMS_GATING_WEIGHTS_PATH,
                        AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE)
from src.data.metadata import load_metadata, filter_by_split

set_seeds(42)
OUTPUT_DIR = "outputs/alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--gate-weights", default=AEMS_GATING_WEIGHTS_PATH)
parser.add_argument("--audio-dbs", required=True,
                    help="comma-separated aligned audio DB filenames (in embeddings/)")
parser.add_argument("--labels", required=True,
                    help="comma-separated labels, same order as --audio-dbs")
parser.add_argument("--out", default="fusion_impact_aligned.json")
args = parser.parse_args()

aligned_dbs = [f.strip() for f in args.audio_dbs.split(",")]
labels = [f.strip() for f in args.labels.split(",")]
assert len(aligned_dbs) == len(labels)


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


records = filter_by_split(load_metadata(AEMS_MANIFEST_PATH), split="test")
queries, qvids, qcats = [], [], []
for rec in records:
    for q in rec["qa_questions"]:
        queries.append(q); qvids.append(rec["video_id"])
        qcats.append(rec["content_fine_category"])

video_db = torch.load(AEMS_VID_EMBEDDINGS_PATH, weights_only=False)
text_db = torch.load(AEMS_TEXT_EMBEDDINGS_FUSED_PATH_TEMPLATE.format(split="test"),
                     weights_only=False)
test_ids = {r["video_id"] for r in records}
common = sorted(set(video_db) & set(text_db) & test_ids)

qvids_f = [v for v in qvids if v in common]
qcats_f = [c for v, c in zip(qvids, qcats) if v in common]
qwithgt = [q for q, v in zip(queries, qvids) if v in common]

print(f"[DATA] queries={len(qwithgt)} videos={len(common)}")

clip_model, _ = clip.load("ViT-B/32", device=DEVICE); clip_model.eval()
BS = 64
qembs_list = []
for i in range(0, len(qwithgt), BS):
    with torch.no_grad():
        e = clip_model.encode_text(clip.tokenize(qwithgt[i:i+BS], truncate=True).to(DEVICE))
    qembs_list.append(normalize(e).cpu())
qembs = torch.cat(qembs_list)
del clip_model, qembs_list
torch.cuda.empty_cache()

video_matrix = torch.stack([normalize(video_db[v].float()).cpu() for v in common])
text_matrix = torch.stack([normalize(text_db[v].float()).cpu() for v in common])

sim_v = qembs.float() @ video_matrix.T
sim_t = qembs.float() @ text_matrix.T   # CLIP-text query vs fused-text gallery

gate = None
if os.path.exists(args.gate_weights):
    gate = GatingNetwork(input_dim=512, hidden_dim=128).to(DEVICE)
    gate.load_state_dict(torch.load(args.gate_weights, map_location=DEVICE,
                                    weights_only=False), strict=False)
    gate.eval()
    ws = torch.cat([gate(qembs[i:i+BS].float().to(DEVICE)).cpu()
                    for i in range(0, len(qembs), BS)])


def audio_sim_for(db):
    am = torch.stack([normalize(db[v].float()).cpu() for v in common])
    return am, normalize(qembs.float() @ am.T)


raw_audio_db = torch.load(AEMS_AUDIO_EMBEDDINGS_PATH, weights_only=False)

variants = {}
db_objects = {}
for name, fname in zip(labels, aligned_dbs):
    db_objects[name] = torch.load("embeddings/" + fname, weights_only=False)

variants["raw"] = {}
am_raw, sim_aclip_raw = audio_sim_for(raw_audio_db)
variants["raw"]["audio_only_cliptxtq"] = sim_aclip_raw
variants["raw"]["equal_fusion"] = (sim_v + sim_t + sim_aclip_raw) / 3
if gate is not None:
    variants["raw"]["adaptive_gating"] = (ws[:, 0:1]*sim_v + ws[:, 1:2]*sim_t + ws[:, 2:3]*sim_aclip_raw)

for name in labels:
    db = db_objects[name]
    _, sim_aclip = audio_sim_for(db)
    variants[name] = {"audio_only_cliptxtq": sim_aclip,
                      "equal_fusion": (sim_v + sim_t + sim_aclip) / 3}
    if gate is not None:
        variants[name]["adaptive_gating"] = (ws[:, 0:1]*sim_v + ws[:, 1:2]*sim_t + ws[:, 2:3]*sim_aclip)

systems = ["audio_only_cliptxtq", "equal_fusion", "adaptive_gating"]
print("\nSYSTEM COMPARISON  (CLIP-text query; audio branch variants)")
print("=" * 100)
hdr = f"{'system':<24}" + "".join(f"{lb:>13}" for lb in ["raw"] + labels)
print(hdr)
summary = {}
for sysname in systems:
    if sysname not in variants["raw"]:
        continue
    row = {}
    for vname in ["raw"] + labels:
        m = evaluate_retrieval(variants[vname][sysname], qvids_f, common, ks=[1, 5, 10])
        row[vname] = m["R@1"]
    delta = {lb: row[lb] - row["raw"] for lb in labels}
    summary[sysname] = {**{f"{lb}_R@1": row[lb] for lb in ["raw"] + labels},
                        **{f"{lb}_delta": delta[lb] for lb in labels}}
    print(f"{sysname:<24}" + "".join(f"{row[vname]:>13.4f}" for vname in ["raw"] + labels))

summary["note"] = ("'audio_only_cliptxtq' = CLIP-text query vs aligned-audio gallery "
                   "(the raw variant was unusable/0 because spaces were incommensurable)")
with open(os.path.join(OUTPUT_DIR, args.out), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[DONE] outputs/alignment/{args.out}")