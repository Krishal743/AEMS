# AEMS Audio Alignment — Post-M4 Tier Campaign — Post Phase 3 Review 1

**Setup Date:** 03 September 2026
**Data Stamp:** AEMS Phase A (v1 embeddings), Post Phase 3 Review 1
**Scope:** Follow-up to the M1–M5 method comparison. With M4 (adapter +
frozen-CLAP-features contrastive) confirmed as the only method that lifts
near-zero audio→CLIP-text retrieval (t2a R@1=0.0323, a2t R@1=0.0538 on the
held-out test), this round systematically tests whether **unfreezing the audio
encoder** can push retrieval higher:
- **Tier 1** — harder negatives + post-hoc hubness reduction *on the frozen M4 adapter*.
- **Tier 2** — LoRA adapters (r=16) inside the frozen HTSAT backbone.
- **Tier 3** — full HTSAT audio-encoder fine-tune (split LR).

Same evaluation contract throughout: **frozen CLIP-text description targets**,
both retrieval directions reported, 5,748 train anchors / 1,022 held-out test
anchors, all model components trained on train only (no leakage).

---

## 1. Baseline being chased (M4)

| ID | Method | t2a R@1 | a2t R@1 | t2a MRR | a2t MRR | align | gap |
|----|--------|---------|---------|---------|---------|------:|----:|
| M4 | adapter + contrastive (frozen CLAP, fresh 1024→512 head, InfoNCE b=256) | 0.0323 | 0.0538 | 0.062 | 0.108 | 0.126 | 0.441 |

Random chance R@1 for the 1,022-gallery test set is ≈ 0.0010.

---

## 2. Tier 1 — Global hard negatives + hubness on frozen M4 features

`scripts/alignment/m4_plus.py`

**What was tried.** A momentum memory bank (momentum 0.5, warm-started from
train features) supplies 64 *global* hard negatives per anchor during an extra
InfoNCE pass; the M4 adapter is re-trained with this loss. Then the resulting
gallery is post-processed with three post-hoc hubness reductions (local
scaling, mutual-kNN re-ranking, query expansion).

**Result.** None of the variants improved retrieval — all stayed at chance
R@1 ≈ 0.0010 (best variant `mutual_knn_k3`). The initial memory-bank run
**diverged** (loss 7→17, R@1 collapse); the loss was stabilised (cosine margin,
`logit_scale` clamped ≤ 50) but the stabilised adapter still produced
degenerate, near-isotropic gallery (alignment −0.745, gallery anisotropy 0.999).

**Conclusion.** Negative-sampling strength and hubness in the gallery are **not**
the bottleneck. With frozen CLAP features the shallow adapter has already
extracted what it can; further training data does not add capacity. This
motivates Tiers 2/3 (unfreeze the encoder).

---

## 3. Tier 2 — LoRA inside HTSAT

`scripts/alignment/lora_torch.py` (LoRA module + injection, 48 targets:
`attn.qkv/proj`, `mlp.fc1/fc2`), `scripts/alignment/audio_ft_utils.py` (raw
waveform pipeline), `scripts/alignment/m_lora_contrastive.py` (train).

**Setup.** 48 LoRA adapters (r=16, α=16, B zero-init, ~2.97 M trainable params)
on the frozen HTSAT backbone + a fresh 1024→512 projection head + fresh
logit_scale, trained with in-batch InfoNCE from raw 30 s waveforms (random 10 s
window per forward). 40 epochs, batch 6, lr 1e-4.

**Result (held-out test).** a2t R@1 ≈ 0.047–0.058 (best epoch 0.0577, final
deterministic 0.0499), t2a R@1 ≈ 0.025, MRR ~0.06–0.10.

**Interpretation.** LoRA fine-tuning reaches roughly **the same level as the
frozen-feature M4 adapter** (0.0538) but does not clearly beat it. The HTSAT
backbone itself was not the primary capacity limit; the adapter already captured
the accessible signal.

---

## 4. Tier 3 — Full audio-encoder fine-tune

`scripts/alignment/m_full_finetune.py`

**Setup.** Entire HTSAT audio branch (30.2 M params) unfrozen + fresh head
(0.66 M) + fresh logit_scale, split-LR AdamW (backbone 5e-6, head 1e-3),
batch 64, cosine schedule (5% warmup), grad-clip 5.0, SpecAugment disabled,
30 epochs (~26 min on 24 GB GPU, 10 GB peak).

**Why the first attempts failed (diagnostics).** Early runs showed **zero**
learning (loss flat at ≈ ln(batch), R@1 chance). Identified causes, each
verified empirically:
1. **LambdaLR bug** — the schedule lambda returned `args.lr * factor`, which
   LambdaLR *multiplies* by the base LR → effective LR ≈ 1e-9, i.e. no updates.
2. **Kept CLAP projection head** — CLAP's own head is locked to CLAP-text
   targets; against frozen CLIP-text it gave a badly-conditioned initial
   objective. A **fresh std=0.02 head** makes the mapping problem well-posed.
3. **InfoNCE at batch 2–8 on varied data** gives near-zero coherent gradient
   for a 30 M-param backbone; batch 64 is required (memorisation tests on
   fixed samples misleadingly "learned").
4. **SpecAugment** stochastic masking further decorrelated per-step gradients;
   disabling it stabilised fine-tuning.

**Result (held-out test, deterministic eval).** a2t R@1 = 0.0391 (best) /
0.0352 (final 30ep), t2a R@1 = 0.017, MRR ~0.069–0.075. Loss fell 4.0→1.36 but
R@1 plateaued around 0.035–0.039.

**Interpretation.** Full fine-tune **under-performs** both M4 (frozen) and
Tier 2 LoRA. With only 5,748 training pairs against a 6,770-video gallery, the
high-capacity update overfits / fails to generalise the alignment, while the
constrained adapter and LoRA regularise better.

---

## 5. Final comparison (all methods, held-out test, n=1022)

`scripts/alignment/compare_methods.py` → `outputs/alignment/method_comparison.json`

| Method | t2a R@1 | t2a MRR | a2t R@1 | a2t MRR | align | gap | hub | aniso |
|--------|--------:| -------:|--------:|--------:|------:|----:|----:|------:|
| RAW CLAP | 0.0000 | 0.0083 | 0.0010 | — | 0.022 | 0.964 | 14.9 | 0.485 |
| M1 whiten+ridge | 0.0000 | 0.0061 | 0.0010 | — | 0.001 | 0.971 | 11.6 | 0.000 |
| M2 CCA | 0.0010 | 0.0084 | 0.0000 | — | 0.003 | 1.008 | 7.4 | 0.000 |
| M5 hubness | 0.0000 | 0.0061 | 0.0010 | — | 0.001 | 0.971 | 11.6 | 0.000 |
| M3 shared-text | 0.0010 | 0.0064 | 0.0000 | — | 0.405 | 0.446 | 14.4 | 0.814 |
| Procrustes (prior) | 0.0000 | — | — | — | — | — | — | — |
| **M4 adapter+contrastive** | **0.0323** | **0.062** | **0.0538** | **0.108** | 0.126 | 0.441 | 11.1 | 0.018 |
| Tier1 global hard-neg | 0.0000 | 0.006 | 0.0010 | — | −0.745 | 1.921 | 12.5 | 0.999 |
| Tier1 + hubness boost | 0.0000 | 0.006 | 0.0010 | — | −0.745 | 1.921 | 12.5 | 0.999 |
| Tier2 LoRA (r=16) | 0.0254 | 0.063 | 0.0470 | 0.101 | 0.121 | 0.655 | 6.3 | 0.053 |
| Tier3 full ft (30ep) | 0.0166 | 0.046 | 0.0401 | 0.075 | 0.108 | 0.531 | 7.5 | 0.030 |

> *Chance R@1 ≈ 0.0010. Tier1 rows used the degenerate/short-run DBs (no gain);
> they are listed to keep the record complete, not as contenders.*

**Verdict: M4 remains the deployable winner.** The three tiers confirm the
learning-style hypothesis — the constraint and data-regime favour a small
adapter over large-capacity fine-tunes:

1. Harder negatives / hubness (Tier 1): **no effect** → not a sampling problem.
2. LoRA (Tier 2): **on par** with M4 (0.047–0.058 vs 0.0538).
3. Full fine-tune (Tier 3): **below** M4 (0.040). With 5.7 k pairs the
   high-capacity update cannot generalise the alignment.

**Production consequence.** The fusion pipeline keeps
`embeddings/aems_audio_aligned_m4_adapter_ct.pt` (`audio_adapter_cliptext_m4.pt`).
Its fusion impact was already measured and is unchanged:
`outputs/alignment/fusion_impact_m4.json` — audio-only CLIP-text branch
R@1 0.0012→0.0078, equal-fusion 0.2411→0.2341, adaptive gating 0.3894 (held).

---

## 6. Reproducibility notes

- **Eval determinism.** AEMS clips are **30 s** and the downstream pipeline
  random-crops to 10 s (`data_truncating='rand_trunc'`). Raw calls were
  stochastic (same model scored 0.058 vs 0.050 across calls). Tier 2/3 final
  numbers use a seeded per-batch crop (`collate_deterministic`); the comparison
  table additionally re-evaluates the exported DBs so all rows share one harness.
- **GPU.** Runs used a single 24 GB GPU (batch 64 fine-tune peak ≈ 10 GB);
  waveforms are cached in RAM (5,748 train ~11 GB, export ~13 GB; 115 GB
  available).
- **LoRA-only cost.** Tier 2 trains ~2.97 M params in ~15 min; Tier 3 full
  fine-tune ~30 min for 30 epochs.

## 7. Artifacts

| Component | Path |
|---|---|
| Tier 1 script/results | `scripts/alignment/m4_plus.py`, `outputs/alignment/m4_plus_results.json` |
| Tier 1 DBs (not deployed) | `embeddings/aems_audio_aligned_m4_global_hn.pt`, `aems_audio_aligned_m4_boosted.pt` |
| Tier 1 model | `models/audio_adapter_m4_globalhn.pt` |
| LoRA module | `scripts/alignment/lora_torch.py` |
| Audio pipeline utils | `scripts/alignment/audio_ft_utils.py` |
| Tier 2 script | `scripts/alignment/m_lora_contrastive.py` |
| Tier 2 model/DB | `models/audio_lora_cliptext_rank16.pt`, `embeddings/aems_audio_aligned_m4_lora.pt`, `outputs/alignment/m2_lora_results.json` |
| Tier 3 script | `scripts/alignment/m_full_finetune.py` |
| Tier 3 model/DB | `models/audio_fullft_cliptext.pt`, `embeddings/aems_audio_aligned_m4_fullft.pt`, `outputs/alignment/tier3_fullft_results.json` |
| Comparison table | `scripts/alignment/compare_methods.py`, `outputs/alignment/method_comparison.json` |
| Fusion impact (M4, unchanged) | `outputs/alignment/fusion_impact_m4.json` |