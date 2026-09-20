# Audio↔CLIP-Text Alignment via Linear Projection — Post Phase 3 Review 1

**Setup Date:** 03 September 2026
**Data Stamp:** AEMS Phase A (v1 embeddings), Post Phase 3 Review 1
**Scope:** Applying linear cross-modal alignment (Wang 2019; Zhang & Saligrama
2016 Procrustes; Grave 2019; Rasiwasia 2010 CCA) to unify CLAP-audio
embeddings into CLIP-text space, enabling cross-modal CLIP-text→audio
retrieval and audio fusion with text/visual.

---

## 1. Objective

Unify all three modalities into **CLIP-text space** so that a single
CLIP-text query can retrieve visual, text, **and** audio, and so audio can be
fused meaningfully in the gating network (which previously relied on a
separate, non-commensurable CLAP query space).

**Target text embedding:** the **description** embedding
(`aems_text_embeddings_description_{split}.pt`), which is a short,
semantically dense caption per video — matching the distribution of a
CLIP-text query.

---

## 2. Method

Learn a linear map `W, b` such that `W·CLAP_audio + b ≈ CLIP_text(description)`
for matched (audio, text) pairs.

- **Anchor set:** each video = one CLAP-audio embedding + one CLIP-text
  (description) embedding.
- **Train / test discipline (no leakage):**
  - **Train anchors:** 5,748 (train split) — used to fit `W, b`.
  - **Test anchors:** 1,022 (test split) — **held out**, disjoint (verified).
- **Candidate methods** (per referenced papers), evaluated on the **train** set:
  - **Ridge regression** (Wang 2019): `min_W ||WX−Y||² + λ||W||²`. Result:
    projected positive-cosine **0.056** (baseline 0.021).
  - **Orthogonal Procrustes** (Zhang & Saligrama 2016; Grave 2019):
    mean-center both spaces, `W = UVᵀ` from SVD of `XᵀY`. Result: projected
    positive-cosine **0.608** — far better. **Chosen.**
  - **CCA** (Rasiwasia 2010): reported as reference only; top-3 canonical
    correlations ≈ 9.6/9.1/7.5.

---

## 3. Results (held-out TEST, no leakage)

| Metric | Before (raw CLAP) | After (Procrustes) |
|--------|:------------------:|:-------------------:|
| **Alignment** positive-pair cosine (Wang & Isola) | 0.0216 | **0.6088** |
| **Modality gap** centroid cos-dist to text (Liang) | 0.9637 | **0.0007** |
| **Hubness** skew K=5 (Radovanović) | 14.86 | 10.94 |
| **Retrieval** CLIP-text→audio R@1 | 0.0000 | **0.0000** |
| CLIP-text→audio R@5 | 0.0059 | 0.0010 |
| CLIP-text→audio R@10 | 0.0157 | 0.0049 |

**Diagnosis of the discrepancy — alignment up, retrieval unchanged:**

After projection (quantified on test):
- Positive cosine mean = **0.609**, negative cosine mean = **0.608**
  → the whole similarity distribution is lifted but **positives remain
  indistinguishable from negatives**.
- Best-negative mean (**0.716**) **exceeds** the positive mean (0.609): for
  the average query, some other video's projected audio is closer to the text
  than the correct one.
- Only **0.29%** of positives beat their hardest negative.
- Projected audio gallery anisotropy (mean pairwise-cos) = **0.566**, *higher*
  than raw (0.485).

**Conclusion (matches the referenced papers' caveats):** An orthogonal
Procrustes map is **distance-preserving** — it aligns centroids and raises
absolute similarity, but it **cannot add discrimination**. The source CLAP
audio space is too collinear/anisotropic in the discriminating directions, so
a pure rotation keeps the cone intact and every query still collapses onto
the same α-hub. Linear projection is "often sufficient" **only when the
underlying semantics are already well separated** — which is precisely *not*
the case here (see prior anisotropy/hubness diagnostics).

---

## 4. Artifacts Produced

| Artifact | Path |
|----------|------|
| Fit script | `scripts/alignment/fit_projection.py` |
| Export script | `scripts/alignment/export_projected_audio.py` |
| Validation script | `scripts/alignment/validate_projection.py` |
| Learned projection | `models/audio_to_cliptext_projection.pt` |
| Projected audio DB (6770 videos) | `embeddings/aems_audio_projected_cliptext.pt` |
| Fit report | `outputs/alignment/projection_fit.json` |
| Export report | `outputs/alignment/export_report.json` |
| Validation report | `outputs/alignment/projection_validation.json` |

**Reproduce:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/alignment/fit_projection.py
python scripts/alignment/export_projected_audio.py
python scripts/alignment/validate_projection.py
```

---

## 5. Decision & Escalation Path

**Verdict:** Pure orthogonal linear projection **is insufficient** — it fixes
alignment and the modality gap but **not retrieval**, because the underlying
audio embeddings lack the per-query discrimination that a rotation could
expose.

**Recommended next steps (in the project plan, not yet implemented):**

1. **Anisotropy correction before/with the linear map** — combine whitening
   (ZCA/mean-centering, shown in the anisotropy diagnostic to restore
   effective dim 192→429 and cosine 0.83→~0) with a learned map. This is still
   *linear* but non-orthogonal and directly targets the cone that blocking
   discrimination.
2. **Learned adapter / contrastive fine-tuning** (AudioCLIP / Wav2CLIP
   approach): a small MLP head (or adapter on CLAP) trained with a contrastive
   loss pulling projected audio to the frozen CLIP-text of the same video.
   This can compress the hub cone and create real margins.
3. **Hubness reduction** (local scaling / normalization) on the gallery, which
   the hubness diagnostic showed is severe (single audio returns for ~92% of
   queries).
4. Re-run the full alignment/uniformity/hubness/modality-gap diagnostic suite
   after any of the above to confirm retrieval (R@1/MRR) improves, not just
   absolute similarity.

---

*Stamp: Setup date 03 September 2026 · Post Phase 3 Review 1 · AEMS Phase A (v1 embeddings)*
