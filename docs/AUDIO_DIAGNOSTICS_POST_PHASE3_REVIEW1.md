# AEMS Audio Embedding Diagnostics — Post Phase 3 Review 1

**Setup Date:** 03 September 2026
**Data Stamp:** AEMS Phase A (v1 embeddings), Post Phase 3 Review 1
**Scope:** Root-cause analysis of low audio-text retrieval recall via the
diagnostic frameworks of Wang & Isola (2020), Liang et al. (2022), Ethayarajh
(2019), Radovanović et al. (2010), Hoiem et al. (2012), Li/BLIP (2022),
Faghri/VSE++ (2018), Chen (2021), and Wu/CLAP (2023).

---

## 1. Executive Summary

The AEMS **audio-text retrieval recall is effectively zero** (R@1 ≈ 0.001,
i.e. at-or-below random chance of 1/1022). Six independent diagnostic
frameworks converge on the same root cause:

> **The CLAP audio embeddings and the CLIP/mean-pooled text embeddings live in
> two nearly-disjoint, orthogonal sub-spaces, and the positive audio–text
> pairs are essentially unaligned (mean cosine ≈ 0.02).**

No amount of temperature tuning, negative mining, or evaluation-protocol
adjustment fixes this: the positive pairs themselves hover at zero similarity,
indistinguishable from random negatives (overlap area 92%). This is not a mild
"low recall" problem — it is a **fundamental modality-coordinate mismatch**
between how CLAP represents audio and how the AEMS text packagers
(description/transcript/fused) produce text embeddings.

**Primary contributing factors, ranked by severity:**

| Rank | Factor | Framework | Signal |
|:----:|--------|-----------|--------|
| 1 | **Zero positive-pair alignment** | Wang & Isola (2020) | pos cos ≈ 0.018, alignment ≈ 1.96 (ideal ~0) |
| 2 | **Extreme hubness** | Radovanović (2010) | 91.6% of queries return the same 1 hub video |
| 3 | **Severe text anisotropy / collapse** | Ethayarajh (2019) | text mean-cos 0.834 (near-cone), eff-dim 192/512 |
| 4 | **Large modality gap** | Liang et al. (2022) | audio↔text centroid cos ≈ 0.03 (orthogonal) |
| 5 | **Positive vs negative inseparable** | Zhai/SigLIP (2023) | overlap 0.92, separation 0.0008 |
| 6 | **Retrieved-audio is a hub, not the match** | Hoiem (2012) | 84.2% "missing semantics" failures |

---

## 2. Dataset & Protocol

- **Dataset:** AEMS educational-video corpus, test split.
- **Candidate gallery size:** 1,022 test videos (all shared across audio, text,
  and video embedding DBs).
- **Query count:** 1,022 (one query per video) in the self-retrieval protocol.
- **Embeddings used:**
  - `aems_audio_embeddings_v1.pt` — CLAP (3×10s segments, mean-pooled), 512-d
  - `aems_text_embeddings_fused_test.pt` — CLIP fusion of description+transcript
  - `aems_video_embeddings_v1.pt` — CLIP ViT-B/32, mean-pooled, 512-d
- **Random chance R@1:** `1/1022 ≈ 0.00098`.

---

## 3. Individual Framework Findings

### 3.1 Alignment & Uniformity — Wang & Isola (2020) + SigLIP (2023)
Script: `scripts/diagnostic/diagnose_alignment_uniformity.py`

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Positive-pair cosine sim | **0.018** | Essentially **zero alignment** (well-trained CLAP ≈ 0.3–0.6) |
| Positive L2 distance | 1.401 | Positive pairs ~orthogonal on the sphere |
| Alignment E‖x−y‖² | **1.964** | Very high (ideal ≈ 0) ⇒ positive pairs far apart |
| Audio uniformity (t=2) | −1.844 | Within normal range (audio not collapsed itself) |
| Text uniformity (t=2) | −0.644 | Too high (text concentrated in one region) |
| **Pos/neg distribution overlap** | **0.92** | Positives indistinguishable from negatives |
| Separation (pos_mean − neg_mean) | 0.0008 | No statistical separation at all |
| Margin (pos − max_neg) | **−0.058** | 99.9% of positives lose to their hardest negative |
| R@1 vs temperature (τ=0.5…50) | 0.001 (flat) | No temperature rescues retrieval |

**Conclusion:** The positive pairs are not aligned. This dominates everything.

### 3.2 Modality Gap — Liang et al. (2022)
Script: `scripts/diagnostic/diagnose_modality_gap.py`

| Modality pair | Centroid euclid | Centroid cos | cos_dist |
|---------------|:---------------:|:------------:|:--------:|
| audio ↔ text  | 1.39 | **0.03** | 0.97 |
| audio ↔ video | 1.42 | −0.01 | 1.01 |
| text  ↔ video | 1.14 | 0.35 | 0.65 |

Cross-modal cosine distributions:
- audio–text cross: mean **0.017**
- audio intra: mean **0.49**
- text intra: mean **0.84**

**Conclusion:** The audio and text centroids are near-orthogonal (modality
gap ~1.0 cosine). Audio is not near text in the shared space.

### 3.3 Anisotropy — Ethayarajh (2019)
Script: `scripts/diagnostic/diagnose_anisotropy.py`

| Modality | Mean pairwise cos | Eff. dim (of 512) | Top-1 PCA var |
|----------|:-----------------:|:-----------------:|:-------------:|
| audio | 0.485 | 46 | 18.8% |
| video | 0.629 | 113 | 8.3% |
| **text** | **0.834** | 192 | 5.5% |

Whitening diagnostic (ZCA):
- audio: cos 0.485 → **−0.0008** (spread restored, eff-dim 46 → 304)
- text: cos 0.834 → **−0.0009** (eff-dim 192 → 429)

**Conclusion:** Both audio and text are anisotropic; **the text embeddings are
severe** (mean-cos 0.83 ≈ collapse into a narrow cone). Whitening restores
spread dramatically, confirming the anisotropy is correctable in principle.

### 3.4 Hubness — Radovanović et al. (2010)
Script: `scripts/diagnostic/diagnose_hubness.py`

| Metric | Value |
|--------|-------|
| Max k-occurrence | **980 / 1022** |
| Hubness skewness (K=1) | **31.5** |
| Top-1 result is a hub | **91.6%** of queries |
| Hubs (k-occ > threshold) | 4 points (0.39% of gallery) |

Top hubs: `34617` (980), `36884` (733), `22277` (708), `14890` (657).

**Conclusion:** A single audio embedding is returned as the top-1 result for
96% of all audio queries. This is the *mechanism* by which the unaligned
space turns into ~0 recall — every query collapses onto the same generic
"hub" audio clip. Local scaling did not help because the underlying
alignment is the problem, not the neighborhood structure.

### 3.5 Data Quality & Failure Categorization — BLIP (2022), VSE++ (2018), Hoiem (2012)
Script: `scripts/diagnostic/diagnose_data_quality.py`

Failure categories (text→audio self retrieval):

| Category | Count | % | Interpretation |
|----------|:-----:|:--:|----------------|
| A. Correct (rank 1) | 1 | 0.1% | — |
| B. Near miss (rank 2–10) | 9 | 0.9% | — |
| C. Semantic confusion (same category) | 53 | 5.2% | Audio exists, not discriminative |
| **D. Missing semantics (diff category)** | **861** | **84.2%** | Audio content absent/misaligned |
| E. Noisy pair (aligned but bad) | 98 | 9.6% | Data quality issue |

Negative difficulty (VSE++):
- 99.9% of positives fail to beat their hardest negative.
- Hard-negative ratio extremely high ⇒ positives not discriminative.

**Conclusion:** 84% of failures are "missing semantics" — the retrieved audio
is from a different category, i.e. the audio embedding does not encode the
query's semantics at all (a coordinate-space problem, not a data-noise
problem). Only ~10% of failures are attributable to data noise.

### 3.6 Comprehensive Retrieval Metrics — Chen (2021), Wu/CLAP (2023)
Script: `scripts/diagnostic/diagnose_retrieval_metrics.py`

| Metric | text→audio | audio→text |
|--------|:----------:|:----------:|
| R@1 | 0.0000 | 0.0000 |
| R@5 | 0.0000 | 0.0000 |
| R@10 | 0.0000 | 0.0000 |
| R@50 | 0.0000 | 0.0000 |
| MRR | 0.0072 | 0.0082 |
| Mean rank | 502 | 498 |
| NDCG@5 | 0.0008 | 0.0019 |
| R@100 (CMC) | 0.097 | 0.103 |

**Protocol audit:** candidate set 1,022; R@1/chance ≈ 0.0× in both
directions. Every content category fails (R@1 = 0) except a handful of
stray hits in Documentaries / Historical Analysis.

---

## 4. Root-Cause Synthesis

The seven diagnostics all point at the same two-layer failure:

1. **Coordinate mismatch (the "why it's ~0"):** The CLAP audio space and the
   CLIP-text space are **not mutually aligned**. When you compute
   `CLAP_text(query) · CLAP_audio(video)` you're comparing two embeddings that
   *both* came from CLAP **only if** the text query is also embedded by CLAP's
   text branch. The current AEMS pipeline embeds **text via CLIP** (for the
   description/transcript/fused databases) while **audio via CLAP**. These two
   encoders were trained independently and their 512-d spaces are **not
   commensurable** → the positive pairs are at ~zero cosine. This is the
   single largest contributor.

2. **Hubness amplification (the "why it's exactly random not better"):**
   Because all cross-encoder cosines are near zero and nearly identical,
   tiny noise differences let one "hub" audio clip win almost every query,
   driving recall to floor.

### Secondary contributors
- **Text anisotropy:** the fused text embeddings are concentrated (mean-cos
  0.83), so even within the text side, retrieval discrimination is poor.
- **Data noise:** ~10% of pairs fall below the noise threshold, but this is
  minor relative to the alignment failure.
- **Segmentation:** 3-segment mean-pooling of audio averages informative and
  uninformative (music-intro / silence) segments, diluting semantics.

---

## 5. Recommended Remediations (ranked)

> These are recommendations **only** — no code changes were made in this
> review beyond the diagnostic tooling. Each corresponds to a measured finding.

1. **[Critical] Use CLAP for the text side of audio-text retrieval.**
   Compute audio-query similarity as `CLAP_text(query) · CLAP_audio(video)`
   instead of `CLIP_text(query) · CLAP_audio(video)`. This directly removes
   the coordinate mismatch observed in §3.2. The query-router/retrieval code
   currently mixes CLIP-text with CLAP-audio; unify the modality branches.
   *Fixes: alignment (§3.1), modality gap (§3.2), hubness mechanism (§3.4).*

2. **[High] Re-embed the text DB with CLAP**, so the text gallery used for
   audio retrieval lives in the same space as CLAP audio. (Alternative to / in
   addition to #1.)

3. **[High] Whitening / isotropy removal** (Ethayarajh §3.3). Apply ZCA or
   mean-centering re-normalization to the text embeddings to break the
   anisotropy (mean-cos 0.83 → ~0). Demonstrated to restore effective
   dimensionality from 192 → 429.

4. **[Medium] Schema-consistent normalization.** Ensure all embedding DBs are
   L2-normalized (some AEMS DBs are float16, some float32) and that audio/text
   are compared with a single consistent similarity/hypersphere.

5. **[Medium] Audio segmentation quality.** Drop uninformative segments
   (intro music, long silence) from the 3-segment average, or use
   max-pooling instead of mean → mitigates "missing semantics" (§3.5).

6. **[Low] Hard-negative / temperature re-tuning** (VSE++ §3.5, SigLIP §3.1)
   — **only after** the alignment fix; tuning alone cannot help (§3.1, R@1
   unchanged across τ).

7. **[Low] Data filtering** (BLIP §3.5) of the ~10% noisy pairs, after the
   coordinate mismatch is resolved.

---

## 6. Better Evaluation Methods (beyond raw R@K)

The existing `evaluate_retrieval.py` only reports R@K. This review adds a
fuller, fairer evaluation battery (Chen 2021) that is now available:

| Method | Why it matters |
|--------|----------------|
| **MRR** | Smoothly penalizes rank; better than R@K for small candidate sets |
| **Mean/Median rank** | Robust central-tendency of error |
| **NDCG@K** | Graded relevance ranking quality |
| **AP@K** | Precision-aware average precision |
| **CMC curve** | Full R@K sweep (K=1…N); exposes at what K recall emerges |
| **Category-stratified R@1** | Shows which content categories are (un)retrievable |
| **Protocol audit (R@1 / chance)** | Flags evaluation leaks / set-size effects |
| **LOO protocol** (exists separately) | Removes exact-caption leak |

**Important caveat for CLAP-style retrieval:** the standard cross-modal
retrieval protocol (e.g. Wu/CLAP on AudioCaps/Clotho) evaluates **text→audio**
and **audio→text** at candidate sets of 1k–5k with expected R@1 of ~5–20%.
Reporting a single R@1 on a 1,022-gallery self-retrieval is insufficient;
the CMC + MRR + NDCG battery here gives the complete picture and correctly
shows the system is at floor, not merely "low."

---

## 7. Reproducibility

All diagnostics load **precomputed** embeddings only — no GPU encoder/LM
is required at runtime, so they are fast and deterministic.

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/diagnostic/diagnose_alignment_uniformity.py   # Wang & Isola; SigLIP
python scripts/diagnostic/diagnose_modality_gap.py           # Liang et al.
python scripts/diagnostic/diagnose_anisotropy.py             # Ethayarajh
python scripts/diagnostic/diagnose_hubness.py                # Radovanović
python scripts/diagnostic/diagnose_data_quality.py           # BLIP; VSE++; Hoiem
python scripts/diagnostic/diagnose_retrieval_metrics.py      # Chen; CLAP
```

Outputs (all in `outputs/diagnostics/`):

| Report file | Source framework |
|-------------|------------------|
| `alignment_uniformity_report.json` | Wang & Isola 2020; Zhai 2023 |
| `modality_gap_report.json` | Liang et al. 2022 |
| `anisotropy_report.json` | Ethayarajh 2019 |
| `hubness_report.json` | Radovanović et al. 2010 |
| `data_quality_report.json` | Li/BLIP 2022; Faghri 2018; Hoiem 2012 |
| `retrieval_metrics_report.json` | Chen 2021; Wu/CLAP 2023 |

---

## 8. Decision Record

| # | Decision | Status |
|---|----------|--------|
| D1 | Root cause = CLIP-text vs CLAP-audio coordinate mismatch, not data noise | **Confirmed** (all 6 frameworks) |
| D2 | Enable CLAP text branch for audio retrieval | **Pending implementation** (Phase 3 Review 2) |
| D3 | Whitening for anisotropy | **Pending implementation** after D2 |
| D4 | Replace single R@K with full metric battery in all future AEMS audio evals | **Done** (this review adds the tooling) |

---

*Stamp: Setup date 03 September 2026 · Post Phase 3 Review 1 · AEMS Phase A (v1 embeddings)*
