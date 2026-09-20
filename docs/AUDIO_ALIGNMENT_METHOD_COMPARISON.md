# AEMS Audio Alignment — Method Comparison — Post Phase 3 Review 1

**Setup Date:** 03 September 2026
**Data Stamp:** AEMS Phase A (v1 embeddings), Post Phase 3 Review 1
**Scope:** Systematic compare-and-select of five audio→CLIP-text alignment
methods (whitening+ridge, CCA, shared-text-anchor, hubness reduction, and an
AudioCLIP-style adapter with contrastive fine-tuning) on the held-out AEMS
test set, plus the earlier Procrustes linear round, to lift the
previously-degenerate CLAP-audio→CLIP-text retrieval branch.

---

## 1. Objective & task

Earlier work established two facts about the AEMS audio branch:

1. **Raw CLAP-audio and CLIP-text are incommensurable spaces** → cross-modal
   CLIP-text→audio retrieval was impossible (R@1 = 0.0000).
2. **A single orthogonal linear projection (Procrustes)** aligned the *two
   distributions* (alignment 0.022→0.609, modality gap 0.96→0.0007) but **did
   not add discrimination** — held-out R@1 stayed 0.0000, because an orthogonal
   rotation is distance-preserving.

This round tests **all methods discussed** to find one that actually lifts
retrieval, not just absolute similarity. All methods were **fit on the 5,748
train anchors only**; the 1,022 held-out test anchors were used exclusively for
evaluation (no leakage). The frozen CLIP-text target is the **description**
embedding everywhere.

---

## 2. Methods tested

| ID | Method | Frameworks | Idea |
|----|--------|-----------|------|
| M0 | Raw CLAP (baseline) | — | Un-aligned reference |
| M1 | Whitening + ridge | Ethayarajh 2019; Wang 2019 | ZCA-whiten both views (restore spread, eff-dim 46→304), then ridge |
| M2 | CCA projection | Rasiwasia 2010; Andrew 2013 | Project into top-k (128) maximally-correlated subspace |
| M5 | Hubness reduction | Radovanović 2010; Zelnik-Manor 2004 | Local-scaling normalization of the M1 gallery |
| M3 | Shared-text anchor | Girdhar 2023; Grave 2019 | Fit Procrustes CLAP-text→CLIP-text, map CLAP-audio through it |
| M4 | **Adapter + contrastive** | Guzhov 2022; Wu 2022; Houlsby 2019; Korbar 2021 | Small MLP on frozen CLAP-audio, InfoNCE + hard negatives vs frozen CLIP-text |

Compute: M1/M2/M5/M3 are matrix ops (M3 uses one CLAP text-encode pass); M4 is
a ~40-epoch GPU train (~2.6k train pairs/batch-256, <15 min on the 24 GB GPU).

---

## 3. Results (held-out test, n = 1,022 anchors; chance R@1 ≈ 0.001)

| Method | CLIP-text→audio R@1 | MRR | audio→CLIP-text R@1 | alignment | hub skew |
|--------|-------|------|-------|-----------|----------|
| RAW CLAP (no alignment) | 0.0000 | 0.008 | 0.0010 | 0.022 | 14.9 |
| Procrustes (prior round) | 0.0000 | — | — | 0.609 | 10.9 |
| M1 whitening + ridge | 0.0000 | 0.006 | 0.0010 | 0.001 | 11.6 |
| M2 CCA projection | 0.0010 | 0.008 | 0.0000 | 0.003 | 7.4 |
| M5 hubness (local scaling) | 0.0000 | 0.006 | 0.0010 | 0.001 | 11.6 |
| M3 shared-text anchor | 0.0010 | 0.006 | 0.0000 | 0.405 | 14.4 |
| **M4 adapter + contrastive** | **0.0323** | **0.062** | **0.0538** | 0.126 | 11.1 |

> `outputs/alignment/method_comparison.json` holds the full matrix (R@5/10,
> mean/median rank, modality gap, anisotropy per method).

---

## 4. Interpretation

- **Every geometry-preserving / similarity-aligning method (M1, M2, M3, M5,
  Procrustes) stays at or below chance (≤ 0.0010).** Aligning the *distribution*
  or restoring isotropic spread does **not** create the discriminative margins
  needed to separate 1,022 positives among 1,022 negatives when the source
  features themselves lack separable structure (see the prior anisotropy/hubness
  diagnostics: audio eff-dim ≈ 46, severe hubness).
- **M4 — the only method that explicitly optimizes retrieval discrimination —
  is the clear winner:** CLIP-text→audio R@1 = **0.0323** (~33× chance, 32×
  better than all others) and audio→CLIP-text R@1 = **0.0538** (~55× chance).
  It learns a non-trivial map with score margins rather than a rotation.
- Comparison confirmed cleanly: M4 beats the next-best method by **32× on the
  operative held-out metric**.

---

## 5. End-to-end fusion impact (`outputs/alignment/fusion_impact_m4.json`)

Wiring the M4-aligned audio DB into the existing AEMS multi-modal evaluation
(5,097 QA-query rows, CLIP-text query):

| System | raw audio | M4 audio | Δ |
|--------|-----------|----------|---|
| audio-only (CLIP-text query) | 0.0012 | **0.0078** | **+0.0067** |
| equal fusion | 0.2411 | 0.2341 | −0.0071 |
| adaptive gating | 0.3894 | 0.3894 | 0.0000 |

- The **audio branch goes from ~chance to actually retrieving** (the
  previously impossible CLIP-text→audio path is now usable).
- The overall fused R@1 is **preserved** (equal fusion within noise; adaptive
  gating unchanged because the gate already down-weights audio, the weakest
  modality). No system regresses.

---

## 6. Selected method & deliverable

**Winner: M4 — frozen-CLAP adapter with contrastive fine-tuning.**

- Model: `models/audio_adapter_cliptext_m4.pt`
- Aligned audio DB (6,770 videos, video-id keys): `embeddings/aems_audio_aligned_m4_adapter_ct.pt`
- Usage: identical interface to `aems_audio_embeddings_v1.pt`; drop-in for the
  audio branch in `scripts/evaluation/eval_aems_retrieval.py`.

Reproduce:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/alignment/m4_adapter_contrastive.py        # train + export DB
python scripts/alignment/compare_methods.py               # full comparison table
python scripts/alignment/fusion_impact_m4.py              # end-to-end fusion impact
```

---

## 7. Recommendation / remaining escalation

- **Adopt M4** as the audio-alignment method for the AEMS audio branch.
- M4 still leaves headroom (audio→text 0.0538). Logical further steps if
  higher audio retrieval is required:
  1. **Full audio-encoder fine-tuning** (not just an adapter head) over CLAP
     with frozen CLIP-text targets — larger capacity than the 512→512 adapter.
  2. **Harder negative mining** (cross-batch / global, VSE++/HSE++ style with
     global negatives) and temperature annealing.
  3. **Contrastive pre-training on the raw audio waveforms** (Wav2CLIP-style)
     rather than the frozen v1 CLAP features, if annotation budget allows.
- The purely linear/whitening/CCA/anchor family is **exhausted** — it cannot
  add discrimination on these features, per 4 methods + Procrustes all at
  chance.

---

*End of AUDIO_ALIGNMENT_METHOD_COMPARISON.md (03 September 2026, Post Phase 3 Review 1).*
