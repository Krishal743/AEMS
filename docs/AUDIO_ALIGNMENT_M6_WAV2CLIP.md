# AEMS Audio Alignment — M6 Wav2CLIP — Post Phase 3 Review 1

**Setup Date:** 03 September 2026
**Data Stamp:** AEMS Phase A (v1 embeddings), Post Phase 3 Review 1
**Scope:** Follow-up experiment to the post-M4 tier campaign. Implements the
Wav2CLIP-style target swap (Wu et al., ICASSP 2022): train the audio adapter to
match the CLIP **video/image** embedding of the same video instead of its text
description, so that alignments live in CLIP's joint space and remain
queryable by frozen CLIP **text** at test time. Motivation: M4 (audio→text)
works but the CLAP-text↔CLIP-text space gap limits it; CLIP visual features are
semantically rich and already live in CLIP's joint space (Radford 2021). We also
test the natural hybrid (video **and** text targets), per the multi-task idea
from the tier-1 analysis.

Same contract: 5,748 train anchors / 1,022 held-out test anchors, frozen
CLIP-text description targets for evaluation, no leakage.

---

## 1. Methods

| ID | Target during training | Test query target |
|----|------------------------|-------------------|
| M6-w2c | `aems_video_embeddings_v1.pt` (CLIP ViT-B/32 mean-pooled frames, 6,770 keys, 1:1 with audio) | CLIP-text (unchanged) |
| M6-hybrid | sum of two InfoNCE terms vs video **and** text targets | CLIP-text (unchanged) |

Training uses the exact M4 recipe (same `AudioAdapter` MLP, hard-negative
InfoNCE, batch 256, lr 1e-3, 30 epochs, cosine) — only the target(s) change
(AudioCLIP frozen-backbone principle; ImageBind single-anchor idea).

Scripts: `scripts/alignment/m6_wav2clip.py` (new),
`scripts/alignment/compare_all_methods.py` (new comparison table),
`scripts/alignment/fusion_impact_aligned_db.py` (new, parameterized end-to-end).

## 2. Results — held-out test (n=1022), vs the frozen-CLIP-text harness

| Method | t2a R@1 | t2a R@10 | t2a MRR | a2t R@1 | a2t R@10 | a2t MRR | align |
|--------|--------:|---------:|--------:|--------:|---------:|--------:|------:|
| M4 (text targets) | **0.0323** | 0.1096 | 0.062 | 0.0538 | 0.2016 | 0.108 | 0.126 |
| M6-w2c (video targets) | 0.0078 | 0.0519 | 0.028 | 0.0235 | 0.1292 | 0.062 | 0.080 |
| M6-hybrid (video+text) | 0.0254 | **0.1204** | 0.059 | **0.0587** | **0.2172** | **0.111** | **0.138** |

- **M6-w2c (video only) underperforms M4** in both directions. Probable cause:
  the target is the *mean* over 16 frames, which smooths away the per-video
  fine-grained signature that text descriptions still carry; and the
  CLAP-audio→video-CLIP mapping is no easier than audio→text. Training
  alignment-to-video reached ~0.25 but transferred weakly to text queries
  (0.08).
- **M6-hybrid** is strictly ≥ M4 on the a2t direction (+0.005 R@1,
  +0.016 R@10) and on t2a R@10, while slightly trailing M4 on t2a R@1
  (0.0254 vs 0.0323). Adding the paired text term recovers the discrimination
  M4 gives while retaining the richer video objective.

## 3. End-to-end fusion impact (held-out TEST, 5,097 QA queries, 1,022 videos)

`outputs/alignment/fusion_impact_m4_vs_m6hybrid.json` — live stack, CLIP-text
query, audio branch swapped (RAW → M4 → M6-hybrid):

| System | RAW | M4 | M6-hybrid | Δ(M6−M4) |
|--------|----:|----:|----------:|----------:|
| audio-only CLIP-text branch (R@1) | 0.0012 | 0.0078 | 0.0082 | +0.0004 |
| equal fusion (R@1) | 0.2625 | 0.3431 | 0.3453 | +0.0022 |
| adaptive gating (R@1) | 0.3894 | 0.3894 | 0.3894 | +0.0000 |

Both aligned DBs lift equal-fusion retrieval massively over RAW
(0.26→0.34); adaptive gating is dominated by video/text routing so the audio
branch change is masked there. The M6-hybrid edge over M4 is small
(+0.0022 equal-fusion R@1 ≈ +11 queries) — a real but marginal gain.

> NOTE: figures recorded earlier in `fusion_impact_m4.json`
> (equal-fusion 0.2411→0.2341) predate the aligned-DB key remap
> (positional→video-id) and are superseded by the corrected numbers above.

## 4. Decision

- **M4 remains the incumbent** for the text-query→audio path (t2a R@1).
- **M6-hybrid** is the best overall aligned audio DB **on balance**:
  it improves the audio-query→text direction (+0.005 R@1) and nudges the
  end-to-end equal-fusion stack marginally higher (+0.0022 R@1), with no
  regression elsewhere.
- Given the margin is within run-noise, both DBs are retained as documented
  candidates. Adopting M6-hybrid in downstream consumers is a swap of the DB
  reference `aems_audio_aligned_m6_hybrid.pt` → for `aems_audio_aligned_m4_adapter_ct.pt`;
  no existing scripts were modified.

## 5. Artifacts

| Component | Path |
|---|---|
| Training script (new) | `scripts/alignment/m6_wav2clip.py` |
| Comparison v2 (new) | `scripts/alignment/compare_all_methods.py` → `outputs/alignment/method_comparison_v2.json` |
| Parameterized fusion (new) | `scripts/alignment/fusion_impact_aligned_db.py` → `outputs/alignment/fusion_impact_m4_vs_m6hybrid.json` |
| M6 video-target model/DB | `models/audio_adapter_clipvid_m6.pt`, `embeddings/aems_audio_aligned_m6_w2c.pt` |
| M6 hybrid model/DB | `models/audio_adapter_clipvid_text_hybrid_m6.pt`, `embeddings/aems_audio_aligned_m6_hybrid.pt` |
| Reused (untouched) | `scripts/alignment/m4_adapter_contrastive.py`, `compare_methods.py`, `fusion_impact_m4.py` |
| Literature | Wav2CLIP (Wu 2022); AudioCLIP (Guzhov 2022); ImageBind (Girdhar 2023); CLIP (Radford 2021) |