# Known Bugs

## Bug 1: eval_transformer_baseline.py truncates captions — Obsolete

**Status**: Obsolete. The script and the MSR-VTT data it evaluated were removed; AEMS evaluation runs through `bin/evaluation/eval_aems_retrieval.py`, which has one caption embedding per video and no deduplication step.

**File**: `scripts/evaluation/eval_transformer_baseline.py`

The `valid_mask` filtering logic at lines 51–53 incorrectly deduplicates captions via a set-based intersection, processing only ~2,727 of 59,794 test captions.

**Fix**: Replicate `run_clip_baseline.py`'s approach: encode all 59,794 test captions, no filtering.

## Bug 2: Gating retraining has no train caption embeddings

**File**: `scripts/training/retrain_gating.py`

`caption_embeddings.pt` only stores test split videos (2,990). The gating script needs train split caption embeddings (7,010 videos) to train.

**Status: Resolved — not root cause.** Re-training with properly split caption pairs (train vs test) produced identical collapsed weights (w_v ≈ 0.015, w_a ≈ 0.535 on transformer embeddings). The gating collapse is confirmed to be an architectural limitation, not a data bug.

**Fix**: Either:
- (a) Precompute caption embeddings for the train split: `bin/embeddings/precompute_caption_embeddings_legacy.py` with `--split train`
- (b) Encode train text queries on-the-fly with CLIP and cache to disk

## Bug 3: run_final_eval.sh references wrong gate weight paths ✅ Fixed

**File**: `bin/evaluation/run_final_eval.sh`

The original script referenced `eval_results/gating_weights_old.pth` and `eval_results/gating_weights_transformer.pth` — paths that didn't exist.

**Status**: ✅ Fixed during restructuring — paths now point to `models/`.

## Bug 4: Audio retrieval near-zero R@1

Audio retrieval R@1 ≈ 0.0003 — 12% of test entries had no matching audio embeddings.

**Partially fixed**: Filtered test entries to only include videos present in `audio_embeddings.pt`. Root cause may be a CLAP encoding issue.

**Update (AEMS)**: The query scripts scored the audio branch with a CLIP vector against CLAP embeddings (or the reverse for audio queries). That was fixed first; audio-only R@1 stayed at 0.004 on the full test set, so the space mismatch was not the cause.

**Root cause**: CLAP itself. A clean comparison (validation-selected checkpoints, real QA test queries, 3 seeds) put zero-shot CLAP at R@1 0.004 against WavLM-Large + adapter at 0.053. AEMS is entirely speech-heavy educational video, so an audio encoder adds little beyond the transcript already in the text branch.

**Status**: Addressed on `feat/new-audio-pipeline` by replacing CLAP with WavLM + adapter. Audio remains the weakest branch by design; see Bug 10.

## Bug 5: LEXICON_LOGGING undefined in run_query_routing.py ✅ Fixed

A variable `LEXICON_LOGGING` was used in the curriculum phase check but never defined.

**Status**: ✅ Fixed during restructuring — defaults to `False`.

## Bug 6: Query scripts crashed and scored audio in the wrong space ✅ Fixed

**Files**: `bin/queries/query_*.py`, `bin/demo/demo.py`, `bin/evaluation/behavioural_test.py`

They defaulted to MSR-VTT gate checkpoints that no longer load, max-pooled caption embeddings in an MSR-VTT-only shape, and compared CLIP queries against CLAP audio embeddings.

**Status**: ✅ Fixed. All of them now go through `src/routing/query_router.py`, which keeps separate CLIP and CLAP query vectors and masks out branches a query type cannot reach. Covered by `tests/test_query_router.py`.

## Bug 7: Gate checkpoints loaded with `strict=False` ✅ Fixed

**Files**: `src/routing/query_router.py`, `bin/evaluation/eval_aems_retrieval.py`

A checkpoint whose weights didn't match the architecture loaded silently, leaving those layers randomly initialised.

**Status**: ✅ Fixed. `load_gate` raises if any learnable parameter is missing or any key is unexpected. It still tolerates the temperature/scale buffers that `forward()` never reads.

## Bug 8: Gating trainer evaluated train queries against test labels ✅ Fixed

**File**: `bin/training/train_gating_network.py`

After freeing CLIP, the test pass scored the first N *train* query embeddings against *test* ground truth, so every test metric it printed was meaningless.

**Status**: ✅ Fixed. Test queries are now encoded before the encoders are freed. The ablation scripts were already correct.

## Bug 9: Imports from the removed `scripts` package ✅ Fixed

28 `from scripts.alignment…` / `from scripts.ablation…` imports across 17 files in `experiments/` crashed on import after the restructure.

**Status**: ✅ Fixed. They now point at the `experiments` package.

## Bug 10: Branch scores were fused on incompatible scales ✅ Fixed

**Files**: `src/routing/query_router.py`, `bin/evaluation/eval_aems_retrieval.py`, `bin/training/train_gating_network.py`

Visual, text and audio cosine similarities have different means and spreads, so
summing them raw let one branch dominate: equal fusion scored R@1 0.083 against
0.389 for text alone, and the trained gate collapsed onto text for every query
(making "adaptive gating" identical to text-only search).

**Status**: ✅ Fixed on `feat/new-audio-pipeline`. Each branch is z-scored per
query across the gallery before fusion, with weights from `AEMS_FUSION_WEIGHTS`
or the gate (`--fusion gate`). The gating trainer now uses the same fusion it is
deployed with; the old angular-similarity and per-modality scale constants,
which inference never applied, are gone.

## Bug 11: CLAP precompute saved the previous video's embedding for short clips ✅ Fixed

**File**: `bin/embeddings/precompute_audio_embeddings.py`

For audio of 10 s or less the script computed `emb` but stored `audio_embed`,
a variable left over from the previous iteration (or undefined on the first
video).

**Status**: ✅ Fixed — the script now extracts WavLM features and has a single
code path for all clip lengths.
