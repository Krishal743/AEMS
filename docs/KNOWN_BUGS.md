# Known Bugs

## Bug 1: eval_transformer_baseline.py truncates captions

**File**: `scripts/evaluation/eval_transformer_baseline.py`

The `valid_mask` filtering logic at lines 51–53 incorrectly deduplicates captions via a set-based intersection, processing only ~2,727 of 59,794 test captions.

**Fix**: Replicate `run_clip_baseline.py`'s approach: encode all 59,794 test captions, no filtering.

## Bug 2: Gating retraining has no train caption embeddings

**File**: `scripts/training/retrain_gating.py`

`caption_embeddings.pt` only stores test split videos (2,990). The gating script needs train split caption embeddings (7,010 videos) to train.

**Fix**: Either:
- (a) Precompute caption embeddings for the train split: `scripts/embeddings/precompute_caption_embeddings.py` with `--split train`
- (b) Encode train text queries on-the-fly with CLIP and cache to disk

## Bug 3: run_final_eval.sh references wrong gate weight paths (FIXED)

**File**: `scripts/evaluation/run_final_eval.sh`

The original script referenced `eval_results/gating_weights_old.pth` and `eval_results/gating_weights_transformer.pth` — paths that didn't exist.

**Status**: Fixed during restructuring — paths now point to `models/`.

## Bug 4: Audio retrieval near-zero R@1

Audio retrieval R@1 ≈ 0.0003 — 12% of test entries had no matching audio embeddings.

**Partially fixed**: Filtered test entries to only include videos present in `audio_embeddings.pt`. Root cause may be a CLAP encoding issue.

## Bug 5: LEXICON_LOGGING undefined in run_query_routing.py (FIXED)

A variable `LEXICON_LOGGING` was used in the curriculum phase check but never defined.

**Status**: Fixed during restructuring — defaults to `False`.
