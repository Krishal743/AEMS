# Experimental protocol

Rules for producing a number in this project. They exist because each one has
already been violated here at a measurable cost.

## Splits

| split | videos | source | used for |
|---|---|---|---|
| fit | 4,886 | train minus validation | fitting every model |
| validation | 862 | 15% of train, `validation_split(..., seed=0)` | every choice: checkpoints, hyperparameters, fusion weights |
| test | 1,022 | manifest `split == "test"` | scored once per finished configuration |

`validation_split` in `src/training/query_data.py` is deterministic and shared
by the audio adapter, the gating network and the ablation ladder, so the same
862 videos are held out everywhere. A model is never selected on data another
model in the same pipeline was fitted on.

Queries are the manifest's `qa_questions`: 24,338 for fit, 4,290 for validation,
5,097 for test. Each query's ground truth is its own video, and the gallery is
the split's videos (so validation and test numbers are not directly comparable —
different gallery sizes).

## Rules

**1. Selection happens on validation.** Checkpoint epochs, temperature, learning
rate, hidden width, fusion weights, reranker depth: all chosen by validation
R@1 or MRR. Test is scored once, after the configuration is frozen.

*Violated by:* `experiments/audio_alignment/m4_contrastive_ssl.py` and
`m4_adapter_contrastive.py`, which pick their best epoch by scoring the **test**
set (`quick_a2t_r1(model, X_te, Y_te)`) and then report that same test set. Any
number in `docs/AUDIO_ALIGNMENT_*.md` inherits that bias.
*Also violated by:* `experiments/ablations/sweep_const_weights.py`, which grid
searches fusion weights on test; `sweep_train_weights.py` is its clean sibling.

**2. Cross-fit any model whose scores train a downstream model.** The audio
adapter is fitted on the fit videos, so on those same videos its similarities
are optimistic. Training the gate on them teaches it that audio is more reliable
than it is. Split the fit videos into K folds, fit an adapter per fold on the
other K-1, and project each fold with the adapter that did not see it.

*Cost of violating it, measured:* the gate's mean audio weight rises from 0.14
to 0.35 and test R@1 falls from 0.4616 to 0.4177 — **-0.0439 R@1
[-0.0520, -0.0361]**. Reproduce with `--no-cross-fit` (row A5' of the ablation).

**3. Claim a difference only when a paired bootstrap CI excludes zero.** Use
`paired_bootstrap` from `src/evaluation/evaluate_retrieval.py`: both systems see
the same queries, so pairing removes query-difficulty variance. A point
difference without an interval is not a result.

*Cost of ignoring it:* "adaptive gating beats fixed weights" looked true at
+0.0020 R@1 until the interval showed it spanning zero. The honest version is
that the gate wins on R@5 (+0.0141, CI excludes zero) and on validation, not on
R@1.

**4. Compare against a *tuned* baseline.** A change that only beats an untuned
configuration has not been shown to help.

*Cost of ignoring it:* per-query z-scoring looked worth ~1.2 R@1 points when
compared against equal-weight fusion and a collapsed gate. Against fixed weights
tuned per configuration it is worth **-0.0063** (row A2). Its actual value is
enabling the gate, not raw recall.

**5. Report multiple seeds for anything trained.** `--seeds 42 100 200`; report
mean ± std; deploy the best on validation. Current gate: 0.5016 / 0.5002 /
0.5000 val R@1 (mean 0.5006 ± 0.0009).

**6. A gate that collapses is not adaptive.** The gating trainer prints the mean
and standard deviation of predicted weights and warns when the weights are
nearly constant or fail to beat tuned fixed weights. A collapsed gate reproduces
single-modality search while claiming to be adaptive — the deployed gate did
exactly this before this work (`w_text = 1.000 ± 0.000`).

## Reproducing the headline numbers

```bash
python bin/evaluation/eval_aems_retrieval.py --bootstrap   # ~6s, CIs + deltas vs text-only
python bin/evaluation/run_ablations.py                     # ablation ladder A0-A5'
python bin/training/train_gating_network.py --seeds 42 100 200
```

Outputs land in `outputs/aems/`: `eval_results_<visual>_<text>_v1.json`,
`summary_table_<visual>_<text>_v1.md`, `ablations.json`,
`gating_training_summary.json`.

## Known-stale results

`experiments/ablations/` and `experiments/audio_alignment/` still run but no
longer describe the shipped system: they fuse 3 branches (no passage branch),
score audio with CLAP in CLAP space rather than WavLM projected into CLIP space,
and use angular similarity with raw weighted sums instead of z-scored fusion.
Their tuned constants (e.g. `[0.30, 0.69, 0.01]`) belong to that older regime.
Treat `docs/AUDIO_ALIGNMENT_*.md`, `docs/ABLATIONS.md` and
`docs/FINAL_VERIFICATION.md` as history, not current results; `docs/RESULTS_*.md`
and `outputs/aems/` are current.
