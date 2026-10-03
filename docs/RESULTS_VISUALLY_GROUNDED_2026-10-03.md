# The visually-grounded query subset — 2026-10-03

Phase 1 of `docs/RESEARCH_READINESS_2026-10-03.md`. The aggregate benchmark is
dominated by lexical overlap between a question and the text it was generated
from, so this carves out the queries where that shortcut is unavailable and
measures the system on them.

## Construction

Three signals, none trusted alone (`src/data/query_subsets.py`):

* **answer overlap** — the manifest carries a written answer for every
  question. If the answer's content words are absent from the transcript and
  description, the answer is not in the text. This is the strongest signal, and
  it is the one the project was not previously using at all.
* **question overlap** — how much of the question's wording appears in its
  source text. Removes the retrieval-time lexical shortcut.
* **visual markers** — a keyword pattern (colour, count, on-screen text,
  clothing, position). Crude, used only to label the strict tier, never alone.

| Tier | Rule | Test queries |
|---|---|---|
| all | — | 5040 |
| `text_blind` | question overlap ≤ 0.3 | 1498 |
| `answer_grounded` | answer overlap ≤ 0.3 | 1708 |
| `visually_grounded` | both, plus a visual marker | **384** (285 videos) |

Examples drawn from the strict tier — the answers genuinely are not in the
transcript:

> *What animal is shown alongside humans waiting in a queue?* — Penguins
> *What visual element appears at the very end of the video, suggesting a message of hope?* — A rainbow appears on the projected screen behind the singer.
> *What is the color of the stuffed animal in the background?* — The stuffed animal is Elmo, who is red.

## Results (full 1,022-video gallery, stage 1, deployed weights)

| Tier | n | R@1 | R@10 | MRR | recall@100 |
|---|---|---|---|---|---|
| all | 5040 | 0.7145 | 0.8171 | 0.7530 | 0.8855 |
| `answer_grounded` | 1708 | 0.4982 | 0.6423 | 0.5498 | 0.7769 |
| `text_blind` | 1498 | 0.2824 | 0.4379 | 0.3384 | 0.6295 |
| **`visually_grounded`** | 384 | **0.2734** | 0.5052 | 0.3547 | 0.7135 |

The deployed system answers roughly **27%** of genuinely visual questions at
rank 1, against 71% on the benchmark as a whole.

Unlike the zero-overlap slice (recall@100 = 0.334), recall here is 0.714, so
stage 2 has real headroom on this tier — the right videos are usually in the
shortlist, just not at the top.

## Which branches survive the shortcut being removed

| Branch | R@1 all | R@1 visually-grounded | degradation |
|---|---|---|---|
| bm25 | 0.548 | 0.089 | **6.2×** |
| chunk (CLIP passages) | 0.363 | 0.076 | 4.8× |
| text (CLIP caption) | 0.388 | 0.086 | 4.5× |
| dense (E5, fine-tuned) | 0.675 | 0.255 | 2.6× |
| audio | 0.053 | 0.023 | 2.3× |
| **visual** | 0.268 | 0.151 | **1.8×** |

The ordering is the result. The lexical branch collapses hardest, because it is
the one the artifact was feeding; the visual branch degrades least, because it
was never relying on the shortcut in the first place. Dense retrieval sits in
between — still the strongest single branch at 0.255, but losing more than half
its advantage.

This is the first evidence in the project that the visual branch carries
something the text branches do not. On the aggregate benchmark it looks nearly
worthless (+0.001 by leave-one-out); here it is the second-strongest branch and
the most robust one.

## The optimal weighting depends on the query type

Weights were re-tuned by coordinate ascent on the 302 `visually_grounded`
queries falling in the **validation** split, then applied unchanged to test:

| Weights | visual | bm25 | dense | test R@1 (visually-grounded) | test R@1 (all) |
|---|---|---|---|---|---|
| deployed | 0.3 | 0.5 | 1.1 | 0.2734 | **0.7145** |
| visually-tuned | **0.5** | **0.1** | 1.1 | **0.2995** | 0.6919 |

| Tier | paired Δ (tuned − deployed) | |
|---|---|---|
| visually_grounded | **+0.0260** [+0.0026, +0.0521] | significant |
| all | **−0.0226** [−0.0292, −0.0157] | significant |
| text_blind | +0.0087 [−0.0020, +0.0194] | ns |
| answer_grounded | −0.0012 [−0.0123, +0.0100] | ns |

**Significant in both directions.** One weighting is measurably better for
visual questions, the other measurably better overall, and each is measurably
worse on the other's territory. No single fixed vector serves both.

This matters for the project's central claim. Adaptive fusion was worth +0.0009
on the aggregate — indistinguishable from noise — and the reasonable conclusion
was that query-dependent weighting had little to offer. That conclusion was an
artifact of the benchmark: on the aggregate, one branch (dense, fed by lexical
overlap) is reliably right, so there is nothing to adapt to. Once the shortcut
is removed, the branches genuinely trade off and adaptation has something to do.

## Stage 2 per tier, and whether it composes with re-weighting

The chain (per-candidate gate → fine-tuned MiniLM, K=100, 8 passages) applied
to each tier, and then applied on top of the visually-tuned stage 1. All deltas
are paired against that tier's deployed stage-1 baseline.

| Tier | stage 1 | + chain | tuned weights | tuned + chain |
|---|---|---|---|---|
| all (5040) | 0.7145 | **0.7294** sig | 0.6919 sig(−) | 0.7244 sig |
| answer_grounded (1708) | 0.4982 | **0.5199** sig | 0.4971 ns | 0.5193 sig |
| text_blind (1498) | 0.2824 | 0.2977 sig | 0.2911 ns | **0.3044** sig |
| **visually_grounded (384)** | 0.2734 | 0.2969 ns | 0.2995 sig | **0.3047** sig |

Three things come out of this.

**The reranker is not artifact-dependent.** It adds +0.015 to +0.023 on every
tier, including the hardest. In relative terms it is worth *more* where the
benchmark is hardest: +2.1% on the aggregate against +8.6% on the
visually-grounded tier. Unlike stage 1, which collapses when lexical overlap is
removed, the cross-encoder transfers. That makes it the most robust component
in the system, and it is the one piece whose aggregate number was not
flattering it.

**On visual queries, fixing the weights is worth as much as running stage 2.**
Re-weighting alone reaches 0.2995 and the full cross-encoder pass reaches
0.2969 — statistically indistinguishable, but the first is a change to six
numbers in a config file and the second is a 22M-parameter model over 100
candidates × 8 passages per query. Where the two interventions are equal in
effect, they are about three orders of magnitude apart in cost.

**They compose, but sub-additively.** Together they reach 0.3047 (+0.0312
[+0.0051, +0.0599]) where simple addition would predict +0.049. They are
partly fixing the same errors — both are, in effect, down-weighting a lexical
match that should not have won.

The best configuration differs by tier: `chain` alone on the aggregate,
`tuned + chain` on both hard tiers. That is the same conclusion the modality
weights reached from the other direction — no single configuration is best
everywhere, which is the case for query-adaptive routing.

## Threshold sensitivity

The tier is defined by two cut-offs and a keyword list, so the branch ordering
has to survive moving them or it is a fact about the cut-offs
(`bin/evaluation/sweep_subset_thresholds.py`, 16 cells, all ≥ 250 queries):

| Q ≤ | A ≤ | n | R@1 | visual | bm25 | dense | visual/bm25 |
|---|---|---|---|---|---|---|---|
| — | — | 5040 | 0.7145 | 0.268 | 0.548 | 0.675 | **0.49** |
| 0.5 | 0.3 | 540 | 0.4204 | 0.174 | 0.217 | 0.385 | 0.80 |
| 0.4 | 0.3 | 476 | 0.3655 | 0.174 | 0.158 | 0.338 | 1.11 |
| **0.3** | **0.3** | **384** | **0.2734** | **0.151** | **0.089** | **0.255** | **1.71** |
| 0.2 | 0.3 | 299 | 0.1839 | 0.130 | 0.040 | 0.177 | 3.25 |
| 0.2 | 0.2 | 250 | 0.1920 | 0.128 | 0.036 | 0.184 | 3.56 |

The ordering is stable and the trend is monotonic: visual beats BM25 in **12 of
16 cells**, and the ratio rises smoothly from 0.49 on all queries to 3.56 at the
strictest setting. The four cells where BM25 still wins are the loosest corner
(Q ≤ 0.5), where the shortcut is only partly removed.

What the sweep shows most clearly is **which** cut-off is doing the work. Hold
Q at 0.3 and vary A across 0.2–0.5: the ratio moves 1.81 → 1.69, essentially
flat. Hold A and vary Q: it moves 3.56 → 0.84. The effect is driven almost
entirely by **question overlap**, not answer overlap.

That corrects the framing above. Answer grounding is the better *definition* of
"this question needs the video" — it is about the question's semantics rather
than its phrasing — but it is not what reorders the branches. The reordering is
a retrieval-side effect: when a question stops sharing wording with the
transcript, the lexical branch loses its grip. Both filters earn their place,
for different reasons, and the subset should not be described as if answer
grounding were producing the headline effect.

The underlying reason is visible in the columns: the **visual branch barely
moves** across the entire grid (0.125–0.179) while **BM25 swings 6×**
(0.036–0.231). Visual branch quality is essentially independent of lexical
overlap, which is the honest statement of the finding; BM25's quality is almost
entirely determined by it.

## Honest limitations

1. **The subset is heuristic and not yet verified by hand.** 300 queries are
   written to `aems_query_subsets_test_to_verify.jsonl` for manual checking;
   until that is done, the tier is defined by thresholds and a keyword list,
   both judgement calls. Threshold sensitivity is now measured (above); the
   keyword list's precision is not.
2. **384 queries is small.** The CI on 0.2734 is [0.2317, 0.3203], and the
   weight-tuning gain (+0.0260) has a lower bound of +0.0026 — real, but thin.
   Tuning used only 302 validation queries.
3. **Visual markers both over- and under-fire.** "What is shown..." matches
   questions that are answerable from the transcript; genuinely visual
   questions phrased without a marker are missed.
4. **This measures the subset, not a fixed benchmark.** Changing the thresholds
   changes the number, so the tier definition has to ship with any result
   quoted from it.
5. These are stage-1 numbers; the reranker was not run per tier.

## Reproduce

```bash
python bin/data/build_visually_grounded_subset.py --split test
python bin/data/build_visually_grounded_subset.py --split train
python bin/evaluation/eval_query_subset.py
python bin/training/tune_fusion_weights.py --floor 0.05 \
    --subset data/processed/aems/metadata/aems_query_subsets_train.json \
    --tier visually_grounded --output outputs/aems/fusion_weight_tuning_visual.json
python bin/evaluation/eval_query_subset.py \
    --weights-json outputs/aems/fusion_weight_tuning_visual.json
```
