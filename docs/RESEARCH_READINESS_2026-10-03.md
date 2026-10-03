# Research readiness: assessment and plan — 2026-10-03

An honest assessment of what stands between the current system and something
defensible as research, and what to do about it.

## Verdict

The engineering is sound and the measurement discipline is better than most
student projects: validation-only selection, paired bootstrap CIs, multi-seed
training, cross-fitting, oracle bounds, 81 tests. None of that is the problem.

The problem is that **the benchmark does not measure multimodal retrieval, and
the headline score is mostly an artifact of lexical overlap**. Until that is
fixed, a stronger number is a worse outcome, because it is a more confident
measurement of the wrong thing. Everything else in this document is secondary.

## Finding 1 — the headline number is an overlap artifact (decisive)

Each test query was stratified by how many of its content words appear in its
own video's transcript or description, then scored with the deployed stage-1
system:

| Query slice | n | R@1 | R@10 | best branch |
|---|---|---|---|---|
| **All** | 5097 | **0.7161** [0.7034, 0.7279] | 0.8187 | dense 0.677 |
| overlap = 0 | 758 | **0.0343** [0.0224, 0.0475] | 0.1003 | *visual* 0.054 |
| overlap (0, 0.2] | 405 | 0.4568 [0.4073, 0.5037] | 0.7383 | dense 0.457 |
| overlap (0.2, 0.6] | 2212 | 0.8038 [0.7871, 0.8205] | 0.9417 | dense 0.761 |
| overlap > 0.6 | 1722 | **0.9646** [0.9553, 0.9733] | 0.9959 | dense 0.904 |

**R@1 goes from 0.96 to 0.035 as lexical overlap goes away.** On the 750
queries (14.9%) that share no content word with their own source text, the
system is near-useless — and the best branch there is *visual* at 0.055, ahead
of the dense retriever at 0.033.

This is not only a BM25 effect. The dense E5 branch, which is supposed to match
meaning rather than words, collapses just as hard (0.905 → 0.033). What the
system has learned is to match a question against the text it was generated
from.

The query set makes this almost inevitable. The questions are LLM-generated and
overlap their own **description at 10.8×** and their own **transcript at 5.7×**
the rate they overlap a random video's. They also ask overwhelmingly about
*visual* detail — "What color is the tractor?", "What tool does the barber
use?", "What type of microphone was used?" — which the system answers without
looking at the video, by matching words to the transcript.

A reviewer can reproduce this stratification in twenty minutes. It has to be
addressed before anything is written up.

## Finding 2 — multimodality contributes almost nothing

Leave-one-out on validation: visual ≈ +0.001, audio ≈ +0.005. Two text branches
(dense, BM25) account for −0.049 and −0.044. The system is a text retriever over
a multimodally indexed corpus, which the docs already say.

The zero-overlap slice makes the point sharper: it is precisely where a
multimodal system would have to earn its keep, and the visual branch manages
0.055 while the audio branch manages 0.007.

The audio branch additionally does not encode content at all. The granularity
probe (`docs/RESULTS_MULTIMODAL_2026-10-03.md`) shows its identity signal
(+0.1683) exceeds its entire content signal (+0.1296): WavLM represents voice
and channel, not meaning.

## Finding 3 — there is no architectural novelty

Component by component, against prior work:

| Component | Status |
|---|---|
| Lexical + dense hybrid fusion | Standard since ~2021 |
| Max-sim late interaction over passages | ColBERT (2020) |
| Cross-encoder reranking of a shortlist | Standard two-stage IR |
| Per-query z-score score fusion | Standard |
| Fine-tuning a bi-encoder on in-domain pairs | Standard |
| **Adaptive gating (the claimed novelty)** | Worth **+0.0009**, at the edge of noise |

The one piece framed as the contribution is the one that does not work. As an
architecture paper this would not survive review.

## Finding 4 — no external validity

One dataset, one split, no public benchmark, no baseline from the literature
implemented or cited numerically. There is no evidence the approach transfers,
and no way for a reader to place the numbers.

## What is genuinely worth keeping

These are real contributions and should be foregrounded rather than buried:

1. **The identity-vs-content probe** — a reusable diagnostic showing an audio
   encoder can score 0.91 retrieving its own video and 0.05 against text,
   because it encodes channel identity. This explains a class of "multimodal
   fusion didn't help" results.
2. **Near-duplicate leakage in query-by-example evaluation** — an even/odd
   frame split looks rigorous and inflates image R@1 by 7.5 points. Many papers
   split this way.
3. **Modality-dependent weighting** — renormalizing over *reachable* branches
   is what everyone does; we showed it costs 0.69 R@1 for audio queries.
4. **The overlap stratification above** — a cheap, general test for whether a
   generated-query video benchmark measures anything.
5. The protocol discipline and reproducibility.

Notice that all five are *evaluation* contributions, not architectural ones.
That is the honest shape of what exists.

## Three paths

**A. Analysis / evaluation paper.** "What actually matters in multimodal video
retrieval, and why your benchmark may not measure it." Uses everything above,
needs no new architecture, and is defensible at a workshop. Lowest risk, modest
ceiling. Realistic in 3–4 weeks.

**B. Change the task so multimodality is necessary: temporal grounding.**
6,769 of 6,770 videos carry timecoded transcripts, which are not used at all
today. Moment retrieval — given a query, return the *span* — cannot be solved by
whole-video text matching, forces frame- and segment-level reasoning, and has
established baselines and metrics (R@1 at IoU 0.5/0.7). This is where real
novelty is available. 4–8 weeks.

**C. Fix the benchmark.** Build a query set that cannot be answered from the
transcript: filter to visually-grounded questions, verify a subsample by hand,
and report the zero-overlap slice as a headline rather than a footnote. Needed
under either A or B. 1–2 weeks.

**Recommendation: C first (it is required either way), then B, with A as the
fallback if B does not produce results in time.** C alone converts the project
from "reports an inflated number" to "honestly characterises a hard problem",
which is the minimum bar for defensibility.

## Phased plan

### Phase 0 — stop reporting the inflated number ✅ DONE

* Make the overlap stratification a first-class script
  (`bin/evaluation/eval_overlap_stratified.py`), not a scratch file.
* Add the zero-overlap and ≤0.2 slices to every results table.
* Put BM25-only in every table as a mandatory baseline — it is 0.55 of the
  0.72, and omitting it overstates the contribution.
* Rewrite the README headline to lead with both numbers: 0.7310 overall,
  0.035 without lexical overlap.
* **Acceptance:** no table in the repo reports an aggregate score without the
  low-overlap slice beside it.

### Phase 1 — a benchmark that cannot be gamed (in progress)

First result in `docs/RESULTS_VISUALLY_GROUNDED_2026-10-03.md`: the
visually-grounded tier (384 test queries) scores 0.2734 against 0.7145
on the aggregate, the visual branch degrades least of all six, and the
optimal weighting differs significantly by query type. Manual
verification of the 300-query sample is the outstanding item.

* Build a **visually-grounded query subset**: keep questions that require the
  video (colour, count, object, action, on-screen text), drop ones answerable
  from the transcript. An LLM classifier plus manual verification of ~300.
* Hand-verify a **gold set of ~300 queries** across categories; report it
  separately as the trustworthy number.
* Add **distractor-hard negatives**: evaluate within content category, so a
  query competes against same-topic videos rather than mostly-unrelated ones.
* **Acceptance:** a slice where BM25 scores near chance and the system still
  works, or clear evidence that it does not.

### Phase 2 — make multimodality necessary (3–6 weeks, the novelty)

* Parse `timecoded_text_to_speech` into aligned segments; build a segment-level
  index (frames + transcript spans).
* Define moment retrieval on it: query → (video, start, end). Metrics R@1 at
  IoU 0.5/0.7, mIoU.
* Baselines: whole-video retrieval then uniform span; transcript-only span
  matching; frame-only.
* The contribution: query-dependent modality routing *at segment level*, where
  the earlier result (0.69 R@1 swing from modality-aware weights) says routing
  genuinely matters — unlike the whole-video text task where it is worth 0.0009.
* **Acceptance:** a configuration using visual or audio evidence beats the
  transcript-only span baseline with a CI excluding zero.

### Phase 3 — external validity (1–2 weeks)

* Evaluate on **MSR-VTT 1k-A** (the standard split) and ideally DiDeMo.
* Report published numbers for CLIP4Clip / X-CLIP / LanguageBind alongside, and
  reimplement at least one as a run-in-house baseline.
* **Acceptance:** our pipeline on MSR-VTT lands within a stated distance of
  published CLIP4Clip, or the gap is explained.

### Phase 4 — paper-grade rigor (1 week, overlaps Phase 3)

* Three seeds minimum for every trained component, report mean ± std.
* One sealed test run per frozen configuration; stop printing test metrics on
  development runs (currently they are observed repeatedly — an acknowledged
  weakness).
* Efficiency table: latency and memory per stage, which the two-stage design
  makes a real contribution if reported.
* Failure analysis: sample and categorise 100 errors.
* Limitations section stating the overlap artifact plainly.

### Phase 5 — writing (1–2 weeks)

Lead with the diagnostic findings; they are the strongest material. Do not
lead with 0.7310.

## What a reviewer will ask

1. What does BM25 alone score? *(0.550 — must be in the main table.)*
2. How were the queries generated, and can they be answered without the video?
   *(LLM-generated; largely yes — this is the central threat to validity.)*
3. What happens without lexical overlap? *(0.035.)*
4. What does each modality contribute? *(Visual +0.001, audio +0.005.)*
5. How does this compare to published work on a standard benchmark? *(No
   answer today.)*
6. What is novel? *(Today: the evaluation findings, not the architecture.)*
7. Is the test set sealed? *(No — observed repeatedly.)*

Questions 1–4 are answerable now and the answers are uncomfortable, which is
precisely why they belong in the paper rather than being left for a reviewer to
find.

## Reproduce the central finding

```bash
python bin/evaluation/eval_overlap_stratified.py
```

(Written already, since the plan rests on it. Phase 0 is then about putting its
output into every table rather than about producing it.)
