# Temporal grounding — 2026-10-04

Phase 2 of `docs/RESEARCH_READINESS_2026-10-03.md`. Whole-video retrieval on
AEMS is largely solvable by matching a question against the transcript it was
generated from. Localisation should not be: every candidate window belongs to
the same video and shares its vocabulary, so a method has to discriminate
*within* a transcript rather than between documents.

## The task, and why its labels are not circular

AEMS ships no temporal annotations. 6,769 of 6,770 videos carry a timecoded
transcript that the project had never used, and labels are derived from it:

**A span is found using the answer; a method only ever sees the question.**
For each QA pair the label is the run of consecutive transcript segments best
covering the *answer*'s content words. The answer is privileged — available
when building the benchmark, never at inference.

That split is what stops the task being circular. The residual confound is that
questions share wording with their answers, and it is small: median question
overlap with its own labelled window is **0.20**, against **0.56** answer
coverage.

| | test | validation (from train) |
|---|---|---|
| labelled moments | 2409 | 2015 |
| videos | 867 | 720 |
| median span | 9.5 s | 9.2 s |
| median span / video | 0.034 | — |

Spans are moments, not sections: 3.4% of a video at the median.

## Results (test, labels frozen, visual weight tuned on validation)

| Method | mIoU | IoU≥0.3 | IoU≥0.5 | IoU≥0.7 | IoU≥0.5 CI |
|---|---|---|---|---|---|
| whole_video | 0.058 | 0.020 | **0.000** | 0.000 | [0.000, 0.000] |
| center prior | 0.049 | 0.073 | 0.036 | 0.011 | [0.029, 0.043] |
| **bm25** | **0.214** | **0.296** | **0.183** | **0.108** | [0.167, 0.198] |
| dense (E5) | 0.195 | 0.274 | 0.155 | 0.087 | [0.140, 0.169] |
| visual (CLIP, 3 s grid) | 0.076 | 0.101 | 0.064 | 0.044 | [0.055, 0.074] |
| dense+visual | 0.195 | 0.274 | 0.155 | 0.087 | [0.140, 0.169] |

**The task is hard and not solvable by priors.** Predicting the whole video
scores exactly 0.000 at IoU 0.5 and the centre prior scores 0.036. Compare the
whole-video benchmark, where trivial text matching reaches 0.96. This is a
benchmark with room in it.

**BM25 beats dense retrieval here (0.183 vs 0.155)** — a reversal of the
whole-video task, where dense led 0.675 to 0.550, and of the general
expectation that a fine-tuned bi-encoder beats bag-of-words. Within a single
transcript every candidate shares the topic and vocabulary, so semantic
similarity is nearly constant across candidates while exact term overlap still
discriminates. Dense retrieval's advantage is between documents, not inside one.

## The visual branch: a negative result, and it is not the sampling

The first run used the deployed 16 uniform frames per video. That is one frame
per ~20 s against a ~9.5 s moment, so **40.3% of spans contained no frame at
all** and the visual score on those was 0.002. No conclusion about visual
evidence could be drawn from that; it would have been a conclusion about the
sampling rate.

Frames were therefore re-encoded on a **3 s grid** for all 867 test and 720
validation videos (168,728 frames, 0 failures), lifting coverage from 59.7% to
**99.1%**.

| | 16 uniform frames | 3 s grid |
|---|---|---|
| spans containing a frame | 59.7% | **99.1%** |
| visual IoU≥0.5, all spans | 0.049 | **0.064** |
| visual IoU≥0.5, spans with a frame | 0.081 | 0.065 |

Coverage nearly doubled and the overall score rose only 31%, because the two
effects partly cancel: a denser grid supplies the right frame more often, and
also supplies many more wrong frames to be fooled by. Conditional on a frame
being present the dense grid is actually *worse* (0.065 against 0.081), which
is what more distractors per video looks like.

**Tuning the fusion weight on validation returned exactly zero.** Every
non-zero weight degraded the fusion monotonically (IoU≥0.5: 0.169 at w=0, 0.160
at 0.1, 0.145 at 0.5, 0.103 at 2.0), so `dense+visual` on test is identical to
`dense` by construction. The visual branch contributes nothing to this task
that transcript matching does not already have.

This was the acceptance criterion in the Phase 2 plan — "a configuration using
visual or audio evidence beats the transcript-only span baseline with a CI
excluding zero". **It fails.** Recording that plainly is the point of having
set it in advance.

### A prediction that was half right

Before running it, the stated expectation was that fixing coverage would
"roughly double the visual score to ~0.08–0.10 and it will still lose badly to
BM25". The direction and the conclusion held; the magnitude did not — 0.049 to
0.064 is a 31% rise, not a doubling, and the distractor effect above was not
anticipated.

## What this does and does not establish

**Does:** CLIP frame-question matching cannot localise these moments, and that
is a property of the representation rather than of the frame sampling, which
has now been controlled. A 3 s grid is dense enough that 99% of moments are
visible, and the result barely moves.

**Does not:** this is not evidence that *visual* information is useless for
temporal grounding in general. The labels mark where an answer is **spoken**,
because that is the only signal the transcript can supply. Questions whose
answers are only shown — the visually-grounded tier of
`docs/RESULTS_VISUALLY_GROUNDED_2026-10-03.md` — get no label at all here. The
benchmark is structurally biased toward transcript methods, which was stated
before it was built and remains its main limitation.

A genuinely visual grounding benchmark needs labels that do not come from text.
The honest options are human annotation, or a second label source derived from
the answer against frames — which would need a stronger image-text model than
CLIP ViT-B/32, given it scores 0.32 on image→caption retrieval here.

## Limitations

1. **Labels come from the transcript**, so the task measures grounding of
   spoken content and favours transcript methods by construction.
2. **No reranking.** Stage 2 was not applied to window candidates.
3. **Proposal set is transcript-derived**, so segment boundaries are given.
   A method is ranking ~5n candidates, not regressing boundaries freely; the
   numbers are not comparable to Charades-STA or ActivityNet Captions.
4. **One frame encoder.** CLIP ViT-B/32 only; SigLIP or a video-native encoder
   was not tried, and the negative result is specific to this representation.
5. **Low-coverage labels discarded.** 2,534 of 5,040 test QA pairs had answer
   coverage below 0.3 and are excluded, which biases the set toward questions
   whose answers are spoken verbatim.

## Reproduce

```bash
python bin/data/build_moment_benchmark.py --split test
python bin/data/build_moment_benchmark.py --split train
python bin/embeddings/precompute_dense_frames.py \
    --videos-from data/processed/aems/metadata/aems_moments_test.json --split test
python bin/training/tune_moment_fusion.py
python bin/evaluation/eval_moment_retrieval.py \
    --dense-frames embeddings/aems_dense_frames_test.pt
```
