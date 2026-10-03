# Multimodal query benchmark — 2026-10-03

The system is described as any-to-any retrieval, but every number the project
reported came from text queries. The image, video, audio and mixed query paths
existed in `src/routing/query_router.py` and had **never been measured**. This
is that measurement.

Run with `bin/evaluation/eval_multimodal_queries.py --bootstrap --per-branch`.
Gallery: all 1,022 test videos, identical for every row.

## The headline is a trap

With per-modality weights (`--weights modality`), against the text-tuned
weights the router used for everything before:

| Query | n | R@1 | text-tuned weights | paired Δ | branches reached |
|---|---|---|---|---|---|
| text | 1022 | 0.7505 | 0.7505 | — (falls back) | all 6 |
| image (1 frame) | 1022 | **0.7329** | 0.6791 | +0.0538 [+0.0352, +0.0734] sig | visual+text+chunk |
| video (8 frames) | 1022 | **0.8904** | 0.8523 | +0.0382 [+0.0186, +0.0558] sig | visual+text+chunk |
| **audio (10 s)** | 886 | **0.9131** | 0.9131 | — (one branch) | audio |
| audio as CLIP query | 886 | **0.9131** | 0.2201 | +0.6930 [+0.6625, +0.7223] sig | visual+text+chunk+audio |
| image + text | 1022 | **0.9041** | 0.8063 | +0.0978 [+0.0793, +0.1184] sig | all 6 |

Read naively this says audio and video queries beat text queries by a wide
margin and the system is strongest in exactly the modalities it never measured.
**That reading is wrong**, and the rest of this document is why.

## These rows measure a different, easier task

Text retrieval here is *semantic*: a question the user wrote has to be matched
to a video it never appeared in. Image, video and audio retrieval as benchmarked
here is **query-by-example**: a piece cut out of the target video is matched
back to the video it came from. Finding the source of a clip is a real and
useful task — reverse clip search, duplicate detection, "what video is this
from" — but it is not the same problem, and the numbers are not comparable.

Hold-out was still necessary and is enforced:

* **frames** — 16 per video. The gallery is rebuilt from the first 8 (first half
  of the runtime), queries come from the last 8. No frame is on both sides.
* **audio** — the indexed embedding averages fixed 10 s segments at the start,
  middle and end. Queries use a 10 s segment at the 25% mark, which clears all
  three for videos of 60 s or more; the 136 shorter test videos are excluded
  from the audio rows (but remain in the gallery as distractors).

Disjoint is not the same as independent, which the frame diagnostic shows
directly. Querying with the odd frames against an even-frame gallery — formally
disjoint, but every query frame sits one sampling step from an indexed one —
inflates the result:

| frame split | own-video best sim | best other video | margin | image R@1 |
|---|---|---|---|---|
| interleave (adjacent) | 0.8630 | 0.7735 | +0.0896 | 0.7544 |
| temporal (half apart) | 0.8225 | 0.7710 | +0.0516 | **0.6791** |

The near-duplicate effect is worth 7.5 R@1 points on its own. The temporal split
is used throughout; the interleaved number is recorded only to show the size of
the trap. Even the temporal split cannot remove the fact that a lecture keeps
the same speaker, room and slide template for its whole runtime.

## Per-branch scores separate appearance from meaning

Scoring each reachable branch alone separates "these pixels resemble those
pixels" from "this image depicts what this text describes".

| Query | visual | caption | passage | bm25 | dense | audio |
|---|---|---|---|---|---|---|
| text | 0.3014 | 0.4403 | 0.3894 | 0.5558 | **0.7153** | — |
| image | **0.6928** | 0.3209 | 0.3053 | — | — | — |
| video | **0.8160** | 0.4863 | 0.4501 | — | — | — |
| image+text | 0.7202 | 0.6008 | 0.4247 | 0.5558 | 0.7153 | — |
| audio | — | — | — | — | — | **0.9131** |
| audio as CLIP query | 0.0305 | 0.0722 | 0.0666 | — | — | 0.9131 |

Image and video queries are carried almost entirely by the visual branch, which
is frames matched against frames. The genuinely cross-modal part — an image
scored against the *text* describing the video — reaches 0.3209, well below the
appearance match but far above the 1/1022 ≈ 0.001 chance rate. Cross-modal
semantic retrieval is real here, and it is roughly symmetric: a text query
scores 0.3014 on the visual branch, an image query 0.3209 on the caption branch.

## The audio number is acoustic identity, not content

0.9131 is the highest figure in the project and the least meaningful. The same
audio embeddings, queried by *text*, score **0.0532** — the weakest branch in
the system. An embedding cannot simultaneously be an excellent and a near-
useless representation of what a video is about. What it can be is an excellent
representation of *who is speaking and in what room*: WavLM is a speech model,
and a 10 s segment from the middle of a lecture matches its own recording's
speaker, microphone and acoustics almost perfectly, whatever is being said.

So the audio row measures acoustic fingerprinting. That is genuinely useful for
deduplication and source attribution, and it should not be reported as the audio
branch understanding content. The 0.0532 figure remains the honest statement of
how much the audio branch contributes to semantic retrieval.

## The defect this exposed, and the fix

Under the old text-tuned weights, `audio as CLIP query` scored **0.2201** while
the audio branch alone scored **0.9131**. Adding information made the system
four times worse.

The cause: the deployed weights were tuned on text queries, where the audio
branch is nearly worthless and is weighted accordingly. When a query reaches
only visual, caption, passage and audio, those weights renormalize over that
subset and the audio branch gets 0.1/0.7 = **14%** of the mass while three
branches that are near-chance for audio take the other 86%. The router
renormalized over *reachable* branches but had no notion that which branches
are *reliable* depends on what kind of query it is.

`AEMS_MODALITY_WEIGHTS` in `src/config.py` fixes this: one weight vector per
modality, tuned by `bin/training/tune_modality_weights.py` with the same
coordinate ascent used for the text weights, on validation videos carved from
**train**. Test results above.

Two details worth recording:

* For `audio_clip` the tuner drove the visual, caption and passage weights to
  **exactly zero**, recovering the full 0.6930. Reading an audio query as a CLIP
  query adds nothing over the audio branch alone — the adapter's output is in
  CLIP space geometrically but carries no usable signal against image or text
  embeddings (0.03-0.07 R@1 alone, against a 1/886 chance rate).
* `text` is deliberately **absent** from the table and falls back to the
  deployed weights. The benchmark's own text row wants a higher visual weight
  (0.5 against 0.3, +0.0070 on validation), but it was tuned on this reduced
  8-frame gallery with one question per video and must not be transplanted onto
  the deployed path. Whether the deployed visual weight is genuinely too low is
  worth re-testing properly on the full gallery.

This is also the first measurement where query-dependent weighting has
something substantial to do. On text queries the per-candidate gate is worth
+0.0009, at the edge of noise; here, getting the weights right for the query's
modality is worth up to 0.69 R@1.

## What this changes in the project's claims

* "Any-to-any retrieval" is now **measured** rather than asserted, and the
  non-text paths do work.
* The strong non-text numbers are query-by-example, not semantic retrieval, and
  must be labelled as such wherever they are quoted.
* Caveat 3 of `docs/RESULTS_2026-10-03.md` ("only text queries are evaluated")
  is now partly discharged, and caveat 4 ("the multimodal branches contribute
  little") is unchanged and in fact reinforced: the branches are good at
  recognising their own material and weak at carrying meaning across modalities.
* Mixed image+text queries beat either alone (0.8063 vs 0.7505 and 0.6791),
  which is the cleanest evidence in the project that fusion across modalities
  helps rather than merely averaging.

## Limitations

1. **No external query set.** Every query is cut from its own target video.
   A genuine benchmark would use images and audio from outside the corpus with
   human relevance judgements; none exists for AEMS.
2. **One query per video**, so the text row (0.7505) is not the headline number
   (0.7310 over 5,097 questions with reranking). It is the matched baseline for
   these rows and nothing else.
3. **No reranking in any row.** The cross-encoder scores text against text and
   cannot serve an image or audio query at all; stage 2 is simply unavailable
   outside text queries, which is itself a gap.
4. **Audio covers 886 of 1,022 videos.** Shorter videos cannot yield a disjoint
   segment. The gallery is still all 1,022, so the task is not made easier.

## Reproduce

```bash
python bin/evaluation/eval_multimodal_queries.py --bootstrap --per-branch --diagnose-frames
python bin/evaluation/eval_multimodal_queries.py --frame-split interleave --modalities image
```


## What the embeddings actually encode

`bin/verification/probe_embedding_granularity.py` resolves the contradiction
between audio→audio 0.9131 and text→audio 0.0532. For 300 test videos it
compares three similarities per branch: two disjoint stretches of the **same
video** (same speaker and room, different words), the **content twin** (the
different video whose transcript is most similar — mean dense similarity 0.7496,
so genuinely about the same subject), and a **random other** video.

| Branch | same video | content twin | random other | share of the signal that is *content* |
|---|---|---|---|---|
| dense text (E5) | 0.9002 | 0.8595 | 0.6493 | **84%** |
| audio (WavLM→adapter) | 0.3186 | 0.1503 | 0.0207 | 43% |
| visual (CLIP frames) | 0.9289 | 0.7613 | 0.6531 | 39% |

(Content share = (twin − random) / (same video − random): how much of the
distance from a stranger to the video itself is already covered by another
video about the same subject.)

**The text branch encodes what is said.** A different video on the same topic
reaches 0.8595 against the video's own other half at 0.9002 — almost all the
way. Two speakers delivering the same content produce nearly the same text
embedding, which is exactly what a retrieval encoder should do.

**The audio branch encodes who is speaking, and where.** Sharing a recording is
worth more than sharing a subject: same-video similarity is 2.1× the content
twin, and the identity gap (+0.1683) is larger than the entire content signal
(+0.1296). This is why it can retrieve its own video from a 10 s clip at 0.9131
while being the system's worst branch (0.0532) at matching text. WavLM is a
speech model; it represents voice and channel, not meaning.

**The visual branch sits in between**, closer to the audio case: a lecture keeps
its speaker, room and slide template throughout, so frames identify the
recording more than the topic.

### So: can they tell two speakers apart saying the same thing?

| | Two speakers, same words | One speaker, different words |
|---|---|---|
| text / BM25 / dense | **No** — near-identical embeddings | Distinguishes them well |
| audio | **Yes, strongly** — this is mostly what it encodes | Often confuses them |
| visual | Yes, if they look or film differently | Often confuses them |

The two failure modes are mirror images, and the system has no mechanism that
covers both. It is also worth being clear about granularity: these are
**whole-video** representations (one audio vector, one caption vector, 16 frame
vectors, N passage vectors). Nothing is diarized or time-aligned, so the system
cannot say *who* said something or *when* — only that a video, taken as a whole,
matches.
