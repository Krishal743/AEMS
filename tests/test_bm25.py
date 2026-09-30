import torch

from src.retrieval.bm25 import BM25PassageIndex, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("What IS the Bellman-equation?") == ["what", "is", "the", "bellman", "equation"]


def test_retrieves_the_video_containing_a_rare_term():
    index = BM25PassageIndex([
        ["today we cover the bellman equation in dynamic programming"],
        ["a lecture about renaissance painting and sculpture"],
        ["how to knead bread dough properly"],
    ])
    scores = index.score("what is the bellman equation")
    assert scores.index(max(scores)) == 0


def test_a_video_scores_by_its_best_passage_not_its_average():
    """One strongly matching passage must carry the video, as in the dense branch."""
    index = BM25PassageIndex([
        ["unrelated chatter", "unrelated chatter", "photosynthesis converts light to sugar"],
        ["photosynthesis"],
    ])
    scores = index.score("photosynthesis converts light to sugar")
    assert scores[0] > scores[1]


def test_query_with_no_matching_terms_scores_zero():
    index = BM25PassageIndex([["alpha beta"], ["gamma delta"]])
    assert index.score("nothing here matches") == [0.0, 0.0]


def test_common_terms_contribute_less_than_rare_ones():
    index = BM25PassageIndex([[f"the lecture covers topic{i}"] for i in range(20)])
    common = index.score("the")
    rare = index.score("topic7")
    assert max(rare) > max(common)
    assert rare.index(max(rare)) == 7


def test_score_batch_shape_and_agreement():
    index = BM25PassageIndex([["alpha beta"], ["gamma delta"], ["beta gamma"]])
    queries = ["alpha", "gamma delta"]
    batch = index.score_batch(queries)
    assert batch.shape == (2, 3)
    for i, q in enumerate(queries):
        assert torch.allclose(batch[i], torch.tensor(index.score(q)))


def test_empty_corpus_is_handled():
    index = BM25PassageIndex([])
    assert index.score("anything") == []


def test_batch_matches_single_with_repeated_query_terms():
    """Duplicate terms must count twice in both paths, or the two disagree."""
    index = BM25PassageIndex([["alpha beta gamma"], ["beta beta delta"], ["gamma"]])
    for query in ("beta beta", "alpha alpha gamma", "beta"):
        assert torch.allclose(index.score_batch([query])[0],
                              torch.tensor(index.score(query)), atol=1e-5)
