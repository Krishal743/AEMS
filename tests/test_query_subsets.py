from src.data.query_subsets import (ANSWER_GROUNDED, TEXT_BLIND, VISUALLY_GROUNDED,
                                    VISUAL_MARKERS, content_words, overlap,
                                    source_pool, tiers_for)


def test_content_words_drops_stopwords_and_short_tokens():
    assert content_words("What is the COLOR of the tractor?") == {"color", "tractor"}


def test_video_is_a_stopword_because_it_is_in_every_question_and_description():
    assert "video" not in content_words("What happens in the video?")


def test_overlap_is_none_when_a_query_has_no_content_words():
    """Guards the division: such queries must be skipped, not counted as zero."""
    assert overlap("What is it?", {"anything"}) is None


def test_overlap_counts_distinct_content_words_present_in_the_source():
    assert overlap("barber tractor", {"barber", "unrelated"}) == 0.5


def test_source_pool_merges_description_and_transcript():
    pool = source_pool({"text_description": "penguins queue", "text_transcript": "gondolas"})
    assert {"penguins", "queue", "gondolas"} <= pool


def test_a_question_answerable_from_the_transcript_is_not_visually_grounded():
    assert tiers_for(question_overlap=0.9, answer_overlap=0.9, has_visual_marker=True) == []


def test_low_question_overlap_alone_only_earns_text_blind():
    assert tiers_for(0.1, 0.9, True) == [TEXT_BLIND]


def test_absent_answer_alone_only_earns_answer_grounded():
    assert tiers_for(0.9, 0.1, True) == [ANSWER_GROUNDED]


def test_visually_grounded_requires_both_overlaps_and_a_marker():
    assert tiers_for(0.1, 0.1, True) == [TEXT_BLIND, ANSWER_GROUNDED, VISUALLY_GROUNDED]
    assert VISUALLY_GROUNDED not in tiers_for(0.1, 0.1, False)


def test_visually_grounded_is_a_subset_of_the_other_two_tiers():
    tiers = tiers_for(0.1, 0.1, True)
    assert TEXT_BLIND in tiers and ANSWER_GROUNDED in tiers


def test_thresholds_are_inclusive_at_the_boundary():
    assert TEXT_BLIND in tiers_for(0.3, 0.9, False, max_question_overlap=0.3)
    assert TEXT_BLIND not in tiers_for(0.31, 0.9, False, max_question_overlap=0.3)


def test_visual_markers_fire_on_the_question_types_the_subset_targets():
    for q in ["What color is the tractor?", "How many people are shown?",
              "What is written on the board?", "What is the man wearing?"]:
        assert VISUAL_MARKERS.search(q), q


def test_visual_markers_do_not_fire_on_a_purely_conceptual_question():
    assert not VISUAL_MARKERS.search("Why is teamwork important in classroom management?")


def test_content_words_is_cached_without_letting_callers_corrupt_the_cache():
    """It is memoised for the moment labeller; a mutable return would be unsafe."""
    first = content_words("barber clippers")
    assert isinstance(first, frozenset)
    assert first is content_words("barber clippers")
    assert first | {"extra"} == {"barber", "clippers", "extra"}
    assert content_words("barber clippers") == {"barber", "clippers"}
