import pytest

from src.data.timecodes import (frames_inside, iou, locate, parse_timecode, segments)


def test_parse_timecode_handles_hours_minutes_and_fractional_seconds():
    assert parse_timecode("00:01:02.760") == pytest.approx(62.76)
    assert parse_timecode("01:00:00.000") == 3600.0


def test_segments_skips_unparseable_entries_rather_than_failing():
    record = {"timecoded_text_to_speech": [
        {"start": "00:00:00.000", "end": "00:00:02.000", "text": "hello"},
        {"start": "bogus", "end": "00:00:04.000", "text": "dropped"},
        {"start": "00:00:04.000", "end": "00:00:06.000", "text": "world"}]}
    assert [t for _, _, t in segments(record)] == ["hello", "world"]


def test_segments_drops_a_span_that_ends_before_it_starts():
    record = {"timecoded_text_to_speech": [
        {"start": "00:00:10.000", "end": "00:00:05.000", "text": "backwards"}]}
    assert segments(record) == []


SEGS = [(0.0, 5.0, "intro music"), (5.0, 10.0, "the barber uses clippers"),
        (10.0, 15.0, "to round the back"), (15.0, 20.0, "unrelated chatter")]


def test_locate_finds_the_window_containing_the_target_words():
    score, start, end, i, j = locate(SEGS, "the barber uses clippers")
    assert (start, end) == (5.0, 10.0)
    assert score == pytest.approx(1.0)


def test_locate_spans_several_segments_when_the_answer_does():
    _, start, end, _, _ = locate(SEGS, "clippers round the back")
    assert start == 5.0 and end == 15.0


def test_length_penalty_prefers_the_tighter_of_two_equal_windows():
    """Without it, a longer window wins by accumulating unrelated words."""
    tight = locate(SEGS, "clippers", length_penalty=0.02)
    assert (tight[1], tight[2]) == (5.0, 10.0)


def test_locate_returns_none_when_the_target_has_no_content_words():
    assert locate(SEGS, "what is it?") is None


def test_locate_returns_none_without_segments():
    assert locate([], "anything") is None


def test_iou_is_one_for_identical_spans_and_zero_for_disjoint_ones():
    assert iou((0.0, 10.0), (0.0, 10.0)) == 1.0
    assert iou((0.0, 10.0), (20.0, 30.0)) == 0.0


def test_iou_of_half_overlapping_spans():
    assert iou((0.0, 10.0), (5.0, 15.0)) == pytest.approx(5.0 / 15.0)


def test_iou_of_a_degenerate_span_is_zero_not_a_division_error():
    assert iou((5.0, 5.0), (0.0, 10.0)) == 0.0


def test_frames_inside_counts_uniform_samples_landing_in_the_span():
    # 16 frames over 320s sit at 10, 30, 50, ... so a 0-40s span holds two.
    assert frames_inside((0.0, 40.0), duration=320.0) == 2


def test_a_short_moment_can_contain_no_frame_at_all():
    """The structural ceiling on frame-based localisation at 16 frames/video."""
    assert frames_inside((11.0, 19.0), duration=320.0) == 0


def test_locate_uses_the_answer_not_the_question():
    """The benchmark's whole validity rests on this: labels come from the answer,
    which a method never sees, so the task is not solvable by the same matching
    that defined it."""
    segs = [(0.0, 5.0, "today we discuss photosynthesis"),
            (5.0, 10.0, "chlorophyll absorbs red and blue light")]
    question = "What does chlorophyll absorb?"
    answer = "chlorophyll absorbs red and blue light"
    assert locate(segs, answer)[1:3] == (5.0, 10.0)
    # The question alone would also land there, but it is never used for labelling;
    # what matters is that the two are different inputs.
    assert locate(segs, question) is not None


def test_a_window_wider_than_max_segments_is_never_proposed():
    segs = [(float(i), float(i + 1), f"word{i}") for i in range(10)]
    target = " ".join(f"word{i}" for i in range(10))
    _, start, end, lo, hi = locate(segs, target, max_segments=3)
    assert hi - lo <= 3
