import pytest
import torch

from src.evaluation.holdout import (INTERLEAVE, TEMPORAL, frame_holdout,
                                    audio_segment_is_disjoint)


def _frames(n=16):
    return torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 4)


def test_neither_split_lets_a_frame_be_both_query_and_index():
    """The whole point: a frame on both sides scores a trivial perfect match."""
    for split in (TEMPORAL, INTERLEAVE):
        index, query = frame_holdout(_frames(), split)
        shared = {tuple(r.tolist()) for r in index} & {tuple(r.tolist()) for r in query}
        assert not shared, f"{split} leaked {len(shared)} frames into both sides"


def test_both_splits_halve_the_frames():
    for split in (TEMPORAL, INTERLEAVE):
        index, query = frame_holdout(_frames(16), split)
        assert index.shape[0] == query.shape[0] == 8


def test_temporal_split_separates_query_from_index_in_time():
    """Every query frame comes after every indexed one, so none is a neighbour."""
    index, query = frame_holdout(_frames(), TEMPORAL)
    assert index[:, 0].max() < query[:, 0].min()


def test_interleave_split_leaves_every_query_frame_adjacent_to_an_indexed_one():
    """Formally disjoint, but each query frame is one step from an indexed frame."""
    index, query = frame_holdout(_frames(), INTERLEAVE)
    gaps = [min(abs(q - i) for i in index[:, 0].tolist()) for q in query[:, 0].tolist()]
    assert max(gaps) == 1


def test_odd_frame_counts_do_not_lose_a_frame():
    index, query = frame_holdout(_frames(15), TEMPORAL)
    assert index.shape[0] + query.shape[0] == 15


def test_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="unknown split"):
        frame_holdout(_frames(), "random")


def test_rejects_too_few_frames_to_hold_any_out():
    with pytest.raises(ValueError, match="at least 2 frames"):
        frame_holdout(_frames(1), TEMPORAL)


def test_audio_segment_clears_all_three_indexed_segments_when_long_enough():
    assert audio_segment_is_disjoint(duration=300.0, position=0.25, clip_sec=10.0)


def test_audio_segment_overlaps_when_the_video_is_short():
    """At 40s the 25% mark sits inside the start and middle segments."""
    assert not audio_segment_is_disjoint(duration=40.0, position=0.25, clip_sec=10.0)


def test_audio_segment_shorter_than_one_clip_cannot_be_held_out():
    assert not audio_segment_is_disjoint(duration=8.0, position=0.25, clip_sec=10.0)


def test_sixty_seconds_is_the_threshold_the_benchmark_relies_on():
    """The benchmark excludes videos under 60s on this basis; verify it holds."""
    assert audio_segment_is_disjoint(duration=60.0, position=0.25, clip_sec=10.0)
    assert not audio_segment_is_disjoint(duration=55.0, position=0.25, clip_sec=10.0)
