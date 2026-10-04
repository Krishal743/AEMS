import numpy as np
import torch

from src.evaluation.moments import METHODS, _z, windows_for

SEGS = [(0.0, 5.0, "a"), (5.0, 10.0, "b"), (10.0, 15.0, "c")]


def test_windows_cover_every_run_up_to_the_width_limit():
    spans, texts, ranges = windows_for(SEGS, max_segments=2)
    assert spans == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0),
                     (0.0, 10.0), (5.0, 15.0)]
    assert texts[3] == "a b"
    assert ranges[3] == (0, 2)


def test_every_single_segment_is_a_candidate():
    """The label can be one segment, so the proposal set must contain each alone."""
    spans, _, _ = windows_for(SEGS, max_segments=3)
    for start, end, _ in SEGS:
        assert (start, end) in spans


def test_the_full_span_is_reachable_when_width_allows():
    spans, _, _ = windows_for(SEGS, max_segments=3)
    assert (0.0, 15.0) in spans


def test_width_limit_excludes_longer_runs():
    spans, _, _ = windows_for(SEGS, max_segments=1)
    assert (0.0, 15.0) not in spans and len(spans) == 3


def test_no_windows_for_an_empty_transcript():
    assert windows_for([], max_segments=5) == ([], [], [])


def test_z_centres_each_vector_so_branches_are_comparable():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(_z(x).mean()) == __import__("pytest").approx(0.0, abs=1e-5)


def test_z_is_scale_invariant():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(_z(a), _z(a * 100), atol=1e-4)


def test_z_survives_a_constant_vector_without_dividing_by_zero():
    out = _z(torch.ones(5))
    assert torch.isfinite(out).all()


def test_method_list_matches_what_the_scripts_expect():
    assert set(METHODS) == {"whole_video", "center", "bm25", "dense", "visual",
                            "dense+visual"}
