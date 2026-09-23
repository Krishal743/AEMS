import pytest
import torch

from src.evaluation.evaluate_retrieval import (
    bootstrap_ci,
    evaluate_retrieval,
    ground_truth_ranks,
    hits_at_k,
    metrics_from_ranks,
    paired_bootstrap,
)


def reference_recall(sim, text_video_ids, video_ids, ks=(1, 5, 10)):
    """The original loop-and-argsort implementation, kept as the oracle."""
    results = {f"R@{k}": 0.0 for k in ks}
    for i in range(sim.size(0)):
        ranked = torch.argsort(sim[i], descending=True)
        for k in ks:
            if text_video_ids[i] in [video_ids[j] for j in ranked[:k].tolist()]:
                results[f"R@{k}"] += 1.0
    return {k: v / sim.size(0) for k, v in results.items()}


def test_matches_the_original_loop_implementation():
    torch.manual_seed(0)
    n_queries, n_videos = 40, 25
    sim = torch.randn(n_queries, n_videos)
    video_ids = [f"v{i}" for i in range(n_videos)]
    gt = [video_ids[i % n_videos] for i in range(n_queries)]
    got = evaluate_retrieval(sim, gt, video_ids)
    expected = reference_recall(sim, gt, video_ids)
    assert got == pytest.approx(expected, abs=1e-7)


def test_ranks_count_strictly_better_candidates():
    sim = torch.tensor([[0.1, 0.9, 0.5], [0.7, 0.2, 0.3]])
    ranks = ground_truth_ranks(sim, [0, 0])
    assert ranks.tolist() == [2, 0]      # 0.1 is worst; 0.7 is best
    assert metrics_from_ranks(ranks, ks=(1,))["R@1"] == 0.5
    assert abs(metrics_from_ranks(ranks, ks=(1,))["MRR"] - (1 / 3 + 1) / 2) < 1e-6


def test_ground_truth_outside_the_gallery_counts_as_a_miss():
    sim = torch.randn(3, 4)
    video_ids = [f"v{i}" for i in range(4)]
    out = evaluate_retrieval(sim, ["v0", "missing", "v2"], video_ids, ks=(10,))
    assert out["R@10"] == pytest.approx(2 / 3, abs=1e-7)


def test_hits_and_metrics_agree():
    torch.manual_seed(1)
    sim = torch.randn(50, 30)
    ranks = ground_truth_ranks(sim, list(range(50 % 30)) * 0 + [i % 30 for i in range(50)])
    for k in (1, 5, 10):
        assert abs(hits_at_k(ranks, k).mean().item() - metrics_from_ranks(ranks)[f"R@{k}"]) < 1e-9


def test_bootstrap_ci_brackets_the_point_estimate():
    torch.manual_seed(0)
    hits = (torch.rand(2000) < 0.4).float()
    low, high = bootstrap_ci(hits, iters=500)
    assert low < hits.mean().item() < high
    assert high - low < 0.1          # 2000 queries -> a tight interval


def test_bootstrap_ci_is_deterministic_for_a_seed():
    hits = (torch.rand(500) < 0.5).float()
    assert bootstrap_ci(hits, iters=200, seed=7) == bootstrap_ci(hits, iters=200, seed=7)


def test_paired_bootstrap_detects_a_consistent_win():
    n = 1000
    b = (torch.rand(n) < 0.3).float()
    a = b.clone()
    a[:80] = 1.0                      # a wins on 80 queries it shares with b
    delta, low, high = paired_bootstrap(a, b, iters=500)
    assert delta > 0 and low > 0      # CI excludes zero: a real difference


def test_paired_bootstrap_reports_no_difference_for_identical_systems():
    hits = (torch.rand(400) < 0.5).float()
    delta, low, high = paired_bootstrap(hits, hits, iters=200)
    assert delta == 0.0 and low == 0.0 and high == 0.0


def test_paired_bootstrap_requires_matching_queries():
    with pytest.raises(ValueError):
        paired_bootstrap(torch.zeros(5), torch.zeros(6))
