import pytest
import torch
import torch.nn.functional as F

from src.rerank import per_candidate, stage1
from src.routing.query_router import BRANCHES

N_VIDEOS, D, NB = 12, 512, len(BRANCHES)


def branch_sims(n_queries=4, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(n_queries, N_VIDEOS) for _ in range(NB)]


def test_top_k_candidates_are_ordered_best_first():
    scores = torch.tensor([[0.1, 0.9, 0.4, 0.7]])
    assert stage1.top_k_candidates(scores, 3).tolist() == [[1, 3, 2]]


def test_candidate_recall_and_positive_position():
    candidates = torch.tensor([[3, 1, 2], [5, 6, 7]])
    gt = torch.tensor([1, 9])
    assert stage1.candidate_recall(candidates, gt) == 0.5
    assert stage1.positive_position(candidates, gt).tolist() == [1, -1]


def test_gather_branch_scores_shape_and_values():
    sims = branch_sims()
    candidates = stage1.top_k_candidates(sims[0], 5)
    feats = stage1.gather_branch_scores(sims, candidates)
    assert feats.shape == (4, 5, NB)
    assert torch.allclose(feats[2, :, 1], sims[1][2, candidates[2]])


def test_reranking_only_reorders_the_shortlist():
    """A reranked candidate must never fall below a video the reranker never saw."""
    torch.manual_seed(0)
    stage1_scores = torch.randn(3, N_VIDEOS)
    candidates = stage1.top_k_candidates(stage1_scores, 4)
    rerank = torch.randn(3, 4)
    full = stage1.rerank_scores_to_ranking(stage1_scores, candidates, rerank)
    for q in range(3):
        chosen = set(candidates[q].tolist())
        others = [i for i in range(N_VIDEOS) if i not in chosen]
        assert full[q, candidates[q]].min() > full[q, others].max()
    # and within the shortlist, order follows the reranker
    assert full[0, candidates[0]].argmax() == rerank[0].argmax()


def test_reranking_preserves_stage1_order_outside_the_shortlist():
    stage1_scores = torch.tensor([[0.5, 0.1, 0.9, 0.3]])
    candidates = torch.tensor([[2]])
    full = stage1.rerank_scores_to_ranking(stage1_scores, candidates, torch.tensor([[0.0]]))
    assert full[0, 0] > full[0, 3] > full[0, 1]


def test_per_candidate_gate_predicts_weights_per_candidate():
    model = per_candidate.build(NB, hidden_dim=32).eval()
    query = F.normalize(torch.randn(3, D), dim=1)
    feats = torch.randn(3, 7, NB)
    with torch.no_grad():
        weights = model(query, feats)
    assert weights.shape == (3, 7, NB)
    assert torch.allclose(weights.sum(-1), torch.ones(3, 7), atol=1e-5)
    # weights genuinely vary across candidates, which is the point of the model
    assert weights.std(dim=1).max() > 0


def test_per_candidate_accepts_the_legacy_positional_call():
    """experiments/ablations/run_ablation.py passes one tensor per branch."""
    model = per_candidate.build(3, hidden_dim=32).eval()
    query = torch.randn(2, D)
    sims = [torch.randn(2, 5) for _ in range(3)]
    with torch.no_grad():
        stacked = model(query, torch.stack(sims, dim=-1))
        positional = model(query, *sims)
    assert torch.allclose(stacked, positional)


def test_per_candidate_rejects_the_wrong_branch_count():
    model = per_candidate.build(NB, hidden_dim=32).eval()
    with pytest.raises(ValueError, match="branches"):
        model(torch.randn(2, D), torch.randn(2, 5, NB - 1))


def test_per_candidate_score_matches_manual_weighting():
    model = per_candidate.build(NB, hidden_dim=32).eval()
    query = torch.randn(2, D)
    feats = torch.randn(2, 6, NB)
    with torch.no_grad():
        scores = per_candidate.score(model, query, feats)
        weights = model(query, feats)
    assert scores.shape == (2, 6)
    assert torch.allclose(scores, (weights * feats).sum(-1), atol=1e-6)


def test_per_candidate_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = per_candidate.build(NB, hidden_dim=64).eval()
    path = tmp_path / "pc.pth"
    torch.save(model.state_dict(), path)
    loaded = per_candidate.load(path, "cpu", n_branches=NB)
    query, feats = torch.randn(2, D), torch.randn(2, 5, NB)
    with torch.no_grad():
        assert torch.allclose(loaded(query, feats), model(query, feats))


def test_per_candidate_load_rejects_a_mismatched_branch_count(tmp_path):
    model = per_candidate.build(3, hidden_dim=64).eval()
    path = tmp_path / "pc3.pth"
    torch.save(model.state_dict(), path)
    with pytest.raises(RuntimeError, match="branches"):
        per_candidate.load(path, "cpu", n_branches=4)


def test_fuse_accepts_per_query_and_shared_weights():
    sims = branch_sims(n_queries=3)
    shared = stage1.fuse(torch.tensor([1.0, 0.0, 0.0, 0.0]), sims)
    assert torch.allclose(shared, sims[0])
    per_query = stage1.fuse(torch.ones(3, NB), sims)
    assert torch.allclose(per_query, sum(sims))
