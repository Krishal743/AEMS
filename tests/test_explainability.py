import torch

from src.explainability.explain_retrieval import (explain_gating_decision,
                                                  explain_modality_contributions,
                                                  format_explanation, label)
from src.routing.query_router import BRANCHES

N = 5


def test_every_branch_has_a_display_name():
    """A branch added without a LABELS entry must not break the output."""
    for branch in BRANCHES:
        assert isinstance(label(branch), str) and label(branch)
    assert label("a_brand_new_branch") == "a_brand_new_branch"


def test_format_explanation_covers_all_branches():
    torch.manual_seed(0)
    weights = torch.full((len(BRANCHES),), 1.0 / len(BRANCHES))
    sims = [torch.randn(N) for _ in BRANCHES]
    ids = [f"v{i}" for i in range(N)]
    text = format_explanation(explain_modality_contributions(weights, sims, ids, top_k=2),
                              explain_gating_decision(weights), top_k=2)
    for branch in BRANCHES:
        assert label(branch) in text
    assert "Rank1 vs Rank2" in text


def test_contributions_sum_to_the_fused_score():
    weights = torch.rand(len(BRANCHES))
    sims = [torch.randn(N) for _ in BRANCHES]
    top = explain_modality_contributions(weights, sims, [f"v{i}" for i in range(N)], top_k=1)[0]
    parts = sum(top[f"{b}_score"] for b in BRANCHES)
    assert abs(parts - top["fused_score"]) < 1e-5
