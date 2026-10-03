import torch

from src.config import AEMS_FUSION_WEIGHTS, AEMS_MODALITY_WEIGHTS
from src.evaluation.multimodal_queries import fuse
from src.routing.query_router import BRANCHES, fixed_weights, modality_weights


def test_text_path_is_unchanged_by_the_modality_table():
    """The deployed text weights were tuned on the full gallery; nothing may shadow them."""
    assert modality_weights() == AEMS_FUSION_WEIGHTS
    assert modality_weights("text") == AEMS_FUSION_WEIGHTS
    assert "text" not in AEMS_MODALITY_WEIGHTS


def test_an_unknown_modality_falls_back_rather_than_raising():
    assert modality_weights("hologram") == AEMS_FUSION_WEIGHTS


def test_image_queries_lean_harder_on_the_visual_branch():
    assert modality_weights("image")["visual"] > AEMS_FUSION_WEIGHTS["visual"]


def test_audio_queries_do_not_pay_attention_to_clip_branches():
    """Tuning drove these to zero: they are near-chance for audio and act as noise."""
    w = modality_weights("audio_clip")
    assert w["visual"] == w["text"] == w["chunk"] == 0.0
    assert w["audio"] > 0


def test_every_modality_entry_covers_every_branch():
    for modality, w in AEMS_MODALITY_WEIGHTS.items():
        assert set(w) == set(BRANCHES), f"{modality} is missing {set(BRANCHES) - set(w)}"


def test_fixed_weights_orders_values_by_branch():
    w = fixed_weights(modality="image")
    assert torch.allclose(w, torch.tensor([float(modality_weights("image")[b])
                                           for b in BRANCHES]))


def test_fuse_ignores_unreachable_branches_and_renormalizes():
    """An unreachable branch must not silently absorb weight and dilute the rest."""
    sims = [None] * len(BRANCHES)
    sims[BRANCHES.index("audio")] = torch.tensor([[1.0, 0.0, -1.0]])
    weights = {b: 1.0 for b in BRANCHES}
    fused = fuse(sims, weights, "cpu")
    assert torch.allclose(fused, torch.tensor([[1.0, 0.0, -1.0]]))


def test_fuse_falls_back_when_every_reachable_branch_has_zero_weight():
    """An audio-only query under weights that zero the audio branch must still rank."""
    sims = [None] * len(BRANCHES)
    sims[BRANCHES.index("visual")] = torch.tensor([[1.0, 0.0]])
    fused = fuse(sims, {b: 0.0 for b in BRANCHES}, "cpu")
    assert torch.allclose(fused, torch.tensor([[1.0, 0.0]]))


def test_fuse_rejects_a_query_that_reached_nothing():
    try:
        fuse([None] * len(BRANCHES), AEMS_FUSION_WEIGHTS, "cpu")
    except ValueError as exc:
        assert "reached no branch" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
