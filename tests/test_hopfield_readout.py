import torch

from hei_nw.store import HopfieldReadout


def test_inference_is_read_only() -> None:
    patterns = torch.randn(4, 3)
    hopfield = HopfieldReadout(patterns, steps=2)
    before = {k: v.clone() for k, v in hopfield.state_dict().items()}
    cue = torch.randn(3)
    _ = hopfield(cue)
    after = hopfield.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])
    assert all(not p.requires_grad for p in hopfield.parameters())


def test_refinement_changes_query() -> None:
    patterns = torch.eye(2)
    hopfield = HopfieldReadout(patterns, steps=1)
    cue = torch.tensor([1.0, 0.1])
    refined = hopfield(cue)
    assert not torch.allclose(refined, cue)
