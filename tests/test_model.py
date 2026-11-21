"""Shape / smoke tests for the model — no training, no real data."""

from __future__ import annotations

import torch

from src.model import Creb1CNN


def test_forward_shape_default_length():
    model = Creb1CNN()
    out = model(torch.zeros(2, 4, 500))
    assert out.shape == (2,)


def test_forward_shape_other_length():
    """The AdaptiveAvgPool head must make the model length-agnostic."""
    model = Creb1CNN()
    out = model(torch.zeros(3, 4, 300))
    assert out.shape == (3,)


def test_output_is_logits_not_probs():
    """Random inputs should produce values outside [0, 1] in at least one dim."""
    torch.manual_seed(0)
    model = Creb1CNN()
    out = model(torch.randn(16, 4, 500))
    # logits are unbounded; if these were sigmoid outputs, all would be in [0,1]
    assert (out < 0).any() or (out > 1).any()