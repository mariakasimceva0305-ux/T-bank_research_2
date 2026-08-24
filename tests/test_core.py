"""Invariants the experiment depends on. Run with: pytest -q"""

from __future__ import annotations

import pytest
import torch

from msteer.concepts import CONCEPTS, DEV_CONCEPTS, HELDOUT_CONCEPTS
from msteer.denoiser import Denoiser, DenoiserConfig, corrupt, make_intervention
from msteer.metrics import concept_rate, distinct_n, pareto_front


def test_seed_and_probe_lexicons_are_disjoint():
    """If the words the direction is built from were also counted by the
    metric, the metric would score the objective the direction was fitted to."""
    seeds = {w for c in CONCEPTS for w in c.seed}
    probes = {w for c in CONCEPTS for w in c.probe}
    assert not seeds & probes


def test_concept_split_is_a_partition():
    assert not {c.name for c in DEV_CONCEPTS} & {c.name for c in HELDOUT_CONCEPTS}
    assert len(DEV_CONCEPTS) + len(HELDOUT_CONCEPTS) == len(CONCEPTS)


def test_corruption_covers_the_range():
    z = torch.randn(4096, 16)
    gen = torch.Generator().manual_seed(0)
    noisy = corrupt(z, gen)
    per_example = (noisy - z).norm(dim=-1)
    assert per_example.min() < 0.5 * per_example.max()


def _denoiser(d: int = 8) -> Denoiser:
    torch.manual_seed(0)
    return Denoiser(DenoiserConfig(d_model=d, d_hidden=16, n_blocks=1)).eval()


def test_orthogonal_variant_preserves_the_injected_component():
    """The defining property of the method: whatever the denoiser does, the
    component along v must be exactly what plain steering would have produced."""
    d = 8
    torch.manual_seed(0)
    v = torch.randn(d)
    v = v / v.norm()
    h = torch.randn(3, d)
    den = _denoiser(d)
    alpha, scale = 2.0, 1.5
    plain = make_intervention(v, alpha, scale, den, "steer")(h)
    orth = make_intervention(v, alpha, scale, den, "denoise_orth")(h)
    assert torch.allclose(plain @ v, orth @ v, atol=1e-4)
    naive = make_intervention(v, alpha, scale, den, "denoise")(h)
    assert not torch.allclose(plain @ v, naive @ v, atol=1e-4)


def test_zero_alpha_is_the_identity_for_every_mode():
    d = 8
    v = torch.zeros(d)
    v[0] = 1.0
    h = torch.randn(3, d)
    for mode in ("steer", "denoise", "denoise_orth"):
        fn = make_intervention(v, 0.0, 1.0, _denoiser(d), mode)
        assert torch.allclose(fn(h), h)


def test_metrics_behave():
    assert concept_rate(["the market fell"], ("market",)) == pytest.approx(100 / 3)
    assert distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)


def test_pareto_front_keeps_only_undominated():
    pts = [(1.0, 10.0), (2.0, 20.0), (1.5, 30.0)]
    assert pareto_front(pts) == [0, 1]
