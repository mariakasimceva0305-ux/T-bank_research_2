"""A cheap prior over residual-stream states, and how it is applied.

Training. The denoiser sees natural activations only. It never sees a steering
vector, so nothing about the evaluation directions can leak into it:

    h_noisy = t * h + (1 - t) * eps,   t ~ U[0, 1],  eps ~ N(0, I)
    L = || h - D(h_noisy) ||^2

in a space standardised by the training mean and standard deviation. The
interpolation form, rather than h + eps, makes the corruption level cover the
whole range from clean to pure noise with a bounded input scale, which matters
because a steered state h + alpha*v is far outside the range that a fixed
sigma would produce.

Application. Two variants are implemented and compared.

    denoise      h~ = D(x)
    denoise_orth h~ = x + P_perp_v [ D(x) - x ],   x = h + alpha*v

The second was proposed on the following argument: a prior fitted to natural
activations treats the injected component itself as the anomaly, so much of
D(x) - x points along -v and undoes the intervention; keeping only the
orthogonal part of the correction should preserve the steering at full strength
while removing the artefacts. The measurement confirms the premise -- the
denoiser cancels 52 to 92 per cent of what was injected -- and refutes the
conclusion: denoise_orth is worse on both axes. The cancellation is the
mechanism, not a defect. Holding the injection at full strength keeps the state
off-manifold and the text collapses into repetition.

A third mode, ``shrink``, is the control for whether a learned prior is needed
at all; see make_shrinkage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DenoiserConfig:
    d_model: int = 768
    d_hidden: int = 2048
    n_blocks: int = 2


class ResidualBlock(nn.Module):
    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_hidden, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class Denoiser(nn.Module):
    """Operates in standardised activation space; mean and scale are buffers,
    so a checkpoint is self-contained."""

    def __init__(self, cfg: DenoiserConfig | None = None,
                 mean: torch.Tensor | None = None,
                 std: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg or DenoiserConfig()
        self.blocks = nn.ModuleList([ResidualBlock(self.cfg)
                                     for _ in range(self.cfg.n_blocks)])
        self.out = nn.Linear(self.cfg.d_model, self.cfg.d_model)
        self.register_buffer("mean", torch.zeros(self.cfg.d_model)
                             if mean is None else mean)
        self.register_buffer("std", torch.ones(self.cfg.d_model)
                             if std is None else std)

    def standardise(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.mean) / self.std

    def restore(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    def forward_standardised(self, z: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            z = block(z)
        return self.out(z)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.restore(self.forward_standardised(self.standardise(h)))


def corrupt(z: torch.Tensor, generator: torch.Generator | None = None
            ) -> torch.Tensor:
    """t * z + (1 - t) * eps with a per-example t ~ U[0, 1]."""
    shape = (z.shape[0],) + (1,) * (z.dim() - 1)
    t = torch.rand(shape, device=z.device, generator=generator)
    eps = torch.randn(z.shape, device=z.device, generator=generator)
    return t * z + (1 - t) * eps


def make_shrinkage(mean: torch.Tensor, gamma: float):
    """Linear contraction of the state towards the mean of natural activations.

    The control for the claim that the denoiser does anything a learned prior
    is needed for. Steering inflates the norm of the state; if a one-line
    contraction towards the centre of the activation distribution reproduced
    the denoiser's front, the 6.9M-parameter prior would be redundant. Sweeping
    alpha does not answer this: alpha changes the size of the injection before
    it happens, while a contraction acts afterwards and also shrinks the
    natural part of h.
    """

    def shrink(x: torch.Tensor) -> torch.Tensor:
        mu = mean.to(x.dtype).to(x.device)
        return mu + gamma * (x - mu)

    return shrink


def make_intervention(v: torch.Tensor, alpha: float, scale: float,
                      denoiser: Denoiser | None, mode: str,
                      shrink_gamma: float = 0.5):
    """Returns the function applied to the residual stream at the hook point.

    ``alpha`` is measured in units of the mean activation norm, so the injected
    vector has norm alpha * scale.
    """
    step = (alpha * scale) * v

    def steer(h: torch.Tensor) -> torch.Tensor:
        return h + step.to(h.dtype).to(h.device)

    if mode == "steer" or alpha == 0.0 or denoiser is None:
        return steer

    def denoise(h: torch.Tensor) -> torch.Tensor:
        x = steer(h)
        return denoiser(x).to(h.dtype)

    def denoise_orth(h: torch.Tensor) -> torch.Tensor:
        x = steer(h)
        delta = denoiser(x).to(h.dtype) - x
        u = v.to(h.dtype).to(h.device)
        parallel = (delta @ u).unsqueeze(-1) * u
        return x + (delta - parallel)

    def shrink_only(h: torch.Tensor) -> torch.Tensor:
        return make_shrinkage(denoiser.mean, shrink_gamma)(steer(h))

    if mode == "denoise":
        return denoise
    if mode == "denoise_orth":
        return denoise_orth
    if mode == "shrink":
        return shrink_only
    raise ValueError(f"unknown mode: {mode}")


MODES = ("steer", "denoise", "denoise_orth", "shrink")
