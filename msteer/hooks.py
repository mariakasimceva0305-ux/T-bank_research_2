"""GPT-2 with a single intervention point after the sixth block.

Every experiment writes into the residual stream at the same place, so the only
thing that varies between conditions is the function applied there.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
LAYER = 6           # intervene on the output of block 6 (of 12)


def load(device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@contextmanager
def intervene(model, fn: Callable[[torch.Tensor], torch.Tensor] | None,
              layer: int = LAYER):
    """Applies ``fn`` to the hidden states leaving ``layer``.

    GPT-2 blocks return a tuple whose first element is the residual stream.
    ``fn`` receives (batch, seq, d_model) and must return the same shape.
    """
    if fn is None:
        yield
        return

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (fn(output[0]),) + output[1:]
        return fn(output)

    handle = model.transformer.h[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture(model, store: list, layer: int = LAYER):
    """Collects the hidden states leaving ``layer`` without modifying them."""

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        store.append(h.detach().to(torch.float32).cpu())

    handle = model.transformer.h[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
