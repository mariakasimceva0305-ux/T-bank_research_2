"""Batched generation under an intervention, plus the neutral prompt set."""

from __future__ import annotations

import torch

from .hooks import intervene

PROMPTS = (
    "The following morning,",
    "In the years that followed,",
    "She looked at the table and said",
    "It is worth noting that",
    "After a long pause, he replied",
    "The committee decided that",
    "Once the noise had died down,",
    "By the end of the week,",
    "There was no obvious reason why",
    "Later that evening, the group",
    "The letter began with",
    "Nobody had expected that",
    "On the third day,",
    "The question that remained was",
    "Standing near the window, she",
    "Everything changed when",
)


@torch.no_grad()
def generate(model, tok, prompts: list[str], fn, max_new_tokens: int = 32,
             device: str = "cpu", seed: int = 0,
             temperature: float = 1.0, top_p: float = 0.9) -> list[str]:
    """Returns continuations only, with the prompt stripped."""
    torch.manual_seed(seed)
    batch = tok(list(prompts), return_tensors="pt", padding=True).to(device)
    with intervene(model, fn):
        out = model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tok.eos_token_id)
    new = out[:, batch["input_ids"].shape[1]:]
    return [tok.decode(row, skip_special_tokens=True) for row in new]
