"""Steering directions as a difference in means at the intervention layer.

For each concept the direction is

    v = normalise( mean over concept sentences - mean over neutral sentences )

taken over the residual stream at layer 6, averaged across token positions. The
two sentence sets share the same templates and differ only in the noun that
fills the slot, so the difference isolates the noun rather than the syntax.

SAE decoder columns are the other standard source of such directions. A
difference in means was chosen because it keeps the experiment self-contained
and because the question under study is what to do *after* a direction has been
injected, which does not depend on where the direction came from. The seed
words are disjoint from the words used to score the generations.
"""

from __future__ import annotations

import torch

from .concepts import Concept
from .hooks import LAYER, capture

TEMPLATES = (
    "The report mentioned the {} in some detail.",
    "She spent the afternoon thinking about the {}.",
    "According to the article, the {} was important.",
    "He wrote a long note about the {} yesterday.",
    "There was a short discussion of the {} at the meeting.",
    "The book devotes a whole chapter to the {}.",
)

NEUTRAL_WORDS = ("thing", "place", "person", "item", "object",
                 "matter", "area", "case")


def _mean_activation(model, tok, sentences: list[str], device: str,
                     layer: int = LAYER) -> torch.Tensor:
    batch = tok(sentences, return_tensors="pt", padding=True).to(device)
    store: list[torch.Tensor] = []
    with torch.no_grad(), capture(model, store, layer):
        model(**batch)
    h = store[0]                                   # (b, t, d)
    mask = batch["attention_mask"].to(torch.float32).cpu().unsqueeze(-1)
    return (h * mask).sum((0, 1)) / mask.sum()


def concept_vector(model, tok, concept: Concept, device: str = "cpu",
                   layer: int = LAYER) -> torch.Tensor:
    positive = [t.format(w) for t in TEMPLATES for w in concept.seed]
    negative = [t.format(w) for t in TEMPLATES for w in NEUTRAL_WORDS]
    v = (_mean_activation(model, tok, positive, device, layer)
         - _mean_activation(model, tok, negative, device, layer))
    return v / v.norm()


def activation_scale(model, tok, sentences: list[str], device: str = "cpu",
                     layer: int = LAYER) -> float:
    """Mean L2 norm of a residual-stream vector at the intervention layer.

    Steering strength is reported in these units, so that alpha means the same
    thing for every concept and is comparable to the norm of the state it is
    added to.
    """
    batch = tok(sentences, return_tensors="pt", padding=True).to(device)
    store: list[torch.Tensor] = []
    with torch.no_grad(), capture(model, store, layer):
        model(**batch)
    mask = batch["attention_mask"].to(torch.bool).cpu()
    return float(store[0][mask].norm(dim=-1).mean())
