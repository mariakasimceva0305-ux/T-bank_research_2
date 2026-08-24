"""Two axes: how much of the concept is present, and how intact the text is.

Concept presence is counted with the probe lexicon, which is disjoint from the
seed words the steering direction was built from.

Fluency is the perplexity of the generated continuation under the *unmodified*
model. Scoring the continuation under the steered model would be circular: the
intervention changes the distribution the perplexity is measured against, and a
strongly steered model finds its own degenerate output perfectly predictable.
"""

from __future__ import annotations

import re

import torch

WORD = re.compile(r"[a-z']+")

_encoder = None


def _sentence_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _encoder


def topic_similarity(texts: list[str], topic: str) -> float:
    """Mean cosine similarity between each generation and a topic sentence.

    Independent of the model being steered and of the words the steering
    direction was built from, so it measures whether the text is on topic
    rather than whether it repeats the seed tokens.
    """
    enc = _sentence_encoder()
    vecs = enc.encode(list(texts) + [topic], convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
    return float((vecs[:-1] @ vecs[-1]).mean())


def concept_rate(texts: list[str], probe: tuple[str, ...]) -> float:
    """Probe-lexicon words per 100 generated words."""
    lexicon = set(probe)
    hits = total = 0
    for text in texts:
        words = WORD.findall(text.lower())
        total += len(words)
        hits += sum(w in lexicon for w in words)
    return 100.0 * hits / max(total, 1)


def distinct_n(texts: list[str], n: int = 2) -> float:
    grams, seen = 0, set()
    for text in texts:
        words = WORD.findall(text.lower())
        for i in range(len(words) - n + 1):
            seen.add(tuple(words[i: i + n]))
            grams += 1
    return len(seen) / max(grams, 1)


@torch.no_grad()
def continuation_perplexity(model, tok, prompts: list[str],
                            continuations: list[str], device: str = "cpu",
                            batch_size: int = 16) -> float:
    """Perplexity of the continuation tokens only, under the clean model."""
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(prompts), batch_size):
        chunk_p = prompts[i: i + batch_size]
        chunk_c = continuations[i: i + batch_size]
        for prompt, cont in zip(chunk_p, chunk_c):
            p_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            full = tok(prompt + cont, return_tensors="pt").input_ids.to(device)
            if full.shape[1] <= p_ids.shape[1]:
                continue
            logits = model(full).logits[:, :-1]
            targets = full[:, 1:]
            start = p_ids.shape[1] - 1
            lp = torch.log_softmax(logits[:, start:], dim=-1)
            picked = lp.gather(-1, targets[:, start:].unsqueeze(-1)).squeeze(-1)
            total_nll += float(-picked.sum())
            total_tokens += picked.numel()
    if total_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated on (concept up, perplexity down)."""
    keep = []
    for i, (c_i, p_i) in enumerate(points):
        dominated = any(c_j >= c_i and p_j <= p_i and (c_j, p_j) != (c_i, p_i)
                        for c_j, p_j in points)
        if not dominated:
            keep.append(i)
    return keep
