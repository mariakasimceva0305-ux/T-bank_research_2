"""Concept presence against fluency, over a grid of steering strengths.

One generation set per (concept, mode, alpha). Prompts, sampling seed and
decoding parameters are shared across modes, so a difference between modes is a
difference in the intervention and not in the sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from msteer.concepts import CONCEPTS, by_name
from msteer.denoiser import MODES, make_intervention
from msteer.generate import PROMPTS, generate
from msteer.hooks import load
from msteer.metrics import (concept_rate, continuation_perplexity,
                            distinct_n, topic_similarity)
from msteer.vectors import activation_scale, concept_vector
from scripts.train_denoiser import load_denoiser

RESULTS = Path("results")
# The effect saturates by alpha ~ 1.5 and the text is already degenerate
# there; the grid is dense where the trade-off actually happens.
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--denoiser", default="checkpoints/denoiser.pt")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--out", default="results/pareto.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    model, tok = load()
    denoiser = load_denoiser(args.denoiser)
    scale = activation_scale(model, tok, list(PROMPTS))
    print(f"mean activation norm at the intervention layer: {scale:.2f}")

    names = ([c.name for c in CONCEPTS] if args.concepts == "all"
             else args.concepts.split(","))
    out = Path(args.out).open("a", encoding="utf-8")

    for name in names:
        concept = by_name(name)
        v = concept_vector(model, tok, concept)
        for mode in MODES:
            for alpha in ALPHAS:
                if alpha == 0.0 and mode != "steer":
                    continue          # identical to the unsteered baseline
                texts: list[str] = []
                prompts: list[str] = []
                fn = make_intervention(v, alpha, scale,
                                       None if mode == "steer" else denoiser,
                                       mode)
                for r in range(args.repeats):
                    texts += generate(model, tok, list(PROMPTS), fn,
                                      max_new_tokens=args.max_new_tokens,
                                      seed=1000 + r)
                    prompts += list(PROMPTS)
                row = dict(
                    concept=name, mode=mode, alpha=alpha,
                    topic_similarity=topic_similarity(texts, concept.topic),
                    concept_rate=concept_rate(texts, concept.probe),
                    seed_rate=concept_rate(texts, concept.seed),
                    perplexity=continuation_perplexity(model, tok, prompts, texts),
                    distinct_2=distinct_n(texts, 2),
                    n_samples=len(texts),
                    sample=texts[0].replace("\n", " ")[:160],
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{name:9s} {mode:13s} a={alpha:4.1f} "
                      f"topic={row['topic_similarity']:5.3f} "
                      f"seed={row['seed_rate']:5.2f} "
                      f"ppl={row['perplexity']:8.1f} "
                      f"d2={row['distinct_2']:.3f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
