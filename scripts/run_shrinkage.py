"""Control: does a linear contraction reproduce what the denoiser does?

The denoiser's measured action on a steered state is an attenuation of the
injected component plus a correction orthogonal to it. If a one-line
contraction of the state towards the mean of natural activations,
``x <- mu + gamma (x - mu)``, reproduced its Pareto front, the learned prior
would be redundant and the finding would reduce to "steering inflates the norm
and any contraction repairs it".

Only the region where the denoiser wins is swept, over two contraction
strengths, so the control is answered at the points the claim rests on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msteer.concepts import CONCEPTS, by_name
from msteer.denoiser import make_intervention
from msteer.generate import PROMPTS, generate
from msteer.hooks import load
from msteer.metrics import (concept_rate, continuation_perplexity,
                            distinct_n, topic_similarity)
from msteer.vectors import activation_scale, concept_vector
from scripts.train_denoiser import load_denoiser

RESULTS = Path("results")
ALPHAS = (0.6, 1.0, 1.5)
GAMMAS = (0.4, 0.7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--out", default="results/pareto.jsonl")
    args = ap.parse_args()

    model, tok = load()
    denoiser = load_denoiser()
    scale = activation_scale(model, tok, list(PROMPTS))
    out = Path(args.out).open("a", encoding="utf-8")

    for concept in CONCEPTS:
        v = concept_vector(model, tok, concept)
        for gamma in GAMMAS:
            for alpha in ALPHAS:
                fn = make_intervention(v, alpha, scale, denoiser, "shrink",
                                       shrink_gamma=gamma)
                texts, prompts = [], []
                for r in range(args.repeats):
                    texts += generate(model, tok, list(PROMPTS), fn,
                                      max_new_tokens=args.max_new_tokens,
                                      seed=1000 + r)
                    prompts += list(PROMPTS)
                row = dict(
                    concept=concept.name, mode=f"shrink_g{gamma:g}", alpha=alpha,
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
                print(f"{concept.name:9s} gamma={gamma:g} a={alpha:4.1f} "
                      f"topic={row['topic_similarity']:5.3f} "
                      f"probe={row['concept_rate']:5.2f} "
                      f"ppl={row['perplexity']:8.1f} "
                      f"d2={row['distinct_2']:.3f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
