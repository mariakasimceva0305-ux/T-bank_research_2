"""Collects natural residual-stream activations at the intervention layer.

These are the only data the denoiser ever sees. No steering vector is involved
at any point here, which is what makes the evaluation directions held out by
construction rather than by a split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from msteer.hooks import LAYER, capture, load

RESULTS = Path("results")


def texts(n: int) -> list[str]:
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    out = [t.strip() for t in data["text"] if len(t.strip()) > 200]
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_texts", type=int, default=1500)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="results/activations.pt")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    model, tok = load()
    corpus = texts(args.n_texts)
    chunks: list[torch.Tensor] = []
    for i in range(0, len(corpus), args.batch_size):
        batch = tok(corpus[i: i + args.batch_size], return_tensors="pt",
                    padding=True, truncation=True, max_length=args.max_len)
        store: list[torch.Tensor] = []
        with torch.no_grad(), capture(model, store, LAYER):
            model(**batch)
        mask = batch["attention_mask"].to(torch.bool)
        chunks.append(store[0][mask])
        if (i // args.batch_size) % 20 == 0:
            print(f"{i}/{len(corpus)} texts, "
                  f"{sum(c.shape[0] for c in chunks)} vectors", flush=True)

    acts = torch.cat(chunks)
    perm = torch.randperm(acts.shape[0], generator=torch.Generator().manual_seed(0))
    acts = acts[perm]
    n_val = min(20_000, acts.shape[0] // 10)
    torch.save({"train": acts[n_val:], "val": acts[:n_val], "layer": LAYER},
               args.out)
    print(f"saved {args.out}: train {acts.shape[0]-n_val}, val {n_val}, "
          f"d={acts.shape[1]}")


if __name__ == "__main__":
    main()
