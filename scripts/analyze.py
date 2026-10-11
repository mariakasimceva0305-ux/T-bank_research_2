"""Tables, Pareto fronts and the decomposition that explains the difference."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from msteer.concepts import DEV_CONCEPTS, HELDOUT_CONCEPTS, by_name  # noqa: E402
from msteer.denoiser import make_intervention  # noqa: E402
from msteer.generate import PROMPTS  # noqa: E402
from msteer.hooks import LAYER, capture, intervene, load  # noqa: E402
from msteer.metrics import pareto_front  # noqa: E402
from msteer.vectors import activation_scale, concept_vector  # noqa: E402
from scripts.train_denoiser import load_denoiser  # noqa: E402

RESULTS = Path("results")
FIGS = RESULTS / "figures"
MODE_LABEL = {"steer": "steering",
              "denoise": "denoiser",
              "denoise_orth": "denoiser, orthogonal correction",
              "shrink_g0.4": "linear shrinkage, gamma=0.4",
              "shrink_g0.7": "linear shrinkage, gamma=0.7"}


def rows() -> list[dict]:
    return [json.loads(line) for line in
            (RESULTS / "pareto.jsonl").read_text(encoding="utf-8").splitlines()
            if line]


def _mean_curve(data: list[dict], names: set[str], mode: str):
    alphas = sorted({r["alpha"] for r in data if r["mode"] == mode}
                    | ({0.0} if mode != "steer" else set()))
    concept, ppl = [], []
    for a in alphas:
        sel = [r for r in data if r["concept"] in names and r["alpha"] == a
               and (r["mode"] == mode or a == 0.0)]
        if not sel:
            concept.append(np.nan)
            ppl.append(np.nan)
            continue
        concept.append(float(np.mean([r["topic_similarity"] for r in sel])))
        ppl.append(float(np.mean([r["perplexity"] for r in sel])))
    return alphas, concept, ppl


def figure_pareto(data: list[dict]) -> None:
    groups = (("development", {c.name for c in DEV_CONCEPTS}),
              ("held out", {c.name for c in HELDOUT_CONCEPTS}))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (title, names) in zip(axes, groups):
        for mode in MODE_LABEL:
            _, concept, ppl = _mean_curve(data, names, mode)
            ax.plot(ppl, concept, marker="o", label=MODE_LABEL[mode])
        ax.set_xscale("log")
        ax.set_xlabel("perplexity of the continuation (clean model)")
        ax.set_title(f"{title} concepts", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("topic similarity of the continuation")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "pareto.png", dpi=160)


def figure_alpha(data: list[dict]) -> None:
    names = {r["concept"] for r in data}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode in MODE_LABEL:
        alphas, concept, ppl = _mean_curve(data, names, mode)
        axes[0].plot(alphas, concept, marker="o", label=MODE_LABEL[mode])
        axes[1].plot(alphas, ppl, marker="o", label=MODE_LABEL[mode])
    axes[0].set_ylabel("topic similarity")
    axes[1].set_ylabel("perplexity")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("steering strength alpha (units of mean activation norm)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "alpha.png", dpi=160)


def decomposition(out: dict) -> None:
    """How much of the denoiser's correction simply cancels the intervention.

    For a steered state x = h + alpha*v, the correction delta = D(x) - x is
    split into its component along v and the rest. A large negative parallel
    component means the prior is treating the injected direction itself as the
    anomaly.
    """
    model, tok = load()
    denoiser = load_denoiser()
    scale = activation_scale(model, tok, list(PROMPTS))
    batch = tok(list(PROMPTS), return_tensors="pt", padding=True)
    mask = batch["attention_mask"].to(torch.bool)

    table = {}
    for concept in DEV_CONCEPTS:
        v = concept_vector(model, tok, concept)
        for alpha in (0.0, 1.0, 2.0, 4.0):
            store: list[torch.Tensor] = []
            fn = make_intervention(v, alpha, scale, None, "steer")
            # The capture hook must be registered after the intervention hook:
            # forward hooks fire in registration order, so the reverse nesting
            # records the state before the intervention and every alpha looks
            # identical.
            with torch.no_grad(), intervene(model, fn), capture(model, store, LAYER):
                model(**batch)
            x = store[0][mask]
            delta = denoiser(x) - x
            parallel = (delta @ v)
            key = f"{concept.name}|alpha={alpha:g}"
            table[key] = {
                "parallel_mean": float(parallel.mean()),
                "parallel_over_injected": float(parallel.mean()
                                                / (alpha * scale)) if alpha else 0.0,
                "orthogonal_norm": float(
                    (delta - parallel.unsqueeze(-1) * v).norm(dim=-1).mean()),
                "delta_norm": float(delta.norm(dim=-1).mean()),
            }
    out["decomposition"] = table
    for key, value in table.items():
        print(f"{key:22s} parallel/injected={value['parallel_over_injected']:+.3f} "
              f"|delta|={value['delta_norm']:7.2f} "
              f"|orth|={value['orthogonal_norm']:7.2f}")


def table_markdown(data: list[dict]) -> str:
    lines = ["| concepts | mode | alpha | topic similarity | seed words/100 "
             "| probe words/100 | perplexity | distinct-2 |",
             "|---|---|---|---|---|---|---|---|"]
    for title, names in (("development", {c.name for c in DEV_CONCEPTS}),
                         ("held out", {c.name for c in HELDOUT_CONCEPTS})):
        for mode in MODE_LABEL:
            for a in sorted({r["alpha"] for r in data}):
                sel = [r for r in data if r["concept"] in names
                       and r["mode"] == mode and r["alpha"] == a]
                if not sel:
                    continue
                lines.append(
                    f"| {title} | {MODE_LABEL[mode]} | {a:g} | "
                    f"{np.mean([r['topic_similarity'] for r in sel]):.3f} | "
                    f"{np.mean([r['seed_rate'] for r in sel]):.2f} | "
                    f"{np.mean([r['concept_rate'] for r in sel]):.2f} | "
                    f"{np.mean([r['perplexity'] for r in sel]):.1f} | "
                    f"{np.mean([r['distinct_2'] for r in sel]):.3f} |")
    return "\n".join(lines)


def main() -> None:
    global FIGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoiser-seed", type=int, default=0)
    parser.add_argument("--decomposition", action="store_true")
    args = parser.parse_args()
    output = RESULTS / f"analysis_denoiser_{args.denoiser_seed}"
    FIGS = output / "figures"
    FIGS.mkdir(parents=True, exist_ok=True)
    data = [r for r in rows() if r.get("denoiser_seed", 0) == args.denoiser_seed]
    if not data:
        raise ValueError("No rows for selected denoiser")
    summary: dict = {}
    text = table_markdown(data)
    (output / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    figure_pareto(data)
    figure_alpha(data)
    if args.decomposition:
        if args.denoiser_seed != 0:
            raise ValueError("Decomposition currently loads base checkpoint only")
        decomposition(summary)
    (output / "analysis.json").write_text(json.dumps(summary, indent=2))
    front = pareto_front([(r["topic_similarity"], r["perplexity"]) for r in data])
    print(f"\nnon-dominated configurations: "
          f"{sorted({(data[i]['mode'], data[i]['alpha']) for i in front})}")


if __name__ == "__main__":
    main()
