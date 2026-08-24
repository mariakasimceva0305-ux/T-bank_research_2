"""Trains the denoiser on natural activations."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from msteer.denoiser import Denoiser, DenoiserConfig, corrupt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="results/activations.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/denoiser.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    blob = torch.load(args.activations, weights_only=True)
    train, val = blob["train"], blob["val"]
    mean, std = train.mean(0), train.std(0).clamp_min(1e-6)

    model = Denoiser(DenoiserConfig(d_model=train.shape[1]), mean, std)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200)
        * 0.5 * (1 + math.cos(math.pi * s / args.steps)))

    z_train = model.standardise(train)
    z_val = model.standardise(val)
    gen = torch.Generator().manual_seed(args.seed)
    history, t0 = [], time.time()

    for step in range(args.steps):
        model.train()
        idx = torch.randint(0, z_train.shape[0], (args.batch_size,), generator=gen)
        z = z_train[idx]
        loss = F.mse_loss(model.forward_standardised(corrupt(z, gen)), z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % 500 == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                zv = z_val[:8192]
                vloss = F.mse_loss(model.forward_standardised(corrupt(zv, gen)), zv)
                identity = F.mse_loss(model.forward_standardised(zv), zv)
            history.append(dict(step=step + 1, train=float(loss),
                                val=float(vloss), val_clean=float(identity)))
            print(f"step {step+1:5d}  train {float(loss):.4f}  "
                  f"val {float(vloss):.4f}  val_on_clean {float(identity):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "cfg": vars(model.cfg)}, path)
    path.with_suffix(".json").write_text(json.dumps(
        dict(vars(args), n_parameters=sum(p.numel() for p in model.parameters()),
             history=history), indent=2))
    print(f"saved {path}")


def load_denoiser(path: str = "checkpoints/denoiser.pt") -> Denoiser:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = Denoiser(DenoiserConfig(**blob["cfg"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


if __name__ == "__main__":
    main()
