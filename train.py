"""Train the Weather benchmark and compare the PatchTST FFN projection."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch import nn
from torchcst import CSTAdamW, CSTLinear

from data import Weather
from model import build_model, shared_sha256


UPSTREAM = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
TORCHCST = "1750c5ea6cbefe3d4a7d8f6ba02fd9272df3cf98"
LR = 1e-4


def make_optimizer(model, variant, atom_lr):
    if variant == "baseline":
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, foreach=True)
    atoms = [m.atoms.p for m in model.modules() if isinstance(m, CSTLinear)]
    atom_ids = {id(p) for p in atoms}
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in atom_ids]},
        {"params": atoms, "lr": atom_lr},
    ]
    return CSTAdamW(groups, model=model, lr=LR, weight_decay=0.0, foreach=True)


def step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.inference_mode()
def evaluate(model, data, origins, batch_size):
    model.eval()
    squared = torch.zeros((), device=data.values.device, dtype=torch.float64)
    absolute = squared.clone()
    total = 0
    for indices in origins.split(batch_size):
        inputs, targets = data.batch(indices)
        error = model(inputs) - targets
        squared += error.square().sum(dtype=torch.float64)
        absolute += error.abs().sum(dtype=torch.float64)
        total += targets.numel()
    metrics = {"mse": squared.item() / total, "mae": absolute.item() / total}
    if not all(map(isfinite, metrics.values())):
        raise ValueError("Non-finite evaluation")
    return metrics


def pilot(model, optimizer, data, order, args):
    train = data.splits[0]
    shuffled = train[torch.randperm(len(train), generator=order, device=train.device)]
    durations = []
    losses = []
    model.train()
    for i, indices in enumerate(shuffled.split(args.batch_size)[: args.pilot + 8]):
        inputs, targets = data.batch(indices)
        torch.cuda.synchronize()
        if i == 8:
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        loss = step(model, optimizer, inputs, targets)
        torch.cuda.synchronize()
        if i >= 8:
            durations.append((perf_counter() - start) * 1000)
        losses.append(loss)
    assert torch.stack(losses).isfinite().all()
    return {
        "updates": len(durations), "warmup_updates": 8,
        "median_ms": median(durations), "step_ms": durations,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "last_loss": losses[-1].item(),
    }


def train(model, optimizer, data, order, args):
    train_origins, validation, test = data.splits
    checkpoint = args.output / "best.pt"
    best = float("inf")
    best_epoch = None
    history = []
    rates = [group["lr"] for group in optimizer.param_groups]
    for epoch in range(1, args.epochs + 1):
        model.train()
        for group, initial in zip(optimizer.param_groups, rates):
            group["lr"] = initial * 0.9 ** max(0, epoch - 4)
        shuffled = train_origins[torch.randperm(len(train_origins), generator=order, device=train_origins.device)]
        error = data.values.new_zeros(())
        for indices in shuffled.split(args.batch_size):
            inputs, targets = data.batch(indices)
            error += step(model, optimizer, inputs, targets) * len(indices)
        train_mse = error.item() / len(train_origins)
        if not isfinite(train_mse):
            raise ValueError("Non-finite training loss")
        validation_metrics = evaluate(model, data, validation, args.batch_size)
        row = {"epoch": epoch, "train_mse": train_mse, "validation": validation_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_metrics["mse"] < best:
            best = validation_metrics["mse"]
            best_epoch = epoch
            pending = checkpoint.with_suffix(".tmp")
            torch.save(model.state_dict(), pending)
            pending.replace(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return {
        "history": history,
        "best_epoch": best_epoch,
        "test": evaluate(model, data, test, args.batch_size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "cst"), required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--atoms", default=512, type=int)
    parser.add_argument("--atom-lr", default=0.00048, type=float)
    parser.add_argument("--pilot", default=0, type=int)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.atoms) < 1 or args.pilot < 0:
        parser.error("Invalid epoch, batch size, Atom count or pilot length")
    if not isfinite(args.atom_lr) or args.atom_lr <= 0:
        parser.error("Atom learning rate must be finite and positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = Weather(args.data, "cuda")
    torch.manual_seed(args.seed)
    model = build_model(args.variant, args.atoms)
    digest = shared_sha256(model)
    model.cuda()
    optimizer = make_optimizer(model, args.variant, args.atom_lr)
    order = torch.Generator(device="cuda").manual_seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "variant": args.variant, "epochs": args.epochs, "batch_size": args.batch_size,
        "seed": args.seed, "atoms": args.atoms if args.variant == "cst" else None,
        "initial_learning_rates": [group["lr"] for group in optimizer.param_groups],
        "parameters": sum(p.numel() for p in model.parameters()),
        "shared_initial_sha256": digest, "dataset": data.config,
        "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "precision": "FP32, TF32 enabled",
        "upstream_commit": UPSTREAM, "torchcst_commit": TORCHCST,
    }
    start = perf_counter()
    action = pilot if args.pilot else train
    result["pilot" if args.pilot else "training"] = action(model, optimizer, data, order, args)
    result["wall_seconds"] = perf_counter() - start
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["pilot"] if args.pilot else result["training"]["test"]), flush=True)


if __name__ == "__main__":
    main()
