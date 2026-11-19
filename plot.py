"""Plot training and validation MSE for the Weather comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figure, (train, validation) = plt.subplots(
        1, 2, figsize=(9, 3.8), layout="constrained"
    )
    figure.suptitle("PatchTST on Weather · 336 input, 96 forecast steps")
    for variant, label, color, line in (
        ("baseline", "Original PatchTST", "#175d95", "-"),
        ("cst", "CST FFN output", "#ad470c", "--"),
    ):
        record = json.loads((args.run / variant / "metrics.json").read_text())
        run = record["training"]
        rows = run["history"]
        epochs = [r["epoch"] for r in rows]
        if epochs != list(range(1, record["epochs"] + 1)):
            raise ValueError(f"Incomplete epoch history for {variant}")
        style = dict(label=label, color=color, linestyle=line)
        train.plot(epochs, [r["train_mse"] for r in rows], **style)
        values = [r["validation"]["mse"] for r in rows]
        validation.plot(epochs, values, **style)
        best = run["best_epoch"]
        validation.scatter(best, values[best - 1], color=color, s=24)

    train.set_ylabel("Training MSE")
    validation.set_ylabel("Validation MSE")
    for axis in (train, validation):
        axis.set(xlabel="Epoch")
        axis.grid(axis="y", color="#d9e1e7", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    train.legend(frameon=False)
    figure.savefig(args.output, dpi=190, facecolor="white")


if __name__ == "__main__":
    main()
