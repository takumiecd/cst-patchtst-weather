"""Forecast the next 96 Weather measurements from a CSV window."""

import argparse
import csv
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torchcst.hub import from_pretrained, load_safetensors

from model import CONTEXT, HORIZON, build_model


HUB = "takumiecd/cst-patchtst-weather"


def read_weather(path, columns):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"date", *columns} <= set(reader.fieldnames or []):
            raise ValueError("Expected the date and all 21 Weather feature columns")
        rows = list(reader)
    times = np.asarray([row["date"] for row in rows], dtype="datetime64[m]")
    values = np.asarray([[float(row[name]) for name in columns] for row in rows])
    if len(rows) < CONTEXT or not np.isfinite(values).all() or np.isnat(times).any():
        raise ValueError("Expected at least 336 valid Weather rows")
    if (np.diff(times) < np.timedelta64(0, "m")).any():
        raise ValueError("Weather rows must be in chronological order")
    return times, values


@torch.inference_mode()
def forecast(model, config, times, values, origin, channel):
    data = config["dataset"]
    column = data["channels"].index(channel)
    mean, std = np.asarray(data["mean"]), np.asarray(data["std"])
    x = (values[origin - CONTEXT : origin] - mean) / std
    device = next(model.parameters()).device
    inputs = torch.tensor(x.T[None], dtype=torch.float32, device=device)
    prediction = model(inputs)[0, column].tolist()
    rows = []
    for step, value in enumerate(prediction):
        row = {"time": str(times[origin - 1] + np.timedelta64((step + 1) * 10, "m")),
               "prediction": value * std[column] + mean[column]}
        if origin + step < len(values):
            row["time"] = str(times[origin + step])
            row["observed"] = float(values[origin + step, column])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--variant", choices=("baseline", "cst"), default="cst")
    parser.add_argument("--channel", default="T (degC)")
    parser.add_argument("--test-index", type=int, help="Zero-based forecast index on the benchmark test split")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local", type=Path, help="Directory with config.json and Safetensors")
    args = parser.parse_args()
    config_path = args.local / "config.json" if args.local else Path(
        hf_hub_download(HUB, "config.json", revision=args.revision, token=False)
    )
    config = json.loads(config_path.read_text())
    times, values = read_weather(args.csv, config["dataset"]["channels"])
    if args.channel not in config["dataset"]["channels"]:
        parser.error("Unknown Weather channel")
    origin = len(values)
    if args.test_index is not None:
        actual_sha = hashlib.sha256(args.csv.read_bytes()).hexdigest()
        test_count = config["dataset"]["split_window_counts"][-1]
        if config["dataset"]["sha256"] != actual_sha or not 0 <= args.test_index < test_count:
            parser.error("--test-index requires the reference Weather CSV and a valid test index")
        origin = len(values) - test_count - HORIZON + 1 + args.test_index
    factory = partial(build_model, **config["variants"][args.variant])
    name = f"{args.variant}.safetensors"
    if args.local:
        model = load_safetensors(args.local / name, model_factory=factory, device=args.device)
    else:
        model = from_pretrained(
            HUB, filename=name, model_factory=factory, device=args.device,
            revision=args.revision, token=False,
        )
    rows = forecast(model, config, times, values, origin, args.channel)
    print(json.dumps({"channel": args.channel, "forecast": rows}, indent=2))


if __name__ == "__main__":
    main()
