"""Read the PatchTST Weather benchmark with chronological target splits."""

import csv
import hashlib

import numpy as np
import torch

from model import CHANNELS, CONTEXT, HORIZON


class Weather:
    def __init__(self, path, device):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            rows = list(reader)
        timestamps = np.asarray([row[0] for row in rows], dtype="datetime64[m]")
        values = np.asarray([row[1:] for row in rows], dtype=np.float64)
        differences = np.diff(timestamps).astype(np.int64)
        if (
            values.shape != (52_696, CHANNELS)
            or not np.isfinite(values).all()
            or (differences < 0).any()
        ):
            raise ValueError("Expected the 52,696-row PatchTST Weather benchmark")

        count = len(values)
        uncommon, frequencies = np.unique(differences[differences != 10], return_counts=True)
        train_end = int(count * 0.7)
        validation_end = count - int(count * 0.2)
        mean = values[:train_end].mean(axis=0)
        std = values[:train_end].std(axis=0)
        std = np.where(std == 0, 1, std)
        self.values = torch.tensor((values - mean) / std, dtype=torch.float32, device=device)
        self.offsets = torch.arange(-CONTEXT, HORIZON, device=device)
        self.splits = tuple(
            torch.arange(start, stop - HORIZON + 1, device=device)
            for start, stop in (
                (CONTEXT, train_end), (train_end, validation_end), (validation_end, count)
            )
        )
        self.config = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "channels": header[1:],
            "context": CONTEXT, "horizon": HORIZON,
            "split_window_counts": [len(split) for split in self.splits],
            "split_timestamps": [str(timestamps[index]) for index in (0, train_end, validation_end, count - 1)],
            "nonstandard_intervals": {str(gap): int(n) for gap, n in zip(uncommon, frequencies)},
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    def batch(self, origins):
        block = self.values[origins[:, None] + self.offsets].transpose(1, 2).contiguous()
        return block[..., :CONTEXT], block[..., CONTEXT:]
