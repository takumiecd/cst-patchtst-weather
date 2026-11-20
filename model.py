"""The reference Weather PatchTST backbone with an optional CST FFN output."""

import hashlib

import torch
from torch import nn
from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, chart_presets, kernels

from upstream.layers.PatchTST_backbone import PatchTST_backbone, TSTEncoderLayer


CONTEXT = 336
HORIZON = 96
CHANNELS = 21


class CSTProjection(nn.Module):
    def __init__(self, dense, atoms):
        super().__init__()
        self.bias = dense.bias
        kernel = kernels.product(
            kernels.l2(TriweightSpec()),
            coordinates=kernels.polar(
                amplitude_max=2.0,
                w_c=0.3,
                bounds=BandwidthBounds(minimum=0.5, birth=3.5, maximum=6.0, upper_floor=0.5),
            ),
        )
        self.linear = CSTLinear(
            charts=(chart_presets.regular_grid((8, 16)), chart_presets.regular_grid((16, 16))),
            atoms=atoms, kernel=kernel, backend="materialized",
        )
        with torch.no_grad():
            # Preserve the Polar radius while matching the dense weight scale.
            target = dense.weight.square().mean().sqrt()
            current = self.linear.operator.weight().square().mean().sqrt()
            polar = self.linear.atoms.p[:, :2]
            radius = polar.norm(dim=1)
            angle = (polar[:, 0] / radius * (target / current)).clamp(-1 + 1e-6, 1 - 1e-6)
            polar[:, 0] = radius * angle
            polar[:, 1] = radius * (1 - angle.square()).sqrt()

    def forward(self, x):
        return self.linear(x) + self.bias


def build_model(variant, atoms):
    model = PatchTST_backbone(
        c_in=CHANNELS, context_window=CONTEXT, target_window=HORIZON, patch_len=16, stride=8,
        n_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2,
        head_dropout=0.0, padding_patch="end", revin=True, affine=False, subtract_last=False,
    )
    if variant == "cst":
        for block in list(model.modules()):
            if isinstance(block, TSTEncoderLayer):
                block.ff[-1] = CSTProjection(block.ff[-1], atoms)
    return model


def shared_sha256(model):
    digest = hashlib.sha256()
    for name, weight in model.named_parameters():
        if ".ff.3." in name and not name.endswith(".ff.3.bias"):
            continue
        digest.update(name.encode())
        digest.update(weight.detach().numpy().tobytes())
    return digest.hexdigest()
