# Adapted from:
# https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
# https://github.com/TheMrGhostman/InceptionTime-Pytorch
import torch
import torch.nn as nn

supervised = True
classification = True


class _Inception(nn.Module):
    """Single inception module: bottleneck --> 3 parallel convs + maxpool branch."""

    def __init__(self, in_channels, n_filters=32, kernel_sizes=(39, 19, 9),
                 bottleneck_channels=32):
        super().__init__()
        # bottleneck (skip if in_channels == 1)
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels,
                                        kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in = 1

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, n_filters, kernel_size=k,
                      padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_mp = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(4 * n_filters)
        self.act = nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(x) if self.bottleneck is not None else x
        branches = [conv(bottleneck) for conv in self.convs]
        branches.append(self.conv_mp(self.maxpool(x)))
        out = torch.cat(branches, dim=1)
        return self.act(self.bn(out))


class _InceptionBlock(nn.Module):
    """Three inception modules with a residual shortcut."""

    def __init__(self, in_channels, n_filters=32, kernel_sizes=(39, 19, 9),
                 bottleneck_channels=32):
        super().__init__()
        self.inception1 = _Inception(in_channels, n_filters, kernel_sizes, bottleneck_channels)
        self.inception2 = _Inception(4 * n_filters, n_filters, kernel_sizes, bottleneck_channels)
        self.inception3 = _Inception(4 * n_filters, n_filters, kernel_sizes, bottleneck_channels)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, 4 * n_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(4 * n_filters),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        shortcut = self.residual(x)
        z = self.inception1(x)
        z = self.inception2(z)
        z = self.inception3(z)
        return self.act(z + shortcut)


class _InceptionTime(nn.Module):
    """
    InceptionTime classifier.
    Depth = number of InceptionBlocks; each block contains 3 inception modules,
    matching the Keras depth=6 with residuals every 3 layers.
    input: (N, C, T)
    """

    def __init__(self, in_channels, out_channels, n_filters=32,
                 kernel_sizes=(39, 19, 9), depth=2, bottleneck_channels=32):
        super().__init__()
        blocks = [
            _InceptionBlock(
                in_channels,
                n_filters,
                kernel_sizes,
                bottleneck_channels=bottleneck_channels,
            )
        ]
        for _ in range(1, depth):
            blocks.append(
                _InceptionBlock(
                    4 * n_filters,
                    n_filters,
                    kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4 * n_filters, out_channels)

        # Kept as an attribute so downstream heads can share the encoder without
        # depending on the classifier output layer.
        self.feature_dim = 4 * n_filters

    def forward_features(self, x):
        """Return the pooled InceptionTime representation (N, feature_dim)."""
        x = self.blocks(x)
        return self.gap(x).squeeze(-1)

    def forward(self, x):
        # Preserve the classifier's historical behavior while exposing the
        # representation for survival heads.
        return self.fc(self.forward_features(x))


def _to_odd(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value - 1


def _normalize_kernel_sizes(kernel_sizes=None, base_kernel_size=None):
    if kernel_sizes is not None:
        if not isinstance(kernel_sizes, (list, tuple)):
            raise ValueError("kernel_sizes must be a list/tuple of ints")
        vals = [_to_odd(v) for v in kernel_sizes]
        vals = sorted(set(vals), reverse=True)
        if len(vals) != 3:
            raise ValueError("kernel_sizes must contain exactly 3 unique values")
        return tuple(vals)

    if base_kernel_size is None:
        return (39, 19, 9)

    k1 = _to_odd(base_kernel_size)
    vals = []
    cur = k1
    while len(vals) < 3:
        cur = max(3, _to_odd(cur))
        if cur not in vals:
            vals.append(cur)
        if cur == 3:
            break
        cur = cur // 2
    while len(vals) < 3:
        vals.append(3)
    return tuple(vals)


def make_model(
    input_shape,
    output_shape=1,
    output_bias=None,
    n_filters=32,
    depth=2,
    bottleneck_channels=32,
    kernel_sizes=None,
    base_kernel_size=None,
):
    """
    input_shape: (T, C) — timesteps x channels, as returned by utils.load_data
    Returns an nn.Module that accepts (N, C, T) input.
    """
    in_channels = input_shape[1]
    model = _InceptionTime(
        in_channels=in_channels,
        out_channels=output_shape,
        n_filters=int(n_filters),
        kernel_sizes=_normalize_kernel_sizes(kernel_sizes, base_kernel_size),
        depth=int(depth),
        bottleneck_channels=int(bottleneck_channels),
    )
    if output_bias is not None:
        with torch.no_grad():
            model.fc.bias.fill_(float(output_bias))
    return model
