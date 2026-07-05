"""
Temporal Convolutional Network (TCN)
"""
import torch
import torch.nn as nn


def _as_tuple(value):
    if isinstance(value, int):
        return (value,)
    return tuple(value)


class TemporalConvNet(nn.Module):
    """
    Multi-scale Temporal Convolutional Network with dilated temporal branches.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=(3, 5, 7), dilations=(1, 2, 3), dropout=0.3):
        super(TemporalConvNet, self).__init__()
        kernel_sizes = _as_tuple(kernel_size)
        dilation_sizes = _as_tuple(dilations)

        if len(kernel_sizes) != len(dilation_sizes):
            raise ValueError("kernel_size and dilations must have the same number of values")
        if len(kernel_sizes) == 0:
            raise ValueError("kernel_size must contain at least one value")

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            level_dilations = tuple(dilation * (2 ** i) for dilation in dilation_sizes)
            layers += [
                MultiScaleTemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    dilations=level_dilations,
                    dropout=dropout
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, F) - batch, time, nodes, features
        Returns:
            out: (B, T, N, F_out)
        """
        B, T, N, num_features = x.shape

        # Reshape for TCN: (B*N, F, T)
        x = x.permute(0, 2, 3, 1).reshape(B * N, num_features, T)

        # Apply TCN
        out = self.network(x)  # (B*N, F_out, T)

        # Reshape back: (B, T, N, F_out)
        _, out_features, T_out = out.shape
        out = out.reshape(B, N, out_features, T_out).permute(0, 3, 1, 2)

        return out


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-scale temporal block with gated dilated Conv1d branches.
    """
    def __init__(self, n_inputs, n_outputs, kernel_sizes=(3, 5, 7), dilations=(1, 2, 3), dropout=0.3):
        super(MultiScaleTemporalBlock, self).__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same number of values")

        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(kernel_sizes, dilations):
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("all temporal kernel sizes must be positive odd integers")
            if dilation <= 0:
                raise ValueError("all temporal dilations must be positive")

            padding = (kernel_size - 1) * dilation
            self.branches.append(
                nn.Conv1d(
                    n_inputs,
                    n_outputs * 2,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation
                )
            )

        self.mix = nn.Conv1d(n_outputs * len(self.branches), n_outputs, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (B, F_in, T)
        Returns:
            out: (B, F_out, T)
        """
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            # Keep the same right-truncation convention as the original TCN block.
            out = out[:, :, :x.size(2)]
            out, gate = out.chunk(2, dim=1)
            branch_outputs.append(out * torch.sigmoid(gate))

        out = torch.cat(branch_outputs, dim=1)
        out = self.mix(out)
        out = self.dropout(self.gelu(out))

        res = x if self.downsample is None else self.downsample(x)
        return self.gelu(out + res)
