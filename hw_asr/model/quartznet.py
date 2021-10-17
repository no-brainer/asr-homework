import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_asr.base import BaseModel
from hw_asr.model.utils import get_same_padding


def pointwise_conv(in_channels, out_channels, dilation=1, stride=1, use_bias=False):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=1, dilation=dilation, stride=stride, bias=use_bias
    )


def depthwise_conv(in_channels, out_channels, kernel_size, stride=1, use_bias=False):
    padding = get_same_padding(kernel_size)
    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     padding=padding, stride=stride, bias=use_bias, groups=in_channels)


def get_conv_block(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        is_separable=True,
        has_activation=True,
):
    if is_separable:
        layers = [
            depthwise_conv(in_channels, in_channels, kernel_size, stride=stride),
            pointwise_conv(in_channels, out_channels, stride=stride),
        ]
    else:
        padding = get_same_padding(kernel_size)
        layers = [
            nn.Conv1d(in_channels, out_channels, stride=stride, dilation=dilation,
                      kernel_size=kernel_size, padding=padding, bias=False)
        ]

    layers.append(nn.BatchNorm1d(out_channels))
    if has_activation:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class QuartzNetBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            num_conv_blocks,
            *args,
            **kwargs
    ):
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        curr_channels = in_channels
        for i in range(num_conv_blocks):
            has_activation = i + 1 < num_conv_blocks
            conv_block = get_conv_block(curr_channels, out_channels, kernel_size, has_activation=has_activation)
            self.conv_blocks.append(conv_block)
            curr_channels = out_channels

        self.residual_transformation = nn.Sequential(
            pointwise_conv(in_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, spectrogram, *args, **kwargs):
        out = spectrogram.clone()
        for block in self.conv_blocks:
            out = block(out)
        out += self.residual_transformation(spectrogram)
        return F.relu(out)


class QuartzNetModel(BaseModel):
    def __init__(
            self,
            n_feats,
            n_class,
            m_quartz_blocks,
            r_conv_blocks,
            channel_nums,
            kernel_sizes,
            *args,
            **kwargs
    ):
        super().__init__(n_feats, n_class, *args, **kwargs)
        layers = [
            get_conv_block(n_feats, channel_nums[0], kernel_sizes[0])
        ]
        for i in range(m_quartz_blocks):
            layers.append(
                QuartzNetBlock(channel_nums[i], channel_nums[i + 1], kernel_sizes[i + 1], r_conv_blocks)
            )

        layers.extend([
            get_conv_block(channel_nums[-3], channel_nums[-2], kernel_sizes[-2]),
            get_conv_block(channel_nums[-2], channel_nums[-1], kernel_sizes[-1]),
            pointwise_conv(channel_nums[-1], n_class, dilation=2, use_bias=True)
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, spectrogram, *args, **kwargs):
        out = self.net(spectrogram.transpose(-2, -1))
        return out.transpose(-2, -1)

    def transform_input_lengths(self, input_lengths):
        return input_lengths / 2
