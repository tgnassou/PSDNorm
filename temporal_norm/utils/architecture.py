# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import copy
import math
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from temporal_norm.utils._psdnorm import PSDNorm

from braindecode.models.base import EEGModuleMixin


def _crop_tensors_to_match(x1, x2, axis=-1):
    """Crops two tensors to their lowest-common-dimension along an axis."""
    dim_cropped = min(x1.shape[axis], x2.shape[axis])

    x1_cropped = torch.index_select(
        x1, dim=axis, index=torch.arange(dim_cropped).to(device=x1.device)
    )
    x2_cropped = torch.index_select(
        x2, dim=axis, index=torch.arange(dim_cropped).to(device=x1.device)
    )
    return x1_cropped, x2_cropped


class _EncoderBlock(nn.Module):
    """Encoding block for a timeseries x of shape (B, C, T)."""

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        kernel_size=9,
        downsample=2,
        activation: nn.Module = nn.ELU,
        filter_size=None,
        norm="BatchNorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Minimal fix: ensure kernel_size is odd to avoid PyTorch warning
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.downsample = downsample

        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm1d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        elif norm == "InstantNormLearn":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels, affine=True)
        elif norm == "LayerNorm":
            norm_layer = nn.LayerNorm(normalized_shape=[out_channels, filter_size])
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        self.block_prepool = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            norm_layer,
        )

        self.pad = nn.ConstantPad1d(padding=1, value=0)
        self.maxpool = nn.MaxPool1d(kernel_size=self.downsample, stride=self.downsample)

    def forward(self, x):
        x = self.block_prepool(x)
        residual = x
        if x.shape[-1] % 2:
            x = self.pad(x)
        x = self.maxpool(x)
        return x, residual


class _DecoderBlock(nn.Module):
    """Decoding block for a timeseries x of shape (B, C, T)."""

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        kernel_size=9,
        upsample=2,
        with_skip_connection=True,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Minimal fix: ensure kernel_size is odd to avoid PyTorch warning
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection

        self.block_preskip = nn.Sequential(
            nn.Upsample(scale_factor=upsample),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            activation(),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.block_postskip = nn.Sequential(
            nn.Conv1d(
                in_channels=(
                    2 * out_channels if with_skip_connection else out_channels
                ),
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.BatchNorm1d(num_features=out_channels),
        )

    def forward(self, x, residual):
        x = self.block_preskip(x)
        if self.with_skip_connection:
            x, residual = _crop_tensors_to_match(
                x, residual, axis=-1
            )  # in case of mismatch
            x = torch.cat([x, residual], axis=1)  # (B, 2 * C, T)
        x = self.block_postskip(x)
        return x


class USleepNorm(EEGModuleMixin, nn.Module):
    """
    Sleep staging architecture from Perslev et al. (2021) [1]_.

    .. figure:: https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41746-021-00440-5/MediaObjects/41746_2021_440_Fig2_HTML.png
        :align: center
        :alt: USleep Architecture

    U-Net (autoencoder with skip connections) feature-extractor for sleep
    staging described in [1]_.

    For the encoder ('down'):
        - the temporal dimension shrinks (via maxpooling in the time-domain)
        - the spatial dimension expands (via more conv1d filters in the time-domain)

    For the decoder ('up'):
        - the temporal dimension expands (via upsampling in the time-domain)
        - the spatial dimension shrinks (via fewer conv1d filters in the time-domain)

    Both do so at exponential rates.

    Parameters
    ----------
    n_chans : int
        Number of EEG or EOG channels. Set to 2 in [1]_ (1 EEG, 1 EOG).
    sfreq : float
        EEG sampling frequency. Set to 128 in [1]_.
    depth : int
        Number of conv blocks in encoding layer (number of 2x2 max pools).
        Note: each block halves the spatial dimensions of the features.
    n_time_filters : int
        Initial number of convolutional filters. Set to 5 in [1]_.
    complexity_factor : float
        Multiplicative factor for the number of channels at each layer of the U-Net.
        Set to 2 in [1]_.
    with_skip_connection : bool
        If True, use skip connections in decoder blocks.
    n_outputs : int
        Number of outputs/classes. Set to 5.
    input_window_seconds : float
        Size of the input, in seconds. Set to 30 in [1]_.
    time_conv_size_s : float
        Size of the temporal convolution kernel, in seconds. Set to 9 / 128 in
        [1]_.
    ensure_odd_conv_size : bool
        If True and the size of the convolutional kernel is an even number, one
        will be added to it to ensure it is odd, so that the decoder blocks can
        work. This can be useful when using different sampling rates from 128
        or 100 Hz.
    activation : nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    References
    ----------
    .. [1] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C.
       U-Sleep: resilient high-frequency sleep staging. *npj Digit. Med.* 4, 72 (2021).
       https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py
    """

    def __init__(
        self,
        n_chans=None,
        sfreq=None,
        depth=12,
        n_time_filters=5,
        complexity_factor=1.67,
        with_skip_connection=True,
        n_outputs=5,
        input_window_seconds=None,
        time_conv_size_s=9 / 128,
        ensure_odd_conv_size=False,
        activation: nn.Module = nn.ELU,
        chs_info=None,
        n_times=None,
        depth_norm=None,
        filter_size=None,
        norm="BatchNorm",
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.mapping = {
            "clf.3.weight": "final_layer.0.weight",
            "clf.3.bias": "final_layer.0.bias",
            "clf.5.weight": "final_layer.2.weight",
            "clf.5.bias": "final_layer.2.bias",
        }

        max_pool_size = 2  # Hardcoded to avoid dimensional errors
        time_conv_size = int(np.round(time_conv_size_s * self.sfreq))
        if time_conv_size % 2 == 0:
            if ensure_odd_conv_size:
                time_conv_size += 1
            else:
                raise ValueError(
                    "time_conv_size must be an odd number to accommodate the "
                    "upsampling step in the decoder blocks."
                )

        channels = [self.n_chans]
        n_filters = n_time_filters
        for _ in range(depth + 1):
            channels.append(int(n_filters * np.sqrt(complexity_factor)))
            n_filters = int(n_filters * np.sqrt(2))
        self.channels = channels

        # Instantiate encoder
        encoder = list()
        for idx in range(depth):
            if norm != "BatchNorm" and idx + 1 <= depth_norm:
                if norm == "PSDNorm":
                    filter_size_layer = filter_size // 2**idx
                    if filter_size_layer % 2 == 0:
                        filter_size_layer += 1
                elif norm == "LayerNorm":
                    filter_size_layer = 105000 // 2**idx
                else:
                    filter_size_layer = None
                norm_ = norm
            else:
                norm_ = "BatchNorm"
                filter_size_layer = None
            encoder += [
                _EncoderBlock(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=time_conv_size,
                    downsample=max_pool_size,
                    activation=activation,
                    filter_size=filter_size_layer,
                    norm=norm_,
                )
            ]
        self.encoder = nn.Sequential(*encoder)

        # Instantiate bottom (channels increase, temporal dim stays the same)
        self.bottom = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=time_conv_size,
                padding=(time_conv_size - 1) // 2,
            ),  # preserves dimension
            activation(),
            nn.BatchNorm1d(num_features=channels[-1]),
        )

        # Instantiate decoder
        decoder = list()
        channels_reverse = channels[::-1]
        for idx in range(depth):
            decoder += [
                _DecoderBlock(
                    in_channels=channels_reverse[idx],
                    out_channels=channels_reverse[idx + 1],
                    kernel_size=time_conv_size,
                    upsample=max_pool_size,
                    with_skip_connection=with_skip_connection,
                    activation=activation,
                )
            ]
        self.decoder = nn.Sequential(*decoder)

        self.clf = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=channels[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # output is (B, C, 1, S * T)
            nn.Tanh(),
            nn.AvgPool1d(self.n_times),  # output is (B, C, S)
        )

        self.final_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=self.n_outputs,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # output is (B, n_classes, S)
            activation(),
            nn.Conv1d(
                in_channels=self.n_outputs,
                out_channels=self.n_outputs,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Identity(),
            # output is (B, n_classes, S)
        )

    def forward(self, x):
        """If input x has shape (B, S, C, T), return y_pred of shape (B, n_classes, S).
        If input x has shape (B, C, T), return y_pred of shape (B, n_classes).
        """
        # reshape input
        if x.ndim == 4:  # input x has shape (B, S, C, T)
            x = x.permute(0, 2, 1, 3)  # (B, C, S, T)
            x = x.flatten(start_dim=2)  # (B, C, S * T)

        # encoder
        residuals = []
        for down in self.encoder:
            x, res = down(x)
            residuals.append(res)

        # bottom
        x = self.bottom(x)

        # decoder
        residuals = residuals[::-1]  # flip order
        for up, res in zip(self.decoder, residuals):
            x = up(x, res)

        # classifier
        x = self.clf(x)
        y_pred = self.final_layer(x)  # (B, n_classes, seq_length)

        if y_pred.shape[-1] == 1:  # seq_length of 1
            y_pred = y_pred[:, :, 0]

        return y_pred


class _SmallCNN(nn.Module):
    """
    Smaller filter sizes to learn temporal information.

    Parameters
    ----------
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    drop_prob : float, default=0.5
        The dropout rate for regularization. Values should be between 0 and 1.
    """

    def __init__(
        self,
        activation: nn.Module = nn.ReLU,
        drop_prob: float = 0.5,
        norm="BatchNorm",
        filter_size=None,
    ):
        super().__init__()
        out_channels = 64
        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(1, 50),
                stride=(1, 6),
                padding=(0, 22),
                bias=False,
            ),
            norm_layer,
            activation(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=(0, 2))
        self.dropout = nn.Dropout(p=drop_prob)
        out_channels = 128
        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size // 2, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 9),
                stride=1,
                padding="same",
                bias=False,
            ),
            norm_layer,
            activation(),
        )

        out_channels = 128
        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size // 2, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 9),
                stride=1,
                padding="same",
                bias=False,
            ),
            norm_layer,
            activation(),
        )

        out_channels = 128
        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size // 2, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 9),
                stride=1,
                padding="same",
                bias=False,
            ),
            norm_layer,
            activation(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 1))

    def forward(self, x):
        x = x[:, :, :1, :]
        x = self.conv1(x)
        x = self.dropout(self.pool1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        return x


class _LargeCNN(nn.Module):
    """
    Larger filter sizes to learn frequency information.

    Parameters
    ----------
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    """

    def __init__(
        self,
        activation: nn.Module = nn.ELU,
        drop_prob: float = 0.5,
        norm="BatchNorm",
        filter_size=None,
    ):
        super().__init__()
        out_channels = 64
        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(1, 400),
                stride=(1, 50),
                padding=(0, 175),
                bias=False,
            ),
            norm_layer,
            activation(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout = nn.Dropout(p=drop_prob)

        out_channels = 128
        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size // 2, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 7),
                stride=1,
                padding="same",
                bias=False,
            ),
            norm_layer,
            activation(),
        )

        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size // 2, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 7),
                stride=1,
                padding="same",
                bias=False,
            ),
            norm_layer,
            activation(),
        )

        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d(num_features=out_channels)
        elif norm == "PSDNorm":
            norm_layer = PSDNorm(filter_size // 2, n_channels=out_channels)
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 7),
                stride=1,
                padding="same",
                bias=False,
            ),
            norm_layer,
            activation(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        x = x[:, :, :1, :]
        x = self.conv1(x)
        x = self.dropout(self.pool1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        return x


class _BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        return out


class DeepSleepNet(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Supratak et al. (2017) [Supratak2017]_.

    Convolutional neural network and bidirectional-Long Short-Term
    for single channels sleep staging described in [Supratak2017]_.

    Parameters
    ----------
    activation_large: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.
    activation_small: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    drop_prob : float, default=0.5
        The dropout rate for regularization. Values should be between 0 and 1.


    References
    ----------
    .. [Supratak2017] Supratak, A., Dong, H., Wu, C., & Guo, Y. (2017).
       DeepSleepNet: A model for automatic sleep stage scoring based
       on raw single-channel EEG. IEEE Transactions on Neural Systems
       and Rehabilitation Engineering, 25(11), 1998-2008.
    """

    def __init__(
        self,
        n_outputs=5,
        return_feats=False,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        activation_large: nn.Module = nn.ELU,
        activation_small: nn.Module = nn.ReLU,
        drop_prob: float = 0.5,
        norm="BatchNorm",
        filter_size=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.cnn1 = _SmallCNN(
            activation=activation_small,
            drop_prob=drop_prob,
            norm=norm,
            filter_size=filter_size,
        )
        self.cnn2 = _LargeCNN(activation=activation_large, drop_prob=drop_prob)
        self.dropout = nn.Dropout(0.5)
        self.bilstm = _BiLSTM(input_size=103680, hidden_size=512, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(103680, 1024, bias=False), nn.BatchNorm1d(num_features=1024)
        )

        self.features_extractor = nn.Identity()
        self.len_last_layer = 1024
        self.return_feats = return_feats

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Linear(1024, self.n_outputs)
        else:
            self.final_layer = nn.Identity()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        if x.ndim == 4:  # input x has shape (B, S, C, T)
            x = x.permute(0, 2, 1, 3)  # (B, C, S, T)
            x = x.flatten(start_dim=2)  # (B, C, S * T)
            x = x.unsqueeze(1)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        x1 = self.cnn1(x)
        x1 = x1.flatten(start_dim=1)

        x2 = self.cnn2(x)
        x2 = x2.flatten(start_dim=1)

        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        temp = x.clone()
        temp = self.fc(temp)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = x.squeeze()
        x = torch.add(x, temp)
        x = self.dropout(x)

        feats = self.features_extractor(x)

        if self.return_feats:
            return feats
        else:
            return self.final_layer(feats)


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.transpose(1, 2)


class MergeWindows(nn.Module):
    def __init__(self, n_windows):
        super().__init__()
        self.n_windows = n_windows

    def forward(self, x):
        # x: (n_batch * n_windows, n_chans, n_times)
        n_batch_times_n_windows, n_chans, n_times = x.shape
        n_batch = n_batch_times_n_windows // self.n_windows
        x = x.view(n_batch, self.n_windows, n_chans, n_times)
        x = x.permute(0, 2, 3, 1)  # (n_batch, n_chans, n_times, n_windows)
        x = x.reshape(n_batch, n_chans, n_times * self.n_windows)
        return x


class UnmergeWindows(nn.Module):
    def __init__(self, n_windows):
        super().__init__()
        self.n_windows = n_windows

    def forward(self, x):
        # x: (n_batch, n_chans, n_times * n_windows)
        n_batch, n_chans, total_time = x.shape
        n_times = total_time // self.n_windows
        x = x.view(n_batch, n_chans, n_times, self.n_windows)
        x = x.permute(0, 3, 1, 2)  # (n_batch, n_windows, n_chans, n_times)
        x = x.reshape(n_batch * self.n_windows, n_chans, n_times)
        return x


class MSDconv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding,
        groups,
        n_windows=None,
        norm="BatchNorm",
        filter_size=None,
    ):
        super(MSDconv, self).__init__()
        self.downsample = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=1,
            stride=stride,
            bias=False,
            groups=groups,
        )
        if norm == "BatchNorm":
            self.norm0 = nn.BatchNorm1d(out_planes)
        elif norm == "PSDNorm":
            self.norm0 = nn.Sequential(
                MergeWindows(n_windows),
                PSDNorm(filter_size, n_channels=out_planes),
                UnmergeWindows(n_windows),
            )
        elif norm == "InstanceNorm":
            self.norm0 = nn.InstanceNorm1d(num_features=out_planes)
        elif norm == "LayerNorm":
            self.norm0 = nn.LayerNorm(normalized_shape=[out_planes, filter_size])
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.dconv1 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=1,
            groups=groups,
        )
        if norm == "BatchNorm":
            self.norm1 = nn.BatchNorm1d(out_planes)
        elif norm == "PSDNorm":
            self.norm1 = nn.Sequential(
                MergeWindows(n_windows),
                PSDNorm(filter_size, n_channels=out_planes),
                UnmergeWindows(n_windows),
            )
        elif norm == "InstanceNorm":
            self.norm1 = nn.InstanceNorm1d(num_features=out_planes)
        elif norm == "LayerNorm":
            self.norm1 = nn.LayerNorm(normalized_shape=[out_planes, filter_size])
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        self.dconv2 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding * 2,
            bias=False,
            dilation=2,
            groups=groups,
        )
        if norm == "BatchNorm":
            self.norm2 = nn.BatchNorm1d(out_planes)
        elif norm == "PSDNorm":
            self.norm2 = nn.Sequential(
                MergeWindows(n_windows),
                PSDNorm(filter_size, n_channels=out_planes),
                UnmergeWindows(n_windows),
            )
        elif norm == "InstanceNorm":
            self.norm2 = nn.InstanceNorm1d(num_features=out_planes)
        elif norm == "LayerNorm":
            self.norm2 = nn.LayerNorm(normalized_shape=[out_planes, filter_size])
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.dconv3 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding * 4,
            bias=False,
            dilation=4,
            groups=groups,
        )
        if norm == "BatchNorm":
            self.norm3 = nn.BatchNorm1d(out_planes)
        elif norm == "PSDNorm":
            self.norm3 = nn.Sequential(
                MergeWindows(n_windows),
                PSDNorm(filter_size, n_channels=out_planes),
                UnmergeWindows(n_windows),
            )
        elif norm == "InstanceNorm":
            self.norm3 = nn.InstanceNorm1d(num_features=out_planes)
        elif norm == "LayerNorm":
            self.norm3 = nn.LayerNorm(normalized_shape=[out_planes, filter_size])
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(out_planes, eps=1e-6)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        down = self.norm0(self.downsample(x))
        x1 = F.gelu(self.norm1(self.dconv1(x)))
        x2 = F.gelu(self.norm2(self.dconv2(x)))
        x3 = F.gelu(self.norm3(self.dconv3(x)))
        out = down + x1 + x2 + x3
        out = self.dropout(out)
        out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)

        return out


class UniConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        # vocab_size_or_config_json_file,
        hidden_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
    ):
        # self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style
        (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attention_probs


class UniEncoder(nn.Module):
    def __init__(self, config):
        super(UniEncoder, self).__init__()
        self.config = config
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)
        self.attention_probs = None

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers, attention_probs = self.encoder(
            x,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        self.attention_probs = attention_probs
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers[-1]


class EpochEncoder(nn.Module):
    def __init__(self, in_plane, n_windows=None, norm="BatchNorm", filter_size=None):
        super(EpochEncoder, self).__init__()
        self.encoder = nn.Sequential(
            MSDconv(
                in_plane,
                64,
                kernel_size=49,
                stride=12,
                padding=24,
                groups=1,
                norm=norm,
                filter_size=filter_size,
                n_windows=n_windows,
            ),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            MSDconv(64, 128, kernel_size=9, stride=1, padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            Transpose(),
            UniEncoder(
                UniConfig(
                    hidden_size=128,
                    intermediate_size=512,
                )
            ),
            Transpose(),
            MSDconv(128, 256, kernel_size=9, stride=1, padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            Transpose(),
            UniEncoder(UniConfig(hidden_size=256, intermediate_size=1024)),
            Transpose(),
            MSDconv(256, 512, kernel_size=9, stride=1, padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            Transpose(),
            UniEncoder(UniConfig(hidden_size=512, intermediate_size=2048)),
            Transpose(),
        )

        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.tensor):
        x = self.encoder(x)
        x = self.avg(x).squeeze()
        return x


class CareSleepNet(nn.Module):
    def __init__(
        self,
        n_chans,
        n_windows,
        dropout=0.1,
        n_outputs=5,
        filter_size=None,
        norm="BatchNorm",
    ):
        super(CareSleepNet, self).__init__()
        self.n_chans = n_chans
        self.n_windows = n_windows
        self.epoch_encoder_eeg = EpochEncoder(
            n_chans // 2, norm=norm, filter_size=filter_size, n_windows=n_windows
        )
        self.epoch_encoder_eog = EpochEncoder(
            n_chans // 2, norm=norm, filter_size=filter_size, n_windows=n_windows
        )
        self.eog2eeg_encoder = CMTransformerEncoder(
            seq_length=n_windows,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=dropout,
            attention_dropout=dropout,
        )
        self.eeg2eog_encoder = CMTransformerEncoder(
            seq_length=n_windows,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=dropout,
            attention_dropout=dropout,
        )
        self.sequence_encoder = TransformerEncoder(
            seq_length=n_windows,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=dropout,
            attention_dropout=dropout,
        )
        self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        batch_size = x.shape[0]
        x_eeg = x[:, :, : self.n_chans // 2, :]
        x_eeg = x_eeg.view(batch_size * self.n_windows, self.n_chans // 2, -1)
        x_eeg = self.epoch_encoder_eeg(x_eeg)
        x_eeg = x_eeg.view(batch_size, self.n_windows, -1)
        x_eog = x[:, :, self.n_chans // 2:, :]
        x_eog = x_eog.view(batch_size * self.n_windows, self.n_chans // 2, -1)
        x_eog = self.epoch_encoder_eog(x_eog)
        x_eog = x_eog.view(batch_size, self.n_windows, -1)
        x_eeg_ = self.eog2eeg_encoder(x_eeg, x_eog)
        x_eog_ = self.eeg2eog_encoder(x_eog, x_eeg)
        x = x_eeg_ + x_eog_

        x = self.sequence_encoder(x)

        return self.classifier(x).transpose(1, 2)


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class CMEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_1_ = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, clue):
        torch._assert(
            input.dim() == 3,
            f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        z = self.ln_1_(clue)
        x, _ = self.self_attention(query=z, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class CMTransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layers = CMEncoderBlock(
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, clue):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        return self.ln(self.layers(self.dropout1(input), self.dropout2(clue)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style(epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
