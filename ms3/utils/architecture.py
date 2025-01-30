# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import torch
from torch import nn

from ms3.utils._tmanorm import TMANorm

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
        self.kernel_size = kernel_size
        self.downsample = downsample

        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm1d(num_features=out_channels)
        elif norm == "TMANorm":
            norm_layer = TMANorm(filter_size, n_channels=out_channels)
        elif norm == "InstantNorm":
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
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection

        self.block_preskip = nn.Sequential(
            nn.Upsample(scale_factor=upsample),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
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


class USleepTMA(EEGModuleMixin, nn.Module):
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
        depth_tma=None,
        filter_size=None,
        filter_size_input=None,
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

        if filter_size_input:
            self.tmainput = TMANorm(filter_size=filter_size_input, bary_learning=bary_learning)
        else:
            self.tmainput = nn.Identity()

        # Instantiate encoder
        encoder = list()
        for idx in range(depth):
            if norm != "BatchNorm" and idx + 1 <= depth_tma:
                if norm == "TMANorm":
                    filter_size_layer = filter_size // 2 ** idx
                elif norm == "LayerNorm":
                    filter_size_layer = 105000 // 2 ** idx
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

        # The temporal dimension remains unchanged
        # (except through the AvgPooling which collapses it to 1)
        # The spatial dimension is preserved from the end of the UNet, and is mapped to n_classes

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

        x = self.tmainput(x)
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


class ChambonTMA(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Chambon et al. (2018) [Chambon2018]_.

    .. figure:: https://braindecode.org/dev/_static/model/SleepStagerChambon2018.jpg
        :align: center
        :alt: SleepStagerChambon2018 Architecture

    Convolutional neural network for sleep staging described in [Chambon2018]_.

    Parameters
    ----------
    n_conv_chs : int
        Number of convolutional channels. Set to 8 in [Chambon2018]_.
    time_conv_size_s : float
        Size of filters in temporal convolution layers, in seconds. Set to 0.5
        in [Chambon2018]_ (64 samples at sfreq=128).
    max_pool_size_s : float
        Max pooling size, in seconds. Set to 0.125 in [Chambon2018]_ (16
        samples at sfreq=128).
    pad_size_s : float
        Padding size, in seconds. Set to 0.25 in [Chambon2018]_ (half the
        temporal convolution kernel size).
    drop_prob : float
        Dropout rate before the output dense layer.
    apply_batch_norm : bool
        If True, apply batch normalization after both temporal convolutional
        layers.
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    n_channels : int
        Alias for `n_chans`.
    input_size_s:
        Alias for `input_window_seconds`.
    n_classes:
        Alias for `n_outputs`.
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.

    References
    ----------
    .. [Chambon2018] Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., &
           Gramfort, A. (2018). A deep learning architecture for temporal sleep
           stage classification using multivariate and multimodal time series.
           IEEE Transactions on Neural Systems and Rehabilitation Engineering,
           26(4), 758-769.
    """

    def __init__(
        self,
        n_chans=None,
        sfreq=None,
        n_conv_chs=8,
        time_conv_size_s=0.5,
        max_pool_size_s=0.125,
        pad_size_s=0.25,
        activation: nn.Module = nn.ReLU,
        input_window_seconds=None,
        n_outputs=5,
        drop_prob=0.25,
        apply_batch_norm=False,
        return_feats=False,
        chs_info=None,
        n_times=None,
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

        self.mapping = {
            "fc.1.weight": "final_layer.1.weight",
            "fc.1.bias": "final_layer.1.bias",
        }

        time_conv_size = np.ceil(time_conv_size_s * self.sfreq).astype(int)
        max_pool_size = np.ceil(max_pool_size_s * self.sfreq).astype(int)
        pad_size = np.ceil(pad_size_s * self.sfreq).astype(int)

        if self.n_chans > 1:
            self.spatial_conv = nn.Conv2d(1, self.n_chans, (self.n_chans, 1))

        CMLN = TMANorm(filter_size=filter_size) if filter_size else nn.Identity()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
            CMLN,
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chs, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)
            ),
            CMLN,
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
        )
        self.len_last_layer = self._len_last_layer(self.n_chans, self.n_times)
        self.return_feats = return_feats

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Sequential(
                nn.Dropout(p=drop_prob),
                nn.Linear(self.len_last_layer, self.n_outputs),
            )

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(torch.Tensor(1, 1, n_channels, input_size))
        self.feature_extractor.train()
        return len(out.flatten())

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.n_chans > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        feats = self.feature_extractor(x).flatten(start_dim=1)

        if self.return_feats:
            return feats
        else:
            return self.final_layer(feats)
