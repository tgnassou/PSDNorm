import numpy as np

import torch
from torch import nn

from temporal_norm.utils._psdnorm import PSDNorm

from braindecode.models.base import EEGModuleMixin


def select_target_init(filter_size):
    if filter_size == 17:
        target_init = [
            torch.tensor(
                [
                    [
                        1.0000e-07,
                        1.5218e00,
                        3.8543e-01,
                        1.3336e-01,
                        9.7329e-02,
                        6.2896e-02,
                        3.3644e-02,
                        2.9016e-02,
                        4.7016e-02,
                    ],
                    [
                        1.0000e-07,
                        5.5084e-01,
                        2.1004e-01,
                        6.0484e-02,
                        2.5701e-02,
                        2.5206e-02,
                        1.3861e-02,
                        9.3079e-03,
                        1.3325e-02,
                    ],
                    [
                        1.0000e-07,
                        4.0258e-01,
                        2.0834e-01,
                        7.5002e-02,
                        2.6815e-02,
                        2.3768e-02,
                        1.6284e-02,
                        1.2785e-02,
                        3.4157e-02,
                    ],
                    [
                        1.0000e-07,
                        9.9586e-01,
                        4.8442e-01,
                        1.7131e-01,
                        7.0263e-02,
                        3.5709e-02,
                        2.0898e-02,
                        1.7793e-02,
                        2.6394e-02,
                    ],
                    [
                        1.0000e-07,
                        3.0178e-01,
                        2.4154e-01,
                        1.2182e-01,
                        4.8453e-02,
                        1.8119e-02,
                        1.0477e-02,
                        1.0457e-02,
                        2.5501e-02,
                    ],
                    [
                        1.0000e-07,
                        1.4321e00,
                        1.0159e00,
                        2.8292e-01,
                        9.7466e-02,
                        5.5544e-02,
                        2.8466e-02,
                        2.2320e-02,
                        3.2318e-02,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [1.0000e-07, 1.0755e-01, 5.3018e-02, 1.4860e-02, 1.0115e-02],
                    [1.0000e-07, 1.1965e-01, 5.1918e-02, 1.6391e-02, 1.3251e-02],
                    [1.0000e-07, 1.7397e-01, 1.1142e-01, 1.8944e-02, 1.0277e-02],
                    [1.0000e-07, 6.8926e-02, 3.0401e-02, 1.4521e-02, 8.9498e-03],
                    [1.0000e-07, 9.7194e-02, 4.4463e-02, 2.3899e-02, 1.2920e-02],
                    [1.0000e-07, 4.8570e-02, 4.9113e-02, 1.7233e-02, 8.2650e-03],
                    [1.0000e-07, 1.3280e-01, 6.6853e-02, 2.6432e-02, 1.4009e-02],
                    [1.0000e-07, 5.5601e-02, 6.0454e-02, 2.4330e-02, 7.9644e-03],
                    [1.0000e-07, 8.1042e-02, 5.4672e-02, 1.8944e-02, 6.3596e-03],
                ]
            ),
            torch.tensor(
                [
                    [1.0000e-07, 1.7779e-02, 8.1376e-03],
                    [1.0000e-07, 2.6136e-02, 1.0732e-02],
                    [1.0000e-07, 9.6504e-03, 6.2814e-03],
                    [1.0000e-07, 1.6744e-02, 1.4272e-02],
                    [1.0000e-07, 3.6129e-02, 4.3019e-02],
                    [1.0000e-07, 1.4267e-02, 8.8366e-03],
                    [1.0000e-07, 1.4415e-02, 5.5146e-03],
                    [1.0000e-07, 2.2225e-02, 1.1866e-02],
                    [1.0000e-07, 7.5579e-03, 8.9970e-03],
                    [1.0000e-07, 2.5746e-02, 1.1362e-02],
                    [1.0000e-07, 1.4739e-02, 1.9769e-02],
                ]
            ),
        ]

    elif filter_size == 33:
        target_init = [
            torch.tensor(
                [
                    [
                        1.0000e-07,
                        3.7747e00,
                        1.3316e00,
                        7.1869e-01,
                        2.7739e-01,
                        1.4327e-01,
                        1.0046e-01,
                        9.1310e-02,
                        8.0411e-02,
                        6.2588e-02,
                        4.6771e-02,
                        2.9268e-02,
                        2.0067e-02,
                        1.8590e-02,
                        1.9031e-02,
                        2.2508e-02,
                        4.9881e-02,
                    ],
                    [
                        1.0000e-07,
                        1.0884e00,
                        5.0651e-01,
                        3.5606e-01,
                        1.7256e-01,
                        8.9460e-02,
                        4.2904e-02,
                        2.5089e-02,
                        1.7407e-02,
                        1.8204e-02,
                        2.2791e-02,
                        1.6047e-02,
                        7.0061e-03,
                        5.8474e-03,
                        5.7471e-03,
                        6.4723e-03,
                        1.2822e-02,
                    ],
                    [
                        1.0000e-07,
                        7.0462e-01,
                        3.7448e-01,
                        3.0232e-01,
                        1.8377e-01,
                        1.1109e-01,
                        5.8234e-02,
                        3.2054e-02,
                        1.6542e-02,
                        1.3583e-02,
                        2.1992e-02,
                        1.8484e-02,
                        7.8930e-03,
                        6.9863e-03,
                        7.9832e-03,
                        1.1830e-02,
                        4.6203e-02,
                    ],
                    [
                        1.0000e-07,
                        1.7537e00,
                        9.3802e-01,
                        7.2803e-01,
                        4.2569e-01,
                        2.5465e-01,
                        1.3484e-01,
                        8.1129e-02,
                        5.0763e-02,
                        3.3819e-02,
                        2.4121e-02,
                        1.6623e-02,
                        1.2138e-02,
                        1.1061e-02,
                        1.1004e-02,
                        1.2548e-02,
                        2.5717e-02,
                    ],
                    [
                        1.0000e-07,
                        4.7735e-01,
                        2.7440e-01,
                        2.9136e-01,
                        2.2902e-01,
                        1.7467e-01,
                        1.0527e-01,
                        6.5537e-02,
                        3.5507e-02,
                        1.8894e-02,
                        1.1040e-02,
                        7.4248e-03,
                        5.9536e-03,
                        5.7735e-03,
                        6.4135e-03,
                        9.0820e-03,
                        3.2751e-02,
                    ],
                    [
                        1.0000e-07,
                        1.2882e00,
                        1.4351e00,
                        1.5719e00,
                        9.1051e-01,
                        4.8091e-01,
                        2.0482e-01,
                        1.0732e-01,
                        6.6723e-02,
                        5.0537e-02,
                        3.7778e-02,
                        2.3483e-02,
                        1.3981e-02,
                        1.2221e-02,
                        1.1999e-02,
                        1.3622e-02,
                        2.8524e-02,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        1.0000e-07,
                        2.4246e-01,
                        8.5448e-02,
                        7.4021e-02,
                        4.2877e-02,
                        1.7467e-02,
                        1.0309e-02,
                        7.7810e-03,
                        8.5329e-03,
                    ],
                    [
                        1.0000e-07,
                        1.9868e-01,
                        1.0972e-01,
                        7.5309e-02,
                        4.2901e-02,
                        2.1187e-02,
                        1.0341e-02,
                        1.2203e-02,
                        1.0003e-02,
                    ],
                    [
                        1.0000e-07,
                        1.5581e-01,
                        1.7703e-01,
                        2.0901e-01,
                        7.7872e-02,
                        2.1266e-02,
                        1.1695e-02,
                        8.6491e-03,
                        5.6986e-03,
                    ],
                    [
                        1.0000e-07,
                        1.6622e-01,
                        5.3750e-02,
                        5.7761e-02,
                        1.7027e-02,
                        1.4459e-02,
                        1.1949e-02,
                        7.5323e-03,
                        7.6410e-03,
                    ],
                    [
                        1.0000e-07,
                        1.8493e-01,
                        9.8125e-02,
                        8.4673e-02,
                        2.8702e-02,
                        3.0076e-02,
                        1.8196e-02,
                        1.0498e-02,
                        1.2352e-02,
                    ],
                    [
                        1.0000e-07,
                        8.6804e-02,
                        4.3481e-02,
                        4.8775e-02,
                        5.0219e-02,
                        2.8238e-02,
                        1.0750e-02,
                        8.5906e-03,
                        5.5477e-03,
                    ],
                    [
                        1.0000e-07,
                        3.4292e-01,
                        1.0398e-01,
                        1.1342e-01,
                        4.5280e-02,
                        2.5204e-02,
                        2.2691e-02,
                        1.4592e-02,
                        8.9524e-03,
                    ],
                    [
                        1.0000e-07,
                        9.5001e-02,
                        5.1999e-02,
                        7.7432e-02,
                        5.4241e-02,
                        3.4767e-02,
                        1.7652e-02,
                        8.8095e-03,
                        4.4033e-03,
                    ],
                    [
                        1.0000e-07,
                        8.5285e-02,
                        8.1401e-02,
                        7.7311e-02,
                        4.6140e-02,
                        2.6419e-02,
                        1.2925e-02,
                        5.8946e-03,
                        3.6037e-03,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [1.0000e-07, 3.6807e-02, 1.7118e-02, 1.0878e-02, 6.1283e-03],
                    [1.0000e-07, 7.9626e-02, 1.6302e-02, 1.2957e-02, 6.8192e-03],
                    [1.0000e-07, 1.6600e-02, 8.7723e-03, 8.9609e-03, 3.8068e-03],
                    [1.0000e-07, 3.5653e-02, 1.4818e-02, 1.6442e-02, 1.1742e-02],
                    [1.0000e-07, 1.7490e-02, 3.7872e-02, 7.2853e-02, 2.0029e-02],
                    [1.0000e-07, 2.0802e-02, 1.4204e-02, 1.0495e-02, 7.1520e-03],
                    [1.0000e-07, 4.2042e-02, 1.1798e-02, 7.0751e-03, 3.9016e-03],
                    [1.0000e-07, 6.2069e-02, 1.2527e-02, 1.7102e-02, 5.7292e-03],
                    [1.0000e-07, 1.6309e-02, 6.9625e-03, 7.9680e-03, 9.2882e-03],
                    [1.0000e-07, 1.8379e-02, 3.1622e-02, 1.6510e-02, 8.4795e-03],
                    [1.0000e-07, 2.4859e-02, 9.8397e-03, 1.4391e-02, 2.2607e-02],
                ]
            ),
        ]
    elif filter_size == 9:
        target_init = [
            torch.tensor(
                [
                    [1.0000e-07, 5.1971e-01, 1.1685e-01, 5.4177e-02, 4.7519e-02],
                    [1.0000e-07, 2.4881e-01, 3.9475e-02, 2.2886e-02, 1.5022e-02],
                    [1.0000e-07, 2.2980e-01, 4.4370e-02, 2.5802e-02, 2.8604e-02],
                    [1.0000e-07, 5.3758e-01, 1.0358e-01, 3.5436e-02, 3.0409e-02],
                    [1.0000e-07, 2.3606e-01, 6.9061e-02, 1.8105e-02, 2.2853e-02],
                    [1.0000e-07, 1.0515e00, 1.5709e-01, 5.4556e-02, 4.1592e-02],
                ]
            ),
            torch.tensor(
                [
                    [1.0000e-07, 5.8747e-02, 1.2767e-02],
                    [1.0000e-07, 6.6396e-02, 1.6592e-02],
                    [1.0000e-07, 1.3206e-01, 1.6049e-02],
                    [1.0000e-07, 3.7394e-02, 1.1137e-02],
                    [1.0000e-07, 5.8855e-02, 1.7345e-02],
                    [1.0000e-07, 4.5149e-02, 1.2281e-02],
                    [1.0000e-07, 7.4174e-02, 1.8892e-02],
                    [1.0000e-07, 5.7194e-02, 1.4382e-02],
                    [1.0000e-07, 6.2579e-02, 1.2846e-02],
                ]
            ),
            torch.tensor(
                [
                    [1.0000e-07, 1.0881e-02],
                    [1.0000e-07, 1.2725e-02],
                    [1.0000e-07, 7.2575e-03],
                    [1.0000e-07, 1.3975e-02],
                    [1.0000e-07, 4.5442e-02],
                    [1.0000e-07, 1.0957e-02],
                    [1.0000e-07, 7.0194e-03],
                    [1.0000e-07, 1.3115e-02],
                    [1.0000e-07, 7.7635e-03],
                    [1.0000e-07, 1.9002e-02],
                    [1.0000e-07, 1.7185e-02],
                ]
            ),
        ]
    else:
        target_init = None
    return target_init


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
        norm,
        in_channels=2,
        out_channels=2,
        kernel_size=9,
        downsample=2,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Minimal fix: ensure kernel_size is odd to avoid PyTorch warning
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.downsample = downsample

        self.block_prepool = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            norm,
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
        norm,
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
            norm,
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


class USleep(EEGModuleMixin, nn.Module):
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
        filter_size=None,
        norm_apply_to="encoder",
        bias_learnable=False,
        target_learnable=False,
        track_running_stats=True,
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
        target_init = select_target_init(filter_size)
        # Instantiate encoder
        encoder = list()
        for idx in range(depth):
            if filter_size is None or norm_apply_to == "decoder":
                norm = nn.BatchNorm1d(channels[idx + 1])
            else:
                if idx in [0, 1, 2]:
                    if filter_size == 1:
                        norm = nn.InstanceNorm1d(channels[idx + 1])
                    else:
                        filter_size_ = filter_size // 2**idx
                        if filter_size_ < 1:
                            filter_size_ = 1
                        if filter_size_ % 2 == 0:
                            filter_size_ += 1

                        norm = PSDNorm(
                            filter_size=filter_size_,
                            n_channels=channels[idx + 1],
                            bias_learnable=bias_learnable,
                            target_learnable=target_learnable,
                            target_init=(
                                target_init[idx] if target_init is not None else None
                            ),
                            track_running_stats=track_running_stats,
                        )
                else:
                    norm = nn.BatchNorm1d(channels[idx + 1])
            encoder += [
                _EncoderBlock(
                    norm=norm,
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=time_conv_size,
                    downsample=max_pool_size,
                    activation=activation,
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
            if filter_size is None or norm_apply_to == "encoder":
                norm = nn.BatchNorm1d(channels_reverse[idx + 1])
            else:
                if idx in [9, 10, 11]:
                    if filter_size == 1:
                        norm = nn.InstanceNorm1d(channels_reverse[idx + 1])
                    else:
                        filter_size_ = filter_size // 2 ** (11 - idx)
                        if filter_size_ < 1:
                            filter_size_ = 1
                        if filter_size_ % 2 == 0:
                            filter_size_ += 1

                        norm = PSDNorm(
                            filter_size=filter_size_,
                            n_channels=channels_reverse[idx + 1],
                            bias_learnable=bias_learnable,
                            target_learnable=target_learnable,
                            target_init=target_init[11 - idx],
                            track_running_stats=track_running_stats,
                        )
                else:
                    norm = nn.BatchNorm1d(channels_reverse[idx + 1])
            decoder += [
                _DecoderBlock(
                    in_channels=channels_reverse[idx],
                    out_channels=channels_reverse[idx + 1],
                    kernel_size=time_conv_size,
                    upsample=max_pool_size,
                    with_skip_connection=with_skip_connection,
                    activation=activation,
                    norm=norm,
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
