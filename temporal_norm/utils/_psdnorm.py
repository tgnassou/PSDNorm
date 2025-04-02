import torch
from torch import nn
import torch.fft
import torch.nn.functional as F


def welch_psd(signal, fs=1.0, nperseg=None, noverlap=None, window="hamming", axis=-1):
    if nperseg is None:
        nperseg = 256
    if noverlap is None:
        noverlap = nperseg // 2

    # Move the specified axis to the last dimension for easier processing
    signal = signal.transpose(axis, -1)

    # Define the window function
    if window == "hamming":
        window_vals = torch.hamming_window(
            nperseg, periodic=False, device=signal.device
        )
    elif window == "hann":
        window_vals = torch.hann_window(nperseg, periodic=False, device=signal.device)
    elif window is None:
        window_vals = torch.ones(nperseg, device=signal.device)
    else:
        raise ValueError("Unsupported window type")

    scaling = (window_vals * window_vals).sum()

    # Calculate step size and number of segments
    step = nperseg - noverlap
    num_segments = (signal.shape[-1] - noverlap) // step

    # Generate indices for all segments in one batch operation
    indices = torch.arange(nperseg, device=signal.device).unsqueeze(
        0
    ) + step * torch.arange(num_segments, device=signal.device).unsqueeze(1)

    # Extract and process all segments in one batch
    segments = signal[..., indices]  # Shape: (..., num_segments, nperseg)
    segments = segments - segments.mean(dim=-1, keepdim=True)  # Detrend
    windowed_segments = segments * window_vals  # Apply window

    # Compute FFT for all segments in parallel using real output to avoid complex dtype
    segment_fft = torch.view_as_real(torch.fft.rfft(windowed_segments, dim=-1))

    # Compute magnitude squared manually (Re^2 + Im^2) / scaling
    segment_psd = (segment_fft[..., 0] ** 2 + segment_fft[..., 1] ** 2) / (fs * scaling)

    # Adjust for one-sided spectrum
    if nperseg % 2:
        segment_psd[..., 1:] *= 2
    else:
        segment_psd[..., 1:-1] *= 2

    # Average over segments
    psd = segment_psd.mean(dim=-2)

    # Compute frequency axis
    freqs = torch.fft.rfftfreq(nperseg, d=1 / fs)

    # Reshape PSD to match the original dimensions
    return freqs, psd.transpose(axis, -1)


class PSDNorm(nn.Module):
    def __init__(
        self,
        filter_size,
        momentum=0.01,
        track_running_stats=True,
        reg=1e-7,
        barycenter_init=None,
        bary_learning=False,
        center=True,
        n_channels=1,
    ):
        # This layer is not always well compiled.
        # The following two lines make sure it's run in eager mode if
        # the compilation fails.
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        super(PSDNorm, self).__init__()
        self.filter_size = filter_size
        self.momentum = momentum
        if bary_learning:
            self.register_parameter(
                "barycenter",
                torch.nn.Parameter(torch.zeros(n_channels, filter_size // 2 + 1))
            )
        else:
            self.register_buffer(
                "barycenter",
                torch.zeros(1),
            )
        self.first_iter = True
        self.track_running_stats = track_running_stats
        self.reg = reg
        self.barycenter_init = barycenter_init
        self.bary_learning = bary_learning
        self.center = center

    def _update_barycenter(self, barycenter):
        if self.first_iter:
            self.barycenter = barycenter
            self.first_iter = False
        else:
            self.barycenter = (
                (1 - self.momentum)**2 * self.barycenter
                + self.momentum**2 * barycenter
                + 2 * self.momentum * (1 - self.momentum) *
                torch.exp(0.5 * (torch.log(self.barycenter) + torch.log(barycenter)))
            )

    def forward(self, x):
        if x.dim() == 4:
            squeeze = True
            x = x.squeeze(2)
        else:
            squeeze = False
        # x: (B, C, T)
        # centered x
        if self.center:
            x = x - torch.mean(x, dim=-1, keepdim=True)
        # compute psd for each channel using welch method
        # psd: (B, C, F)

        psd = welch_psd(x, window=None, nperseg=self.filter_size)[1] + self.reg

        # compute running barycenter of psd
        # barycenter: (C, F,)
        # update running barycenter
        if self.training and self.track_running_stats and not self.bary_learning:
            weights = torch.ones_like(psd) / psd.shape[-1]
            new_barycenter = torch.sum(weights * torch.sqrt(psd), axis=0) ** 2
            self._update_barycenter(new_barycenter.detach())

        if self.bary_learning:
            target = torch.exp(self.barycenter)
        else:
            target = self.barycenter
        # compute filtermodel
        # H: (B, C, F)

        D = torch.sqrt(target) / torch.sqrt(psd)
        H = torch.fft.irfft(D, dim=-1, n=self.filter_size)
        H = torch.fft.fftshift(H, dim=-1)

        # apply filter, convolute H with x
        # x_filtered: (B, C, T)
        H = torch.flip(H, dims=[-1])

        B, C, T = x.shape
        filters = H.view(-1, 1, H.shape[-1])
        input_x = x.view(1, -1, T)
        x_filtered = F.conv1d(input_x, filters, padding="same", groups=filters.shape[0])
        x_filtered = x_filtered.view(B, C, -1)

        if squeeze:
            x_filtered = x_filtered.unsqueeze(2)
        return x_filtered


def welch_psd_2d(signal, fs=1.0, nperseg=None, noverlap=None, window="hamming", axis=-1):
    if nperseg is None:
        nperseg = 256
    if noverlap is None:
        noverlap = nperseg // 2

    # Move the specified axis to the last dimension for easier processing
    signal = signal.transpose(axis, -1)

    # Define the window function
    if window == "hamming":
        window_vals = torch.hamming_window(
            nperseg, periodic=False, device=signal.device
        )
    elif window == "hann":
        window_vals = torch.hann_window(nperseg, periodic=False, device=signal.device)
    elif window is None:
        window_vals = torch.ones(nperseg, device=signal.device)
    else:
        raise ValueError("Unsupported window type")

    scaling = (window_vals * window_vals).sum()

    # Calculate step size and number of segments
    step = nperseg - noverlap
    num_segments = (signal.shape[-1] - noverlap) // step

    # Generate indices for all segments in one batch operation
    indices = torch.arange(nperseg, device=signal.device).unsqueeze(
        0
    ) + step * torch.arange(num_segments, device=signal.device).unsqueeze(1)

    # Extract and process all segments in one batch
    segments = signal[..., indices]  # Shape: (..., num_segments, nperseg)
    segments = segments - segments.mean(dim=-1, keepdim=True)  # Detrend
    windowed_segments = segments * window_vals  # Apply window

    # Compute FFT for all segments in parallel
    segment_fft = torch.fft.rfft(windowed_segments, dim=-1)
    segment_psd = torch.abs(segment_fft) ** 2 / (fs * scaling)

    # Adjust for one-sided spectrum
    if nperseg % 2:
        segment_psd[..., 1:] *= 2
    else:
        segment_psd[..., 1:-1] *= 2

    # Average over segments
    psd = segment_psd.mean(dim=-2)

    # Compute frequency axis
    freqs = torch.fft.rfftfreq(nperseg, d=1 / fs)

    # Reshape PSD to match the original dimensions
    return freqs, psd.transpose(axis, -1)


class PSDNorm2D(nn.Module):
    def __init__(
        self,
        filter_size,
        momentum=0.01,
        track_running_stats=True,
        reg=1e-7,
        barycenter_init=None,
        bary_learning=False,
        center=True,
        n_channels=1,
    ):
        super(PSDNorm, self).__init__()
        self.filter_size = filter_size
        self.momentum = momentum
        if bary_learning:
            self.register_parameter(
                "barycenter",
                torch.nn.Parameter(torch.zeros(n_channels, filter_size // 2 + 1))
            )
        else:
            self.register_buffer(
                "barycenter",
                torch.zeros(1),
            )
        self.first_iter = True
        self.track_running_stats = track_running_stats
        self.reg = reg
        self.barycenter_init = barycenter_init
        self.bary_learning = bary_learning
        self.center = center

    def _update_barycenter(self, barycenter,):
        if self.first_iter:
            self.barycenter = barycenter
            self.first_iter = False
        else:
            self.barycenter = (
                (1 - self.momentum)**2 * self.barycenter
                + self.momentum**2 * barycenter
                + 2 * self.momentum * (1 - self.momentum) *
                torch.exp(0.5 * (torch.log(self.barycenter) + torch.log( barycenter)))
            )

    def forward(self, x):
        # x: (B, C, T)
        # centered x
        if self.center:
            x = x - torch.mean(x, dim=-1, keepdim=True)
        # compute psd for each channel using welch method
        # psd: (B, C, F)

        psd = welch_psd_2d(x, window=None, nperseg=self.filter_size)[1] + self.reg

        # compute running barycenter of psd
        # barycenter: (C, F,)
        # update running barycenter
        if self.training and self.track_running_stats and not self.bary_learning:
            weights = torch.ones_like(psd) / psd.shape[-1]
            new_barycenter = torch.sum(weights * torch.sqrt(psd), axis=0) ** 2
            self._update_barycenter(new_barycenter.detach())

        if self.bary_learning:
            target = torch.exp(self.barycenter)
        else:
            target = self.barycenter
        # compute filtermodel
        # H: (B, C, F)

        D = torch.sqrt(target) / torch.sqrt(psd)
        H = torch.fft.irfft(D, dim=-1)
        H = torch.fft.fftshift(H, dim=-1)

        # apply filter, convolute H with x
        # x_filtered: (B, C, T)
        H = torch.flip(H, dims=[-1])
        n_chan = x.shape[1]
        n_batch = x.shape[0]
        x_filtered = torch.cat(
            [
                torch.nn.functional.conv1d(
                    x[i : i + 1],
                    H[i : i + 1].view(n_chan, 1, -1),
                    padding="same",
                    groups=n_chan,
                )
                for i in range(n_batch)
            ]
        )

        return x_filtered