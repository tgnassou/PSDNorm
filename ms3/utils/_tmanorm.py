import torch
from torch import nn
import torch.fft


def welch_psd(signal, fs=1.0, nperseg=None, noverlap=None, window="hamming", axis=-1):
    """
    Compute the Power Spectral Density (PSD) of a signal using Welch's method along a specified axis.

    Parameters:
    - signal (torch.Tensor): Tensor of the input signal (can be multi-dimensional).
    - fs (float): Sampling frequency of the signal.
    - nperseg (int): Length of each segment.
    - noverlap (int): Number of points to overlap between segments.
    - window (str): Window function to apply on each segment.
    - axis (int): Axis along which to compute the PSD.

    Returns:
    - freqs (torch.Tensor): Array of sample frequencies.
    - psd (torch.Tensor): Power spectral density of each frequency component along the specified axis.
    """
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
    # Calculate step size and number of segments along the last axis
    step = nperseg - noverlap
    num_segments = (signal.shape[-1] - noverlap) // step

    # Pre-allocate array for the PSD, retaining all other dimensions
    psd_sum = torch.zeros(*signal.shape[:-1], nperseg // 2 + 1, device=signal.device)

    # Iterate over segments along the last axis
    for i in range(num_segments):
        # Extract the segment
        segment = signal[..., i * step : i * step + nperseg]

        # detrend
        segment = segment - torch.mean(segment, axis=-1, keepdim=True)

        # Apply window function
        windowed_segment = segment * window_vals
        # Compute the FFT and PSD for the segment along the last axis
        segment_fft = torch.fft.rfft(windowed_segment, dim=-1)
        segment_psd = torch.abs(segment_fft) ** 2 / (fs * scaling)
        if nperseg % 2:
            segment_psd[..., 1:] *= 2
        else:
            segment_psd[..., 1:-1] *= 2
        # Accumulate PSDs from each segment
        psd_sum += segment_psd

    # Average PSD over all segments
    psd = psd_sum / num_segments

    # Compute frequency axis
    freqs = torch.fft.rfftfreq(nperseg, d=1 / fs)

    # Reshape PSD to match the original dimensions with the last axis replaced by frequency components
    return freqs, psd.transpose(axis, -1)


class TMANorm(nn.Module):
    def __init__(self, filter_size, momentum=0.1):
        super(TMANorm, self).__init__()
        self.filter_size = filter_size
        self.momentum = momentum
        self.register_buffer("running_barycenter", torch.zeros(1))
        self.first_iter = True

    def forward(self, x):
        # x: (B, C, T)

        # compute psd for each channel using welch method
        # psd: (B, C, F)
        psd = welch_psd(x, window=None, nperseg=self.filter_size)[1]

        # compute running barycenter of psd
        # barycenter: (C, F,)
        weights = torch.ones_like(psd) / psd.shape[-1]
        new_barycenter = torch.sum(weights * torch.sqrt(psd), axis=0) ** 2

        # update running barycenter
        if self.first_iter:
            self.running_barycenter = new_barycenter
            self.first_iter = False
        else:
            self.running_barycenter = (
                1 - self.momentum
            ) * self.running_barycenter + self.momentum * new_barycenter

        # compute filter
        # H: (B, C, F)
        D = torch.sqrt(self.running_barycenter) / torch.sqrt(psd)
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
