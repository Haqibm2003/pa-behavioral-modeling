"""Complex-baseband test signals and spectral tools.

Everything here works on the complex baseband envelope, which is the
representation behavioural PA modelling is done in: the RF carrier is stripped
away and only the modulation remains, so a 3.5 GHz signal and a 28 GHz signal
look identical to the model. What matters is bandwidth, peak-to-average ratio,
and how fast the envelope moves.

Two details drive everything downstream:

**Oversampling.** A nonlinearity of order *K* spreads the spectrum to roughly
*K* times the signal bandwidth. Simulating at the Nyquist rate of the input
aliases that spectral regrowth back into the channel, which destroys exactly the
measurement you are trying to make. A 5th-order model needs at least 5x
oversampling; this module defaults to 8x.

**Peak-to-average power ratio.** OFDM sums many independent subcarriers, so by
the central limit theorem the envelope is close to Rayleigh and the PAPR is
high, typically 8-12 dB. That is what forces a linear PA to run backed off from
saturation, and it is the reason predistortion is worth doing at all.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "qam_symbols",
    "multicarrier_signal",
    "occupied_edge",
    "papr_db",
    "rms",
    "set_rms",
    "welch_psd",
    "band_power",
]


def rms(x: np.ndarray) -> float:
    """Root-mean-square magnitude of a complex signal."""
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def set_rms(x: np.ndarray, target: float) -> np.ndarray:
    """Scale a signal to a target RMS level."""
    current = rms(x)
    if current == 0:
        return x
    return x * (target / current)


def papr_db(x: np.ndarray) -> float:
    """Peak-to-average power ratio in dB."""
    power = np.abs(x) ** 2
    mean_power = float(np.mean(power))
    if mean_power == 0:
        return 0.0
    return float(10.0 * np.log10(np.max(power) / mean_power))


def qam_symbols(
    count: int, order: int = 16, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Draw `count` symbols from a square QAM constellation, unit average power."""
    rng = rng or np.random.default_rng(0)
    side = int(round(np.sqrt(order)))
    if side * side != order:
        raise ValueError(f"QAM order must be a perfect square, got {order}")

    levels = np.arange(-(side - 1), side, 2, dtype=float)
    real = rng.choice(levels, size=count)
    imaginary = rng.choice(levels, size=count)
    symbols = real + 1j * imaginary
    return symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))


def multicarrier_signal(
    num_samples: int = 32768,
    occupied_fraction: float = 0.125,
    qam_order: int = 16,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a multicarrier complex baseband signal by a single IFFT.

    Random QAM symbols are placed on a contiguous block of subcarriers centred
    on DC, occupying ``occupied_fraction`` of the sample rate, and one inverse
    FFT produces the whole record. The unoccupied bins are guard band.

    **Why one long IFFT rather than concatenated OFDM symbols.** Independently
    generated symbols laid end to end are discontinuous at every boundary, and
    those discontinuities splatter energy across the whole spectrum. Measured
    here, that approach left the guard band only about 31 dB below the occupied
    band; adding a cyclic prefix and raised-cosine windowing improved it to
    about 51 dB. A single IFFT is exactly periodic, so it has no boundaries at
    all and the guard band sits roughly 127 dB down, at the numerical noise
    floor.

    That headroom is not a luxury. Adjacent-channel ratios of interest run from
    25 to 50 dBc, so a generator whose own leakage is 31 dB down would be the
    dominant term in the measurement: you would be characterising the test
    signal instead of the amplifier. The cost is that the signal is periodic,
    which is standard practice in the predistortion literature and is harmless
    here because nothing in the modelling depends on the record being aperiodic.

    ``occupied_fraction`` also sets the oversampling ratio, as its reciprocal.
    The default of 1/8 leaves room for fifth- and seventh-order regrowth to fall
    inside the simulated band rather than aliasing back on top of the signal.
    """
    rng = rng or np.random.default_rng(0)

    occupied = int(round(num_samples * occupied_fraction))
    if occupied < 4:
        raise ValueError(
            f"occupied_fraction={occupied_fraction} gives only {occupied} "
            f"subcarriers; use a larger fraction or more samples"
        )
    if occupied >= num_samples:
        raise ValueError(
            "occupied_fraction must be below 1.0 so that guard band remains for "
            "spectral regrowth to be measured in"
        )

    half = occupied // 2
    # Skip DC: a carrier at zero frequency is conventionally left empty, and it
    # keeps the signal zero-mean.
    indices = np.concatenate(
        [np.arange(1, half + 1), np.arange(num_samples - half, num_samples)]
    )

    spectrum = np.zeros(num_samples, dtype=complex)
    spectrum[indices] = qam_symbols(len(indices), qam_order, rng)
    signal = np.fft.ifft(spectrum) * num_samples / np.sqrt(len(indices))
    return set_rms(signal, 1.0)


def occupied_edge(occupied_fraction: float = 0.125) -> float:
    """Normalised frequency of the occupied band edge, for setting ACPR bands."""
    return occupied_fraction / 2.0


def welch_psd(
    x: np.ndarray,
    segment: int = 2048,
    overlap: float = 0.5,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density by Welch's method, in normalised frequency.

    Averaging periodograms over overlapping windowed segments trades frequency
    resolution for variance. That trade matters here: a single periodogram of a
    random signal has 100% standard deviation per bin, which is far too noisy to
    read an adjacent-channel ratio off.

    Returns ``(frequencies, psd)`` with frequencies in cycles/sample spanning
    -0.5 to 0.5, and the PSD in linear power units.
    """
    x = np.asarray(x)
    segment = min(segment, len(x))
    step = max(1, int(segment * (1.0 - overlap)))

    if window == "hann":
        taper = np.hanning(segment)
    elif window == "rect":
        taper = np.ones(segment)
    else:
        raise ValueError(f"unknown window {window!r}; use 'hann' or 'rect'")
    # Normalise so windowing does not bias the power estimate.
    taper = taper / np.sqrt(np.mean(taper ** 2))

    starts = range(0, len(x) - segment + 1, step)
    accumulator = np.zeros(segment)
    count = 0
    for start in starts:
        chunk = x[start : start + segment] * taper
        accumulator += np.abs(np.fft.fft(chunk)) ** 2
        count += 1
    if count == 0:
        raise ValueError("signal is shorter than one segment")

    psd = np.fft.fftshift(accumulator / (count * segment))
    frequencies = np.fft.fftshift(np.fft.fftfreq(segment))
    return frequencies, psd


def band_power(
    frequencies: np.ndarray, psd: np.ndarray, low: float, high: float
) -> float:
    """Integrate PSD over a frequency band, in linear power units."""
    mask = (frequencies >= low) & (frequencies < high)
    if not np.any(mask):
        raise ValueError(f"no PSD bins in band [{low}, {high})")
    return float(np.sum(psd[mask]))
