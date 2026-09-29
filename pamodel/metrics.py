"""Figures of merit for behavioural models and for linearised transmitters.

Two families, and they answer different questions.

**Model accuracy.** NMSE asks how closely a model reproduces the amplifier's
output sample by sample. It is the number you minimise when fitting, and it is
the one that can mislead: a model can post a respectable NMSE while getting the
out-of-band behaviour wrong, because in-band energy dominates the error sum.

**System performance.** ACPR asks how much power the transmitter splatters into
its neighbours' spectrum. That is what a regulator measures and what actually
constrains how hard the amplifier can be driven. A behavioural model is only
useful if improving its NMSE also improves the ACPR you get after predistortion,
and this module exists so that claim can be checked rather than assumed.
"""

from __future__ import annotations

import numpy as np

from .signals import band_power, welch_psd

__all__ = [
    "estimate_delay",
    "align",
    "fit_gain",
    "nmse_db",
    "acpr_db",
    "evm_percent",
    "am_am",
    "am_pm",
    "memory_spread",
]


def estimate_delay(reference: np.ndarray, signal: np.ndarray, max_lag: int = 64) -> int:
    """Integer sample delay of ``signal`` relative to ``reference``.

    Found by maximising the magnitude of the cross-correlation. Real captures
    are never sample-aligned, because cables and converters add latency, and an
    unaligned comparison reports the delay as modelling error. Even in
    simulation the amplifier's own filters impose a group delay, so this is not
    optional.
    """
    reference = np.asarray(reference)
    signal = np.asarray(signal)
    length = min(len(reference), len(signal))
    reference, signal = reference[:length], signal[:length]

    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.empty(len(lags))
    for index, lag in enumerate(lags):
        if lag >= 0:
            a, b = reference[: length - lag], signal[lag:]
        else:
            a, b = reference[-lag:], signal[: length + lag]
        scores[index] = abs(np.vdot(a, b)) if len(a) else 0.0
    return int(lags[int(np.argmax(scores))])


def align(
    reference: np.ndarray, signal: np.ndarray, max_lag: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Trim a pair to their best integer alignment."""
    lag = estimate_delay(reference, signal, max_lag)
    if lag > 0:
        return reference[: len(signal) - lag], signal[lag:]
    if lag < 0:
        return reference[-lag:], signal[: len(reference) + lag]
    length = min(len(reference), len(signal))
    return reference[:length], signal[:length]


def fit_gain(x: np.ndarray, y: np.ndarray) -> complex:
    """Least-squares complex gain such that ``y ≈ G x``."""
    x = np.asarray(x)
    y = np.asarray(y)
    return complex(np.vdot(x, y) / np.vdot(x, x))


def nmse_db(y_true: np.ndarray, y_model: np.ndarray) -> float:
    """Normalised mean-square error in dB.

        NMSE = 10 log10( sum |y_true - y_model|^2 / sum |y_true|^2 )

    Lower is better. As a rule of thumb, -25 dB is a usable model, -35 dB is
    good, and below -45 dB you are usually fitting measurement noise.
    """
    y_true = np.asarray(y_true)
    y_model = np.asarray(y_model)
    length = min(len(y_true), len(y_model))
    y_true, y_model = y_true[:length], y_model[:length]

    error = float(np.sum(np.abs(y_true - y_model) ** 2))
    signal = float(np.sum(np.abs(y_true) ** 2))
    if signal == 0:
        return float("nan")
    if error == 0:
        return float("-inf")
    return float(10.0 * np.log10(error / signal))


def acpr_db(
    y: np.ndarray,
    occupied_fraction: float = 0.125,
    segment: int = 4096,
    channel_spacing: float = 1.2,
) -> tuple[float, float]:
    """Adjacent channel power ratio, lower and upper, in dBc.

    With ``B`` the occupied bandwidth and ``S = channel_spacing * B`` the centre
    spacing between channels, the measurement bands are

        occupied        [-B/2,     B/2]
        lower adjacent  [-S - B/2, -S + B/2]
        upper adjacent  [ S - B/2,  S + B/2]

    **The spacing is not cosmetic.** An adjacent band that begins exactly at the
    occupied band edge measures the analysis window's own spectral leakage
    rather than the amplifier. Measured here, a clean undistorted signal whose
    guard band sits 140 dB down still reported roughly -28 dBc that way, so the
    floor of the measurement was 12 dB worse than any distortion being studied
    and every number was leakage. Real standards define adjacent channels on a
    spacing slightly wider than the occupied bandwidth for the same reason; the
    default of 1.2 leaves a gap of 0.2 B on each side.

    More negative is better. Both sides are returned because AM/PM conversion
    makes regrowth asymmetric, and averaging the two hides that.
    """
    frequencies, psd = welch_psd(y, segment=segment)
    half = occupied_fraction / 2.0
    offset = channel_spacing * occupied_fraction

    occupied = band_power(frequencies, psd, -half, half)
    lower = band_power(frequencies, psd, -offset - half, -offset + half)
    upper = band_power(frequencies, psd, offset - half, offset + half)

    if occupied == 0:
        return float("nan"), float("nan")
    return (
        float(10.0 * np.log10(lower / occupied)),
        float(10.0 * np.log10(upper / occupied)),
    )


def evm_percent(x: np.ndarray, y: np.ndarray, gain: complex | None = None) -> float:
    """Error vector magnitude as a percentage of RMS reference amplitude.

    The linear gain is divided out first, because EVM measures distortion, not
    amplification. If ``gain`` is not supplied it is fitted by least squares,
    which is what a receiver's equaliser would do.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    length = min(len(x), len(y))
    x, y = x[:length], y[:length]

    if gain is None:
        gain = fit_gain(x, y)
    error = y - gain * x
    reference_power = float(np.mean(np.abs(gain * x) ** 2))
    if reference_power == 0:
        return float("nan")
    return float(100.0 * np.sqrt(np.mean(np.abs(error) ** 2) / reference_power))


def am_am(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Input magnitude against output magnitude.

    For a memoryless amplifier this is a single-valued curve. Memory turns it
    into a band, because the output depends on the envelope's recent history as
    well as its present value. The width of that band is a direct visual
    measure of how much memory the device has, and it is the reason a
    memoryless model cannot be enough.
    """
    return np.abs(x), np.abs(y)


def am_pm(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Input magnitude against the phase shift the amplifier applies, in degrees."""
    x = np.asarray(x)
    y = np.asarray(y)
    phase = np.angle(y * np.conj(x))
    return np.abs(x), np.degrees(phase)


def memory_spread(x: np.ndarray, y: np.ndarray, window: int = 101) -> float:
    """Quantify memory as the AM/AM scatter left after removing the curve itself.

    A memoryless amplifier has a single-valued AM/AM characteristic: given |x|,
    the output magnitude is determined. Memory breaks that, because the output
    also depends on the envelope's recent history, so the characteristic becomes
    a band.

    Measuring the band width naively, as the spread of |y| inside a bin of |x|,
    does not work: most of what it measures is the AM/AM curve's own slope
    across the bin, and a memoryless device scores almost the same as one with
    strong memory. Instead the samples are sorted by |x|, a moving average
    estimates the underlying curve, and the residual scatter about that curve is
    what is reported. On the device used here this separates a memoryless
    amplifier from one with memory by a factor of about five, where the naive
    binned measure separated them by a factor of one.

    Returns the residual standard deviation as a fraction of mean output
    magnitude, so it is dimensionless and comparable across drive levels.
    """
    magnitude_in = np.abs(np.asarray(x))
    magnitude_out = np.abs(np.asarray(y))

    order = np.argsort(magnitude_in)
    sorted_out = magnitude_out[order]

    trend = np.convolve(sorted_out, np.ones(window) / window, mode="same")
    interior = slice(window, -window)
    residual = sorted_out[interior] - trend[interior]
    mean_level = float(np.mean(sorted_out[interior]))
    if mean_level == 0:
        return float("nan")
    return float(np.std(residual) / mean_level)
