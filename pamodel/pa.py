"""Ground-truth power amplifier models: the device under test.

These stand in for a measured amplifier. Everything downstream, the memory
polynomial and the neural model, is trying to learn one of these from its input
and output records alone, without being told the structure.

**The ground truth is deliberately not a memory polynomial.** If it were, a
memory polynomial could fit it exactly and the whole comparison would be rigged:
you would be measuring least squares against itself. Instead the default device
is a Wiener-Hammerstein cascade, an LTI filter followed by a memoryless
nonlinearity followed by another LTI filter. That is a genuinely different
structure, so every model in this project is an approximation of it, which is
the honest situation you are in with real hardware.

Two memoryless nonlinearities are provided:

``RappPA``
    Solid-state amplifier. Smooth compression into saturation, controlled by a
    knee sharpness. This is the usual choice for modern transmitters and is the
    default here, since 6G front ends are solid state.

``SalehPA``
    Travelling-wave tube amplifier. Stronger AM/PM conversion than a typical
    SSPA, kept because AM/PM is what makes predistortion genuinely
    two-dimensional rather than a gain lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RappPA",
    "SalehPA",
    "WienerHammerstein",
    "default_device",
    "input_backoff_db",
    "scale_for_backoff",
]

# Amplitude used to probe small-signal gain: far enough below saturation that
# compression is negligible, far enough above zero to stay clear of round-off.
_SMALL_SIGNAL = 1e-4


@dataclass
class RappPA:
    """Rapp solid-state amplifier model.

    AM/AM follows

        |y| = |x| / (1 + (|x| / Asat)^(2p))^(1/(2p))

    where ``p`` sets how sharply the amplifier bends into saturation. Large
    ``p`` approaches a hard limiter; ``p = 1`` is very soft.

    Real solid-state amplifiers also convert amplitude to phase, so an optional
    AM/PM term is included. It saturates rather than growing without bound,
    which is what measured devices do.
    """

    saturation: float = 1.0
    smoothness: float = 2.0
    am_pm: float = 0.0
    am_pm_saturation: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        magnitude = np.abs(x)
        ratio = magnitude / self.saturation
        gain = 1.0 / (1.0 + ratio ** (2 * self.smoothness)) ** (
            1.0 / (2 * self.smoothness)
        )
        if self.am_pm == 0.0:
            return x * gain
        squared = (magnitude / self.am_pm_saturation) ** 2
        phase = self.am_pm * squared / (1.0 + squared)
        return x * gain * np.exp(1j * phase)


@dataclass
class SalehPA:
    """Saleh travelling-wave-tube model.

        |y|   = alpha_a |x| / (1 + beta_a |x|^2)
        phase = alpha_p |x|^2 / (1 + beta_p |x|^2)
    """

    alpha_a: float = 2.1587
    beta_a: float = 1.1517
    alpha_p: float = 4.0033
    beta_p: float = 9.1040

    def __call__(self, x: np.ndarray) -> np.ndarray:
        magnitude = np.abs(x)
        squared = magnitude ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = self.alpha_a / (1.0 + self.beta_a * squared)
        phase = self.alpha_p * squared / (1.0 + self.beta_p * squared)
        return x * gain * np.exp(1j * phase)


@dataclass
class WienerHammerstein:
    """LTI filter, then a memoryless nonlinearity, then another LTI filter.

    The two filters are what create memory: the amplifier's response to a sample
    depends on the ones before it, so its AM/AM curve is not a curve at all but
    a band, widened by the envelope's recent history. That is precisely the
    effect a memoryless model cannot capture and a memory polynomial exists to
    approximate.

    Filter taps are complex, because the matching networks and bias circuits
    that cause memory in a real amplifier are not phase-flat.
    """

    nonlinearity: object = field(default_factory=RappPA)
    pre_taps: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.30 - 0.12j, -0.08 + 0.05j])
    )
    post_taps: np.ndarray = field(
        default_factory=lambda: np.array([1.0, -0.20 + 0.08j, 0.06 - 0.03j])
    )

    def __post_init__(self) -> None:
        self.pre_taps = np.asarray(self.pre_taps, dtype=complex)
        self.post_taps = np.asarray(self.post_taps, dtype=complex)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        # Causal FIR: out[n] = sum_k tap[k] * in[n-k], truncated to the input
        # length. numpy's 'same' mode centres the kernel instead, which shifts a
        # 3-tap response by one sample and makes the amplifier non-causal. That
        # offset silently corrupts any gain measured from an impulse.
        stage_one = np.convolve(x, self.pre_taps)[: len(x)]
        distorted = self.nonlinearity(stage_one)
        return np.convolve(distorted, self.post_taps)[: len(x)]

    def linear_gain(self, reference: np.ndarray | None = None) -> complex:
        """Complex small-signal gain in the signal band, by least squares.

        Gain is frequency dependent here, because the two FIR filters are not
        phase-flat, so "the gain" is only meaningful relative to a band. The
        estimate drives the amplifier with a heavily attenuated copy of a
        band-limited signal, far below compression, and solves ``y ≈ G x`` in
        the least-squares sense.

        This matters for two things downstream: NMSE is normalised by it, and
        predistortion targets ``y ≈ G x`` rather than ``y ≈ x``, since no
        predistorter can change how much the amplifier amplifies.
        """
        if reference is None:
            from .signals import multicarrier_signal

            reference = multicarrier_signal(
                4096, 0.125, rng=np.random.default_rng(12345)
            )
        reference = np.asarray(reference, dtype=complex)
        scale = _SMALL_SIGNAL / float(np.sqrt(np.mean(np.abs(reference) ** 2)))
        probe = reference * scale
        response = self(probe)
        return complex(np.vdot(probe, response) / np.vdot(probe, probe))


def default_device(am_pm: float = 0.35, seed: int | None = None) -> WienerHammerstein:
    """The device used throughout the project.

    A Rapp nonlinearity with moderate AM/PM inside a Wiener-Hammerstein cascade.
    Chosen so that at a realistic back-off the amplifier is clearly nonlinear
    but not clipping catastrophically, which is the regime predistortion is
    actually used in.
    """
    return WienerHammerstein(
        nonlinearity=RappPA(
            saturation=1.0, smoothness=2.0, am_pm=am_pm, am_pm_saturation=1.0
        )
    )


def input_backoff_db(x: np.ndarray, saturation: float = 1.0) -> float:
    """Input back-off: how far the signal's RMS sits below saturation, in dB."""
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if rms == 0:
        return float("inf")
    return float(20.0 * np.log10(saturation / rms))


def scale_for_backoff(
    x: np.ndarray, backoff_db: float, saturation: float = 1.0
) -> np.ndarray:
    """Scale a signal so its RMS sits ``backoff_db`` below saturation.

    Back-off is the central efficiency knob. Running closer to saturation, a
    smaller back-off, is more power efficient and more nonlinear. The entire
    argument for predistortion is that it buys back linearity so the amplifier
    can be operated at lower back-off.
    """
    target_rms = saturation * 10.0 ** (-backoff_db / 20.0)
    current = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if current == 0:
        return x
    return x * (target_rms / current)


def add_measurement_noise(
    y: np.ndarray, snr_db: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add complex Gaussian noise at a given SNR.

    Measured amplifier records are noisy, and a model that only ever sees clean
    simulated data will look better than it is. Including noise is what makes
    "validate against measurements" mean something.
    """
    rng = rng or np.random.default_rng(0)
    signal_power = float(np.mean(np.abs(y) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y))
    )
    return y + noise
