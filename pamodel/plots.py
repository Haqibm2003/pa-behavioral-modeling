"""Figures.

Matplotlib is imported lazily so the rest of the package keeps working without
it. Every function takes an optional axis so figures can be composed.

The styling choice worth explaining is the use of scatter plots with heavy
transparency for AM/AM and AM/PM. A line plot would imply the amplifier has a
single-valued characteristic, which is exactly the assumption memory violates.
Drawing every sample as a translucent point shows the characteristic as the
band it really is, and the width of that band is the memory.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "plot_am_am",
    "plot_am_pm",
    "plot_spectra",
    "plot_backoff_sweep",
    "plot_model_comparison",
    "plot_training_history",
]


def _axis(ax=None, figsize=(6, 4)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_am_am(x, y, ax=None, gain=1.0, label=None, color="C0", max_points=6000):
    """Input magnitude against output magnitude, normalised by small-signal gain."""
    ax = _axis(ax)
    magnitude_in = np.abs(x)
    magnitude_out = np.abs(np.asarray(y) / gain)
    if len(magnitude_in) > max_points:
        step = len(magnitude_in) // max_points
        magnitude_in, magnitude_out = magnitude_in[::step], magnitude_out[::step]
    ax.scatter(magnitude_in, magnitude_out, s=1, alpha=0.08, color=color, label=label)
    limit = float(np.max(magnitude_in))
    ax.plot([0, limit], [0, limit], "k--", lw=1, alpha=0.6, label="ideal linear")
    ax.set_xlabel("input magnitude |x|")
    ax.set_ylabel("output magnitude |y| / G")
    ax.set_title("AM/AM")
    ax.legend(loc="upper left", fontsize=8, markerscale=6)
    return ax


def plot_am_pm(x, y, ax=None, color="C1", max_points=6000):
    """Input magnitude against the phase shift the amplifier applies."""
    from .metrics import am_pm

    ax = _axis(ax)
    magnitude, phase = am_pm(x, y)
    if len(magnitude) > max_points:
        step = len(magnitude) // max_points
        magnitude, phase = magnitude[::step], phase[::step]
    ax.scatter(magnitude, phase, s=1, alpha=0.08, color=color)
    ax.set_xlabel("input magnitude |x|")
    ax.set_ylabel("phase shift (degrees)")
    ax.set_title("AM/PM")
    return ax


def plot_spectra(signals, labels, ax=None, occupied_fraction=0.125, channel_spacing=1.2):
    """Overlaid power spectra with the ACPR measurement bands marked."""
    from .signals import welch_psd

    ax = _axis(ax, figsize=(7, 4.5))
    for index, (signal, label) in enumerate(zip(signals, labels)):
        frequencies, psd = welch_psd(signal, segment=4096)
        reference = np.max(psd)
        ax.plot(
            frequencies,
            10 * np.log10(np.maximum(psd / reference, 1e-16)),
            lw=1.1,
            label=label,
            color=f"C{index}",
        )

    half = occupied_fraction / 2
    offset = channel_spacing * occupied_fraction
    ax.axvspan(-half, half, color="grey", alpha=0.12)
    for sign in (-1, 1):
        ax.axvspan(
            sign * offset - half, sign * offset + half, color="tab:red", alpha=0.10
        )
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-130, 5)
    ax.set_xlabel("normalised frequency")
    ax.set_ylabel("PSD (dB, peak-normalised)")
    ax.set_title("Spectrum: occupied band (grey), adjacent channels (red)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def plot_backoff_sweep(sweep, ax=None, acpr_target=-45.0):
    """ACPR against input back-off, with and without predistortion."""
    ax = _axis(ax, figsize=(6.5, 4.5))
    ax.plot(sweep["backoff_db"], sweep["acpr_raw"], "o-", label="no DPD", color="C3")
    if not np.all(np.isnan(sweep["acpr_dpd"])):
        ax.plot(
            sweep["backoff_db"], sweep["acpr_dpd"], "s-", label="with DPD", color="C0"
        )
    ax.axhline(acpr_target, color="k", ls="--", lw=1, label=f"{acpr_target:g} dBc limit")
    ax.set_xlabel("input back-off (dB)")
    ax.set_ylabel("worst-side ACPR (dBc)")
    ax.set_title("Lower back-off is more efficient and less linear")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def plot_model_comparison(rows, ax=None):
    """Test NMSE against parameter count, log x-axis.

    ``rows`` is a sequence of ``(label, parameters, nmse_db, family)``.
    """
    ax = _axis(ax, figsize=(6.5, 4.5))
    families = {}
    for label, parameters, nmse, family in rows:
        families.setdefault(family, []).append((parameters, nmse, label))
    for index, (family, points) in enumerate(sorted(families.items())):
        points.sort()
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            "o-",
            label=family,
            color=f"C{index}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("free parameters")
    ax.set_ylabel("test NMSE (dB)")
    ax.set_title("Accuracy against model size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    return ax


def plot_training_history(histories, labels, ax=None):
    """Training loss curves."""
    ax = _axis(ax)
    for index, (history, label) in enumerate(zip(histories, labels)):
        ax.semilogy(history, lw=1.2, label=label, color=f"C{index}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    ax.set_title("Training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    return ax
