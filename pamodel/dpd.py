"""Digital predistortion by the indirect learning architecture.

This is where a behavioural model stops being a description and starts being
useful. Knowing what the amplifier does is only worth the effort if it lets you
cancel what it does.

**The indirect learning trick.** What you actually want is the amplifier's
pre-inverse: a block *f* such that ``PA(f(x)) = G x``. Training that directly is
awkward, because the thing you want to invert sits between your parameters and
your loss, so the gradient has to pass through the amplifier.

The indirect approach sidesteps it. Train a *post*-inverse instead, a block that
takes the amplifier's output and reconstructs its input:

    minimise  || f(y/G) - z ||

where ``z`` is what was fed to the amplifier and ``y`` is what came out. This is
an ordinary regression with no amplifier in the loop, so with a memory
polynomial it is a single least-squares solve. The post-inverse is then deployed
unchanged as the pre-inverse. That substitution is exact only when the
amplifier's behaviour is invertible over the operating range, which is why the
procedure is iterated: each pass linearises further and the next post-inverse is
trained on data closer to the intended operating point.

**Why this is an efficiency technique, not a quality technique.** Amplifier
efficiency rises as you approach saturation, and distortion rises with it. The
spectral mask, not the amplifier, is usually what sets how hard you may drive.
Predistortion buys back linearity, which lets you run at lower back-off for the
same emitted adjacent-channel power. The back-off recovered is the real figure
of merit, and :func:`backoff_sweep` measures it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .memory_polynomial import MemoryPolynomial
from .metrics import acpr_db, fit_gain, nmse_db

__all__ = ["IndirectLearningDPD", "backoff_sweep", "recovered_backoff_db"]


def _default_model() -> MemoryPolynomial:
    return MemoryPolynomial(orders=(1, 3, 5, 7, 9), memory_depth=4)


@dataclass
class IndirectLearningDPD:
    """Predistorter trained by the indirect learning architecture.

    ``model_factory`` builds a fresh behavioural model for each iteration; it
    must expose ``fit(x, y)`` and ``predict(x)``. Any model in this package
    works, though a memory polynomial is the usual choice because its fit is a
    closed-form solve rather than an optimisation.
    """

    model_factory: Callable[[], object] = _default_model
    iterations: int = 4
    preserve_power: bool = True

    model: object | None = field(default=None, init=False)
    gain: complex = field(default=1.0 + 0j, init=False)
    history: list[dict] = field(default_factory=list, init=False)

    def train(
        self,
        amplifier: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        gain: complex | None = None,
        occupied_fraction: float = 0.125,
    ) -> "IndirectLearningDPD":
        """Train against a callable amplifier using the signal ``x``."""
        x = np.asarray(x, dtype=complex)

        if gain is None:
            # Fit the small-signal gain from a heavily attenuated probe, so
            # compression does not bias the target the predistorter aims at.
            probe = x * 1e-4
            gain = fit_gain(probe, amplifier(probe))
        self.gain = gain

        self.history = []
        predistorted = x.copy()

        for iteration in range(self.iterations):
            output = amplifier(predistorted)
            normalised = output / self.gain

            model = self.model_factory()
            # Post-inverse: map the amplifier's (normalised) output back to the
            # input that produced it.
            model.fit(normalised, predistorted)
            self.model = model

            candidate = model.predict(x)
            if self.preserve_power:
                candidate = _match_power(candidate, x)

            lower, upper = acpr_db(amplifier(candidate), occupied_fraction)
            self.history.append(
                {
                    "iteration": iteration + 1,
                    "acpr_lower": lower,
                    "acpr_upper": upper,
                    "inverse_nmse": nmse_db(
                        predistorted[model.transient :],
                        model.predict(normalised)[model.transient :],
                    ),
                }
            )
            predistorted = candidate

        return self

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Predistort a signal."""
        if self.model is None:
            raise RuntimeError("predistorter has not been trained; call train() first")
        z = self.model.predict(np.asarray(x, dtype=complex))
        return _match_power(z, x) if self.preserve_power else z

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x)


def _match_power(z: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rescale so the predistorted signal carries the same average power.

    Predistortion expands the envelope to pre-compensate compression, which
    raises average power and quietly pushes the amplifier further into
    saturation. Holding power constant keeps a back-off comparison honest:
    otherwise the predistorted case is being driven harder than the reference
    and the two ACPR numbers are not measured at the same operating point.
    """
    reference_power = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    current = float(np.sqrt(np.mean(np.abs(z) ** 2)))
    if current == 0:
        return z
    return z * (reference_power / current)


def backoff_sweep(
    amplifier: Callable[[np.ndarray], np.ndarray],
    signal: np.ndarray,
    backoffs_db: np.ndarray,
    predistorter: IndirectLearningDPD | None = None,
    occupied_fraction: float = 0.125,
    retrain_each_point: bool = True,
) -> dict:
    """Measure ACPR against input back-off, with and without predistortion.

    ``retrain_each_point`` retrains the predistorter at every back-off, which is
    what a real transmitter does through its adaptation loop. Training once at
    one operating point and reusing the coefficients everywhere flatters the
    result, because a predistorter is only valid near the drive level it was
    fitted at.
    """
    from .pa import scale_for_backoff

    results = {"backoff_db": [], "acpr_raw": [], "acpr_dpd": []}

    for backoff in backoffs_db:
        driven = scale_for_backoff(signal, float(backoff))

        raw_lower, raw_upper = acpr_db(amplifier(driven), occupied_fraction)
        results["backoff_db"].append(float(backoff))
        results["acpr_raw"].append(max(raw_lower, raw_upper))

        if predistorter is None:
            results["acpr_dpd"].append(float("nan"))
            continue

        if retrain_each_point:
            trained = IndirectLearningDPD(
                model_factory=predistorter.model_factory,
                iterations=predistorter.iterations,
                preserve_power=predistorter.preserve_power,
            ).train(amplifier, driven, occupied_fraction=occupied_fraction)
        else:
            trained = predistorter

        corrected = amplifier(trained.apply(driven))
        dpd_lower, dpd_upper = acpr_db(corrected, occupied_fraction)
        results["acpr_dpd"].append(max(dpd_lower, dpd_upper))

    for key in results:
        results[key] = np.array(results[key])
    return results


def recovered_backoff_db(sweep: dict, acpr_target: float = -45.0) -> float:
    """Back-off saved by predistortion at a given ACPR limit, in dB.

    Interpolates each curve to find the lowest back-off meeting the target, and
    returns the difference. This is the number that translates into efficiency:
    every dB of back-off recovered is a dB less power wasted keeping the
    amplifier linear.
    """
    backoff = np.asarray(sweep["backoff_db"])
    order = np.argsort(backoff)
    backoff = backoff[order]

    def minimum_backoff(acpr: np.ndarray) -> float:
        acpr = np.asarray(acpr)[order]
        meeting = np.where(acpr <= acpr_target)[0]
        if len(meeting) == 0:
            return float("nan")
        index = meeting[0]
        if index == 0:
            return float(backoff[0])
        # Linear interpolation between the bracketing points.
        x0, x1 = backoff[index - 1], backoff[index]
        y0, y1 = acpr[index - 1], acpr[index]
        if y1 == y0:
            return float(x1)
        return float(x0 + (acpr_target - y0) * (x1 - x0) / (y1 - y0))

    raw = minimum_backoff(sweep["acpr_raw"])
    dpd = minimum_backoff(sweep["acpr_dpd"])
    if np.isnan(raw) or np.isnan(dpd):
        return float("nan")
    return float(raw - dpd)
