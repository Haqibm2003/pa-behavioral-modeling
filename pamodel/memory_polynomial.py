"""Memory polynomial and generalised memory polynomial models.

The standard behavioural model for a power amplifier. The memory polynomial is

    y(n) = sum_k sum_m  a_km * x(n-m) * |x(n-m)|^(k-1)

which is a polynomial in the envelope, evaluated on a tap-delay line. The
generalised form adds cross terms, where the sample being weighted and the
envelope doing the weighting come from different taps:

    + sum  b_kml * x(n-m) * |x(n-m-l)|^(k-1)     lagging
    + sum  c_kml * x(n-m) * |x(n-m+l)|^(k-1)     leading

Three properties make this the workhorse of the field.

**It is linear in its coefficients.** However nonlinear the amplifier, the model
is a linear combination of fixed basis functions, so identification is ordinary
least squares. There is no local minimum to get stuck in and no learning rate to
tune; the fit is a single matrix solve and it is reproducible to the last bit.

**Only odd orders matter.** For a bandpass signal, even-order products land far
from the carrier and are filtered away. Including them wastes conditioning on
basis functions that cannot explain in-band behaviour, so the default order set
is (1, 3, 5, 7).

**It is badly conditioned if you are careless.** The columns |x|^(k-1) x for
successive k are strongly correlated, and the condition number grows quickly
with order. Two defences are used here: the input is normalised to unit RMS
before the basis is built, so high powers neither overflow nor vanish, and
ridge regularisation is available. The condition number is reported after every
fit rather than left for the user to discover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MemoryPolynomial", "delay"]


def delay(x: np.ndarray, amount: int) -> np.ndarray:
    """Shift a signal by ``amount`` samples, zero-padding the start.

    Positive shifts move the signal later, so ``delay(x, 1)[n] == x[n-1]``.
    Negative shifts look ahead, which is what the leading cross terms need.
    """
    x = np.asarray(x)
    if amount == 0:
        return x.copy()
    out = np.zeros_like(x)
    if amount > 0:
        out[amount:] = x[:-amount]
    else:
        out[:amount] = x[-amount:]
    return out


@dataclass
class MemoryPolynomial:
    """Memory polynomial / GMP behavioural model, identified by least squares.

    ``orders``
        Polynomial orders to include. Odd-only by default, for the reason above.
    ``memory_depth``
        Number of delay taps, so ``m`` runs from 0 to ``memory_depth - 1``.
    ``lag_cross`` / ``lead_cross``
        Number of GMP cross-term offsets. Zero on both gives a plain memory
        polynomial.
    ``ridge``
        Tikhonov regularisation weight, applied to the normalised basis.
    """

    orders: tuple[int, ...] = (1, 3, 5, 7)
    memory_depth: int = 4
    lag_cross: int = 0
    lead_cross: int = 0
    ridge: float = 0.0

    coefficients: np.ndarray | None = field(default=None, init=False)
    input_scale: float = field(default=1.0, init=False)
    condition_number: float = field(default=float("nan"), init=False)

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------

    @property
    def num_coefficients(self) -> int:
        """Number of basis functions, and so of fitted coefficients."""
        base = len(self.orders) * self.memory_depth
        higher = [k for k in self.orders if k > 1]
        cross = len(higher) * self.memory_depth * (self.lag_cross + self.lead_cross)
        return base + cross

    @property
    def transient(self) -> int:
        """Samples at the start whose basis is contaminated by zero padding."""
        return self.memory_depth + max(self.lag_cross, self.lead_cross)

    def basis(self, x: np.ndarray) -> np.ndarray:
        """Build the regressor matrix, shape ``(len(x), num_coefficients)``.

        The input is scaled to unit RMS first. Without it, a signal whose RMS is
        0.1 makes the seventh-order column smaller than the first-order column
        by a factor of a million, and least squares on that matrix is dominated
        by round-off rather than by the data.
        """
        x = np.asarray(x, dtype=complex)
        x = x / self.input_scale

        columns: list[np.ndarray] = []

        for k in self.orders:
            for m in range(self.memory_depth):
                delayed = delay(x, m)
                columns.append(delayed * np.abs(delayed) ** (k - 1))

        higher = [k for k in self.orders if k > 1]
        for k in higher:
            for m in range(self.memory_depth):
                delayed = delay(x, m)
                for offset in range(1, self.lag_cross + 1):
                    envelope = np.abs(delay(x, m + offset))
                    columns.append(delayed * envelope ** (k - 1))
                for offset in range(1, self.lead_cross + 1):
                    envelope = np.abs(delay(x, m - offset))
                    columns.append(delayed * envelope ** (k - 1))

        return np.column_stack(columns)

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MemoryPolynomial":
        """Identify coefficients from an input/output record by least squares."""
        x = np.asarray(x, dtype=complex)
        y = np.asarray(y, dtype=complex)
        if len(x) != len(y):
            raise ValueError(
                f"input and output must be the same length, got {len(x)} and {len(y)}"
            )

        self.input_scale = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        if self.input_scale == 0:
            raise ValueError("input signal is all zeros")

        matrix = self.basis(x)
        skip = self.transient
        matrix, target = matrix[skip:], y[skip:]

        if len(target) < self.num_coefficients:
            raise ValueError(
                f"{len(target)} usable samples cannot identify "
                f"{self.num_coefficients} coefficients; use a longer record or a "
                f"smaller model"
            )

        singular_values = np.linalg.svd(matrix, compute_uv=False)
        self.condition_number = float(
            singular_values[0] / singular_values[-1]
            if singular_values[-1] > 0
            else np.inf
        )

        if self.ridge > 0:
            # Solve the normal equations with a Tikhonov penalty.
            gram = matrix.conj().T @ matrix
            gram += self.ridge * np.trace(gram).real / len(gram) * np.eye(len(gram))
            self.coefficients = np.linalg.solve(gram, matrix.conj().T @ target)
        else:
            self.coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the identified model to an input signal."""
        if self.coefficients is None:
            raise RuntimeError("model has not been fitted; call fit() first")
        return self.basis(x) @ self.coefficients

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    def describe(self) -> str:
        """One-line summary for result tables."""
        kind = "GMP" if (self.lag_cross or self.lead_cross) else "MP"
        orders = ",".join(str(k) for k in self.orders)
        detail = f"K=[{orders}] M={self.memory_depth}"
        if self.lag_cross or self.lead_cross:
            detail += f" lag={self.lag_cross} lead={self.lead_cross}"
        return f"{kind} ({detail}, {self.num_coefficients} coeff)"
