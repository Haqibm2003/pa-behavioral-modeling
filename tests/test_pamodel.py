"""Tests for the behavioural modelling package.

Expected values are derived from theory rather than captured from a previous
run, so a change in behaviour fails instead of quietly re-baselining.

The load-bearing tests are:

``test_recovers_a_known_memory_polynomial``
    Generate data from a memory polynomial with known coefficients and check
    least squares recovers them to machine precision. If the basis construction
    or the delay convention were wrong, this fails and nothing else would
    obviously look broken.

``test_gradients_match_finite_differences``
    The neural network's backward pass is written by hand, so it is checked
    against numerical differentiation rather than trusted.

``test_dpd_fails_when_peaks_exceed_saturation``
    Asserts a physical limit rather than a software behaviour: a saturated
    amplifier is not invertible, so predistortion must fail there. A version
    that "improved" ACPR in that regime would be wrong.
"""

import numpy as np
import pytest

from pamodel.dpd import IndirectLearningDPD, recovered_backoff_db
from pamodel.memory_polynomial import MemoryPolynomial, delay
from pamodel.metrics import (
    acpr_db,
    align,
    estimate_delay,
    evm_percent,
    fit_gain,
    memory_spread,
    nmse_db,
)
from pamodel.neural import MLP, NeuralPAModel, TorchTDNN, envelope_features
from pamodel.pa import (
    RappPA,
    SalehPA,
    WienerHammerstein,
    default_device,
    input_backoff_db,
    scale_for_backoff,
)
from pamodel.signals import (
    band_power,
    multicarrier_signal,
    occupied_edge,
    papr_db,
    qam_symbols,
    rms,
    set_rms,
    welch_psd,
)

OCCUPIED = 0.125


@pytest.fixture(scope="module")
def signal():
    return multicarrier_signal(8192, OCCUPIED, rng=np.random.default_rng(1))


@pytest.fixture(scope="module")
def test_signal():
    return multicarrier_signal(8192, OCCUPIED, rng=np.random.default_rng(99))


class TestSignals:
    def test_qam_has_unit_average_power(self):
        symbols = qam_symbols(4000, 16, np.random.default_rng(0))
        assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0, rel=1e-9)

    def test_qam_rejects_non_square_order(self):
        with pytest.raises(ValueError, match="perfect square"):
            qam_symbols(10, 18)

    def test_signal_is_unit_rms(self, signal):
        assert rms(signal) == pytest.approx(1.0, rel=1e-12)

    def test_papr_is_realistic_for_multicarrier(self, signal):
        """Summing many subcarriers gives a near-Rayleigh envelope, 8-12 dB PAPR."""
        assert 7.0 < papr_db(signal) < 13.0

    def test_guard_band_is_numerically_clean(self, signal):
        """A single IFFT has no symbol boundaries, so the guard band is empty.

        This headroom is what makes ACPR measurable: adjacent-channel ratios of
        interest are 25-50 dBc, so the generator's own leakage must be far below
        that or it dominates the measurement.
        """
        frequencies, psd = welch_psd(signal, segment=2048)
        edge = occupied_edge(OCCUPIED)
        occupied = band_power(frequencies, psd, -edge, edge)
        guard = band_power(frequencies, psd, 1.5 * edge, 3.5 * edge)
        assert 10 * np.log10(occupied / guard) > 80.0

    def test_set_rms(self):
        x = np.array([1 + 1j, 2 - 1j, 0.5j])
        assert rms(set_rms(x, 3.0)) == pytest.approx(3.0)

    def test_rejects_full_occupancy(self):
        with pytest.raises(ValueError, match="guard band"):
            multicarrier_signal(1024, 1.0)


class TestAmplifier:
    def test_rapp_compresses_monotonically(self):
        amplifier = RappPA(saturation=1.0, smoothness=2.0)
        magnitudes = np.array([0.1, 0.3, 0.6, 0.9, 1.5, 3.0], dtype=complex)
        output = np.abs(amplifier(magnitudes))
        gains = output / np.abs(magnitudes)
        assert np.all(np.diff(gains) < 0), "gain must fall as drive rises"
        assert gains[0] == pytest.approx(1.0, abs=1e-3)

    def test_rapp_saturates(self):
        """Output magnitude approaches Asat however hard it is driven."""
        amplifier = RappPA(saturation=1.0, smoothness=3.0)
        assert abs(amplifier(np.array([1e3 + 0j]))[0]) == pytest.approx(1.0, rel=1e-2)

    def test_rapp_am_pm_is_zero_when_disabled(self):
        amplifier = RappPA(am_pm=0.0)
        x = np.array([0.5 + 0.5j])
        assert np.angle(amplifier(x)[0]) == pytest.approx(np.angle(x[0]))

    def test_saleh_matches_its_formula(self):
        amplifier = SalehPA()
        magnitude = 0.7
        x = np.array([magnitude + 0j])
        expected = amplifier.alpha_a * magnitude / (1 + amplifier.beta_a * magnitude ** 2)
        assert abs(amplifier(x)[0]) == pytest.approx(expected)

    def test_flat_linear_device_has_unit_gain(self):
        """Validates the gain estimator itself against a known answer."""
        device = WienerHammerstein(
            nonlinearity=RappPA(saturation=1e9), pre_taps=[1.0], post_taps=[1.0]
        )
        assert abs(device.linear_gain()) == pytest.approx(1.0, rel=1e-6)

    def test_filtering_is_causal(self):
        """Output must not respond before the input arrives.

        numpy's 'same' convolution centres the kernel, which makes a 3-tap
        response start one sample early. That offset silently corrupts any gain
        measured from an impulse, so causality is asserted.
        """
        device = default_device()
        impulse = np.zeros(64, dtype=complex)
        impulse[32] = 0.01
        response = device(impulse)
        assert np.allclose(response[:32], 0, atol=1e-15)
        assert abs(response[32]) > 0

    def test_memory_widens_the_am_am_characteristic(self):
        """A memoryless device has a single-valued AM/AM curve; memory spreads it."""
        x = multicarrier_signal(4096, OCCUPIED, rng=np.random.default_rng(3))
        x = scale_for_backoff(x, 10.0)

        memoryless = WienerHammerstein(
            nonlinearity=RappPA(am_pm=0.3), pre_taps=[1.0], post_taps=[1.0]
        )
        with_memory = default_device()

        flat = memory_spread(x, memoryless(x))
        banded = memory_spread(x, with_memory(x))

        # The absolute floor of the estimator depends on record length, so the
        # claim under test is the ratio: memory must widen the band markedly.
        assert banded > 2.5 * flat, f"memory spread {banded:.4f} vs memoryless {flat:.4f}"

    def test_backoff_round_trips(self, signal):
        scaled = scale_for_backoff(signal, 8.0)
        assert input_backoff_db(scaled) == pytest.approx(8.0, abs=1e-9)


class TestMemoryPolynomial:
    def test_delay_shifts_and_zero_pads(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(delay(x, 1), [0, 1, 2, 3])
        assert np.allclose(delay(x, -1), [2, 3, 4, 0])
        assert np.allclose(delay(x, 0), x)

    def test_recovers_a_known_memory_polynomial(self):
        """Exact recovery when the data really is a memory polynomial.

        Coefficients are recovered to machine precision, which is only possible
        if the basis ordering, the delay convention and the least-squares solve
        all agree with each other.
        """
        rng = np.random.default_rng(7)
        x = multicarrier_signal(8192, OCCUPIED, rng=rng) * 0.4

        truth = MemoryPolynomial(orders=(1, 3, 5), memory_depth=3)
        truth.input_scale = float(rms(x))
        known = rng.standard_normal(truth.num_coefficients) + 1j * rng.standard_normal(
            truth.num_coefficients
        )
        truth.coefficients = known
        y = truth.predict(x)

        fitted = MemoryPolynomial(orders=(1, 3, 5), memory_depth=3).fit(x, y)
        assert np.allclose(fitted.coefficients, known, rtol=1e-8, atol=1e-8)
        skip = fitted.transient
        assert nmse_db(y[skip:], fitted.predict(x)[skip:]) < -200

    def test_accuracy_improves_with_order(self, signal, test_signal):
        device = default_device()
        x_train = scale_for_backoff(signal, 12.0)
        x_test = scale_for_backoff(test_signal, 12.0)
        y_train, y_test = device(x_train), device(x_test)

        scores = []
        for orders in [(1,), (1, 3), (1, 3, 5, 7)]:
            model = MemoryPolynomial(orders=orders, memory_depth=4).fit(x_train, y_train)
            skip = model.transient
            scores.append(nmse_db(y_test[skip:], model.predict(x_test)[skip:]))
        assert all(later < earlier for earlier, later in zip(scores, scores[1:]))
        assert scores[-1] < -45.0

    def test_excess_order_can_hurt_at_low_drive(self, signal, test_signal):
        """More terms is not automatically better.

        At 12 dB back-off the amplifier is only mildly compressed, so the
        9th- and 11th-order basis columns are barely excited. Least squares
        then fits round-off in those directions and generalisation gets worse,
        here by about 8 dB. Model order has to be matched to drive level.
        """
        device = default_device()
        x_train = scale_for_backoff(signal, 12.0)
        x_test = scale_for_backoff(test_signal, 12.0)
        y_train, y_test = device(x_train), device(x_test)

        def score(orders):
            model = MemoryPolynomial(orders=orders, memory_depth=4).fit(x_train, y_train)
            skip = model.transient
            return nmse_db(y_test[skip:], model.predict(x_test)[skip:])

        assert score((1, 3, 5, 7, 9, 11)) > score((1, 3, 5, 7))

    def test_coefficient_count_matches_basis_width(self):
        model = MemoryPolynomial(orders=(1, 3, 5), memory_depth=4, lag_cross=2, lead_cross=1)
        x = np.ones(64, dtype=complex)
        model.input_scale = 1.0
        assert model.basis(x).shape[1] == model.num_coefficients

    def test_refuses_underdetermined_fit(self):
        x = np.ones(10, dtype=complex)
        with pytest.raises(ValueError, match="cannot identify"):
            MemoryPolynomial(orders=(1, 3, 5, 7), memory_depth=5).fit(x, x)

    def test_ridge_reduces_coefficient_norm(self, signal):
        device = default_device()
        x = scale_for_backoff(signal, 10.0)
        y = device(x)
        plain = MemoryPolynomial(orders=(1, 3, 5, 7, 9, 11), memory_depth=5).fit(x, y)
        ridged = MemoryPolynomial(
            orders=(1, 3, 5, 7, 9, 11), memory_depth=5, ridge=1e-3
        ).fit(x, y)
        assert np.linalg.norm(ridged.coefficients) < np.linalg.norm(plain.coefficients)

    def test_predict_before_fit_is_an_error(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            MemoryPolynomial().predict(np.ones(10, dtype=complex))


class TestNeural:
    def test_feature_matrix_shape(self):
        x = np.ones(50, dtype=complex)
        features = envelope_features(x, memory_depth=4, magnitude_powers=(1, 2))
        assert features.shape == (50, 4 * (2 + 2))

    def test_gradients_match_finite_differences(self):
        """The backward pass is hand-written, so it is checked numerically."""
        rng = np.random.default_rng(0)
        network = MLP([6, 7, 5, 2], rng)
        features = rng.standard_normal((11, 6))
        targets = rng.standard_normal((11, 2))

        def loss_value():
            predictions, _ = network.forward(features)
            return float(np.mean((predictions - targets) ** 2))

        predictions, cache = network.forward(features)
        residual = predictions - targets
        weight_grads, bias_grads = network.backward(cache, 2.0 * residual / residual.size)

        step = 1e-6
        worst = 0.0
        for weight, gradient in zip(network.weights, weight_grads):
            for _ in range(12):
                i = rng.integers(weight.shape[0])
                j = rng.integers(weight.shape[1])
                original = weight[i, j]
                weight[i, j] = original + step
                plus = loss_value()
                weight[i, j] = original - step
                minus = loss_value()
                weight[i, j] = original
                numeric = (plus - minus) / (2 * step)
                analytic = gradient[i, j]
                worst = max(
                    worst,
                    abs(numeric - analytic) / max(1e-12, abs(numeric) + abs(analytic)),
                )
        assert worst < 1e-6, f"worst relative gradient error {worst:.2e}"

    def test_network_learns_the_device(self, signal, test_signal):
        device = default_device()
        x_train = scale_for_backoff(signal, 12.0)
        x_test = scale_for_backoff(test_signal, 12.0)
        model = NeuralPAModel(
            memory_depth=4, hidden=(12, 12), epochs=250, seed=0
        ).fit(x_train, device(x_train))
        skip = model.transient
        score = nmse_db(device(x_test)[skip:], model.predict(x_test)[skip:])
        assert score < -25.0, f"network only reached {score:.1f} dB"

    def test_loss_decreases(self, signal):
        device = default_device()
        x = scale_for_backoff(signal, 12.0)
        model = NeuralPAModel(memory_depth=3, hidden=(8,), epochs=120, seed=0).fit(
            x, device(x)
        )
        assert model.history[-1] < 0.05 * model.history[0]

    @pytest.mark.skipif(not TorchTDNN.available(), reason="torch is not installed")
    def test_numpy_and_torch_backends_agree(self, signal, test_signal):
        """Cross-validates the hand-written gradients against autograd."""
        device = default_device()
        x_train = scale_for_backoff(signal, 12.0)
        x_test = scale_for_backoff(test_signal, 12.0)
        y_train, y_test = device(x_train), device(x_test)

        scores = []
        for cls in (NeuralPAModel, TorchTDNN):
            model = cls(memory_depth=4, hidden=(12, 12), epochs=250, seed=0).fit(
                x_train, y_train
            )
            skip = model.transient
            scores.append(nmse_db(y_test[skip:], model.predict(x_test)[skip:]))
        assert abs(scores[0] - scores[1]) < 4.0, (
            f"backends disagree: numpy {scores[0]:.2f} dB, torch {scores[1]:.2f} dB"
        )


class TestMetrics:
    def test_nmse_of_identical_signals(self):
        x = np.array([1 + 1j, 2, 3j])
        assert nmse_db(x, x) == float("-inf")

    def test_nmse_of_scaled_signal(self):
        """A 10% amplitude error is -20 dB NMSE."""
        x = np.ones(100, dtype=complex)
        assert nmse_db(x, 1.1 * x) == pytest.approx(-20.0, abs=1e-9)

    def test_delay_estimation_and_alignment(self):
        x = multicarrier_signal(2048, OCCUPIED, rng=np.random.default_rng(5))
        shifted = np.concatenate([np.zeros(7, dtype=complex), x[:-7]])
        assert estimate_delay(x, shifted) == 7
        reference, aligned = align(x, shifted)
        assert nmse_db(reference, aligned) < -100

    def test_fit_gain_recovers_a_known_gain(self):
        x = multicarrier_signal(1024, OCCUPIED, rng=np.random.default_rng(6))
        gain = 2.5 - 1.3j
        assert fit_gain(x, gain * x) == pytest.approx(gain, rel=1e-12)

    def test_acpr_floor_on_an_undistorted_signal(self, signal):
        """The measurement must not be limited by its own window leakage.

        Placing the adjacent band flush against the occupied band measured about
        -28 dBc on a clean signal, which is worse than most distortion being
        studied. A channel spacing wider than the occupied bandwidth fixes it.
        """
        lower, upper = acpr_db(signal, OCCUPIED)
        assert max(lower, upper) < -90.0

    def test_acpr_worsens_as_drive_increases(self, signal):
        device = default_device()
        scores = [
            max(acpr_db(device(scale_for_backoff(signal, backoff)), OCCUPIED))
            for backoff in (16.0, 12.0, 8.0)
        ]
        assert scores[0] < scores[1] < scores[2]

    def test_evm_is_zero_for_a_pure_gain(self):
        x = multicarrier_signal(1024, OCCUPIED, rng=np.random.default_rng(8))
        assert evm_percent(x, 3.0 * x) == pytest.approx(0.0, abs=1e-9)


class TestPredistortion:
    def test_dpd_improves_acpr_below_saturation(self, signal):
        """With envelope peaks under saturation, the amplifier is invertible."""
        device = default_device()
        x = scale_for_backoff(signal, 13.0)
        assert np.max(np.abs(x)) < 1.0, "this test requires peaks below saturation"

        before = max(acpr_db(device(x), OCCUPIED))
        predistorter = IndirectLearningDPD(iterations=3).train(
            device, x, occupied_fraction=OCCUPIED
        )
        after = max(acpr_db(device(predistorter.apply(x)), OCCUPIED))
        assert after < before - 10.0, f"DPD gained only {before - after:.1f} dB"

    def test_dpd_fails_when_peaks_exceed_saturation(self, signal):
        """A physical limit, not a software one.

        Past saturation the amplifier maps many inputs to the same output, so no
        pre-inverse exists and predistortion cannot help. An implementation that
        appeared to improve ACPR here would be wrong.
        """
        device = default_device()
        x = scale_for_backoff(signal, 5.0)
        assert np.max(np.abs(x)) > 1.3, "this test requires peaks well past saturation"

        before = max(acpr_db(device(x), OCCUPIED))
        predistorter = IndirectLearningDPD(iterations=3).train(
            device, x, occupied_fraction=OCCUPIED
        )
        after = max(acpr_db(device(predistorter.apply(x)), OCCUPIED))
        assert after > before

    def test_apply_before_train_is_an_error(self):
        with pytest.raises(RuntimeError, match="not been trained"):
            IndirectLearningDPD().apply(np.ones(10, dtype=complex))

    def test_recovered_backoff_reads_the_crossings(self):
        sweep = {
            "backoff_db": np.array([8.0, 10.0, 12.0, 14.0]),
            "acpr_raw": np.array([-30.0, -35.0, -40.0, -50.0]),
            "acpr_dpd": np.array([-30.0, -50.0, -60.0, -70.0]),
        }
        # raw crosses -45 dBc at 13.0 dB (between 12 and 14), DPD at 9.5 dB
        # (between 8 and 10), so 3.5 dB of back-off is recovered.
        assert recovered_backoff_db(sweep, -45.0) == pytest.approx(3.5, abs=0.01)
