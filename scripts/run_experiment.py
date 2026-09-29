"""Run the full study and write figures and results to disk.

    python scripts/run_experiment.py

Produces every number quoted in the README and every figure in ``figures/``.
Deterministic: fixed seeds throughout, so re-running reproduces the table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pamodel.dpd import IndirectLearningDPD, backoff_sweep, recovered_backoff_db  # noqa: E402
from pamodel.memory_polynomial import MemoryPolynomial  # noqa: E402
from pamodel.metrics import acpr_db, fit_gain, nmse_db  # noqa: E402
from pamodel.neural import NeuralPAModel  # noqa: E402
from pamodel.pa import RappPA, WienerHammerstein, default_device, scale_for_backoff  # noqa: E402
from pamodel.plots import (  # noqa: E402
    plot_am_am,
    plot_am_pm,
    plot_backoff_sweep,
    plot_model_comparison,
    plot_spectra,
    plot_training_history,
)
from pamodel.signals import multicarrier_signal, papr_db  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

OCCUPIED = 0.125
SAMPLES = 32768
OPERATING_BACKOFF = 12.0  # dB, chosen below so envelope peaks stay under saturation


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def main() -> None:
    results: dict = {}
    started = time.time()

    train_signal = multicarrier_signal(SAMPLES, OCCUPIED, rng=np.random.default_rng(1))
    test_signal = multicarrier_signal(SAMPLES, OCCUPIED, rng=np.random.default_rng(99))

    device = default_device()
    x_train = scale_for_backoff(train_signal, OPERATING_BACKOFF)
    x_test = scale_for_backoff(test_signal, OPERATING_BACKOFF)
    y_train, y_test = device(x_train), device(x_test)
    gain = device.linear_gain()

    banner("1. Device and test signal")
    peak_ratio = float(np.max(np.abs(x_train)))
    print(f"  samples per record     {SAMPLES}")
    print(f"  occupied bandwidth     {OCCUPIED} of sample rate ({1/OCCUPIED:.0f}x oversampled)")
    print(f"  PAPR                   {papr_db(train_signal):.2f} dB")
    print(f"  input back-off         {OPERATING_BACKOFF:.1f} dB")
    print(f"  envelope peak          {peak_ratio:.3f} (saturation = 1.0)")
    print(f"  small-signal gain      {abs(gain):.4f} at {np.degrees(np.angle(gain)):+.2f} deg")
    print(f"  raw ACPR               {max(acpr_db(y_train, OCCUPIED)):.2f} dBc")
    results["setup"] = {
        "samples": SAMPLES,
        "occupied_fraction": OCCUPIED,
        "papr_db": papr_db(train_signal),
        "backoff_db": OPERATING_BACKOFF,
        "envelope_peak": peak_ratio,
        "raw_acpr_dbc": max(acpr_db(y_train, OCCUPIED)),
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_am_am(x_train, y_train, ax=axes[0], gain=gain, label="device")
    plot_am_pm(x_train, y_train, ax=axes[1])
    figure.suptitle(
        "Wiener-Hammerstein device: the characteristic is a band, not a curve. "
        "Its width is memory.",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "01_device_characteristic.png", dpi=140)
    plt.close(figure)

    banner("2. Behavioural models: accuracy against model size")
    rows = []
    comparison = []

    mp_configs = [
        dict(orders=(1,), memory_depth=1),
        dict(orders=(1, 3), memory_depth=2),
        dict(orders=(1, 3, 5, 7), memory_depth=2),
        dict(orders=(1, 3, 5, 7), memory_depth=5),
        dict(orders=(1, 3, 5, 7, 9, 11), memory_depth=5),
        dict(orders=(1, 3, 5, 7, 9, 11, 13, 15), memory_depth=5),
    ]
    for config in mp_configs:
        model = MemoryPolynomial(**config).fit(x_train, y_train)
        skip = model.transient
        test_nmse = nmse_db(y_test[skip:], model.predict(x_test)[skip:])
        train_nmse = nmse_db(y_train[skip:], model.predict(x_train)[skip:])
        rows.append(
            {
                "model": model.describe(),
                "family": "memory polynomial",
                "parameters": model.num_coefficients,
                "train_nmse_db": train_nmse,
                "test_nmse_db": test_nmse,
                "condition_number": model.condition_number,
            }
        )
        comparison.append(
            (model.describe(), model.num_coefficients, test_nmse, "memory polynomial")
        )
        print(
            f"  {model.describe():<46} {model.num_coefficients:>4}p  "
            f"test {test_nmse:7.2f} dB  cond {model.condition_number:.1e}"
        )

    histories, history_labels = [], []
    for hidden, epochs in [((8, 8), 600), ((16, 16), 600), ((24, 24), 600), ((32, 32), 900)]:
        start = time.time()
        model = NeuralPAModel(
            memory_depth=5, hidden=hidden, epochs=epochs, seed=0
        ).fit(x_train, y_train)
        skip = model.transient
        test_nmse = nmse_db(y_test[skip:], model.predict(x_test)[skip:])
        train_nmse = nmse_db(y_train[skip:], model.predict(x_train)[skip:])
        rows.append(
            {
                "model": model.describe(),
                "family": "neural network",
                "parameters": model.num_parameters,
                "train_nmse_db": train_nmse,
                "test_nmse_db": test_nmse,
                "fit_seconds": time.time() - start,
            }
        )
        comparison.append(
            (model.describe(), model.num_parameters, test_nmse, "neural network")
        )
        histories.append(model.history)
        history_labels.append(f"hidden {hidden[0]}x{hidden[1]}")
        print(
            f"  {model.describe():<46} {model.num_parameters:>4}p  "
            f"test {test_nmse:7.2f} dB  ({time.time() - start:.0f}s)"
        )
    results["models"] = rows

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    plot_model_comparison(comparison, ax=axes[0])
    plot_training_history(histories, history_labels, ax=axes[1])
    figure.tight_layout()
    figure.savefig(FIGURES / "02_model_comparison.png", dpi=140)
    plt.close(figure)

    banner("3. Where the polynomial basis starts to struggle")
    print("  Rapp p controls how sharply the amplifier enters saturation.")
    print(f"  {'p':>5} {'MP-7':>9} {'MP-11':>9} {'MP-15':>9}   (test NMSE, dB)")
    sharpness = []
    for smoothness in [1.0, 2.0, 4.0, 10.0]:
        sharp_device = WienerHammerstein(
            nonlinearity=RappPA(saturation=1.0, smoothness=smoothness, am_pm=0.35)
        )
        yt_train, yt_test = sharp_device(x_train), sharp_device(x_test)
        entry = {"smoothness": smoothness}
        line = []
        for name, orders in [
            ("mp7", (1, 3, 5, 7)),
            ("mp11", (1, 3, 5, 7, 9, 11)),
            ("mp15", (1, 3, 5, 7, 9, 11, 13, 15)),
        ]:
            model = MemoryPolynomial(orders=orders, memory_depth=5).fit(x_train, yt_train)
            skip = model.transient
            value = nmse_db(yt_test[skip:], model.predict(x_test)[skip:])
            entry[name] = value
            line.append(value)
        sharpness.append(entry)
        print(f"  {smoothness:5.1f} {line[0]:9.2f} {line[1]:9.2f} {line[2]:9.2f}")
    results["sharpness_sweep"] = sharpness

    banner("4. Predistortion")
    predistorter = IndirectLearningDPD(iterations=4).train(
        device, x_train, gain=gain, occupied_fraction=OCCUPIED
    )
    for record in predistorter.history:
        print(
            f"  iteration {record['iteration']}  "
            f"ACPR {max(record['acpr_lower'], record['acpr_upper']):7.2f} dBc  "
            f"post-inverse NMSE {record['inverse_nmse']:7.2f} dB"
        )
    predistorted = predistorter.apply(x_test)
    corrected = device(predistorted)
    raw_acpr = max(acpr_db(device(x_test), OCCUPIED))
    dpd_acpr = max(acpr_db(corrected, OCCUPIED))
    print(f"\n  held-out ACPR without DPD  {raw_acpr:7.2f} dBc")
    print(f"  held-out ACPR with DPD     {dpd_acpr:7.2f} dBc  ({raw_acpr - dpd_acpr:.1f} dB better)")
    print(f"  linearisation NMSE         {nmse_db((gain * x_test)[8:], corrected[8:]):7.2f} dB")
    results["dpd"] = {
        "iterations": predistorter.history,
        "raw_acpr_dbc": raw_acpr,
        "dpd_acpr_dbc": dpd_acpr,
        "linearisation_nmse_db": nmse_db((gain * x_test)[8:], corrected[8:]),
    }

    figure = plt.figure(figsize=(7.5, 4.6))
    plot_spectra(
        [gain * x_test, device(x_test), corrected],
        ["ideal linear", "amplifier, no DPD", "amplifier, with DPD"],
        ax=figure.gca(),
        occupied_fraction=OCCUPIED,
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "03_spectra.png", dpi=140)
    plt.close(figure)

    banner("5. Back-off sweep: what predistortion actually buys")
    backoffs = np.arange(4.0, 17.0, 1.0)
    sweep = backoff_sweep(
        device, test_signal, backoffs, predistorter, occupied_fraction=OCCUPIED
    )
    print(f"  {'IBO':>5} {'peak':>7} {'raw':>9} {'with DPD':>9}")
    for backoff, raw, dpd in zip(sweep["backoff_db"], sweep["acpr_raw"], sweep["acpr_dpd"]):
        peak = float(np.max(np.abs(scale_for_backoff(test_signal, backoff))))
        marker = "  <- peaks clip" if peak > 1.0 else ""
        print(f"  {backoff:5.0f} {peak:7.3f} {raw:9.2f} {dpd:9.2f}{marker}")

    recovered = recovered_backoff_db(sweep, acpr_target=-45.0)
    print(f"\n  back-off recovered at a -45 dBc limit: {recovered:.1f} dB")
    results["backoff_sweep"] = {
        "backoff_db": sweep["backoff_db"].tolist(),
        "acpr_raw": sweep["acpr_raw"].tolist(),
        "acpr_dpd": sweep["acpr_dpd"].tolist(),
        "recovered_backoff_db": recovered,
    }

    figure = plt.figure(figsize=(6.8, 4.6))
    plot_backoff_sweep(sweep, ax=figure.gca())
    figure.tight_layout()
    figure.savefig(FIGURES / "04_backoff_sweep.png", dpi=140)
    plt.close(figure)

    with open(ROOT / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    banner("Done")
    print(f"  figures written to {FIGURES}")
    print(f"  results written to {ROOT / 'results.json'}")
    print(f"  total runtime {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
