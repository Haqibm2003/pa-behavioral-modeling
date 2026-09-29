"""Model comparison at two drive levels.

The headline comparison between a memory polynomial and a neural network is not
a single number: it depends on how hard the amplifier is being driven. This
script measures both a mild operating point, where envelope peaks stay below
saturation, and a hard one, where they do not, and appends both to results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pamodel.memory_polynomial import MemoryPolynomial  # noqa: E402
from pamodel.metrics import nmse_db  # noqa: E402
from pamodel.neural import NeuralPAModel  # noqa: E402
from pamodel.pa import RappPA, WienerHammerstein, default_device, scale_for_backoff  # noqa: E402
from pamodel.signals import multicarrier_signal  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OCCUPIED, SAMPLES = 0.125, 32768

train = multicarrier_signal(SAMPLES, OCCUPIED, rng=np.random.default_rng(1))
test = multicarrier_signal(SAMPLES, OCCUPIED, rng=np.random.default_rng(99))
device = default_device()

out = {}
for backoff in (12.0, 6.0):
    x_tr, x_te = scale_for_backoff(train, backoff), scale_for_backoff(test, backoff)
    y_tr, y_te = device(x_tr), device(x_te)
    peak = float(np.max(np.abs(x_tr)))
    print(f"\n=== back-off {backoff:.0f} dB, envelope peak {peak:.2f} "
          f"({'below' if peak < 1 else 'ABOVE'} saturation) ===")

    entry = {"backoff_db": backoff, "envelope_peak": peak, "models": [], "sharpness": []}
    for orders, depth in [((1, 3, 5, 7), 5), ((1, 3, 5, 7, 9, 11), 5)]:
        m = MemoryPolynomial(orders=orders, memory_depth=depth).fit(x_tr, y_tr)
        k = m.transient
        score = nmse_db(y_te[k:], m.predict(x_te)[k:])
        entry["models"].append({"model": m.describe(), "parameters": m.num_coefficients,
                                "test_nmse_db": score, "family": "memory polynomial"})
        print(f"  {m.describe():<44} {m.num_coefficients:>5}p  {score:7.2f} dB")

    for hidden, epochs in [((16, 16), 600), ((32, 32), 900)]:
        n = NeuralPAModel(memory_depth=5, hidden=hidden, epochs=epochs, seed=0).fit(x_tr, y_tr)
        k = n.transient
        score = nmse_db(y_te[k:], n.predict(x_te)[k:])
        entry["models"].append({"model": n.describe(), "parameters": n.num_parameters,
                                "test_nmse_db": score, "family": "neural network"})
        print(f"  {n.describe():<44} {n.num_parameters:>5}p  {score:7.2f} dB")

    print(f"  -- polynomial accuracy vs saturation-knee sharpness --")
    for p in (1.0, 2.0, 4.0, 10.0):
        dev = WienerHammerstein(nonlinearity=RappPA(saturation=1.0, smoothness=p, am_pm=0.35))
        yt_tr, yt_te = dev(x_tr), dev(x_te)
        m = MemoryPolynomial(orders=(1, 3, 5, 7, 9, 11), memory_depth=5).fit(x_tr, yt_tr)
        k = m.transient
        score = nmse_db(yt_te[k:], m.predict(x_te)[k:])
        entry["sharpness"].append({"smoothness": p, "mp11_nmse_db": score})
        print(f"     Rapp p={p:<5.1f} MP-11  {score:7.2f} dB")
    out[f"backoff_{backoff:.0f}dB"] = entry

results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
results["operating_point_comparison"] = out
(ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nappended to results.json")
