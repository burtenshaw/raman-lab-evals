"""Generate the adapted `raman-lab` Harbor dataset.

Adapted from Terminal-Bench 2 `raman-fitting` (harbor-framework/terminal-bench-2@2fd12b8).
Changes vs. the original:
  * several samples (synthetic graphene spectra with known D, G and 2D peaks), not one
  * units and file format stated in the instruction
  * a D peak and the I_D/I_G ratio, the quantity a lab reports for defect density
  * verifier grades fit quality on its own copy of the data, plus I_D/I_G within tolerance
  * reference values and the verifier's data live in tests/, never in the agent's workdir

Usage: python build_dataset.py --out datasets/raman-lab
"""

import argparse
import json
import math
import shutil
import textwrap
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

WINDOWS = {"D": (1250.0, 1450.0), "G": (1500.0, 1700.0), "2D": (2550.0, 2850.0)}
IMAGE = "python:3.13-slim-bookworm"

# (sample id, I_D/I_G amplitude ratio, noise level, G width, split)
SAMPLES = [
    ("s01", 0.05, 40.0, 8.0, "dev"),
    ("s02", 0.30, 60.0, 9.0, "dev"),
    ("s03", 0.80, 80.0, 11.0, "dev"),
    ("s04", 1.40, 60.0, 14.0, "dev"),
    ("s05", 0.10, 120.0, 9.0, "test"),
    ("s06", 0.55, 50.0, 10.0, "test"),
    ("s07", 1.10, 100.0, 13.0, "test"),
    ("s08", 0.20, 40.0, 8.5, "test"),
]


def lorentzian(x, x0, gamma, amplitude, offset):
    return amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset


def make_spectrum(rng, id_ig, noise, g_width):
    x = np.linspace(1100.0, 3000.0, 3800)
    g_amp = rng.uniform(6000, 9000)
    peaks = [
        (rng.uniform(1345, 1355), rng.uniform(12, 20), id_ig * g_amp),  # D
        (rng.uniform(1578, 1586), g_width, g_amp),                      # G
        (rng.uniform(2675, 2700), rng.uniform(15, 30), rng.uniform(0.8, 2.2) * g_amp),  # 2D
    ]
    baseline = rng.uniform(800, 1500) + rng.uniform(-0.15, 0.15) * (x - 1100)
    y = baseline.copy()
    for x0, gamma, amp in peaks:
        y += lorentzian(x, x0, gamma, amp, 0.0)
    y += rng.normal(0, noise, size=x.shape)
    return x, y


def reference_fit(x, y):
    """The oracle's method: one Lorentzian plus constant offset per window."""
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        m = (x > lo) & (x < hi)
        xw, yw = x[m], y[m]
        p0 = [xw[np.argmax(yw)], 15.0, yw.max() - np.median(yw), np.median(yw)]
        p, cov = curve_fit(lorentzian, xw, yw, p0=p0, maxfev=20000)
        out[name] = dict(zip(["x0", "gamma", "amplitude", "offset"], map(float, p)))
        out[name]["_amp_sd"] = float(np.sqrt(cov[2, 2]))
    return out


INSTRUCTION = """\
You are given a Raman spectrum of a graphene sample in `/app/spectrum.csv`.
It has a header row and two columns: `raman_shift_cm-1` (Raman shift in cm⁻¹) and `intensity` (counts).

Fit a single Lorentzian plus a constant offset to each of the D (~1350 cm⁻¹), G (~1580 cm⁻¹)
and 2D (~2690 cm⁻¹) peaks, using the model

    I(x) = amplitude * gamma^2 / ((x - x0)^2 + gamma^2) + offset

Then report the defect ratio I_D/I_G, defined as the D amplitude divided by the G amplitude.

Write the result to `/app/results.json` in this format:

{
  "D":  {"x0": <float>, "gamma": <float>, "amplitude": <float>, "offset": <float>},
  "G":  {"x0": <float>, "gamma": <float>, "amplitude": <float>, "offset": <float>},
  "2D": {"x0": <float>, "gamma": <float>, "amplitude": <float>, "offset": <float>},
  "ID_IG": <float>
}
"""

TASK_TOML = """\
schema_version = "1.1"
artifacts = []

[task]
name = "raman-lab/{task_id}"
description = "Fit D, G and 2D Lorentzians to a graphene Raman spectrum and report I_D/I_G. Adapted from terminal-bench/raman-fitting."
keywords = ["fitting", "physics", "scientific-computing", "raman"]

[metadata]
adapted_from = "terminal-bench/raman-fitting@2fd12b8"
split = "{split}"
difficulty = "medium"
category = "scientific-computing"

[agent]
timeout_sec = 900.0

[verifier]
timeout_sec = 300.0

[environment]
docker_image = "{image}"
workdir = "/app"
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
allow_internet = true
"""

SOLVE_SH = """\
#!/bin/bash
# Oracle: one Lorentzian + constant per window, scipy curve_fit with data-driven initial guesses.
set -euo pipefail
cd /app
pip install -q numpy==2.3.1 scipy==1.16.0
cat > /tmp/oracle_fit.py << 'EOF'
import json
import numpy as np
from scipy.optimize import curve_fit

WINDOWS = {"D": (1250.0, 1450.0), "G": (1500.0, 1700.0), "2D": (2550.0, 2850.0)}

def lorentzian(x, x0, gamma, amplitude, offset):
    return amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset

x, y = np.loadtxt("/app/spectrum.csv", delimiter=",", skiprows=1, unpack=True)
out = {}
for name, (lo, hi) in WINDOWS.items():
    m = (x > lo) & (x < hi)
    xw, yw = x[m], y[m]
    p0 = [xw[np.argmax(yw)], 15.0, yw.max() - np.median(yw), np.median(yw)]
    p, _ = curve_fit(lorentzian, xw, yw, p0=p0, maxfev=20000)
    out[name] = dict(zip(["x0", "gamma", "amplitude", "offset"], map(float, p)))
out["ID_IG"] = out["D"]["amplitude"] / out["G"]["amplitude"]
json.dump(out, open("/app/results.json", "w"), indent=2)
EOF
python /tmp/oracle_fit.py
cat /app/results.json
"""

TEST_SH = """\
#!/bin/bash
# Verifier: grades the fit on its own copy of the data. Writes /logs/verifier/reward.json.
mkdir -p /logs/verifier
pip install -q --target /tmp/verifier_deps numpy==2.3.1 > /dev/null 2>&1
PYTHONPATH=/tmp/verifier_deps python /tests/score.py
"""

SCORE_PY = """\
import json
import math
from pathlib import Path

import numpy as np

WINDOWS = {"D": (1250.0, 1450.0), "G": (1500.0, 1700.0), "2D": (2550.0, 2850.0)}
FIT_SLACK = 1.10      # agent RMSE may be at most 10% above the reference fit's RMSE
REWARD = Path("/logs/verifier/reward.json")

def lorentzian(x, x0, gamma, amplitude, offset):
    return amplitude * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset

def rmse(params, x, y):
    return float(np.sqrt(np.mean((lorentzian(x, **params) - y) ** 2)))

def write(scores):
    scores["reward"] = float(all(v == 1.0 for k, v in scores.items()))
    REWARD.write_text(json.dumps(scores))
    print(json.dumps(scores, indent=2))

x, y = np.loadtxt("/tests/data/spectrum.csv", delimiter=",", skiprows=1, unpack=True)
ref = json.loads(Path("/tests/data/reference.json").read_text())
keys = ["d_fit", "g_fit", "2d_fit", "id_ig"]

try:
    out = json.loads(Path("/app/results.json").read_text())
    scores = {}
    for name, (lo, hi) in WINDOWS.items():
        m = (x > lo) & (x < hi)
        params = {k: float(out[name][k]) for k in ["x0", "gamma", "amplitude", "offset"]}
        ok = all(math.isfinite(v) for v in params.values())
        scores[name.lower() + "_fit"] = float(ok and rmse(params, x[m], y[m]) <= FIT_SLACK * ref["rmse"][name])
    id_ig = float(out["ID_IG"])
    scores["id_ig"] = float(abs(id_ig - ref["ID_IG"]) <= ref["ID_IG_tol"])
except Exception as exc:  # missing file, missing key, wrong type: all score zero
    print(f"invalid submission: {exc!r}")
    scores = {k: 0.0 for k in keys}
write(scores)
"""


def build(out_dir: Path, seed: int):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    manifest = []
    for i, (sid, id_ig, noise, g_width, split) in enumerate(SAMPLES):
        rng = np.random.default_rng(seed + i)
        x, y = make_spectrum(rng, id_ig, noise, g_width)
        fit = reference_fit(x, y)
        ref_rmse = {}
        for name, (lo, hi) in WINDOWS.items():
            m = (x > lo) & (x < hi)
            p = {k: fit[name][k] for k in ["x0", "gamma", "amplitude", "offset"]}
            ref_rmse[name] = float(np.sqrt(np.mean((lorentzian(x[m], **p) - y[m]) ** 2)))
        ratio = fit["D"]["amplitude"] / fit["G"]["amplitude"]
        # tolerance: 3 sigma of the ratio from the fit covariance, floor of 0.02 absolute or 10% relative
        rel_sd = math.hypot(fit["D"]["_amp_sd"] / fit["D"]["amplitude"], fit["G"]["_amp_sd"] / fit["G"]["amplitude"])
        tol = max(0.02, 0.10 * ratio, 3 * ratio * rel_sd)
        reference = {"ID_IG": ratio, "ID_IG_tol": tol, "rmse": ref_rmse, "true_ID_IG": id_ig,
                     "fit": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in fit.items()}}

        task_id = f"graphene-{sid}"
        t = out_dir / task_id
        (t / "environment").mkdir(parents=True)
        (t / "solution").mkdir()
        (t / "tests" / "data").mkdir(parents=True)
        csv = "raman_shift_cm-1,intensity\n" + "\n".join(f"{a:.4f},{b:.3f}" for a, b in zip(x, y)) + "\n"
        (t / "environment" / "spectrum.csv").write_text(csv)
        (t / "tests" / "data" / "spectrum.csv").write_text(csv)
        (t / "tests" / "data" / "reference.json").write_text(json.dumps(reference, indent=2))
        (t / "instruction.md").write_text(INSTRUCTION)
        (t / "task.toml").write_text(TASK_TOML.format(task_id=task_id, split=split, image=IMAGE))
        (t / "solution" / "solve.sh").write_text(SOLVE_SH)
        (t / "tests" / "test.sh").write_text(TEST_SH)
        (t / "tests" / "score.py").write_text(SCORE_PY)
        for f in [t / "solution" / "solve.sh", t / "tests" / "test.sh"]:
            f.chmod(0o755)
        manifest.append({"task": task_id, "split": split, "true_ID_IG": id_ig,
                         "ref_ID_IG": round(ratio, 4), "tol": round(tol, 4), "noise": noise})
    (out_dir / "dataset.toml").write_text(textwrap.dedent("""\
        [dataset]
        name = "raman-lab/raman-lab"
        version = "0.1.0"
        description = "Graphene Raman fitting with D/G/2D peaks and I_D/I_G. Adapted from terminal-bench/raman-fitting."
        """))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/raman-lab")
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    for row in build(Path(a.out), a.seed):
        print(row)
