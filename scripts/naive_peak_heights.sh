#!/bin/bash
# Naive baseline: no fitting. Reads peak heights above the median and guesses widths.
# A realistic shortcut that the verifier should reject on fit quality.
set -euo pipefail
cd /app
pip install -q numpy==2.3.1
cat > /tmp/naive.py << 'EOF'
import json
import numpy as np
WINDOWS = {"D": (1250.0, 1450.0), "G": (1500.0, 1700.0), "2D": (2550.0, 2850.0)}
x, y = np.loadtxt("/app/spectrum.csv", delimiter=",", skiprows=1, unpack=True)
out = {}
for name, (lo, hi) in WINDOWS.items():
    m = (x > lo) & (x < hi)
    xw, yw = x[m], y[m]
    base = float(np.median(yw))
    out[name] = {"x0": float(xw[np.argmax(yw)]), "gamma": 10.0,
                 "amplitude": float(yw.max() - base), "offset": base}
out["ID_IG"] = out["D"]["amplitude"] / out["G"]["amplitude"]
json.dump(out, open("/app/results.json", "w"), indent=2)
EOF
python /tmp/naive.py
