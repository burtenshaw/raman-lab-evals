"""Emulate Harbor trials locally (no Docker) to sanity-check tasks before spending sandbox time.

Each trial gets its own scratch root; copies of the solver and verifier scripts have
/app, /tests and /logs rewritten to point inside that root. Real runs use `harbor run -e hf-sandbox`.

Usage: python scripts/local_check.py datasets/raman-lab {oracle|nop|naive} [--verbose]
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
SOLVERS = {
    "oracle": lambda task: task / "solution" / "solve.sh",
    "naive": lambda task: ROOT / "naive_peak_heights.sh",
    "nop": lambda task: None,
}


def remap(text: str, root: Path) -> str:
    for p in ["/tmp/verifier_deps", "/app", "/tests", "/logs"]:
        text = text.replace(p, f"{root}{p}")
    return text


def run_script(src: Path, root: Path, cwd: Path, verbose: bool):
    dst = root / "bin" / src.name
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(remap(src.read_text(), root))
    r = subprocess.run(["bash", str(dst)], cwd=cwd, capture_output=True, text=True)
    if verbose:
        print(f"--- {src.name} exit={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")


def run(task: Path, solver: str, verbose: bool) -> dict:
    with tempfile.TemporaryDirectory(dir=ROOT / ".trials") as tmp:
        root = Path(tmp)
        shutil.copytree(task / "environment", root / "app")
        script = SOLVERS[solver](task)
        if script is not None:
            run_script(script, root, root / "app", verbose)
        shutil.copytree(task / "tests", root / "tests")
        score = root / "tests" / "score.py"
        score.write_text(remap(score.read_text(), root))
        (root / "logs" / "verifier").mkdir(parents=True)
        run_script(root / "tests" / "test.sh", root, root / "app", verbose)
        return json.loads((root / "logs" / "verifier" / "reward.json").read_text())


if __name__ == "__main__":
    dataset, solver = Path(sys.argv[1]), sys.argv[2]
    verbose = "--verbose" in sys.argv
    (ROOT / ".trials").mkdir(exist_ok=True)
    tasks = sorted(dataset.glob("graphene-*"))
    if verbose:
        tasks = tasks[:1]
    results = {t.name: run(t, solver, verbose) for t in tasks}
    for name, r in results.items():
        print(solver, name, r)
    mean = sum(r["reward"] for r in results.values()) / len(results)
    print(f"{solver}: mean reward {mean:.2f}")
