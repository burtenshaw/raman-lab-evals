"""Summarise Harbor trial folders into per-trial and per-model tables.

Usage: python scripts/analyze.py runs/2026-11-18 results/

Reads every <model>/<harbor job>/<trial>/ folder under the run directory, then writes
  results/trials.csv    one row per rollout
  results/summary.csv   one row per model
  results/summary.md    the summary as a Markdown table
"""

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RUN_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["gpt-oss-120b", "Qwen3.8-27B", "DeepSeek-V4.1-Flash", "Kimi-K3", "MiniMax-M3.1-Flash", "gemma-4-31B-it"]


def read_trajectories(agent_dir: Path):
    """Return (agent steps, tool-call names in the first agent step, final token metrics)."""
    steps, first_calls, metrics = [], None, {}
    for traj in sorted(agent_dir.glob("trajectory*.json")):
        data = json.loads(traj.read_text())
        for key, value in (data.get("final_metrics") or {}).items():
            metrics[key] = metrics.get(key, 0) + (value or 0)
        for step in data.get("steps", []):
            if step.get("source") != "agent":
                continue
            steps.append(step)
            if first_calls is None:
                first_calls = [c.get("function_name") for c in step.get("tool_calls") or []]
    return steps, first_calls or [], metrics


def verifier_output(trial: Path) -> str:
    for name in ("verifier/test-stdout.txt", "verifier/stdout.txt"):
        if (trial / name).exists():
            return (trial / name).read_text()
    return ""


rows = []
for meta_path in sorted(RUN_DIR.glob("*/run_meta.json")):
    meta = json.loads(meta_path.read_text())
    model = meta["model"].split("/")[-1]
    pricing = meta.get("router_pricing") or {}
    for result in sorted(meta_path.parent.glob("*/*/result.json")):
        r = json.loads(result.read_text())
        if "task_name" not in r:
            continue
        trial, job = result.parent, result.parent.parent.name
        is_oracle = job.startswith("oracle")
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        exception = (r.get("exception_info") or {}).get("exception_type")
        steps, first_calls, metrics = read_trajectories(trial / "agent")
        vout = verifier_output(trial)
        reward = rewards.get("reward")
        if reward == 1.0:
            outcome = "passed"
        elif "invalid submission" in vout or not rewards:
            outcome = "no valid results file"
        else:
            outcome = "fit check failed"
        prompt_tokens = metrics.get("total_prompt_tokens") or 0
        completion_tokens = metrics.get("total_completion_tokens") or 0
        started, finished = r.get("started_at"), r.get("finished_at")
        seconds = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds() if started and finished else None
        cost = (prompt_tokens * pricing.get("input", 0) + completion_tokens * pricing.get("output", 0)) / 1e6
        rows.append({
            "model": "oracle" if is_oracle else model,
            "provider": None if is_oracle else meta.get("provider"),
            "dataset": "raman-fitting" if job.endswith("original") else "raman-lab",
            "task": r["task_name"], "trial": r["trial_name"],
            "reward": reward, "d_fit": rewards.get("d_fit"), "g_fit": rewards.get("g_fit"),
            "2d_fit": rewards.get("2d_fit"), "id_ig": rewards.get("id_ig"),
            "outcome": outcome, "exception": exception,
            # Terminus-2 exposes "mark_task_complete" as a tool; calling it in the first response
            # means the model declared the task done before it had seen any command output.
            "done_in_first_response": "mark_task_complete" in first_calls,
            "agent_steps": len(steps),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "est_cost_usd": round(cost, 5),
            "wall_seconds": round(seconds) if seconds is not None else None,
        })

with open(OUT / "trials.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def clustered_se(by_task: dict) -> float:
    """Standard error of the mean with rollouts clustered by task (Miller 2024, eq. 4)."""
    values = [v for vs in by_task.values() for v in vs]
    n, mean = len(values), sum(values) / len(values)
    return math.sqrt(sum(sum(v - mean for v in vs) ** 2 for vs in by_task.values())) / n


summary = []
for model in MODEL_ORDER:
    lab = [r for r in rows if r["model"] == model and r["dataset"] == "raman-lab"]
    orig = [r for r in rows if r["model"] == model and r["dataset"] == "raman-fitting"]
    if not lab:
        continue
    by_task = defaultdict(list)
    for r in lab:
        by_task[r["task"]].append(1.0 if r["reward"] == 1.0 else 0.0)
    passes = sum(sum(v) for v in by_task.values())
    early = [r for r in lab if r["done_in_first_response"]]
    checked = [r for r in lab if not r["done_in_first_response"]]
    summary.append({
        "model": model, "provider": lab[0]["provider"],
        "raman_fitting_passed": sum(r["reward"] == 1.0 for r in orig), "raman_fitting_rollouts": len(orig),
        "raman_lab_passed": int(passes), "raman_lab_rollouts": len(lab),
        "mean": round(passes / len(lab), 3), "clustered_se": round(clustered_se(by_task), 3),
        "pass_at_3": sum(any(v) for v in by_task.values()), "pass_hat_3": sum(all(v) for v in by_task.values()),
        "n_tasks": len(by_task),
        "done_in_first_response": len(early),
        "passed_when_done_early": sum(r["reward"] == 1.0 for r in early),
        "passed_when_checked": sum(r["reward"] == 1.0 for r in checked),
        "no_results_file": sum(r["outcome"] == "no valid results file" for r in lab),
        "fit_check_failed": sum(r["outcome"] == "fit check failed" for r in lab),
        "errors": sum(bool(r["exception"]) for r in lab + orig),
        "median_tokens_per_rollout": sorted(r["prompt_tokens"] + r["completion_tokens"] for r in lab)[len(lab) // 2],
        "median_wall_seconds": sorted(r["wall_seconds"] for r in lab)[len(lab) // 2],
        "est_cost_usd_all_rollouts": round(sum(r["est_cost_usd"] for r in lab + orig), 2),
    })

with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

lines = [
    "| Model | raman-fitting | raman-lab | Mean ± SE | pass@3 | pass^3 | Said done first |",
    "|---|---|---|---|---|---|---|",
]
for s in summary:
    lines.append(
        f"| {s['model']} | {s['raman_fitting_passed']}/{s['raman_fitting_rollouts']} "
        f"| {s['raman_lab_passed']}/{s['raman_lab_rollouts']} | {s['mean']:.2f} ± {s['clustered_se']:.2f} "
        f"| {s['pass_at_3']}/{s['n_tasks']} | {s['pass_hat_3']}/{s['n_tasks']} "
        f"| {s['done_in_first_response']}/{s['raman_lab_rollouts']} |"
    )
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
oracle = [r for r in rows if r["model"] == "oracle"]
print(f"oracle: {sum(r['reward'] == 1.0 for r in oracle)}/{len(oracle)} passed")
print("\n".join(lines))
for s in summary:
    print(s["model"], {k: s[k] for k in ("errors", "no_results_file", "fit_check_failed", "passed_when_done_early", "passed_when_checked",
                                         "median_tokens_per_rollout", "median_wall_seconds", "est_cost_usd_all_rollouts")})
