# /// script
# requires-python = ">=3.12"
# dependencies = ["harbor[hf-sandbox]==0.25.0", "numpy==2.3.1", "scipy==1.16.0", "requests", "huggingface_hub"]
# ///
"""Run one model on raman-fitting (original) and raman-lab (adapted) inside an HF Job.

Harbor orchestrates, each trial runs in its own HF Sandbox, and the model is served by
HF Inference Providers through the OpenAI-compatible router. Launch with scripts/launch.sh,
which inlines build_dataset.py into BUILD_DATASET_PY below.

Environment:
  MODEL          router model id, optionally pinned to a provider: "Qwen/Qwen3.8-27B:deepinfra"
  ROLLOUTS       rollouts per task (default 3)
  ORACLE         "1" to also run the reference solution on both datasets
  RUN_TAG        name of this run; trial folders are uploaded under runs/<RUN_TAG>/
  RESULTS_REPO   HF dataset repo that receives the raw Harbor job folders
  N_CONCURRENT   concurrent sandboxes (default 5)
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import requests

MODEL = os.environ["MODEL"]
K = os.environ.get("ROLLOUTS", "3")
ORACLE = os.environ.get("ORACLE", "0") == "1"
NAMESPACE = os.environ.get("SANDBOX_NAMESPACE", "openenv")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "")
ROUTER = "https://router.huggingface.co/v1"
BASE_ID, _, PROVIDER = MODEL.partition(":")
SLUG = BASE_ID.split("/")[-1].lower()
RUN_TAG = os.environ.get("RUN_TAG", SLUG)
WORK = Path("/tmp/work")
WORK.mkdir(exist_ok=True)

BUILD_DATASET_PY = r'''__BUILD_DATASET__'''

# --- 1. build the adapted dataset --------------------------------------------------------
(WORK / "build_dataset.py").write_text(BUILD_DATASET_PY)
subprocess.run([sys.executable, "build_dataset.py", "--out", "datasets/raman-lab"], cwd=WORK, check=True)
ADAPTED = WORK / "datasets" / "raman-lab"

# --- 2. fail loudly if the router does not serve the model (and provider) ------------------
token = os.environ["HF_TOKEN"]
catalog = {m["id"]: m for m in requests.get(f"{ROUTER}/models", headers={"Authorization": f"Bearer {token}"}, timeout=60).json()["data"]}
if BASE_ID not in catalog:
    sys.exit(f"router does not serve {BASE_ID}")
providers = {p["provider"]: p for p in catalog[BASE_ID]["providers"]}
if PROVIDER and PROVIDER not in providers:
    sys.exit(f"{BASE_ID} is not served by {PROVIDER}; available: {sorted(providers)}")
print("model:", BASE_ID, "provider:", PROVIDER or "router default", json.dumps(providers.get(PROVIDER, {})), flush=True)
os.environ["OPENAI_API_KEY"] = token  # litellm's openai-compatible route sends this key to the router


# --- 3. run harbor in-process so every HF Sandbox is created in NAMESPACE ------------------
def harbor(args, job_name):
    full = ["run"] + args + ["--jobs-dir", str(WORK / "jobs"), "--job-name", job_name, "-q"]
    code = textwrap.dedent(f"""
        import functools, sys
        from huggingface_hub import Sandbox
        Sandbox.create = functools.partial(Sandbox.create.__func__, Sandbox, namespace={NAMESPACE!r})
        from harbor.cli.main import app
        sys.argv = ["harbor"] + {full!r}
        app()
    """)
    print(f"\n=== harbor {' '.join(full)}", flush=True)
    subprocess.run([sys.executable, "-c", code], cwd=WORK, check=False)


common = ["-e", "hf-sandbox", "-n", os.environ.get("N_CONCURRENT", "5"), "--yes"]
agent = ["-a", "terminus-2", "-m", f"openai/{MODEL}", "--ak", f"api_base={ROUTER}"]

if ORACLE:
    harbor(["-p", str(ADAPTED), "-a", "oracle"] + common, "oracle-adapted")
    harbor(["-t", "terminal-bench/raman-fitting", "-a", "oracle"] + common, "oracle-original")
harbor(["-t", "terminal-bench/raman-fitting", "-k", K] + agent + common, f"{SLUG}-original")
harbor(["-p", str(ADAPTED), "-k", K] + agent + common, f"{SLUG}-adapted")

# --- 4. print a one-line summary per trial -------------------------------------------------
for res in sorted((WORK / "jobs").glob("*/*/result.json")):
    r = json.loads(res.read_text())
    if "task_name" in r:
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        exc = (r.get("exception_info") or {}).get("exception_type")
        print("TRIAL", res.parent.parent.name, r["task_name"], rewards.get("reward"), exc, flush=True)

# --- 5. scrub the token and upload the raw job folders -------------------------------------
for f in (WORK / "jobs").rglob("*"):
    if f.is_file():
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        if token in text:
            f.write_text(text.replace(token, "<HF_TOKEN>"))
(WORK / "jobs" / "run_meta.json").write_text(json.dumps(
    {"model": BASE_ID, "provider": PROVIDER or None, "k": int(K), "router_pricing": providers.get(PROVIDER, {}).get("pricing"),
     "harbor": "0.25.0", "agent": "terminus-2", "sandbox_namespace": NAMESPACE}, indent=2))
if RESULTS_REPO:
    from huggingface_hub import HfApi
    HfApi(token=token).upload_folder(repo_id=RESULTS_REPO, repo_type="dataset", folder_path=str(WORK / "jobs"),
                                     path_in_repo=f"runs/{RUN_TAG}", commit_message=f"Add {RUN_TAG}")
    print("uploaded to", RESULTS_REPO, f"runs/{RUN_TAG}", flush=True)
