"""Run successive Kaggle Stage 1 sessions, attaching each previous output.

Run this on an authenticated machine outside Kaggle. Each session gets a new
private notebook, so latest-version status and output cannot refer to an older
session of the same notebook.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "legalqa_main_01_qlora_train.ipynb"
DEFAULT_DATASETS = (
    "lighth/ver3-smoke-output",
    "lighth/uit-dsc-2026-task2-legalqa-train",
)


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def status_name(response):
    value = response.status
    return str(getattr(value, "name", value)).split(".")[-1].upper()


def metadata(kernel, previous, datasets, notebook_name):
    return {
        "id": kernel,
        "title": kernel.split("/", 1)[1],
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": list(datasets),
        "kernel_sources": [previous] if previous else [],
        "competition_sources": [],
        "model_sources": [],
    }


def submit(api, kernel, previous, datasets, notebook):
    with tempfile.TemporaryDirectory(prefix="legalqa-stage1-") as folder:
        folder = Path(folder)
        shutil.copy2(notebook, folder / notebook.name)
        (folder / "kernel-metadata.json").write_text(
            json.dumps(metadata(kernel, previous, datasets, notebook.name), indent=2),
            encoding="utf-8",
        )
        result = api.kernels_push(str(folder))
    if not getattr(result, "ref", None) or not isinstance(getattr(result, "version_number", None), int):
        raise RuntimeError(f"Kaggle did not confirm a version for {kernel}")
    return result.version_number


def wait_for_version(api, kernel, poll_seconds, max_wait_hours, sleep=time.sleep):
    deadline = time.monotonic() + max_wait_hours * 3600
    while True:
        response = api.kernels_status(kernel)
        status = status_name(response)
        print(f"{kernel}: {status}", flush=True)
        if status == "COMPLETE":
            return
        if status not in {"QUEUED", "RUNNING", "PENDING", "INITIALIZING"}:
            raise RuntimeError(f"Kaggle run stopped: {kernel}: {status}: "
                               f"{getattr(response, 'failure_message', '')}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {kernel}; rerun this command to continue polling")
        sleep(poll_seconds)


def read_stage_manifest(api, kernel, destination):
    destination.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel, str(destination), file_pattern=r"stage1_manifest\.json$", force=True)
    found = list(destination.rglob("stage1_manifest.json"))
    if len(found) != 1:
        raise RuntimeError(f"Expected one Stage 1 manifest in {kernel} output, found {len(found)}")
    manifest = json.loads(found[0].read_text(encoding="utf-8"))
    if manifest.get("schema") != 2 or manifest.get("stage") != 1:
        raise RuntimeError(f"Invalid Stage 1 manifest from {kernel}")
    if manifest.get("status") not in {"paused", "complete"}:
        raise RuntimeError(f"Unexpected Stage 1 status from {kernel}: {manifest.get('status')}")
    return manifest


def run(api, *, owner, slug_prefix, datasets=DEFAULT_DATASETS, notebook=NOTEBOOK,
        state_path=ROOT / "runs" / "auto_stage1" / "state.json", max_sessions=20,
        poll_seconds=120, max_wait_hours=15, sleep=time.sleep):
    notebook = Path(notebook)
    state_path = Path(state_path)
    if not notebook.is_file():
        raise FileNotFoundError(notebook)
    identity = {
        "owner": owner,
        "slug_prefix": slug_prefix,
        "datasets": list(datasets),
        "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("Runner settings or notebook changed; use the original files to resume")
    else:
        state = {"identity": identity, "run_id": uuid.uuid4().hex[:8],
                 "next_session": 1, "previous": None, "active": None, "done": False}
        save_state(state_path, state)
    if state["done"]:
        print(f"Stage 1 already complete: {state['previous']}")
        return state

    while True:
        active = state["active"]
        if active is None:
            number = state["next_session"]
            if number > max_sessions:
                raise RuntimeError(f"Stage 1 still paused after {max_sessions} sessions; state: {state_path}")
            kernel = f"{owner}/{slug_prefix}-{state['run_id']}-{number:03d}"
            previous = state["previous"]
            print(f"Starting session {number}: {kernel}; previous={previous}", flush=True)
            version = submit(api, kernel, previous, datasets, notebook)
            active = {"number": number, "kernel": kernel, "version": version}
            state["active"] = active
            save_state(state_path, state)

        kernel = active["kernel"]
        wait_for_version(api, kernel, poll_seconds, max_wait_hours, sleep)
        output = state_path.parent / f"session-{active['number']:03d}-manifest"
        found = list(output.rglob("stage1_manifest.json")) if output.exists() else []
        if found:
            # A restarted controller may have downloaded this manifest already.
            if len(found) != 1:
                raise RuntimeError(f"Ambiguous local manifest download: {output}")
            manifest = json.loads(found[0].read_text(encoding="utf-8"))
            if manifest.get("schema") != 2 or manifest.get("stage") != 1 or manifest.get("status") not in {"paused", "complete"}:
                raise RuntimeError(f"Invalid local Stage 1 manifest: {found[0]}")
        else:
            manifest = read_stage_manifest(api, kernel, output)
        print(f"Stage 1 session {active['number']}: {manifest['status']}", flush=True)
        state["previous"] = kernel
        state["next_session"] = active["number"] + 1
        state["active"] = None
        state["done"] = manifest["status"] == "complete"
        save_state(state_path, state)
        if state["done"]:
            print(f"Stage 1 complete. Attach {kernel} output to Stage 2.", flush=True)
            return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="Kaggle account username")
    parser.add_argument("--slug-prefix", default="legalqa-sft-auto")
    parser.add_argument("--dataset-source", action="append", dest="datasets",
                        help="Repeat for each Kaggle dataset; defaults to this project's two inputs")
    parser.add_argument("--state", type=Path, default=ROOT / "runs" / "auto_stage1" / "state.json")
    parser.add_argument("--max-sessions", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-hours", type=float, default=15)
    args = parser.parse_args()
    if args.max_sessions < 1 or args.poll_seconds < 1 or args.max_wait_hours <= 0:
        parser.error("session, poll, and wait limits must be positive")
    try:
        from kaggle import api
    except ImportError as error:
        raise SystemExit("Install the official Kaggle API first: python -m pip install kaggle") from error
    run(api, owner=args.owner, slug_prefix=args.slug_prefix,
        datasets=args.datasets or DEFAULT_DATASETS, state_path=args.state,
        max_sessions=args.max_sessions, poll_seconds=args.poll_seconds,
        max_wait_hours=args.max_wait_hours)


if __name__ == "__main__":
    main()
