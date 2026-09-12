"""Portable, bounded Kaggle stages. Heavy commands run in child processes.

The notebook kills the entire worker process group at its wall-clock budget.
Snapshots are finalized in a separate, bounded process after the worker exits.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .io import (ROOT, config, copy_file, digest, file_hash, load_questions, read_json,
                 source_hash, validate_predictions, write_json)
from .generation import adapter_identity
from .models import model_lock
from .runtime import should_pause


SCHEMA = 2
RESUME_FILES = {"adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth"}


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def marker(stage):
    return f"stage{stage}_manifest.json"


def verify_snapshot(root, stage=None):
    root = Path(root)
    if stage is None:
        paths = list(root.glob("stage[123]_manifest.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one stage manifest in {root}")
        path = paths[0]
    else:
        path = root/marker(stage)
    manifest = read_json(path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported snapshot schema; use explicit legacy import for old main outputs")
    if stage is not None and manifest.get("stage") != stage:
        raise ValueError("Wrong input stage")
    for relative, expected in manifest["files"].items():
        actual = safe_path(root, relative)
        if not actual.is_file() or actual.stat().st_size != expected["size"] or file_hash(actual) != expected["sha256"]:
            raise ValueError(f"Artifact missing/changed: {actual}")
    # Required data cannot simply be removed from the inventory to bypass checks.
    required = {"config.json", "session.json", "models.lock.json"}
    if manifest["stage"] != 1 or manifest["status"] == "complete" or "sft/training_manifest.json" in manifest["files"]:
        required.add("data/split_manifest.json")
    if not required.issubset(manifest["files"]):
        raise ValueError("Snapshot lacks its required provenance files")
    session = read_json(root/"session.json")
    if manifest["code_commit"] != session["code_commit"] or manifest["source_hash"] != session["source_hash"]:
        raise ValueError("Snapshot/session code identity mismatch")
    return manifest


def copy_artifacts(source, destination, manifest, predicate=lambda relative: True):
    """Copy only inventoried files. Never overwrite divergent local work."""
    for relative in manifest["files"]:
        if not predicate(relative):
            continue
        src, dst = safe_path(source, relative), safe_path(destination, relative)
        if dst.exists():
            if file_hash(dst) != file_hash(src):
                raise ValueError(f"Conflicting local artifact; use a fresh session: {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy_file(src, dst)


def valid_resume(path):
    path = Path(path)
    if not all((path/name).is_file() and (path/name).stat().st_size for name in RESUME_FILES):
        return False
    try:
        return read_json(path/"trainer_state.json")["global_step"] > 0
    except (ValueError, KeyError):
        return False


def epochs(sft, count=2):
    result = []
    for epoch in range(1, count+1):
        path = Path(sft)/f"epoch-{epoch:02d}"
        if not (path/"epoch_complete.json").is_file():
            continue
        state = read_json(path/"trainer_state.json")
        note = read_json(path/"epoch_complete.json")
        if abs(float(state["epoch"])-epoch) > 1e-6 or note["step"] != state["global_step"]:
            raise ValueError(f"Invalid epoch checkpoint: {path}")
        if adapter_identity(path) != note["adapter"]:
            raise ValueError(f"Epoch adapter checksum differs: {path}")
        result.append(path)
    return result


def harvest_epochs(sft):
    """Recover an epoch adapter if the worker stopped just after Trainer saved it."""
    sft = Path(sft)
    for checkpoint in sft.glob("checkpoint-*"):
        if not valid_resume(checkpoint):
            continue
        state = read_json(checkpoint/"trainer_state.json")
        epoch = float(state.get("epoch") or 0)
        if epoch not in (1.0, 2.0):
            continue
        destination = sft/f"epoch-{int(epoch):02d}"
        destination.mkdir(exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
            copy_file(checkpoint/name,destination/name)
        write_json(destination/"epoch_complete.json", {"epoch":epoch,"step":state["global_step"],
                                                       "adapter":adapter_identity(destination)})


def verify_training(sft, c, data, lock, index_hash, origin_code):
    from .training import select_training_questions
    sft = Path(sft)
    manifest = read_json(sft/"training_manifest.json")
    qa = select_training_questions(load_questions(Path(data)/"train.json", answers=True), c)
    if manifest["config"] != c or manifest["models"] != lock or manifest["code"] != origin_code:
        raise ValueError("Training config/model/code differs; refuse to reuse or relabel old training")
    if manifest["qa_hash"] != digest(qa) or set(manifest["qa_ids"]) != set(qa):
        raise ValueError("Training QA/split differs from the saved adapter")
    if manifest["retrieval"]["index_hash"] != index_hash:
        raise ValueError("Training used another index")
    if manifest["retrieval"].get("mode") != c["training"]["retrieval_mode"]:
        raise ValueError("Training used another retrieval mode")
    # Test/dev must stay disjoint from the training IDs used by QLoRA.
    dev = load_questions(Path(data)/f'{c["training"]["selection_split"]}.questions.json')
    if set(qa) & set(dev):
        raise ValueError("Training/selection split leakage")
    return manifest


def validate_selection(root):
    root = Path(root)
    selection = read_json(root/"selection.json")
    expected = selection["prediction_manifest"]["identity"]["adapter"]
    if not expected or adapter_identity(root/"selected_adapter") != expected:
        raise ValueError("Selected adapter bytes do not match the checkpoint evaluated on dev")
    return selection


def generation_ready(path):
    """A JSON written just before a kill is not a completed generation bundle."""
    path = Path(path)
    if not all(p.is_file() for p in (path, path.with_suffix(".audit.json"), path.with_suffix(".manifest.json"))):
        return False
    predictions = read_json(path)
    manifest = read_json(path.with_suffix(".manifest.json"))
    if manifest["prediction_hash"] != digest(predictions):
        raise ValueError("Prediction bundle checksum mismatch")
    if set(read_json(path.with_suffix(".audit.json"))) != set(predictions):
        raise ValueError("Prediction audit is incomplete")
    return True


def finalize(root, outcome="paused"):
    """Publish even partial progress; ZIPs contain diagnostics, never base weights."""
    root = Path(root)
    session_path = root/"session.json"
    if not session_path.is_file():
        raise ValueError("Setup did not finish; no resumable session to publish")
    session = read_json(session_path)
    stage = session["stage"]
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part.endswith(".tmp") for part in path.relative_to(root).parts):
            continue
        if path.suffix in {".tmp", ".zip"} or path.name in {marker(i) for i in (1,2,3)}:
            continue
        # A killed Trainer may leave an incomplete checkpoint; keep it out of the
        # portable snapshot so it cannot supersede the latest complete checkpoint.
        checkpoint = next((p for p in path.parents if p.name.startswith("checkpoint-") and p.parent.name == "sft"), None)
        if checkpoint and not valid_resume(checkpoint):
            continue
        files[relative] = {"size":path.stat().st_size, "sha256":file_hash(path)}
    progress = read_json(root/"progress.json") if (root/"progress.json").is_file() else {}
    # A timeout/error can leave an older 'complete' marker. Never promote that.
    complete = outcome == "ok" and progress.get("complete") is True
    manifest = {"schema":SCHEMA, "stage":stage, "quality_version":"v8",
                "code_commit":session["code_commit"], "source_hash":session["source_hash"],
                "status":"complete" if complete else outcome if outcome=="failed" else "paused",
                "progress":progress, "files":files}
    write_json(root/marker(stage), manifest)
    diagnostic_path = root/f"legalqa_main_stage{stage}_v8_diagnostics.zip"
    with ZipFile(diagnostic_path.with_suffix(".zip.tmp"), "w", compression=ZIP_DEFLATED) as archive:
        archive.write(root/marker(stage), arcname=marker(stage))
        for relative in files:
            path = root/relative
            # Includes dev questions/references, retrieval, predictions, audits,
            # reports, training state, and per-question resume journals.
            if path.suffix in {".json", ".jsonl", ".txt"}:
                archive.write(path, arcname=relative)
    os.replace(diagnostic_path.with_suffix(".zip.tmp"), diagnostic_path)
    print(f'SNAPSHOT {manifest["status"]}: {root/marker(stage)}', flush=True)
    print(f'Diagnostics: {diagnostic_path}', flush=True)
    return manifest


class Stage:
    def __init__(self, options):
        self.o = options
        self.number = int(options["stage"])
        self.root = Path(options["root"])
        self.data = self.root/"data"
        self.c = config()
        self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        self.code = source_hash()
        self.root.mkdir(parents=True, exist_ok=True)
        self.previous = Path(options["previous"]) if options.get("previous") else None
        self.upstream = Path(options["upstream"]) if options.get("upstream") else None
        self.legacy = Path(options["legacy"]) if options.get("legacy") else None
        self.index = Path(options["version3"])/"index"
        self.saved_models = Path(options["version3"])/"models"
        self.models = self.root.parent/f"stage{self.number}_runtime_models"
        self.models.mkdir(exist_ok=True)
        for role in ("embedding", "reranker", "generator"):
            source = self.saved_models/role
            if not (source/"config.json").is_file():
                raise FileNotFoundError(source/"config.json")
            link = self.models/role
            if link.exists() or link.is_symlink():
                if link.resolve() != source.resolve():
                    raise ValueError(f"Wrong model symlink: {link}")
            else:
                link.symlink_to(source, target_is_directory=True)
        shutil.copy2(self.saved_models/"models.lock.json", self.models/"models.lock.json")
        self.lock = model_lock(self.c, self.models)
        index_manifest = read_json(self.index/"index_manifest.json")
        if index_manifest.get("chunks") != 407107 or index_manifest.get("documents") != 8507:
            raise ValueError("Expected completed Version 3 full index")
        self.index_hash = digest(index_manifest)
        if index_manifest["identity"]["embedding"] != self.lock["models"]["embedding"]:
            raise ValueError("Index/model embedding revision mismatch")
        self.identity = {"stage":self.number,"code_commit":self.commit,"source_hash":self.code,
                         "config_hash":digest(self.c),"models":self.lock,"index_hash":self.index_hash}
        self.restore()
        if (self.root/"config.json").is_file() and read_json(self.root/"config.json") != self.c:
            raise ValueError("Saved config differs from pinned code config")
        if (self.root/"models.lock.json").is_file() and read_json(self.root/"models.lock.json") != self.lock:
            raise ValueError("Saved model revisions differ")
        write_json(self.root/"config.json", self.c)
        write_json(self.root/"models.lock.json", self.lock)
        write_json(self.root/"session.json", self.identity)
        self.cfg = self.root/"config.json"
        if self.c["evaluation"]["primary_metric"] != "meteor" or self.c["evaluation"]["target_meteor"] != .65:
            raise ValueError("METEOR objective must remain 0.65")
        if not self.c["training"]["required"] or not self.c["generation"]["load_in_4bit"]:
            raise ValueError("QLoRA is mandatory")
        print("Evaluation objective:", self.c["evaluation"], flush=True)
        print("Retrieval:", self.c["retrieval"], "Generation:", self.c["generation"], flush=True)

    def restore(self):
        existing = self.root/"session.json"
        if self.previous and not existing.exists():
            snapshot = verify_snapshot(self.previous, self.number)
            # The session file is the commit marker for restoration. Publishing
            # it first would make a half-copied output look ready on the next run.
            copy_artifacts(self.previous, self.root, snapshot, lambda p:p != "session.json")
            copy_file(self.previous/"session.json", existing)
        if existing.exists():
            previous = read_json(existing)
            for key, value in self.identity.items():
                if previous.get(key) != value:
                    raise ValueError(f"Local/resumed session identity differs: {key}")
            self.identity = previous
        if self.upstream:
            snap = verify_snapshot(self.upstream, self.number-1)
            source_session = read_json(self.upstream/"session.json")
            for key in ("code_commit", "source_hash", "config_hash", "models", "index_hash"):
                if source_session[key] != self.identity[key]:
                    raise ValueError(f"Upstream provenance mismatch: {key}")
            # Only completed upstream stages can advance; partial snapshots
            # must be resumed as PREVIOUS_OUTPUT of that same stage.
            if snap["status"] != "complete":
                raise ValueError("Upstream stage is partial; resume it before moving to the next stage")
            upstream_id = digest(snap)
            if self.identity.get("upstream_id") not in (None, upstream_id):
                raise ValueError("Previous output belongs to a different upstream version")
            self.identity["upstream_id"] = upstream_id
            if not existing.exists():
                def keep(relative):
                    if relative in {"session.json", "progress.json", "environment.freeze.txt"}:
                        return False
                    if self.number == 3 and (relative.startswith("sft/") or relative.startswith("train.sft")):
                        return False
                    return True
                copy_artifacts(self.upstream, self.root, snap, keep)
            origin = source_session.get("training_origin_code", self.code)
            self.identity["training_origin_code"] = origin
        elif self.number > 1 and not existing.exists():
            raise ValueError("Attach upstream output or a previous cumulative output")

    def command(self, *args, cap=0):
        if should_pause():
            return False
        env = {**os.environ, "LEGALQA_MAX_ITEMS":str(cap), "LEGALQA_INDEX_HASH":self.index_hash,
               "PYTHONUNBUFFERED":"1"}
        command = [sys.executable, "-m", "legalqa", "--config", str(self.cfg),
                   "--models", str(self.models), *map(str,args)]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        audit = self.models/"parameter_audit.json"
        if audit.is_file():
            copy_file(audit, self.root/"parameter_audit.json")
        return True

    def progress(self, **values):
        write_json(self.root/"progress.json", values)
        print("Progress:", values, flush=True)

    def train(self):
        from .data import prepare
        from .training import prepare_training_subset
        if not (self.data/"split_manifest.json").exists():
            dataset = Path(self.o["dataset"])
            fingerprints = {name:file_hash(dataset/name) for name in ("train.json","public-official.json")}
            if self.identity.get("dataset_hashes",fingerprints) != fingerprints:
                raise ValueError("Dataset changed during interrupted data preparation")
            self.identity["dataset_hashes"] = fingerprints
            write_json(self.root/"session.json",self.identity)
            prepare(dataset/"train.json", dataset/"public-official.json", self.data,self.c["seed"])
        sft = self.root/"sft"
        pending = self.identity.get("pending_legacy")
        if pending and self.legacy is None:
            self.legacy = Path(pending["source"])
        if self.legacy and not sft.exists():
            self.import_legacy()
        imported = sft/"import_provenance.json"
        if imported.is_file():
            provenance = read_json(imported)
            self.identity["training_origin_code"] = provenance["code"]
            self.identity["legacy_training_manifest_sha256"] = provenance["manifest_sha256"]
            self.identity.pop("pending_legacy",None)
            write_json(self.root/"session.json",self.identity)
        if (sft/"training_manifest.json").exists() or (sft/"training_result.json").exists():
            verify_training(sft,self.c,self.data,self.lock,self.index_hash,
                            self.identity.get("training_origin_code",self.code))
        harvest_epochs(sft)
        if (sft/"training_result.json").exists():
            if len(epochs(sft)) != 2:
                raise ValueError("Completed training lacks both verified epoch adapters")
            self.progress(complete=True, phase="training_complete")
            return
        if self.legacy:
            raise ValueError("Legacy migration accepts only a fully completed two-epoch run")
        train = self.data/"train.sft.json"
        questions = self.data/"train.sft.questions.json"
        cache = self.root/"train.sft.lexical.retrieval.json"
        prepare_training_subset(self.c,self.data/"train.json",train)
        if not cache.exists():
            self.command("retrieve", "--questions", questions,"--index",self.index,"--output",cache,"--mode","lexical")
        if not cache.exists() or should_pause():
            self.progress(complete=False, phase="train_retrieval")
            return
        args = ["fit", "--train",train,"--retrieval",cache,"--output",sft,"--gpu","0"]
        checkpoints = [p for p in sft.glob("checkpoint-*") if valid_resume(p)]
        if checkpoints:
            latest = max(checkpoints,key=lambda p:read_json(p/"trainer_state.json")["global_step"])
            args += ["--resume",latest]
        elif sft.exists() and any(sft.iterdir()):
            # No optimizer step was ever saved: only setup manifests may be
            # removed, after their identity was checked above. Preserve reports.
            leftovers = list(sft.iterdir())
            allowed = {"training_manifest.json", "training_data_report.json"}
            if any(p.name not in allowed for p in leftovers):
                raise ValueError("Incomplete checkpoint without a valid resume; inspect diagnostics")
            for p in leftovers:
                p.unlink()
        self.command(*args)
        complete = (sft/"training_result.json").is_file() and len(epochs(sft)) == 2
        self.progress(complete=complete, phase="training_complete" if complete else "training_paused")

    def import_legacy(self):
        """Explicit migration: preserve old training origin; regenerate all evaluations."""
        source = self.legacy
        old_config = read_json(source/"config.json")
        if old_config != self.c:
            raise ValueError("Legacy config differs; do not relabel old adapters with current settings")
        if read_json(source/"data_public"/"split_manifest.json") != read_json(self.data/"split_manifest.json"):
            raise ValueError("Legacy data/split differs")
        sft = source/"sft"
        original = read_json(sft/"training_manifest.json")
        verify_training(sft,self.c,self.data,self.lock,self.index_hash,original["code"])
        result = read_json(sft/"training_result.json")
        if result.get("epochs") != 2:
            raise ValueError("Legacy training did not finish two epochs")
        adapter_identity(sft/"adapter_last")
        found = {}
        for p in sft.glob("checkpoint-*"):
            if not (p/"trainer_state.json").is_file():
                continue
            state = read_json(p/"trainer_state.json")
            e = float(state.get("epoch") or 0)
            if e in (1.0,2.0):
                adapter_identity(p)
                found[int(e)] = p
        if set(found) != {1,2}:
            raise ValueError("Legacy output must include complete epoch 1 and 2 adapters")
        provenance = {"code":original["code"],"manifest_sha256":file_hash(sft/"training_manifest.json"),
                      "adapters":{str(e):adapter_identity(p) for e,p in found.items()}}
        pending = {"source":str(source),"provenance":provenance}
        if self.identity.get("pending_legacy",pending) != pending:
            raise ValueError("Pending legacy import differs from its original source")
        self.identity["pending_legacy"] = pending
        write_json(self.root/"session.json",self.identity)
        target = self.root/"legacy_import.tmp"
        shutil.copytree(sft,target,dirs_exist_ok=True,copy_function=copy_file)
        for epoch,p in found.items():
            destination = target/f"epoch-{epoch:02d}"
            destination.mkdir(exist_ok=True)
            for name in ("adapter_config.json","adapter_model.safetensors","trainer_state.json"):
                copy_file(p/name,destination/name)
            state = read_json(p/"trainer_state.json")
            write_json(destination/"epoch_complete.json",{"epoch":epoch,"step":state["global_step"],
                                                          "adapter":adapter_identity(destination)})
        write_json(target/"import_provenance.json",provenance)
        # Either the entire imported SFT appears, or none of it. train() also
        # recovers provenance if interruption happens just after this rename.
        os.replace(target,self.root/"sft")
        self.identity["training_origin_code"] = original["code"]
        self.identity["legacy_training_manifest_sha256"] = provenance["manifest_sha256"]
        self.identity.pop("pending_legacy",None)
        write_json(self.root/"session.json",self.identity)
        print("Imported legacy QLoRA; original training identity retained. No training rerun.",flush=True)

    def select_retrieve(self):
        from .metrics import select_reports
        sft = self.root/"sft"
        verify_training(sft,self.c,self.data,self.lock,self.index_hash,
                        self.identity.get("training_origin_code",self.code))
        checkpoints = epochs(sft)
        if len(checkpoints) != 2:
            raise ValueError("Both completed QLoRA epoch adapters are required")
        split = self.c["training"]["selection_split"]
        questions = self.data/f"{split}.questions.json"
        references = self.data/f"{split}.references.json"
        cache = self.root/f"{split}.retrieval.json"
        if not (self.root/"selection.json").is_file():
            if self.o.get("mode") == "retrieve":
                raise ValueError("First run mode=select or auto; no selected checkpoint yet")
            if not cache.exists():
                self.command("retrieve","--questions",questions,"--index",self.index,"--output",cache)
            if not cache.exists() or should_pause():
                self.progress(complete=False,phase="dev_retrieval")
                return
            baseline = self.root/f"{split}.base.metrics.json"
            reports = []
            for adapter in [None]+checkpoints:
                label = adapter.name if adapter else "base_v8"
                pred = self.root/f"{split}.{label}.json"
                report = baseline if adapter is None else self.root/f"{split}.{label}.metrics.json"
                if not generation_ready(pred):
                    args = ["generate","--questions",questions,"--retrieval",cache,"--output",pred]
                    if adapter:
                        args += ["--adapter",adapter]
                    self.command(*args)
                if not generation_ready(pred) or should_pause():
                    self.progress(complete=False,phase=f"evaluate_{label}")
                    return
                if not report.exists():
                    # Metrics also run in subprocess: they obey the notebook's
                    # process-group timeout, including WordNet initialization.
                    self.command("evaluate","--predictions",pred,"--references",references,
                                 "--output",report,"--label",label)
                if not report.exists():
                    self.progress(complete=False,phase=f"score_{label}")
                    return
                if adapter:
                    self.command("compare","--baseline",baseline,"--candidate",report,
                                 "--output",self.root/f"{split}.{label}.comparison.json")
                    reports.append(report)
            if should_pause():
                self.progress(complete=False,phase="select")
                return
            select_reports(reports,self.root/"selection.pending.json",self.c["evaluation"])
            selected = read_json(self.root/"selection.pending.json")
            # Reports may contain an absolute path from an earlier Kaggle session.
            source = sft/selected["label"]
            if adapter_identity(source) != selected["prediction_manifest"]["identity"]["adapter"]:
                raise ValueError("Evaluation report no longer matches its adapter")
            destination = self.root/"selected_adapter"
            destination.mkdir(exist_ok=True)
            for name in ("adapter_config.json","adapter_model.safetensors","trainer_state.json"):
                copy_file(source/name,destination/name)
            selected["portable_adapter_path"] = "selected_adapter"
            write_json(self.root/"selection.json",selected)
        selection = validate_selection(self.root)
        print("Selected METEOR:",selection["meteor"],"Target:",selection["objective"],flush=True)
        public = self.root/"public.retrieval.json"
        if self.o.get("mode") != "select" and not public.exists():
            self.command("retrieve","--questions",self.data/"test.questions.json","--index",self.index,"--output",public)
        self.progress(complete=public.exists(),phase="public_ready" if public.exists() else "public_retrieval",
                      selection_ready=True)

    def generate_submit(self):
        selection = validate_selection(self.root)
        questions = self.data/"test.questions.json"
        cache = self.root/"public.retrieval.json"
        if not cache.exists():
            raise ValueError("Stage 2 public retrieval is incomplete")
        destination = self.root/"submissions"/"public"
        pred = destination/"submission.json"
        if not generation_ready(pred):
            self.command("generate","--questions",questions,"--retrieval",cache,
                         "--adapter",self.root/"selected_adapter","--output",pred,
                         cap=int(self.o.get("max_new_questions",200)))
        if not generation_ready(pred):
            partial = pred.with_suffix(".partial.json")
            self.progress(complete=False,phase="generate",answered=len(read_json(partial)) if partial.exists() else None,
                          total=len(load_questions(questions)))
            return
        validate_predictions(read_json(pred),load_questions(questions))
        from .generation import package_submission
        package_submission(pred,questions,destination/"submission.zip",self.c["submission_filename"])
        self.progress(complete=True,phase="submission_ready",answers=len(load_questions(questions)),
                      selected_meteor=selection["meteor"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["run","finalize"])
    parser.add_argument("--options",required=True)
    parser.add_argument("--outcome",default="paused",choices=["ok","paused","failed"])
    args = parser.parse_args()
    options = read_json(args.options)
    if args.action == "finalize":
        finalize(options["root"],args.outcome)
        return
    stage = Stage(options)
    freeze = subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (stage.root/"environment.freeze.txt").write_text(freeze,encoding="utf-8")
    [None,stage.train,stage.select_retrieve,stage.generate_submit][stage.number]()


if __name__ == "__main__":
    main()
