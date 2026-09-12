import hashlib
import json
import os
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return json.load(f, object_pairs_hook=_unique_pairs)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def copy_file(source, destination):
    """Expose a copied file only after all bytes have arrived."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+".copying.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return str(destination)


def digest(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hash():
    return digest({str(p.relative_to(ROOT)): file_hash(p)
                   for p in sorted((ROOT / "legalqa").glob("*.py"))})


class Journal:
    """Append-only checkpoints with identity checks and recovery of an interrupted final line."""
    def __init__(self, path, identity):
        self.path = Path(path)
        meta = self.path.with_suffix(self.path.suffix+".meta.json")
        if meta.exists():
            if read_json(meta) != identity:
                raise ValueError("Checkpoint identity differs; choose a new output path")
        else:
            if self.path.exists():
                raise ValueError("Checkpoint data exists without its identity manifest")
            write_json(meta,identity)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.records = {}
        if self.path.exists():
            safe_offset = 0
            with self.path.open("rb") as f:
                for line in f:
                    if not line.endswith(b"\n"):
                        break  # an interrupted final append; only committed complete lines are reused
                    row = json.loads(line,object_pairs_hook=_unique_pairs)
                    if row["id"] in self.records:
                        raise ValueError("Duplicate checkpoint record")
                    self.records[row["id"]] = row["value"]
                    safe_offset = f.tell()
            if self.path.stat().st_size != safe_offset:
                with self.path.open("r+b") as f:
                    f.truncate(safe_offset)

    def append(self, key, value):
        if key in self.records:
            raise ValueError("Checkpoint record already committed")
        line = json.dumps({"id":key,"value":value},ensure_ascii=False,allow_nan=False)+"\n"
        with self.path.open("ab") as f:
            f.write(line.encode("utf-8")); f.flush(); os.fsync(f.fileno())
        self.records[key] = value


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def group_key(text):
    return normalize(text).casefold().rstrip(" ?.!。")


def load_questions(path, answers=False):
    raw = read_json(path)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Expected a nonempty JSON object keyed by question ID")
    result = {}
    for key, value in raw.items():
        if not isinstance(value, dict) or not isinstance(value.get("question"), str):
            raise ValueError(f"Invalid question record: {key}")
        if not value["question"].strip():
            raise ValueError(f"Empty question: {key}")
        item = {"question": value["question"]}
        if answers:
            if not isinstance(value.get("answer"), str) or not value["answer"].strip():
                raise ValueError(f"Missing/empty gold answer: {key}")
            item["answer"] = value["answer"]  # preserve the BTC target verbatim
        result[key] = item
    return result


def validate_predictions(predictions, questions):
    if not isinstance(predictions, dict) or set(predictions) != set(questions):
        got = set(predictions) if isinstance(predictions, dict) else set()
        raise ValueError(f"ID mismatch: missing={len(set(questions)-got)}, extra={len(got-set(questions))}")
    for key, item in predictions.items():
        if not isinstance(item, dict) or set(item) != {"answer"}:
            raise ValueError(f"Expected exactly {{'answer': string}} at {key}")
        if not isinstance(item["answer"], str) or not item["answer"].strip():
            raise ValueError(f"Empty/non-string answer: {key}")


def config(path=None):
    c = read_json(path or ROOT / "config.json")
    if c["parameter_limit"] != 4_000_000_000:
        raise ValueError("This pipeline enforces the competition's strict <4B limit")
    ch = c["chunking"]
    if not 0 <= ch["overlap_tokens"] < ch["child_tokens"] <= 420:
        raise ValueError("Invalid child token window")
    if ch["child_tokens"] + ch["header_tokens"] + 12 > 512:
        raise ValueError("E5 input would exceed 512 tokens")
    if c["generation"]["max_input_tokens"] + c["generation"]["max_new_tokens"] > 8192:
        raise ValueError("Default RAG context policy is at most 8192 total tokens")
    return c
