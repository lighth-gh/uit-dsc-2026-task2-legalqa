import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from .io import digest, file_hash, group_key, load_questions, normalize, read_json, write_json


def split_records(records, seed=2026):
    """Group identical normalized questions OR answers before a deterministic 80/10/10 split."""
    ids = sorted(records)
    parent = {key: key for key in ids}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for key in ids:
        for field in ("question", "answer"):
            signature = (field, group_key(records[key][field]))
            if signature in seen:
                a, b = find(key), find(seen[signature])
                parent[max(a, b)] = min(a, b)
            else:
                seen[signature] = key
    groups = defaultdict(list)
    for key in ids:
        groups[find(key)].append(key)
    if len(groups) < 10:
        raise ValueError("Need at least 10 independent QA groups for the 80/10/10 split")
    ordered = sorted(groups.values(), key=lambda g: digest([seed, sorted(g)]))
    # Assign whole groups by total group count; sizes are deliberately approximate.
    n = len(ordered)
    cut1, cut2 = max(1, round(0.8*n)), min(n-1, round(0.9*n))
    splits = {
        "train": [k for g in ordered[:cut1] for k in g],
        "dev": [k for g in ordered[cut1:cut2] for k in g],
        "holdout": [k for g in ordered[cut2:] for k in g],
    }
    for part in splits:
        splits[part].sort(key=lambda k: digest([seed, part, k]))
    return splits, {"groups": n, "duplicate_group_records": len(ids)-n}


def prepare(train_path, test_path, output, seed):
    train = load_questions(train_path, answers=True)
    test = load_questions(test_path)
    splits, group_info = split_records(train, seed)
    out = Path(output)
    manifest = {"seed": seed, "train_sha256": file_hash(train_path),
                "test_sha256": file_hash(test_path), "grouping": group_info,
                "splits": splits, "source": "user-supplied BTC Task 2 files"}
    splits["dev30"] = splits["dev"][:30]
    splits["dev100"] = splits["dev"][:100]
    splits["train_all"] = sorted(train)
    existing = out / "split_manifest.json"
    if existing.exists() and read_json(existing) != manifest:
        raise ValueError("Existing split differs. Choose a NEW output directory; do not overwrite a locked split.")
    for split, keys in splits.items():
        write_json(out / f"{split}.json", {k: train[k] for k in keys})
        write_json(out / f"{split}.questions.json", {k: {"question": train[k]["question"]} for k in keys})
        write_json(out / f"{split}.references.json", {k: train[k]["answer"] for k in keys})
    write_json(out / "test.questions.json", test)
    write_json(existing, manifest)
    lengths = sorted(len(x["answer"].split()) for x in train.values())
    report = {"samples": len(train), "test_samples": len(test), **group_info,
              "split_sizes": {k: len(v) for k, v in splits.items()},
              "train_answer_words": {"median": lengths[len(lengths)//2],
                                     "p90": lengths[int((len(lengths)-1)*.9)], "max": max(lengths)}}
    write_json(out / "data_report.json", report)
    return report


def iter_documents(path):
    """Read BTC context JSONs recursively, including a nested selected-contexts.zip."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        with ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if name.endswith(".json") and not name.startswith("__MACOSX/"):
                    yield from _parse_document(json.loads(z.read(name).decode("utf-8-sig")), name)
    elif path.is_dir():
        files = sorted(path.rglob("context_*.json"))
        if not files:
            raise ValueError("No context_*.json files found under the corpus directory")
        for file in files:
            yield from _parse_document(read_json(file), str(file.relative_to(path)))
    else:
        raise ValueError("Corpus must be selected-contexts.zip or a directory containing context_*.json")


def _parse_document(raw, source):
    # One BTC document per file. Other schemas fail visibly instead of being silently guessed.
    if not isinstance(raw, dict) or "passage" not in raw or "id" not in raw:
        raise ValueError(f"Expected {{id,name,link,passage}} at {source}")
    if raw["passage"] is not None and not isinstance(raw["passage"], str):
        raise ValueError(f"Invalid passage type at {source}")
    text = unicodedata.normalize("NFC", (raw["passage"] or "").replace("\r", "\n"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()
    yield {"doc_id": str(raw["id"]), "name": str(raw.get("name") or ""),
           "link": str(raw.get("link") or ""), "text": text, "source_file": source}


ARTICLE = re.compile(r"(?m)^[ \t]*(Điều[ \t]+\d+[a-zđ]?(?:[.:][ \t]*|[ \t]+)[^\n]*)", re.I)
DOC_NUMBER = re.compile(r"\b\d{1,5}(?:/\d{4})?/[A-Za-zÀ-ỹĐđ0-9]+(?:-[A-Za-zÀ-ỹĐđ0-9]+)*\b")


def document_number(text):
    m = re.search(r"(?im)^\s*Số\s*:\s*([^\n]+)", text[:6000])
    if m:
        number = DOC_NUMBER.search(m.group(1))
        if number:
            return number.group(0)
    return ""  # never turn a website page ID or slug suffix into a legal document number


def legal_parents(doc):
    text = doc["text"]
    matches = list(ARTICLE.finditer(text))
    starts = sorted(set([0] + [m.start() for m in matches] + [len(text)]))
    number = document_number(text)
    for ordinal, (start, end) in enumerate(zip(starts, starts[1:])):
        body = text[start:end].strip()
        if not body:
            continue
        match = ARTICLE.match(body)
        heading = normalize(match.group(1)) if match else ""
        yield {"parent_id": f"{doc['doc_id']}:{ordinal}", "doc_id": doc["doc_id"],
               "heading": heading, "number": number, "text": body,
               "source_file": doc["source_file"], "link": doc["link"]}


def token_children(parent, tokenizer, size, overlap, header_tokens):
    body = parent["text"]
    offsets = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    header = " ".join(x for x in [parent["number"], parent["heading"]] if x)
    header = tokenizer.decode(tokenizer(header, add_special_tokens=False)["input_ids"][:header_tokens])
    for start in range(0, len(offsets), size-overlap):
        end = min(start+size, len(offsets))
        a, b = offsets[start][0], offsets[end-1][1]
        text = body[a:b].strip()
        if text:
            yield {"parent_id": parent["parent_id"], "start": a, "end": b,
                   "header": header, "text": text}
        if end == len(offsets):
            break
