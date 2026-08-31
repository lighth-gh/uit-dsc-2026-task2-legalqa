from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
import unicodedata
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .text import chunk_passage, query_terms


SCHEMA_VERSION = "4"


def load_qa(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level phải là JSON object")
    for sample_id, item in data.items():
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError(f"{path}: mẫu {sample_id!r} không có question hợp lệ")
    return {str(key): value for key, value in data.items()}


def write_predictions(path: str | Path, predictions: dict[str, str]) -> None:
    output = {str(key): {"answer": str(answer)} for key, answer in predictions.items()}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    with tmp_target.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    tmp_target.replace(target)


def iter_contexts(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        for item_path in sorted(source.rglob("context_*.json")):
            with item_path.open("r", encoding="utf-8") as handle:
                yield json.load(handle)
        return

    if not zipfile.is_zipfile(source):
        raise ValueError("--contexts phải là thư mục context_*.json hoặc tệp .zip")
    with zipfile.ZipFile(source) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".json"))
        for name in names:
            yield json.loads(archive.read(name).decode("utf-8"))


def _connect(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    db_path = Path(path).resolve()
    if readonly:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _create_schema(connection: sqlite3.Connection, force: bool) -> None:
    if force:
        connection.executescript(
            "DROP TABLE IF EXISTS contexts_vocab; DROP TABLE IF EXISTS train_vocab; "
            "DROP TABLE IF EXISTS contexts_fts; DROP TABLE IF EXISTS train_fts; "
            "DROP TABLE IF EXISTS metadata;"
        )
    existing = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='contexts_fts'"
    ).fetchone()
    if existing:
        raise FileExistsError("Index đã tồn tại; thêm --force nếu muốn xây lại")

    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE VIRTUAL TABLE contexts_fts USING fts5(
            context_id UNINDEXED,
            chunk_no UNINDEXED,
            name,
            link UNINDEXED,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE train_fts USING fts5(
            sample_id UNINDEXED,
            question,
            answer UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE contexts_vocab USING fts5vocab(contexts_fts, 'row');
        CREATE VIRTUAL TABLE train_vocab USING fts5vocab(train_fts, 'row');
        """
    )


def build_index(
    contexts_path: str | Path,
    train_path: str | Path,
    db_path: str | Path,
    max_chunk_words: int = 620,
    overlap_words: int = 100,
    force: bool = False,
) -> dict[str, int | float]:
    started = time.time()
    train = load_qa(train_path)
    missing_answers = [
        sample_id
        for sample_id, item in train.items()
        if not str(item.get("answer") or "").strip()
    ]
    if missing_answers:
        preview = ", ".join(repr(sample_id) for sample_id in missing_answers[:5])
        suffix = "..." if len(missing_answers) > 5 else ""
        raise ValueError(
            f"{train_path}: train sample thiếu answer hợp lệ ({preview}{suffix})"
        )

    connection = _connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-200000")
        _create_schema(connection, force=force)

        train_rows = [
            (sample_id, item["question"], str(item.get("answer") or ""))
            for sample_id, item in train.items()
        ]
        connection.executemany(
            "INSERT INTO train_fts(sample_id, question, answer) VALUES (?, ?, ?)",
            train_rows,
        )

        doc_count = 0
        chunk_count = 0
        empty_docs = 0
        corpus_hasher = hashlib.sha256()
        batch: list[tuple[str, int, str, str, str]] = []
        for context in iter_contexts(contexts_path):
            doc_count += 1
            passage = str(context.get("passage") or "")
            chunks = chunk_passage(
                passage,
                max_words=max_chunk_words,
                overlap_words=overlap_words,
            )
            if not chunks:
                empty_docs += 1
                continue
            context_id = str(context.get("id", ""))
            name = str(context.get("name") or "")
            link = str(context.get("link") or "")
            for chunk_no, text in enumerate(chunks):
                corpus_hasher.update(context_id.encode("utf-8"))
                corpus_hasher.update(b"\0")
                corpus_hasher.update(str(chunk_no).encode("ascii"))
                corpus_hasher.update(b"\0")
                corpus_hasher.update(name.encode("utf-8"))
                corpus_hasher.update(b"\0")
                corpus_hasher.update(link.encode("utf-8"))
                corpus_hasher.update(b"\0")
                corpus_hasher.update(text.encode("utf-8"))
                corpus_hasher.update(b"\0")
                batch.append((context_id, chunk_no, name, link, text))
                chunk_count += 1
                if len(batch) >= 1000:
                    connection.executemany(
                        "INSERT INTO contexts_fts(context_id, chunk_no, name, link, text) "
                        "VALUES (?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
            if doc_count % 250 == 0:
                connection.commit()
                print(
                    f"[build] documents={doc_count:,} chunks={chunk_count:,}",
                    file=sys.stderr,
                    flush=True,
                )
        if batch:
            connection.executemany(
                "INSERT INTO contexts_fts(context_id, chunk_no, name, link, text) "
                "VALUES (?, ?, ?, ?, ?)",
                batch,
            )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "max_chunk_words": str(max_chunk_words),
            "overlap_words": str(overlap_words),
            "documents": str(doc_count),
            "chunks": str(chunk_count),
            "train_samples": str(len(train_rows)),
            "corpus_sha256": corpus_hasher.hexdigest(),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        elapsed = time.time() - started
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "empty_documents": empty_docs,
            "train_samples": len(train_rows),
            "elapsed_seconds": round(elapsed, 2),
        }
    finally:
        connection.close()


def _fts_query(terms: list[str]) -> str:
    # Mỗi token được quote nên các ký tự đặc biệt không thể biến thành cú pháp FTS.
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _fts_vocab_form(term: str) -> str:
    # Tương đương remove_diacritics của unicode61; ký tự đ/Đ không phải combining
    # mark nên được giữ nguyên giống SQLite.
    decomposed = unicodedata.normalize("NFD", term.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


class SearchIndex:
    def __init__(self, db_path: str | Path):
        self.connection = _connect(db_path, readonly=True)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SearchIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def metadata(self) -> dict[str, str]:
        return {
            row["key"]: row["value"]
            for row in self.connection.execute("SELECT key, value FROM metadata")
        }

    def _rarest_terms(
        self, question: str, vocab_table: str, max_terms: int
    ) -> list[str]:
        candidates = query_terms(question, max_terms=60)
        if len(candidates) <= max_terms:
            return candidates
        normalized_to_original: dict[str, str] = {}
        for term in candidates:
            normalized_to_original.setdefault(_fts_vocab_form(term), term)
        normalized = list(normalized_to_original)
        placeholders = ",".join("?" for _ in normalized)
        try:
            rows = self.connection.execute(
                f"SELECT term, doc FROM {vocab_table} WHERE term IN ({placeholders})",
                normalized,
            ).fetchall()
        except sqlite3.OperationalError:
            # Tương thích index schema v1 nếu người dùng chưa build lại.
            return candidates[:max_terms]
        doc_frequency = {str(row["term"]): int(row["doc"]) for row in rows}
        available = [term for term in candidates if _fts_vocab_form(term) in doc_frequency]
        available.sort(key=lambda term: (doc_frequency[_fts_vocab_form(term)], candidates.index(term)))
        return available[:max_terms] or candidates[:max_terms]

    def search_contexts(self, question: str, top_k: int = 12) -> list[dict[str, Any]]:
        query = _fts_query(self._rarest_terms(question, "contexts_vocab", max_terms=8))
        if not query:
            return []
        rows = self.connection.execute(
            """
            SELECT context_id, chunk_no, name, link, text,
                   bm25(contexts_fts, 0.0, 0.0, 2.0, 0.0, 1.0) AS bm25_score
            FROM contexts_fts
            WHERE contexts_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (query, int(top_k)),
        ).fetchall()
        return [dict(row) for row in rows]

    def search_train(
        self,
        question: str,
        top_k: int = 5,
        exclude_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = _fts_query(self._rarest_terms(question, "train_vocab", max_terms=12))
        if not query:
            return []
        rows = self.connection.execute(
            """
            SELECT sample_id, question, answer,
                   bm25(train_fts, 0.0, 1.0, 0.0) AS bm25_score
            FROM train_fts
            WHERE train_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (query, int(top_k + 10)),
        ).fetchall()
        output = [dict(row) for row in rows if str(row["sample_id"]) != str(exclude_id)]
        return output[:top_k]
