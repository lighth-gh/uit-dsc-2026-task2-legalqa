from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import unicodedata
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from .text import (
    chunk_passage,
    expand_retrieval_query,
    query_terms,
    retrieval_priority_phrases,
    tokenize,
    validate_chunk_parameters,
)


SCHEMA_VERSION = "5"
CORPUS_HASH_VERSION = "2"
_CORPUS_RECORD_DOMAIN = b"legalqa-corpus-record-v2\0"
_CORPUS_DOMAIN = b"legalqa-corpus-v2\0"
_ORDERED_CORPUS_DOMAIN = b"legalqa-corpus-order-v2\0"


class CorpusHasher:
    """Tạo fingerprint không phụ thuộc thứ tự từ các chunk đã canonical hóa."""

    def __init__(self) -> None:
        self._record_digests: list[bytes] = []

    def update(
        self,
        context_id: str,
        chunk_no: int,
        name: str,
        link: str,
        text: str,
    ) -> None:
        record_hasher = hashlib.sha256(_CORPUS_RECORD_DOMAIN)
        for value in (context_id, str(chunk_no), name, link, text):
            encoded = value.encode("utf-8")
            record_hasher.update(len(encoded).to_bytes(8, "big"))
            record_hasher.update(encoded)
        self._record_digests.append(record_hasher.digest())

    def hexdigest(self) -> str:
        corpus_hasher = hashlib.sha256(_CORPUS_DOMAIN)
        corpus_hasher.update(len(self._record_digests).to_bytes(8, "big"))
        for digest in sorted(self._record_digests):
            corpus_hasher.update(digest)
        return corpus_hasher.hexdigest()

    def ordered_hexdigest(self) -> str:
        """Fingerprint phụ thuộc thứ tự, dùng cho checkpoint theo vị trí."""
        corpus_hasher = hashlib.sha256(_ORDERED_CORPUS_DOMAIN)
        corpus_hasher.update(len(self._record_digests).to_bytes(8, "big"))
        for digest in self._record_digests:
            corpus_hasher.update(digest)
        return corpus_hasher.hexdigest()


def load_qa(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level phải là JSON object")
    for sample_id, item in data.items():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("question"), str)
            or not str(item.get("question") or "").strip()
        ):
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


def _is_context_member(parts: tuple[str, ...]) -> bool:
    if (
        not parts
        or any(part.startswith(".") or part == "__MACOSX" for part in parts)
        or any(part == ".." for part in parts)
    ):
        return False
    file_name = parts[-1]
    return file_name.startswith("context_") and file_name.endswith(".json")


def _validate_context(
    value: Any,
    location: str,
    seen_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: context phải là JSON object")

    raw_id = value.get("id")
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
        raise ValueError(f"{location}: context id phải là chuỗi hoặc số nguyên")
    context_id = str(raw_id).strip()
    if not context_id:
        raise ValueError(f"{location}: context id không được để trống")
    if context_id in seen_ids:
        raise ValueError(f"{location}: context id bị trùng: {context_id!r}")

    passage = value.get("passage")
    if not isinstance(passage, str):
        raise ValueError(f"{location}: passage phải là chuỗi")

    normalized = dict(value)
    normalized["id"] = context_id
    normalized["passage"] = passage
    for field in ("name", "link"):
        field_value = value.get(field)
        if field_value is None:
            normalized[field] = ""
        elif isinstance(field_value, str):
            normalized[field] = field_value
        else:
            raise ValueError(f"{location}: {field} phải là chuỗi hoặc null")

    seen_ids.add(context_id)
    return normalized


def iter_contexts(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    seen_ids: set[str] = set()
    if source.is_dir():
        items: list[tuple[str, Path]] = []
        for item_path in source.rglob("*.json"):
            relative = item_path.relative_to(source)
            if item_path.is_file() and _is_context_member(relative.parts):
                items.append((relative.as_posix(), item_path))
        items.sort(key=lambda item: (PurePosixPath(item[0]).name, item[0]))
        for relative_name, item_path in items:
            with item_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            yield _validate_context(value, str(source / relative_name), seen_ids)
        return

    if not zipfile.is_zipfile(source):
        raise ValueError("--contexts phải là thư mục context_*.json hoặc tệp .zip")
    with zipfile.ZipFile(source) as archive:
        valid_members: list[tuple[str, zipfile.ZipInfo]] = []
        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            parts = PurePosixPath(normalized_name).parts
            if not member.is_dir() and _is_context_member(parts):
                valid_members.append((normalized_name, member))
        valid_members.sort(
            key=lambda item: (PurePosixPath(item[0]).name, item[0])
        )
        for name, member in valid_members:
            value = json.loads(archive.read(member).decode("utf-8"))
            yield _validate_context(value, f"{source}!/{name}", seen_ids)


def _connect(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    db_path = Path(path).resolve()
    if readonly:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        # FTS5 performs many small random reads. Keep its temporary work and a
        # useful page cache in RAM; this matters especially on Kaggle mounts.
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-200000")
        connection.execute("PRAGMA mmap_size=1073741824")
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


def _sqlite_sidecar_paths(db_path: Path) -> tuple[Path, Path, Path]:
    return (
        Path(f"{db_path}-wal"),
        Path(f"{db_path}-shm"),
        Path(f"{db_path}-journal"),
    )


def _cleanup_temporary_database(db_path: Path) -> None:
    for path in (db_path, *_sqlite_sidecar_paths(db_path)):
        path.unlink(missing_ok=True)


def _quiesce_database(db_path: Path) -> None:
    wal_path, shm_path, journal_path = _sqlite_sidecar_paths(db_path)
    if not any(path.exists() for path in (wal_path, shm_path, journal_path)):
        return

    connection = sqlite3.connect(db_path, timeout=1.0)
    try:
        connection.execute("PRAGMA busy_timeout=1000")
        # Reading the schema first lets SQLite recover a hot rollback journal.
        connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is not None and int(checkpoint[0]) != 0:
            raise RuntimeError(
                "Không thể thay index đang được sử dụng; hãy đóng các SearchIndex đang mở"
            )
        journal_mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
        if journal_mode is None or str(journal_mode[0]).lower() != "delete":
            raise RuntimeError("Không thể hợp nhất WAL trước khi thay index")
    finally:
        connection.close()

    remaining = [
        path for path in (wal_path, shm_path, journal_path) if path.exists()
    ]
    if remaining:
        raise RuntimeError(
            "Không thể thay index khi SQLite sidecar vẫn đang được sử dụng"
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
    validate_chunk_parameters(max_chunk_words, overlap_words)
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

    target_path = Path(db_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        raise FileExistsError("Index đã tồn tại; thêm --force nếu muốn xây lại")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".building",
        dir=target_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        connection = _connect(temporary_path)
        try:
            # The temporary database is never queried while it is being built, so
            # DELETE mode keeps the completed index in a single promotable file.
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA cache_size=-200000")
            _create_schema(connection, force=False)

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
            corpus_hasher = CorpusHasher()
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
                    corpus_hasher.update(
                        context_id,
                        chunk_no,
                        name,
                        link,
                        text,
                    )
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

            if chunk_count == 0:
                raise ValueError("Corpus không tạo được chunk nào để xây index")

            metadata = {
                "schema_version": SCHEMA_VERSION,
                "corpus_hash_version": CORPUS_HASH_VERSION,
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
            connection.commit()
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity is None or str(integrity[0]).lower() != "ok":
                raise RuntimeError(f"SQLite index không toàn vẹn: {integrity}")
        finally:
            connection.close()

        temporary_sidecars = [
            path for path in _sqlite_sidecar_paths(temporary_path) if path.exists()
        ]
        if temporary_sidecars:
            raise RuntimeError(
                "SQLite index tạm chưa được hợp nhất thành một file hoàn chỉnh"
            )
        if target_path.exists():
            if not force:
                raise FileExistsError("Index đã tồn tại; thêm --force nếu muốn xây lại")
            _quiesce_database(target_path)
        else:
            orphan_sidecars = [
                path for path in _sqlite_sidecar_paths(target_path) if path.exists()
            ]
            if orphan_sidecars:
                raise RuntimeError(
                    "Không thể tạo index cạnh SQLite sidecar mồ côi; "
                    "hãy khôi phục hoặc dọn các file -wal/-shm/-journal trước"
                )
        temporary_path.replace(target_path)
    finally:
        _cleanup_temporary_database(temporary_path)

    elapsed = time.time() - started
    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "empty_documents": empty_docs,
        "train_samples": len(train_rows),
        "elapsed_seconds": round(elapsed, 2),
    }


def _fts_query(terms: list[str]) -> str:
    # Mỗi token được quote nên các ký tự đặc biệt không thể biến thành cú pháp FTS.
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _fts_vocab_form(term: str) -> str:
    # Tương đương remove_diacritics của unicode61; ký tự đ/Đ không phải combining
    # mark nên được giữ nguyên giống SQLite.
    decomposed = unicodedata.normalize("NFD", term.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên lớn hơn 0")


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
        _validate_top_k(top_k)
        expanded_question = expand_retrieval_query(question)
        query = _fts_query(
            self._rarest_terms(expanded_question, "contexts_vocab", max_terms=8)
        )
        if not query:
            return []
        broad_rows = self.connection.execute(
            """
            SELECT context_id, chunk_no, name, link, text,
                   bm25(contexts_fts, 0.0, 0.0, 2.0, 0.0, 1.0) AS bm25_score
            FROM contexts_fts
            WHERE contexts_fts MATCH ?
            ORDER BY bm25_score, context_id, CAST(chunk_no AS INTEGER), rowid
            LIMIT ?
            """,
            (query, int(top_k)),
        ).fetchall()

        priority_phrases = retrieval_priority_phrases(question)
        if not priority_phrases:
            return [dict(row) for row in broad_rows]

        phrase_query = _fts_query(priority_phrases)
        phrase_rows = self.connection.execute(
            """
            SELECT context_id, chunk_no, name, link, text,
                   bm25(contexts_fts, 0.0, 0.0, 2.0, 0.0, 1.0) AS bm25_score
            FROM contexts_fts
            WHERE contexts_fts MATCH ?
            ORDER BY bm25_score, context_id, CAST(chunk_no AS INTEGER), rowid
            LIMIT ?
            """,
            (phrase_query, int(top_k)),
        ).fetchall()

        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in [*phrase_rows, *broad_rows]:
            item = dict(row)
            key = (str(item["context_id"]), int(item["chunk_no"]))
            haystack = " ".join(
                tokenize(f"{item.get('name') or ''} {item.get('text') or ''}")
            )
            exact_matches = sum(phrase in haystack for phrase in priority_phrases)
            if key in by_key:
                # Keep the broad-query BM25 score when available so scores in
                # the lexical list remain comparable after exact-phrase merge.
                previous_matches = int(by_key[key].get("exact_phrase_matches") or 0)
                item["exact_phrase_matches"] = max(previous_matches, exact_matches)
            else:
                item["exact_phrase_matches"] = exact_matches
            by_key[key] = item

        ranked = sorted(
            by_key.values(),
            key=lambda item: (
                -int(item.get("exact_phrase_matches") or 0),
                float(item.get("bm25_score") or 0.0),
                str(item.get("context_id") or ""),
                int(item.get("chunk_no") or 0),
            ),
        )
        return ranked[:top_k]

    def search_train(
        self,
        question: str,
        top_k: int = 5,
        exclude_id: str | None = None,
    ) -> list[dict[str, Any]]:
        _validate_top_k(top_k)
        query = _fts_query(self._rarest_terms(question, "train_vocab", max_terms=12))
        if not query:
            return []
        rows = self.connection.execute(
            """
            SELECT sample_id, question, answer,
                   bm25(train_fts, 0.0, 1.0, 0.0) AS bm25_score
            FROM train_fts
            WHERE train_fts MATCH ?
            ORDER BY bm25_score, sample_id, rowid
            LIMIT ?
            """,
            (query, int(top_k + 10)),
        ).fetchall()
        output = [dict(row) for row in rows if str(row["sample_id"]) != str(exclude_id)]
        return output[:top_k]

    def get_context_chunks(
        self,
        context_id: str,
        chunk_nos: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Lấy các chunk của một context_id, tùy chọn lọc theo danh sách chunk_no."""
        cid = str(context_id or "").strip()
        if not cid:
            return []
        if chunk_nos is None:
            rows = self.connection.execute(
                """
                SELECT context_id, chunk_no, name, link, text
                FROM contexts_fts
                WHERE context_id = ?
                ORDER BY CAST(chunk_no AS INTEGER), rowid
                """,
                (cid,),
            ).fetchall()
        else:
            valid_nos = [int(no) for no in chunk_nos if isinstance(no, (int, str)) and str(no).isdigit()]
            if not valid_nos:
                return []
            placeholders = ",".join("?" for _ in valid_nos)
            rows = self.connection.execute(
                f"""
                SELECT context_id, chunk_no, name, link, text
                FROM contexts_fts
                WHERE context_id = ? AND CAST(chunk_no AS INTEGER) IN ({placeholders})
                ORDER BY CAST(chunk_no AS INTEGER), rowid
                """,
                [cid, *valid_nos],
            ).fetchall()
        return [dict(row) for row in rows]
