import json
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from .data import iter_documents, legal_parents, token_children
from .io import Journal, digest, file_hash, load_questions, read_json, source_hash, write_json
from .models import Encoder, Reranker, model_lock, release


def rrf(rankings, constant=60):
    scores = Counter()
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1.0/(constant+rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


def diversified(keys, chunks, limit, per_parent):
    result, counts = [], Counter()
    for key in keys:
        parent = chunks[key]["parent_id"]
        if counts[parent] < per_parent:
            result.append(key)
            counts[parent] += 1
            if len(result) == limit:
                break
    return result


def connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def build_index(c, corpus_path, root, output, device):
    import faiss
    import numpy as np
    from transformers import AutoTokenizer
    lock = model_lock(c, root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/"index_manifest.json"
    if manifest_path.exists():
        raise ValueError("Index already completed. Reuse it or build into a NEW directory.")
    identity = {"chunking": c["chunking"], "embedding": lock["models"]["embedding"], "code": source_hash()}
    tokenizer = AutoTokenizer.from_pretrained(Path(root)/"embedding", local_files_only=True, use_fast=True)
    db = out/"corpus.sqlite"
    build_note = out/"corpus_manifest.json"
    if build_note.exists():
        note = read_json(build_note)
        if note["identity"] != identity:
            raise ValueError("Partial index fingerprint differs. Use a new directory.")
        # Verify the supplied source corpus, not merely the directory name.
        source_records = [[doc["source_file"], digest(doc)] for doc in iter_documents(corpus_path)]
        if digest(source_records) != note["corpus_sha256"]:
            raise ValueError("Corpus changed since partial build")
    else:
        temp = out/"corpus.building.sqlite"
        if temp.exists():
            temp.unlink()
        con = connect(temp)
        con.executescript("""
        CREATE TABLE parents(parent_id TEXT PRIMARY KEY, doc_id TEXT, heading TEXT, number TEXT,
                             text TEXT, source_file TEXT, link TEXT);
        CREATE TABLE chunks(chunk_id INTEGER PRIMARY KEY, parent_id TEXT, start INTEGER, end INTEGER,
                            header TEXT, text TEXT);
        CREATE INDEX chunks_parent ON chunks(parent_id);
        CREATE VIRTUAL TABLE search USING fts5(header, text, tokenize='unicode61 remove_diacritics 0');
        """)
        seen_ids, seen_passages, sources = set(), set(), []
        empty, duplicates, docs, child_id = 0, 0, 0, 0
        for doc in iter_documents(corpus_path):
            sources.append([doc["source_file"], digest(doc)])
            if doc["doc_id"] in seen_ids:
                raise ValueError(f"Duplicate corpus document ID {doc['doc_id']}")
            seen_ids.add(doc["doc_id"])
            if not doc["text"]:
                empty += 1
                continue
            content_hash = digest(doc["text"])
            if content_hash in seen_passages:
                duplicates += 1
                continue
            seen_passages.add(content_hash)
            docs += 1
            for parent in legal_parents(doc):
                con.execute("INSERT INTO parents VALUES(:parent_id,:doc_id,:heading,:number,:text,:source_file,:link)", parent)
                for chunk in token_children(parent, tokenizer, c["chunking"]["child_tokens"],
                                             c["chunking"]["overlap_tokens"], c["chunking"]["header_tokens"]):
                    con.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?)", (child_id, chunk["parent_id"], chunk["start"],
                                chunk["end"], chunk["header"], chunk["text"]))
                    con.execute("INSERT INTO search(rowid,header,text) VALUES(?,?,?)", (child_id,chunk["header"],chunk["text"]))
                    child_id += 1
            if docs % 100 == 0:
                con.commit()
                print(f"Corpus: {docs} documents, {child_id} chunks", flush=True)
        if not child_id:
            raise ValueError("Corpus contains no nonempty chunks")
        con.commit()
        con.execute("INSERT INTO search(search) VALUES('optimize')")
        con.commit()
        con.close()
        os.replace(temp, db)
        note = {"identity": identity, "corpus_sha256": digest(sources), "documents": docs,
                "chunks": child_id, "empty_documents": empty, "duplicate_passages": duplicates}
        write_json(build_note, note)
    con = connect(db)
    n = note["chunks"]
    vector_path, progress_path = out/"vectors.npy", out/"embedding_progress.json"
    done = 0
    if progress_path.exists():
        progress = read_json(progress_path)
        if progress["manifest_hash"] != digest(note):
            raise ValueError("Embedding checkpoint fingerprint differs")
        done = progress["completed"]
        vectors = np.load(vector_path, mmap_mode="r+")
        if vectors.shape != (n,384) or not 0 <= done <= n:
            raise ValueError("Invalid partial vectors/checkpoint")
    else:
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype=np.float32, shape=(n,384))
    encoder = Encoder(root, device)
    batch = c["retrieval"]["embedding_batch"]
    for start in range(done, n, batch):
        rows = con.execute("SELECT * FROM chunks WHERE chunk_id>=? ORDER BY chunk_id LIMIT ?", (start,batch)).fetchall()
        texts = [row["header"]+"\n"+row["text"] for row in rows]
        values = encoder.encode(texts, kind="passage", batch_size=batch)
        vectors[start:start+len(rows)] = values
        completed = start+len(rows)
        if completed % (batch*20) == 0 or completed == n:
            vectors.flush()
            write_json(progress_path, {"manifest_hash": digest(note), "completed": completed})
            print(f"Embeddings: {completed}/{n}", flush=True)
    vectors.flush()
    release(encoder)
    faiss_index = faiss.IndexFlatIP(384)
    for start in range(0,n,8192):
        faiss_index.add(np.ascontiguousarray(vectors[start:start+8192]))
    faiss.write_index(faiss_index, str(out/"dense.faiss.tmp"))
    os.replace(out/"dense.faiss.tmp", out/"dense.faiss")
    con.close()
    note["sqlite_sha256"] = file_hash(db)
    note["faiss_sha256"] = file_hash(out/"dense.faiss")
    write_json(manifest_path, note)
    return note


class Retriever:
    def __init__(self, c, index_dir):
        import faiss
        self.c = c
        self.out = Path(index_dir)
        self.manifest = read_json(self.out/"index_manifest.json")
        if self.manifest["identity"]["chunking"] != c["chunking"]:
            raise ValueError("Index chunking does not match current config")
        if file_hash(self.out/"corpus.sqlite") != self.manifest["sqlite_sha256"]:
            raise ValueError("Corpus database hash mismatch")
        if file_hash(self.out/"dense.faiss") != self.manifest["faiss_sha256"]:
            raise ValueError("FAISS index hash mismatch")
        self.con = connect(self.out/"corpus.sqlite")
        self.index = faiss.read_index(str(self.out/"dense.faiss"))

    def bm25(self, question):
        tokens = list(dict.fromkeys(re.findall(r"[^\W_]+", question.casefold(), re.UNICODE)))[:48]
        if not tokens:
            return []
        query = " OR ".join('"'+t+'"' for t in tokens)
        # FTS5 bm25() is lower-is-better; do not reverse it.
        rows = self.con.execute("SELECT rowid FROM search WHERE search MATCH ? ORDER BY bm25(search,2.5,1.0),rowid LIMIT ?",
                                (query, self.c["retrieval"]["bm25_k"])).fetchall()
        return [row[0] for row in rows]

    def chunks(self, ids):
        if not ids:
            return {}
        rows = self.con.execute("SELECT * FROM chunks WHERE chunk_id IN ("+",".join("?" for _ in ids)+")", ids).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def retrieve_one(self, question, dense_ids, reranker):
        start = time.perf_counter()
        bm = self.bm25(question)
        bm_seconds = time.perf_counter()-start
        rc = self.c["retrieval"]
        fusion = rrf([bm, dense_ids], rc["rrf_constant"])
        chunks = self.chunks(fusion)
        pool = diversified(fusion, chunks, rc["pool_k"], rc["max_children_per_parent"])
        t = time.perf_counter()
        scores = reranker.score(question, [chunks[k]["header"]+"\n"+chunks[k]["text"] for k in pool])
        ranked = sorted(zip(pool,scores), key=lambda pair: (-pair[1], pair[0]))
        selected, seen = [], set()
        for key, score in ranked:
            child = chunks[key]
            if child["parent_id"] in seen:
                continue
            seen.add(child["parent_id"])
            parent = dict(self.con.execute("SELECT * FROM parents WHERE parent_id=?", (child["parent_id"],)).fetchone())
            # Cache a bounded verbatim parent window; very large annexes must not inflate every JSON record.
            start_char, end_char = 0, len(parent["text"])
            if end_char > 24000:
                start_char = max(0,min(child["start"]-10000,end_char-24000))
                end_char = start_char+24000
                parent["text"] = parent["text"][start_char:end_char]
            selected.append({**parent, "seed_start": child["start"]-start_char, "seed_end": child["end"]-start_char,
                             "source_window_start": start_char, "source_window_end": end_char,
                             "seed_chunk_id": key, "rerank_score": score})
            if len(selected) == rc["parents_k"]:
                break
        return {"question": question, "contexts": selected,
                "stages": {"bm25": bm, "dense": dense_ids, "rrf": fusion,
                           "reranked": [k for k,_ in ranked]},
                "seconds": {"bm25": bm_seconds, "rerank": time.perf_counter()-t}}


def retrieve(c, questions_path, root, index_dir, output, device):
    import numpy as np
    lock = model_lock(c, root)
    questions = load_questions(questions_path)
    engine = Retriever(c, index_dir)
    if engine.manifest["identity"]["embedding"] != lock["models"]["embedding"]:
        raise ValueError("Dense index was built with a different embedding checkpoint")
    identity = {"questions_hash": digest(questions), "index_hash": digest(engine.manifest),
                "retrieval": c["retrieval"], "models": lock, "code": source_hash()}
    output = Path(output)
    journal = Journal(output.with_suffix(".checkpoint.jsonl"),identity)
    records = journal.records
    keys = [k for k in questions if k not in records]
    if keys:
        encoder = Encoder(root, device)
        start = time.perf_counter()
        query_vectors = encoder.encode([questions[k]["question"] for k in keys], kind="query",
                                       batch_size=c["retrieval"]["embedding_batch"])
        # Batched search amortizes the corpus scan. -1 padding is filtered for small corpora.
        _, neighbours = engine.index.search(np.ascontiguousarray(query_vectors),
                                             min(c["retrieval"]["dense_k"], engine.index.ntotal))
        dense_seconds = (time.perf_counter()-start)/len(keys)
        release(encoder)
        reranker = Reranker(root, device, c["retrieval"]["reranker_batch"], c["retrieval"]["reranker_max_tokens"])
        for position,key in enumerate(keys):
            ids = [int(x) for x in neighbours[position] if x >= 0]
            record = engine.retrieve_one(questions[key]["question"], ids, reranker)
            record["seconds"]["dense_amortized"] = dense_seconds
            journal.append(key,record)
            if (position+1) % 10 == 0 or position+1 == len(keys):
                print(f"Retrieved: {len(records)}/{len(questions)}", flush=True)
        release(reranker)
    engine.con.close()
    if set(records) != set(questions):
        raise ValueError("Retrieval cache ID mismatch")
    payload = {"identity": identity, "records": {k: records[k] for k in questions}}
    write_json(output, payload)
    return {"records": len(records), "output": str(output), "fingerprint": digest(identity)}


def read_retrieval(path, questions, c, root):
    payload = read_json(path)
    records = payload["records"]
    if not set(questions).issubset(records):
        raise ValueError("Missing query IDs in retrieval cache")
    for key,item in questions.items():
        if records[key]["question"] != item["question"]:
            raise ValueError(f"Question text differs in retrieval cache: {key}")
    if payload["identity"]["models"] != model_lock(c, root):
        raise ValueError("Retrieval cache uses different model revisions")
    if payload["identity"]["retrieval"] != c["retrieval"]:
        raise ValueError("Retrieval cache uses different retrieval settings")
    return {k: records[k] for k in questions}, payload["identity"]
