import json
import os
import re
import sqlite3
import time
from collections import Counter, OrderedDict
from pathlib import Path

from .data import DOC_NUMBER, iter_documents, legal_parents, token_children
from .io import Journal, digest, file_hash, load_questions, read_json, source_hash, write_json
from .models import Encoder, Reranker, model_lock, release
from .runtime import should_pause


QUESTION_STOPWORDS = frozenset({
    "ai", "bao", "bằng", "các", "cho", "có", "của", "được", "gì", "hay", "khi",
    "là", "một", "nào", "như", "những", "phải", "quy", "sao", "sẽ", "theo", "thế",
    "thì", "tại", "trong", "và", "về", "với", "đối", "định",
})
RECENCY_MARKERS = ("mới nhất", "hiện hành", "hiện nay", "còn hiệu lực")
SANCTION_PRINCIPLE_MARKERS = (
    "nhiều lần", "mỗi hành vi", "từng hành vi", "một hành vi", "nhiều hành vi",
    "xử phạt bao nhiêu lần", "xử phạt mấy lần",
)
SCOPE_MARKERS = ("quỹ", "hội đồng quản lý quỹ")


def word_tokens(text):
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


def content_terms(text):
    return list(dict.fromkeys(token for token in word_tokens(text)
                              if len(token) > 1 and token not in QUESTION_STOPWORDS))


def legal_document_numbers(text):
    return {match.group(0).casefold() for match in DOC_NUMBER.finditer(text)
            if any(char.isalpha() for char in match.group(0).rsplit("/", 1)[-1])}


def document_year(text):
    years = [int(value) for value in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text)]
    return max(years) if years else None


def legal_intent_queries(question):
    """Return bounded, rule-based query expansions without using an answer/reference."""
    folded = question.casefold()
    variants = []
    if ("xử phạt" in folded or "vi phạm" in folded) and any(
            marker in folded for marker in SANCTION_PRINCIPLE_MARKERS):
        variants.append(
            "nguyên tắc xử phạt vi phạm hành chính một hành vi bị xử phạt một lần "
            "nhiều hành vi xử phạt từng hành vi"
        )
    return variants


def scope_mismatch_adjustment(question, candidate_text, settings):
    """Penalize a narrow named scope only in the opt-in retrieval ablation."""
    weight = float(settings.get("scope_mismatch_penalty", 0.0))
    if weight <= 0:
        return 0.0
    question, candidate_text = question.casefold(), candidate_text.casefold()
    mismatches = sum(marker not in question and marker in candidate_text
                     for marker in SCOPE_MARKERS)
    return -weight * min(1, mismatches)


def retrieval_adjustment(question, candidate, settings, newest_year=None):
    """Small interpretable boosts complement the neural reranker without replacing it."""
    candidate_text = (candidate.get("header", "")+"\n"+candidate.get("text", "")).casefold()
    terms = content_terms(question)
    candidate_terms = set(word_tokens(candidate_text))
    coverage = sum(term in candidate_terms for term in terms)/max(1, len(terms))
    adjustment = settings.get("lexical_score_weight", 0.0)*coverage

    requested_documents = legal_document_numbers(question)
    if requested_documents and any(number in candidate_text for number in requested_documents):
        adjustment += settings.get("exact_document_bonus", 0.0)

    question_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", question))
    if question_years and any(year in candidate_text for year in question_years):
        adjustment += settings.get("year_match_bonus", 0.0)

    tokens = word_tokens(question)
    phrases = [" ".join(tokens[i:i+3]) for i in range(max(0, len(tokens)-2))
               if sum(token not in QUESTION_STOPWORDS for token in tokens[i:i+3]) >= 2]
    if any(phrase in candidate_text for phrase in phrases):
        adjustment += settings.get("phrase_match_bonus", 0.0)

    if newest_year and any(marker in question.casefold() for marker in RECENCY_MARKERS):
        if document_year(candidate.get("header", "")) == newest_year:
            adjustment += settings.get("recency_bonus", 0.0)
    adjustment += scope_mismatch_adjustment(question, candidate_text, settings)
    return adjustment


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
    def close(self):
        readers = getattr(self, '_phrase_readers', None)
        if readers is not None:
            readers.close()
            self._phrase_readers = None
        self.con.close()

    def __init__(self, c, index_dir, load_dense=True):
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
        self._bm25_size = self.con.execute("SELECT coalesce(max(chunk_id),-1)+1 FROM chunks").fetchone()[0]
        self.index = None
        if load_dense:
            import faiss
            self.index = faiss.read_index(str(self.out/"dense.faiss"))

    def bm25(self, question):
        tokens = list(dict.fromkeys(word_tokens(question)))[:48]
        if not tokens:
            return []
        limit = self.c["retrieval"]["bm25_k"]
        if self.c["retrieval"].get("bm25_cache_mb", 0) <= 0:
            return self._fts(" OR ".join('"'+t+'"' for t in tokens), limit)
        if limit <= 0:
            return []
        import numpy as np
        if not hasattr(self, "_bm25_terms"):
            self._bm25_terms = OrderedDict()
            self._bm25_bytes = 0
        if not hasattr(self, "_bm25_size"):
            self._bm25_size = self.con.execute("SELECT coalesce(max(rowid),-1)+1 FROM search").fetchone()[0]
        scores = np.zeros(self._bm25_size, dtype=np.float64)
        budget = int(self.c["retrieval"]["bm25_cache_mb"]*1024*1024)
        for token in tokens:
            postings = self._bm25_terms.pop(token, None)
            if postings is None:
                # FTS5 OR BM25 is the sum of its term contributions, including
                # global IDF, document length normalization and column weights.
                # Cache these exact float64 contributions, not approximations.
                cursor = self.con.execute(
                    "SELECT rowid,bm25(search,2.5,1.0) FROM search WHERE search MATCH ?",
                    ('"'+token+'"',))
                try:
                    postings = np.fromiter((tuple(row) for row in cursor),
                                           dtype=[("id", "<i8"), ("score", "<f8")])
                finally:
                    cursor.close()
                if postings.nbytes <= budget:
                    while self._bm25_terms and self._bm25_bytes+postings.nbytes > budget:
                        _, expired = self._bm25_terms.popitem(last=False)
                        self._bm25_bytes -= expired.nbytes
                    self._bm25_bytes += postings.nbytes
            if postings.nbytes <= budget:
                self._bm25_terms[token] = postings
            scores[postings["id"]] += postings["score"]
        hits = np.flatnonzero(scores < 0)
        if len(hits) > limit:
            cutoff = np.partition(scores[hits], limit-1)[limit-1]
            hits = hits[scores[hits] <= cutoff]
        return [int(key) for key in hits[np.lexsort((hits, scores[hits]))][:limit]]

    def _fts(self, query, limit):
        if limit <= 0:
            return []
        rows = self.con.execute(
            "SELECT rowid FROM search WHERE search MATCH ? "
            "ORDER BY bm25(search,2.5,1.0),rowid LIMIT ?", (query, limit)).fetchall()
        return [row[0] for row in rows]

    def lexical_rankings(self, question):
        rankings, seconds = [], {}
        for name in ("bm25", "bm25_phrases", "bm25_precise"):
            start = time.perf_counter()
            rankings.append(getattr(self, name)(question))
            seconds[name+"_query"] = time.perf_counter()-start
        return rankings, seconds

    @staticmethod
    def _rank_sparse(ids, scores, limit):
        import numpy as np
        if limit <= 0:
            return []
        if len(ids) > limit:
            keep = scores <= np.partition(scores, limit-1)[limit-1]
            ids, scores = ids[keep], scores[keep]
        return [int(key) for key in ids[np.lexsort((ids, scores))][:limit]]

    def _init_phrase_cache(self):
        if not hasattr(self, "_phrase_cache"):
            self._phrase_cache = OrderedDict()
            self._phrase_bytes = 0
            # Phrase evaluation revisits posting/position pages repeatedly.
            # A per-connection page cache avoids churning SQLite's small
            # default cache; this does not write or rebuild the source index.
            self.con.execute("PRAGMA cache_size=-65536")

    def _phrase_scores(self, phrase):
        """Independent phrase-group scores; never evict the word cache."""
        self._init_phrase_cache()
        cached = self._phrase_cache.pop(phrase, None)
        if cached is not None:
            self._phrase_cache[phrase] = cached
            return cached
        postings = self._read_phrase_scores(self.con, phrase)
        self._remember_phrase(phrase, postings)
        return postings

    @staticmethod
    def _read_phrase_scores(con, phrase):
        import numpy as np
        group = phrase if isinstance(phrase, tuple) else (phrase,)
        query = ' OR '.join('"'+part+'"' for part in group)
        cursor = con.execute("SELECT rowid,bm25(search,2.5,1.0) FROM search WHERE search MATCH ?",
                             (query,))
        try:
            return np.fromiter((tuple(row) for row in cursor),
                               dtype=[("id", "<i8"), ("score", "<f8")])
        finally:
            cursor.close()

    def _remember_phrase(self, phrase, postings):
        budget = int(self.c["retrieval"].get("phrase_cache_mb", 64)*1024*1024)
        if budget > 0 and postings.nbytes <= budget:
            # Bound key overhead too, including phrases with no matches.
            while self._phrase_cache and (self._phrase_bytes+postings.nbytes > budget
                                          or len(self._phrase_cache) >= 4096):
                _, expired = self._phrase_cache.popitem(last=False)
                self._phrase_bytes -= expired.nbytes
            self._phrase_cache[phrase] = postings
            self._phrase_bytes += postings.nbytes

    def _phrase_contributions(self, phrases):
        """Score independent phrases on separate read-only SQLite connections."""
        from concurrent.futures import ThreadPoolExecutor
        self._init_phrase_cache()
        cache = self._phrase_cache
        missing = [phrase for phrase in phrases if phrase not in cache]
        workers = min(2, max(1, int(self.c["retrieval"].get("phrase_workers", 2))))
        database = next((row[2] for row in self.con.execute('PRAGMA database_list') if row[1] == 'main'), '') if missing and workers > 1 else ''
        if not database or len(missing) < 2:
            return [self._phrase_scores(phrase) for phrase in phrases]

        def read_batch(batch):
            # Connections are created, used and closed in their worker thread.
            con = sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True)
            try:
                con.execute('PRAGMA cache_size=-32768')
                return [(phrase, self._read_phrase_scores(con, phrase)) for phrase in batch]
            finally:
                con.close()

        computed = {}
        batches = [missing[i::workers] for i in range(workers)]
        if self.c['retrieval'].get('phrase_reuse_readers', True):
            if getattr(self, '_phrase_readers', None) is None:
                from .phrase_sqlite import PhraseReaders
                self._phrase_readers = PhraseReaders(database, workers,
                    self.c['retrieval'].get('phrase_mmap_mb', 1024))
                print(f'Phrase SQLite: reusable_readers={workers}, '
                      f'mmap_bytes={self._phrase_readers.mmap_bytes}', flush=True)
            for batch in self._phrase_readers.read(batches, self._read_phrase_scores):
                computed.update(batch)
        else:
            # Exact pre-optimization path, retained for A/B verification.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for batch in pool.map(read_batch, batches):
                    computed.update(batch)
        result = []
        for phrase in phrases:
            postings = cache.pop(phrase, None)
            if postings is not None:
                cache[phrase] = postings
            else:
                postings = computed[phrase]
            result.append(postings)
        # Defer evictions until all references needed for this query are held.
        for phrase, postings in computed.items():
            self._remember_phrase(phrase, postings)
        return result

    def _cached_conjunction(self, terms, limit):
        """Intersect all matching IDs, then sum term scores in query order.

        Missing postings fall back to SQL, without populating or reordering the
        broad BM25 cache. Phrase/precise optimizations must not displace it.
        """
        import numpy as np
        cache = getattr(self, "_bm25_terms", {})
        if any(term not in cache for term in terms):
            return None
        postings = [cache[term] for term in terms]
        if any(not len(p) for p in postings):
            return []
        # FTS5 currently enumerates rowids ascending. Fail back safely if an
        # alternate SQLite implementation gives an unordered posting list.
        if any(np.any(p["id"][1:] < p["id"][:-1]) for p in postings):
            return None
        smallest = min(postings, key=len)
        hits = smallest["id"]
        for p in sorted(postings, key=len):
            if p is smallest:
                continue
            positions = np.searchsorted(p["id"], hits)
            hits = hits[positions < len(p)]
            positions = positions[positions < len(p)]
            hits = hits[p["id"][positions] == hits]
            if not len(hits):
                return []
        scores = np.zeros(len(hits), dtype=np.float64)
        for p in postings:
            scores += p["score"][np.searchsorted(p["id"], hits)]
        return self._rank_sparse(hits, scores, limit)

    def bm25_phrases(self, question):
        tokens = word_tokens(question)
        phrases = []
        for size in (4, 3):
            for start in range(max(0, len(tokens)-size+1)):
                part = tokens[start:start+size]
                if sum(token not in QUESTION_STOPWORDS for token in part) >= 2:
                    phrases.append(" ".join(part))
        phrases = list(dict.fromkeys(phrases))[:24]
        if not phrases:
            return []
        limit = self.c["retrieval"].get("precise_bm25_k", 40)
        if self.c["retrieval"].get("fast_phrase_precise", False):
            import numpy as np
            if limit <= 0:
                return []
            workers = min(2, max(1, int(self.c["retrieval"].get("phrase_workers", 2))))
            # Keep adjacent, overlapping phrases together to minimize duplicate
            # row materialization. Two SQL groups run concurrently on file indexes.
            width = (len(phrases)+workers-1)//workers
            groups = [tuple(phrases[i:i+width]) for i in range(0,len(phrases),width)]
            contributions = self._phrase_contributions(groups)
            ids = np.concatenate([p["id"] for p in contributions])
            scores = np.concatenate([p["score"] for p in contributions])
            unique, inverse = np.unique(ids, return_inverse=True)
            totals = np.zeros(len(unique), dtype=np.float64)
            np.add.at(totals, inverse, scores)
            return self._rank_sparse(unique, totals, limit)
        query = " OR ".join('"'+phrase+'"' for phrase in phrases)
        return self._fts(query, limit)

    def bm25_precise(self, question):
        terms = content_terms(question)
        if len(terms) < 3:
            return []
        limit = self.c["retrieval"].get("precise_bm25_k", 40)
        if limit <= 0:
            return []
        result = []
        ordered = sorted(terms, key=lambda value: (-len(value), terms.index(value)))
        # Start strict and relax only when needed. Earlier rows remain ahead after deduplication.
        for count in dict.fromkeys([min(12, len(terms)), min(8, len(terms)), min(5, len(terms))]):
            selected = ordered[:count]
            query = " AND ".join('"'+term+'"' for term in selected)
            try:
                cached = self._cached_conjunction(selected, limit) if self.c["retrieval"].get("fast_phrase_precise", False) else None
                result.extend(self._fts(query, limit) if cached is None else cached)
            except sqlite3.OperationalError:
                continue
            if len(dict.fromkeys(result)) >= limit:
                break
        return list(dict.fromkeys(result))[:limit]

    def chunks(self, ids):
        if not ids:
            return {}
        rows = self.con.execute("SELECT * FROM chunks WHERE chunk_id IN ("+",".join("?" for _ in ids)+")", ids).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def _adjacent_candidate_ids(self, question, seed_keys, chunks, limit):
        """Find chunks from the immediately adjacent articles of top seed parents."""
        distance = int(self.c["retrieval"].get("adjacent_articles", 0))
        if distance <= 0 or limit <= 0:
            return []
        parent_ids = list(dict.fromkeys(chunks[key]["parent_id"] for key in seed_keys))
        if not parent_ids:
            return []
        placeholders = ",".join("?" for _ in parent_ids)
        seed_parents = self.con.execute(
            f"SELECT parent_id,doc_id FROM parents WHERE parent_id IN ({placeholders})", parent_ids
        ).fetchall()
        by_document = {}
        for row in seed_parents:
            by_document.setdefault(row["doc_id"], set()).add(row["parent_id"])
        neighbours = []
        for doc_id, seeds in by_document.items():
            ordered = [row[0] for row in self.con.execute(
                "SELECT parent_id FROM parents WHERE doc_id=? ORDER BY rowid", (doc_id,)
            )]
            positions = {parent_id: i for i, parent_id in enumerate(ordered)}
            for seed in seeds:
                position = positions[seed]
                for offset in range(1, distance + 1):
                    for index in (position - offset, position + offset):
                        if 0 <= index < len(ordered):
                            neighbours.append(ordered[index])
        neighbours = list(dict.fromkeys(neighbours))
        if not neighbours:
            return []
        placeholders = ",".join("?" for _ in neighbours)
        rows = self.con.execute(
            f"SELECT * FROM chunks WHERE parent_id IN ({placeholders}) ORDER BY chunk_id", neighbours
        ).fetchall()
        rc = self.c["retrieval"]
        per_parent = int(rc["max_children_per_parent"])
        candidates = [dict(row) for row in rows]
        years = [document_year(row["header"]) for row in candidates]
        newest_year = max((year for year in years if year), default=None)
        candidates.sort(key=lambda row: (
            -retrieval_adjustment(question, row, rc, newest_year), row["chunk_id"]
        ))
        result, counts = [], Counter()
        for row in candidates:
            if counts[row["parent_id"]] >= per_parent:
                continue
            result.append(row["chunk_id"])
            counts[row["parent_id"]] += 1
            if len(result) == limit:
                break
        return result

    def retrieve_one(self, question, dense_ids, reranker):
        start = time.perf_counter()
        (bm, phrase_bm, precise_bm), timings = self.lexical_rankings(question)
        query_variants = [question]
        extra_rankings = []
        if self.c["retrieval"].get("intent_query_expansion", False):
            for position, variant in enumerate(legal_intent_queries(question), 1):
                query_variants.append(variant)
                rankings, variant_timings = self.lexical_rankings(variant)
                extra_rankings.extend(rankings)
                timings.update({f"intent_{position}_{key}": value
                                for key, value in variant_timings.items()})
        bm_seconds = time.perf_counter()-start
        rc = self.c["retrieval"]
        fusion = rrf([bm, phrase_bm, precise_bm, *extra_rankings, dense_ids], rc["rrf_constant"])
        chunks = self.chunks(fusion)
        base_pool = diversified(fusion, chunks, rc["pool_k"], rc["max_children_per_parent"])
        reserve = int(rc.get("adjacent_reserve", max(1, rc["pool_k"] // 4)))
        reserve = min(reserve, rc["pool_k"])
        adjacent = self._adjacent_candidate_ids(
            question, base_pool[:int(rc.get("adjacent_seed_k", 8))], chunks, reserve
        )
        if adjacent:
            chunks.update(self.chunks(adjacent))
            keep = max(0, rc["pool_k"] - len(adjacent))
            pool = list(dict.fromkeys(base_pool[:keep] + adjacent + base_pool[keep:]))[:rc["pool_k"]]
        else:
            pool = base_pool
        t = time.perf_counter()
        scores = reranker.score(question, [chunks[k]["header"]+"\n"+chunks[k]["text"] for k in pool])
        years = [document_year(chunks[key]["header"]) for key in pool]
        newest_year = max((year for year in years if year), default=None)
        scored = [(key, score, retrieval_adjustment(question, chunks[key], rc, newest_year))
                  for key,score in zip(pool,scores)]
        ranked = sorted(scored, key=lambda row: (-(row[1]+row[2]), row[0]))
        selected = self._materialize(ranked,chunks)
        return {"question": question, "contexts": selected,
                "stages": {"bm25": bm, "bm25_phrases": phrase_bm, "bm25_precise": precise_bm,
                           "dense": dense_ids, "rrf": fusion, "pool": pool,
                           "adjacent": adjacent, "reranked": [row[0] for row in ranked]},
                "query_variants": query_variants,
                "candidate_count": len(pool),
                "seconds": {**timings, "bm25": bm_seconds, "rerank": time.perf_counter()-t}}

    def _materialize(self, ranked, chunks):
        rc = self.c["retrieval"]
        selected,seen = [],set()
        for key, score, adjustment in ranked:
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
                             "seed_chunk_id": key, "rerank_score": score,
                             "retrieval_adjustment": adjustment, "final_score": score+adjustment})
            if len(selected) == rc["parents_k"]:
                break
        return selected

    def retrieve_one_lexical(self, question):
        """Fast question-only retrieval for QLoRA training examples; no GPU models."""
        start = time.perf_counter()
        (bm, phrase_bm, precise_bm), timings = self.lexical_rankings(question)
        rc = self.c["retrieval"]
        fusion = rrf([bm, phrase_bm, precise_bm], rc["rrf_constant"])
        chunks = self.chunks(fusion)
        pool_k = self.c.get("training",{}).get("lexical_pool_k",rc["pool_k"])
        pool = diversified(fusion,chunks,pool_k,rc["max_children_per_parent"])
        years = [document_year(chunks[key]["header"]) for key in pool]
        newest_year = max((year for year in years if year),default=None)
        ranked = []
        for rank,key in enumerate(pool):
            rank_score = 4.0/(1.0+rank/8.0)
            adjustment = retrieval_adjustment(question,chunks[key],rc,newest_year)
            ranked.append((key,rank_score,adjustment))
        ranked.sort(key=lambda row:(-(row[1]+row[2]),row[0]))
        return {"question":question,"contexts":self._materialize(ranked,chunks),
                "stages":{"bm25":bm,"bm25_phrases":phrase_bm,"bm25_precise":precise_bm,
                          "dense":[],"rrf":fusion,"reranked":[row[0] for row in ranked]},
                "seconds":{**timings,"bm25":time.perf_counter()-start,"rerank":0.0,"dense_amortized":0.0}}


def retrieve(c, questions_path, root, index_dir, output, device, mode="full"):
    if mode not in {"full","lexical"}:
        raise ValueError("Retrieval mode must be full or lexical")
    lock = model_lock(c, root)
    questions = load_questions(questions_path)
    engine = Retriever(c, index_dir, load_dense=mode == "full")
    print(f"Retrieval mode={mode}, bm25_k={c['retrieval']['bm25_k']}, "
          f"precise_bm25_k={c['retrieval'].get('precise_bm25_k',40)}, "
          f"bm25_cache_mb={c['retrieval'].get('bm25_cache_mb',0)}, "
          f"fast_phrase_precise={c['retrieval'].get('fast_phrase_precise',False)}, "
          f"phrase_workers={c['retrieval'].get('phrase_workers',2)}, "
          f"phrase_cache_mb={c['retrieval'].get('phrase_cache_mb',64)}", flush=True)
    if engine.manifest["identity"]["embedding"] != lock["models"]["embedding"]:
        raise ValueError("Dense index was built with a different embedding checkpoint")
    identity = {"questions_hash": digest(questions), "index_hash": digest(engine.manifest),
                "retrieval": c["retrieval"], "mode":mode,
                "mode_config":c.get("training",{}).get("lexical_pool_k") if mode=="lexical" else None,
                "models": lock, "code": source_hash()}
    output = Path(output)
    journal = Journal(output.with_suffix(".checkpoint.jsonl"),identity)
    records = journal.records
    keys = [k for k in questions if k not in records]
    try:
        if keys and mode == "full":
            import numpy as np
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
                if should_pause(position):
                    break
                ids = [int(x) for x in neighbours[position] if x >= 0]
                record = engine.retrieve_one(questions[key]["question"], ids, reranker)
                record["seconds"]["dense_amortized"] = dense_seconds
                journal.append(key,record)
                if (position+1) % 10 == 0 or position+1 == len(keys):
                    print(f"Retrieved: {len(records)}/{len(questions)}", flush=True)
            release(reranker)
        elif keys:
            for position,key in enumerate(keys):
                if should_pause(position):
                    break
                record = engine.retrieve_one_lexical(questions[key]["question"])
                journal.append(key,record)
                if (position+1) % 10 == 0 or position == 0 or position+1 == len(keys):
                    print(f"Retrieved lexical: {len(records)}/{len(questions)}; seconds={record['seconds']}",flush=True)
    finally:
        engine.close()
    if set(records) < set(questions):
        return {"status":"paused", "records":len(records), "total":len(questions)}
    if set(records) != set(questions):
        raise ValueError("Retrieval cache ID mismatch")
    payload = {"identity": identity, "records": {k: records[k] for k in questions}}
    write_json(output, payload)
    return {"records": len(records), "output": str(output), "fingerprint": digest(identity)}


def read_retrieval(path, questions, c, root, expected_mode=None, expected_index_hash=None):
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
    identity = payload["identity"]
    if identity.get("code") != source_hash():
        raise ValueError("Retrieval cache uses different pipeline code; rebuild the cache")
    if expected_mode and identity.get("mode") != expected_mode:
        raise ValueError("Retrieval cache mode differs from the requested pipeline stage")
    if identity.get("mode") == "lexical" and identity.get("mode_config") != c["training"].get("lexical_pool_k"):
        raise ValueError("Lexical retrieval cache uses different training settings")
    expected_index_hash = expected_index_hash or os.environ.get("LEGALQA_INDEX_HASH")
    if expected_index_hash and identity.get("index_hash") != expected_index_hash:
        raise ValueError("Retrieval cache uses a different index")
    return {k: records[k] for k in questions}, payload["identity"]
