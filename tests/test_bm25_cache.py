import sqlite3
import unittest

from legalqa.retrieval import Retriever, word_tokens


class BM25CacheTests(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(':memory:')
        self.con.execute("CREATE VIRTUAL TABLE search USING fts5(header,text,tokenize='unicode61 remove_diacritics 0')")
        self.con.executemany('INSERT INTO search(rowid,header,text) VALUES(?,?,?)', [
            (0,'thuế 2026','quy định thuế thuế thu nhập cá nhân'),
            (2,'hải quan','thuế nhập khẩu ưu đãi thuế quan'),
            (4,'thuế 2026','quy định thuế thuế thu nhập cá nhân'),
            (5,'hợp đồng','quy định hợp đồng lao động'),
            (8,'khác','nội dung khác'),
        ])
        self.engine = Retriever.__new__(Retriever)
        self.engine.con = self.con
        self.engine.c = {'retrieval':{'bm25_k':3,'bm25_cache_mb':1}}

    def tearDown(self):
        self.con.close()

    def test_cached_scores_preserve_fts5_ranking_ties_and_sparse_ids(self):
        for question in ['thuế thuế thu nhập năm 2026','hợp đồng lao động','quy định thuế quan','thuế','zzmissing','','2026']:
            tokens=list(dict.fromkeys(word_tokens(question)))[:48]
            query=' OR '.join('"'+t+'"' for t in tokens)
            expected=self.engine._fts(query,3) if query else []
            self.assertEqual(self.engine.bm25(question),expected)
            self.assertEqual(self.engine.bm25(question),expected)

    def test_warm_query_needs_no_sql_and_cache_stays_bounded(self):
        expected=self.engine.bm25('thuế quan')
        statements=[];self.con.set_trace_callback(statements.append)
        self.assertEqual(self.engine.bm25('thuế quan'),expected)
        self.assertEqual(statements,[])
        self.engine.c['retrieval']['bm25_cache_mb']=0.00008
        # A fresh engine starts with an empty cache at the new budget.
        del self.engine._bm25_terms
        for question in ['hợp đồng','thuế quan','nội dung','hợp đồng']:
            self.engine.bm25(question)
            self.assertLessEqual(self.engine._bm25_bytes,83)
            self.assertEqual(self.engine._bm25_bytes,sum(p.nbytes for p in self.engine._bm25_terms.values()))

    def test_disabled_cache_uses_original_query(self):
        self.engine.c['retrieval']['bm25_cache_mb']=0
        self.assertEqual(self.engine.bm25('thuế'),self.engine._fts('"thuế"',3))
