import copy
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legalqa.io import config, file_hash
from legalqa.retrieval import Retriever


class PhrasePreciseTests(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(':memory:')
        self.con.execute("CREATE VIRTUAL TABLE search USING fts5(header,text,tokenize='unicode61 remove_diacritics 0')")
        texts = [
            ('thuế thu nhập cá nhân', 'quy định thuế thu nhập cá nhân và thuế nhập khẩu'),
            ('hải quan', 'thuế nhập khẩu ưu đãi thuế quan'),
            ('thuế thu nhập cá nhân', 'quy định thuế thu nhập cá nhân và thuế nhập khẩu'),
            ('thuế thu nhập', 'cá nhân được giảm thuế'),  # A phrase cannot cross columns.
            ('hợp đồng', 'quy định hợp đồng lao động và bảo hiểm xã hội'),
            ('khác', 'nội dung không liên quan'),
        ]
        rng = random.Random(2026)
        self.vocabulary = 'thuế nhập khẩu hải quan bảo hiểm người lao động hợp đồng kinh doanh cá nhân'.split()
        texts += [('', ' '.join(rng.choices(self.vocabulary, k=rng.randrange(8,70)))) for _ in range(120)]
        self.con.executemany('INSERT INTO search(rowid,header,text) VALUES(?,?,?)',
                             [(i*2, header, text) for i,(header,text) in enumerate(texts)])
        self.fast = Retriever.__new__(Retriever)
        self.fast.con = self.con
        self.fast.c = copy.deepcopy(config())
        self.fast.c['retrieval'].update(fast_phrase_precise=True, precise_bm25_k=3)
        self.old = Retriever.__new__(Retriever)
        self.old.con = self.con
        self.old.c = copy.deepcopy(self.fast.c)
        self.old.c['retrieval']['fast_phrase_precise'] = False

    def tearDown(self):
        self.con.close()

    def test_rankings_match_sql_with_overlap_ties_empty_and_random_queries(self):
        questions = ['', 'và là', 'thuế thu nhập cá nhân thuế thu nhập cá nhân',
                     'quy định hợp đồng lao động bảo hiểm xã hội',
                     'hải quan thuế nhập khẩu ưu đãi thuế quan', 'zzmissing không tồn tại']
        rng = random.Random(97)
        questions += [' '.join(rng.choices(self.vocabulary,k=rng.randrange(3,20))) for _ in range(40)]
        for q in questions:
            self.fast.bm25(q)
            for method in ('bm25_phrases','bm25_precise'):
                with self.subTest(question=q,method=method):
                    expected = getattr(self.old,method)(q)
                    self.assertEqual(getattr(self.fast,method)(q),expected)
                    self.assertEqual(getattr(self.fast,method)(q),expected)

    def test_warm_precise_and_phrases_issue_no_sql_or_evict_word_cache(self):
        q = 'quy định thuế thu nhập cá nhân và thuế nhập khẩu'
        self.fast.bm25(q)
        before = list(self.fast._bm25_terms.items())
        self.fast.bm25_phrases(q)
        statements = []
        self.con.set_trace_callback(statements.append)
        self.fast.bm25_phrases(q)
        self.fast.bm25_precise(q)
        self.assertEqual(statements,[])
        self.assertEqual(list(self.fast._bm25_terms),[key for key,_ in before])
        for key, value in before:
            self.assertIs(self.fast._bm25_terms[key],value)

    def test_precise_missing_or_unordered_postings_fall_back_to_sql(self):
        q = 'quy định thuế thu nhập cá nhân'
        expected = self.old.bm25_precise(q)
        with patch.object(self.fast,'_fts',wraps=self.fast._fts) as sql:
            self.assertEqual(self.fast.bm25_precise(q),expected)
            self.assertTrue(sql.called)
        self.fast.bm25(q)
        self.fast._bm25_terms['thuế'] = self.fast._bm25_terms['thuế'][::-1]
        with patch.object(self.fast,'_fts',wraps=self.fast._fts) as sql:
            self.assertEqual(self.fast.bm25_precise(q),expected)
            self.assertTrue(sql.called)

    def test_precise_keeps_strict_then_relaxed_order(self):
        terms = ['longestword','secondword','thirdword','fourthxx','fifthxx','sixthxx',
                 'seventh','eighth','ninth','tenth','last','end']
        for i,n in enumerate((5,8,12)):
            self.con.execute('INSERT INTO search(rowid,header,text) VALUES(?,?,?)',
                             (1000+i,'',' '.join(terms[:n])))
        q = ' '.join(terms)
        self.fast.bm25(q)
        result = self.fast.bm25_precise(q)
        self.assertEqual(result,self.old.bm25_precise(q))
        self.assertEqual(result[0],1002)
        self.assertIn(1001,result)
        self.assertIn(1000,result)

    def test_phrase_cache_budget_and_disabled_cache_preserve_results(self):
        self.fast.c['retrieval']['phrase_cache_mb'] = 64/(1024*1024)
        for phrase in ['thuế thu nhập','thu nhập cá','nhập cá nhân','không tồn tại','thuế thu nhập']:
            self.fast._phrase_scores(phrase)
            self.assertLessEqual(self.fast._phrase_bytes,64)
            self.assertEqual(self.fast._phrase_bytes,sum(p.nbytes for p in self.fast._phrase_cache.values()))
        fresh = Retriever.__new__(Retriever);fresh.con=self.con
        fresh.c=copy.deepcopy(self.fast.c);fresh.c['retrieval']['phrase_cache_mb']=0
        q = 'thuế thu nhập cá nhân'
        self.assertEqual(fresh.bm25_phrases(q),self.old.bm25_phrases(q))
        self.assertEqual(len(fresh._phrase_cache),0)

    def test_zero_limit_returns_no_candidates(self):
        self.fast.c['retrieval']['precise_bm25_k'] = 0
        self.assertEqual(self.fast.bm25_phrases('thuế thu nhập cá nhân'),[])
        self.assertEqual(self.fast.bm25_precise('thuế thu nhập cá nhân'),[])

    def test_parallel_file_queries_match_sql_without_modifying_index(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder)/'corpus.sqlite'
            target = sqlite3.connect(database)
            self.con.commit()
            self.con.backup(target)
            target.close()
            original_hash = file_hash(database)
            con = sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
            engine = Retriever.__new__(Retriever);engine.con=con
            engine.c=copy.deepcopy(self.fast.c)
            engine.c['retrieval'].update(phrase_workers=2,phrase_cache_mb=0.0001)
            try:
                for q in ['quy định thuế thu nhập cá nhân thuế nhập khẩu',
                          'hợp đồng lao động bảo hiểm xã hội',
                          'quy định thuế thu nhập cá nhân thuế nhập khẩu']:
                    self.assertEqual(engine.bm25_phrases(q),self.old.bm25_phrases(q))
                self.assertLessEqual(engine._phrase_bytes,104)
                with patch.object(engine,'_read_phrase_scores',side_effect=sqlite3.OperationalError('test worker error')):
                    with self.assertRaisesRegex(sqlite3.OperationalError,'test worker error'):
                        engine._phrase_contributions(['thuế thu nhập','missing phrase a','missing phrase b'])
            finally:
                engine.close()
            self.assertEqual(file_hash(database),original_hash)
