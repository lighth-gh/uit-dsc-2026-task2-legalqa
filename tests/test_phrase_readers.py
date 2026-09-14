import copy
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from legalqa.io import config, file_hash
from legalqa.phrase_sqlite import PhraseReaders
from legalqa.retrieval import Retriever


class PhraseReadersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'corpus.sqlite'
        con = sqlite3.connect(self.path)
        con.execute("CREATE VIRTUAL TABLE search USING fts5(header,text,tokenize='unicode61 remove_diacritics 0')")
        con.executemany('INSERT INTO search VALUES(?,?)',[
            ('thuế thu nhập cá nhân','thuế thu nhập cá nhân và thuế nhập khẩu'),
            ('hải quan','thuế nhập khẩu ưu đãi thuế quan'),
            ('thuế thu nhập cá nhân','thuế thu nhập cá nhân và thuế nhập khẩu'),
            ('hợp đồng','hợp đồng lao động bảo hiểm xã hội'),
        ])
        con.commit();con.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_connections_are_reused_readonly_and_closed(self):
        original_hash = file_hash(self.path)
        pool = PhraseReaders(self.path,2,64)
        connections = list(pool.connections)
        try:
            batches = [['thuế thu nhập'],['hợp đồng lao động']]
            first = pool.read(batches,Retriever._read_phrase_scores)
            second = pool.read(batches,Retriever._read_phrase_scores)
            for a,b in zip(first,second):
                self.assertEqual(a[0][1].tobytes(),b[0][1].tobytes())
            self.assertEqual(pool.connections,connections)
            for con in connections:
                with self.assertRaisesRegex(sqlite3.OperationalError,'readonly'):
                    con.execute("INSERT INTO search VALUES('x','y')")
        finally:
            pool.close()
        pool.close()  # Closing an already-drained reader is safe.
        for con in connections:
            with self.assertRaises(sqlite3.ProgrammingError):con.execute('SELECT 1')
        self.assertEqual(file_hash(self.path),original_hash)

    def test_failed_batch_drains_other_worker_before_reuse(self):
        pool = PhraseReaders(self.path,2,0)
        finished = threading.Event()
        def score(con,phrase):
            if phrase == 'fail':
                raise sqlite3.OperationalError('worker failure')
            finished.set()
            return Retriever._read_phrase_scores(con,phrase)
        try:
            with self.assertRaisesRegex(sqlite3.OperationalError,'worker failure'):
                pool.read([['fail'],['thuế thu nhập']],score)
            self.assertTrue(finished.is_set())
            result = pool.read([['thuế thu nhập'],['hợp đồng lao động']],score)
            self.assertTrue(len(result[0][0][1]) > 0)
            self.assertEqual(pool.mmap_bytes,[0,0])
        finally:
            pool.close()

    def test_failed_initialization_closes_connections_already_opened(self):
        first = sqlite3.connect(self.path,check_same_thread=False)
        with patch('legalqa.phrase_sqlite.sqlite3.connect',side_effect=[first,sqlite3.OperationalError('open failure')]):
            with self.assertRaisesRegex(sqlite3.OperationalError,'open failure'):
                PhraseReaders(self.path,2,0)
        with self.assertRaises(sqlite3.ProgrammingError):first.execute('SELECT 1')

    def test_retriever_keeps_exact_scores_with_cache_disabled_and_mmap_disabled(self):
        engines = []
        try:
            for reuse in (False,True):
                engine = Retriever.__new__(Retriever)
                engine.con = sqlite3.connect(self.path)
                engine.c = copy.deepcopy(config())
                engine.c['retrieval'].update(phrase_reuse_readers=reuse,phrase_mmap_mb=0,phrase_cache_mb=0)
                engines.append(engine)
            for q in ['quy định thuế thu nhập cá nhân thuế nhập khẩu','hợp đồng lao động bảo hiểm xã hội']*2:
                self.assertEqual(engines[0].bm25_phrases(q),engines[1].bm25_phrases(q))
            self.assertIsNotNone(engines[1]._phrase_readers)
        finally:
            for engine in engines:engine.close()
