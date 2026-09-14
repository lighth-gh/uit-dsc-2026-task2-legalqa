"""Reusable read-only SQLite workers for phrase queries; ranking is unchanged."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class PhraseReaders:
    def __init__(self, database, workers, mmap_mb):
        self.connections = []
        self.executor = None
        self.mmap_bytes = []
        try:
            for _ in range(workers):
                con = sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True,
                                      check_same_thread=False)
                self.connections.append(con)
                con.execute('PRAGMA cache_size=-32768')
                row = con.execute(f'PRAGMA mmap_size={max(0,int(mmap_mb))*1024*1024}').fetchone()
                self.mmap_bytes.append(row[0] if row else 0)
            self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='legalqa-phrase')
        except BaseException:
            self.close()
            raise

    def read(self, batches, score):
        # Retriever submits synchronously: exactly one task per connection,
        # and drains every task before any connection is used again.
        def batch_job(item):
            slot, batch = item
            return [(phrase, score(self.connections[slot], phrase)) for phrase in batch]
        futures = [self.executor.submit(batch_job, item) for item in enumerate(batches)]
        try:
            return [future.result() for future in futures]
        finally:
            # Even when one worker fails, the other must finish before reuse.
            from concurrent.futures import wait
            wait(futures)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        for con in self.connections:
            con.close()
        self.connections.clear()
