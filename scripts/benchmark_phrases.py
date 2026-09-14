"""Compare phrase reader I/O against the previous implementation, including startup."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legalqa.io import config, load_questions, write_json
from legalqa.retrieval import Retriever


class ObservedRetriever(Retriever):
    def _phrase_contributions(self, phrases):
        result = super()._phrase_contributions(phrases)
        self.last_contributions = result
        return result


def benchmark(database, questions, settings):
    import numpy as np
    engines = []
    records = []
    try:
        for reuse in (False, True):
            engine = ObservedRetriever.__new__(ObservedRetriever)
            engine.con = sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)
            engine.c = copy.deepcopy(settings)
            engine.c['retrieval'].update(fast_phrase_precise=True,phrase_reuse_readers=reuse)
            engines.append(engine)
        for position,(key,item) in enumerate(questions.items()):
            row = {'id':key}
            results = {}
            # Alternate execution order to reduce the OS file-cache advantage.
            for index in ((0,1) if position % 2 == 0 else (1,0)):
                engine = engines[index]
                engine.last_contributions = []
                start = time.perf_counter()
                results[index] = engine.bm25_phrases(item['question'])
                row['before_seconds' if index == 0 else 'after_seconds'] = time.perf_counter()-start
            row['identical_ranking'] = results[0] == results[1]
            left, right = [engine.last_contributions for engine in engines]
            row['identical_scores'] = len(left) == len(right) and all(
                np.array_equal(a,b) for a,b in zip(left,right))
            records.append(row)
            print(f'Phrase benchmark {position+1}/{len(questions)}: {row}',flush=True)
        before = sum(r['before_seconds'] for r in records)
        after = sum(r['after_seconds'] for r in records)
        readers = getattr(engines[1],'_phrase_readers',None)
        return {'questions':len(records),'before_seconds':before,'after_seconds':after,
                'speedup':before/max(after,1e-9),
                'before_median_seconds':statistics.median(r['before_seconds'] for r in records),
                'after_median_seconds':statistics.median(r['after_seconds'] for r in records),
                'identical_rankings':all(r['identical_ranking'] for r in records),
                'identical_full_scores':all(r['identical_scores'] for r in records),
                'mmap_bytes':readers.mmap_bytes if readers else [],'records':records}
    finally:
        for engine in engines:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True)
    parser.add_argument('--questions',required=True)
    parser.add_argument('--config')
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('--limit must be positive')
    qa = dict(list(load_questions(args.questions).items())[:args.limit])
    report = benchmark(args.database,qa,config(args.config))
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k != 'records'},indent=2))
    if not report['identical_rankings'] or not report['identical_full_scores']:
        raise SystemExit('Phrase output changed; do not use this optimization for the full run')


if __name__ == '__main__':
    main()
