"""Compare original SQL lexical retrieval with the bounded term cache, without GPUs."""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legalqa.io import config, load_questions, write_json
from legalqa.retrieval import Retriever


def benchmark(database, questions, settings):
    engines = []
    for cache_mb in (0, settings['retrieval']['bm25_cache_mb']):
        engine = Retriever.__new__(Retriever)
        engine.con = sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True)
        engine.c = {**settings, 'retrieval':{**settings['retrieval'], 'bm25_cache_mb':cache_mb}}
        engines.append(engine)
    rows = []
    try:
        for key, item in questions.items():
            row = {'id':key}; results = []
            for name, engine in zip(('original','cached'),engines):
                start = time.perf_counter()
                rankings, timings = engine.lexical_rankings(item['question'])
                row[name] = {**timings, 'total':time.perf_counter()-start}
                results.append(rankings)
            row['identical_rankings'] = results[0] == results[1]
            rows.append(row)
            print(f"{len(rows)}/{len(questions)}: {row}", flush=True)
        totals = {name:sum(row[name]['total'] for row in rows) for name in ('original','cached')}
        return {'questions':len(rows),'seconds':totals,
                'speedup':totals['original']/max(totals['cached'],1e-9),
                'identical_rankings':all(row['identical_rankings'] for row in rows),
                'cached_median_seconds':statistics.median(row['cached']['total'] for row in rows),
                'cache_bytes':getattr(engines[1],'_bm25_bytes',0),'records':rows}
    finally:
        for engine in engines:
            engine.con.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True)
    parser.add_argument('--questions',required=True)
    parser.add_argument('--config')
    parser.add_argument('--limit',type=int,default=30)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.limit <= 0:
        parser.error('--limit must be positive')
    questions=dict(list(load_questions(args.questions).items())[:args.limit])
    report=benchmark(args.database,questions,config(args.config))
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    if not report['identical_rankings']:
        raise SystemExit('Ranking mismatch: inspect benchmark records before running the full stage')


if __name__=='__main__':
    main()
