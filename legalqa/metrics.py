import ast
import contextlib
import io
import re
import sys
from collections import Counter
from pathlib import Path

from .io import ROOT, digest, file_hash, load_questions, read_json, validate_predictions, write_json


DEFAULT_OBJECTIVE = {"primary_metric":"meteor", "secondary_metric":"rougeL", "target_meteor":0.65}


def evaluation_objective(settings=None):
    objective = {**DEFAULT_OBJECTIVE, **(settings or {})}
    if objective["primary_metric"] != "meteor" or objective["secondary_metric"] != "rougeL":
        raise ValueError("This pipeline requires METEOR as primary and ROUGE-L only as secondary")
    target = float(objective["target_meteor"])
    if not 0 <= target <= 1:
        raise ValueError("target_meteor must be between 0 and 1")
    objective["target_meteor"] = target
    return objective


def metric_environment():
    import numpy as np
    import nltk
    from nltk.translate.meteor_score import meteor_score
    vendor = ROOT/"vendor"
    sys.path.insert(0,str(vendor))
    from rouge_score import rouge_scorer
    if vendor not in Path(rouge_scorer.__file__).resolve().parents:
        raise RuntimeError("An unrelated rouge_score was imported first; run evaluation in a fresh process")
    try:
        nltk.corpus.wordnet.ensure_loaded()
    except LookupError as error:
        raise RuntimeError("Missing official METEOR WordNet resource. Run: python -m nltk.downloader wordnet omw-1.4") from error
    # Compile only the unmodified BTC eval_qa function. Do not execute its download calls or /app main().
    tree = ast.parse((vendor/"scoring.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="eval_qa")
    env = {"np":np,"meteor_score":meteor_score,"rouge_scorer":rouge_scorer}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(vendor/"scoring.py"),"exec"),env)
    identity = {"nltk":nltk.__version__, "numpy":np.__version__,
                "scoring":file_hash(vendor/"scoring.py"),
                "rouge_files":{p.name:file_hash(p) for p in sorted((vendor/"rouge_score").glob("*.py"))}}
    return env,identity


def references(path):
    raw = read_json(path)
    if not isinstance(raw,dict) or not raw:
        raise ValueError("Reference must be a nonempty JSON object")
    result = {}
    for key,value in raw.items():
        answer = value.get("answer") if isinstance(value,dict) else value
        if not isinstance(answer,str) or not answer.strip():
            raise ValueError(f"Invalid reference answer {key}")
        result[key] = answer
    return result


def evaluate(prediction_path, reference_path, output, label=None, objective=None):
    import numpy as np
    pred, truth = read_json(prediction_path), references(reference_path)
    validate_predictions(pred,truth)
    env,identity = metric_environment()
    with contextlib.redirect_stdout(io.StringIO()):
        official = env["eval_qa"](pred,truth)
    scorer = env["rouge_scorer"].RougeScorer(["rougeL"],use_stemmer=False)
    rows = {}
    for key in pred:
        answer,gold = pred[key]["answer"],truth[key]
        rows[key] = {
            "meteor":env["meteor_score"]([gold.split()],answer.split()),
            "rougeL":scorer.score(gold,answer)["rougeL"].fmeasure,
            "prediction_words":len(answer.split()), "reference_words":len(gold.split())}
    if abs(np.mean([r["meteor"] for r in rows.values()])-official["meteor"])>1e-12:
        raise AssertionError("Per-item METEOR differs from the original BTC function")
    if abs(np.mean([r["rougeL"] for r in rows.values()])-official["rouge"])>1e-12:
        raise AssertionError("Per-item ROUGE differs from the original BTC function")
    lengths = np.asarray([r["prediction_words"] for r in rows.values()])
    objective = evaluation_objective(objective)
    target = objective["target_meteor"]
    objective_report = {**objective, "target_met":float(official["meteor"]) >= target,
                        "meteor_gap":max(0.0,target-float(official["meteor"]))}
    report = {"label":label or Path(prediction_path).stem,"samples":len(rows),
              "meteor":float(official["meteor"]),"rougeL":float(official["rouge"]),
              "objective":objective_report,
              "prediction_path":str(Path(prediction_path).resolve()), "prediction_hash":digest(pred),
              "reference_hash":digest(truth),"metric_identity":identity,"per_question":rows,
              "lengths":{"mean":float(lengths.mean()),"p90":float(np.quantile(lengths,.9)),
                         "mean_reference":float(np.mean([r["reference_words"] for r in rows.values()]))}}
    manifest_path = Path(prediction_path).with_suffix(".manifest.json")
    if manifest_path.exists():
        report["prediction_manifest"] = read_json(manifest_path)
    audit_path = Path(prediction_path).with_suffix(".audit.json")
    if audit_path.exists():
        audit = read_json(audit_path)
        report["routes"] = dict(Counter(row["route"] for row in audit.values()))
        report["token_limit_rate"] = sum(row["hit_token_limit"] for row in audit.values())/len(audit)
    write_json(output,report)
    return {k:v for k,v in report.items() if k not in {"per_question","metric_identity","prediction_manifest"}}


def select_reports(paths, output, objective=None):
    reports = [read_json(p) for p in paths]
    if not reports:
        raise ValueError("Need at least one evaluation report")
    for report in reports[1:]:
        for key in ("reference_hash","metric_identity"):
            if report[key] != reports[0][key]:
                raise ValueError(f"Cannot compare runs with different {key}")
        if set(report["per_question"]) != set(reports[0]["per_question"]):
            raise ValueError("Cannot compare different validation IDs")
    objective = evaluation_objective(objective)
    primary,secondary = objective["primary_metric"],objective["secondary_metric"]
    best = max(reports,key=lambda r:(r[primary],r[secondary]))
    target = objective["target_meteor"]
    result = {"selection_rule":"highest METEOR; ROUGE-L only for a METEOR tie",
              "objective":{**objective,"target_met":best["meteor"] >= target,
                           "meteor_gap":max(0.0,target-best["meteor"])},
              "label":best["label"],"meteor":best["meteor"],"rougeL":best["rougeL"],
              "reference_hash":best["reference_hash"],"prediction_manifest":best.get("prediction_manifest"),
              "candidates":[{"label":r["label"],"meteor":r["meteor"],"rougeL":r["rougeL"]} for r in reports]}
    write_json(output,result)
    return result


def compare_reports(baseline_path, candidate_path, output, seed=2026, draws=2000):
    import numpy as np
    base,candidate = read_json(baseline_path),read_json(candidate_path)
    for key in ("reference_hash","metric_identity"):
        if base[key] != candidate[key]:
            raise ValueError("Paired comparison requires identical references and metric implementation")
    ids = sorted(base["per_question"])
    if ids != sorted(candidate["per_question"]):
        raise ValueError("Paired comparison requires the same IDs")
    rng = np.random.default_rng(seed)
    result = {"samples":len(ids),"baseline":base["label"],"candidate":candidate["label"],
              "note":"Paired question bootstrap; near-duplicate dependencies can make this interval optimistic."}
    for metric in ("meteor","rougeL"):
        delta = np.array([candidate["per_question"][k][metric]-base["per_question"][k][metric] for k in ids])
        boot = [float(delta[rng.integers(0,len(ids),len(ids))].mean()) for _ in range(draws)]
        result[metric] = {"mean_delta":float(delta.mean()),"95_percent_interval":np.quantile(boot,[.025,.975]).tolist(),
                          "wins":int((delta>0).sum()),"losses":int((delta<0).sum())}
    write_json(output,result)
    return result


def diagnose_retrieval(qa_path, retrieval_path, index_dir, output):
    from .retrieval import connect
    qa = load_questions(qa_path,answers=True)
    cache = read_json(retrieval_path)["records"]
    con = connect(Path(index_dir)/"corpus.sqlite")
    values = Counter()
    n = 0
    for key,record in qa.items():
        if key not in cache or cache[key]["question"] != record["question"]:
            raise ValueError("Diagnostic query/retrieval mismatch")
        gold = Counter(re.findall(r"[^\W_]+",record["answer"].casefold()))
        for stage,ids in cache[key]["stages"].items():
            for k in (1,5,20):
                picked = ids[:k]
                if not picked:
                    continue
                rows = con.execute("SELECT header,text FROM chunks WHERE chunk_id IN ("+",".join("?" for _ in picked)+")",picked)
                words = Counter(re.findall(r"[^\W_]+"," ".join(row[0]+" "+row[1] for row in rows).casefold()))
                values[f"{stage}_answer_token_coverage@{k}"] += sum((gold & words).values())/max(1,sum(gold.values()))
        n += 1
    con.close()
    report = {"samples":n,"metric_type":"answer-token coverage diagnostic, NOT gold retrieval recall",
              "warning":"Lexical overlap is not evidence that a document supports the answer; never use this as gold Recall@k.",
              "values":{k:v/n for k,v in values.items()}}
    write_json(output,report)
    return report
