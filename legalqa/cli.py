import argparse
import json
import os
from pathlib import Path

from .io import config


def main():
    parser = argparse.ArgumentParser(description="UIT DSC 2026 LegalQA <4B pipeline")
    parser.add_argument("--config",default=None)
    parser.add_argument("--models",default="models")
    parser.add_argument("--device",default="cuda:0")
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("fetch-models")
    sub.add_parser("audit-models")
    p = sub.add_parser("prepare")
    p.add_argument("--train",required=True); p.add_argument("--test",required=True); p.add_argument("--output",required=True)
    p = sub.add_parser("build-index")
    p.add_argument("--corpus",required=True); p.add_argument("--output",required=True)
    p = sub.add_parser("retrieve")
    p.add_argument("--questions",required=True); p.add_argument("--index",required=True); p.add_argument("--output",required=True)
    p = sub.add_parser("generate")
    p.add_argument("--questions",required=True); p.add_argument("--retrieval",required=True); p.add_argument("--output",required=True)
    p.add_argument("--adapter"); p.add_argument("--mode",choices=["generate","extractive"],default="generate")
    p = sub.add_parser("fit")
    p.add_argument("--train",required=True); p.add_argument("--retrieval",required=True); p.add_argument("--output",required=True)
    p.add_argument("--resume"); p.add_argument("--gpu",default="0")
    p = sub.add_parser("evaluate")
    p.add_argument("--predictions",required=True); p.add_argument("--references",required=True); p.add_argument("--output",required=True)
    p.add_argument("--label")
    p = sub.add_parser("select")
    p.add_argument("--reports",nargs="+",required=True); p.add_argument("--output",required=True)
    p = sub.add_parser("compare")
    p.add_argument("--baseline",required=True); p.add_argument("--candidate",required=True); p.add_argument("--output",required=True)
    p = sub.add_parser("diagnose-retrieval")
    p.add_argument("--qa",required=True); p.add_argument("--retrieval",required=True)
    p.add_argument("--index",required=True); p.add_argument("--output",required=True)
    p = sub.add_parser("package")
    p.add_argument("--predictions",required=True); p.add_argument("--questions",required=True); p.add_argument("--output",required=True)
    p.add_argument("--filename")
    args = parser.parse_args()
    c = config(args.config)
    cmd = args.command
    if cmd == "fit":
        # Before any torch import. QLoRA Trainer runs on exactly one visible GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.device = "cuda:0"
    if cmd in {"build-index","retrieve","generate","fit","audit-models"}:
        from .models import audit_models
        audit = audit_models(c,args.models)
    if cmd == "audit-models":
        result = audit
    elif cmd == "fetch-models":
        from .models import fetch_models
        result = fetch_models(c,args.models)
    elif cmd == "prepare":
        from .data import prepare
        result = prepare(args.train,args.test,args.output,c["seed"])
    elif cmd == "build-index":
        from .retrieval import build_index
        result = build_index(c,args.corpus,args.models,args.output,args.device)
    elif cmd == "retrieve":
        from .retrieval import retrieve
        result = retrieve(c,args.questions,args.models,args.index,args.output,args.device)
    elif cmd == "generate":
        from .generation import generate
        result = generate(c,args.questions,args.retrieval,args.models,args.output,args.device,args.adapter,args.mode)
    elif cmd == "fit":
        from .training import fit
        result = fit(c,args.train,args.retrieval,args.models,args.output,args.device,args.resume)
    elif cmd == "evaluate":
        from .metrics import evaluate
        result = evaluate(args.predictions,args.references,args.output,args.label)
    elif cmd == "select":
        from .metrics import select_reports
        result = select_reports(args.reports,args.output)
    elif cmd == "compare":
        from .metrics import compare_reports
        result = compare_reports(args.baseline,args.candidate,args.output,c["seed"])
    elif cmd == "diagnose-retrieval":
        from .metrics import diagnose_retrieval
        result = diagnose_retrieval(args.qa,args.retrieval,args.index,args.output)
    elif cmd == "package":
        from .generation import package_submission
        result = package_submission(args.predictions,args.questions,args.output,args.filename or c["submission_filename"])
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
