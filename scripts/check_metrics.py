"""Execute after installing the exact metric requirements and WordNet.

The tiny strings here are metric unit fixtures only. They never enter any training or retrieval data.
"""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from legalqa.metrics import metric_environment

env,identity = metric_environment()
from rouge_score import tokenize

pred = {"fixture":{"answer":"alpha beta gamma delta"}}
truth = {"fixture":"alpha beta gamma delta"}
result = env["eval_qa"](pred,truth)
assert abs(result["rouge"]-1.0)<1e-12
assert abs(result["meteor"]-(1-0.5*(1/4)**3))<1e-12
assert tokenize.tokenize("pháp luật",None)==["ph","p","lu","t"]
print("Official metric runtime self-test passed; identical METEOR need not be exactly 1.0.")
print("NLTK:",identity["nltk"])
