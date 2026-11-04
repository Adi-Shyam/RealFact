"""Verification-quality evaluation harness (methodology §I-B, §III-C).

Label harmonisation exactly as specified:
  FEVER        SUPPORTS -> SUPPORT, REFUTES -> CONTRADICT, NOT ENOUGH INFO -> NEUTRAL
  LIAR         {true, mostly-true} -> SUPPORT, {false, pants-fire} -> CONTRADICT,
               {half-true, barely-true} -> NEUTRAL (middle band also reported separately)
  FakeNewsNet  headline is the claim; real -> SUPPORT, fake -> CONTRADICT;
               CONTRADICT+NEUTRAL also scored as a merged "not-supported" class
System verdicts: Supported -> SUPPORT, Contradicted -> CONTRADICT, Unverified/Conflicting -> NEUTRAL
(and reported disaggregated). Every ablation mode is scored from the same retrieval.

Usage:
  .venv/bin/python -m eval.run_eval --dataset liar --path data/datasets/liar/test.tsv --limit 300
  .venv/bin/python -m eval.run_eval --dataset fever --path data/datasets/fever/shared_task_dev.jsonl
  .venv/bin/python -m eval.run_eval --dataset fakenewsnet --path data/datasets/fakenewsnet   # dir with *_fake.csv / *_real.csv
Results are merged into data/eval_results.json and shown on the Evaluation tab.
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import pipeline  # noqa: E402
from backend.config import CORPUS_DEFAULTS, DATA_DIR, FEVER_CLAIMS_PATH, FEVER_DB_PATH, FEVER_EMB_PATH  # noqa: E402
from backend.ingest import normalise_headline  # noqa: E402
from backend.store import Store  # noqa: E402

LABELS = ["SUPPORT", "CONTRADICT", "NEUTRAL"]
VERDICT_TO_LABEL = {"Supported": "SUPPORT", "Contradicted": "CONTRADICT",
                    "Unverified": "NEUTRAL", "Conflicting Evidence": "NEUTRAL"}
LIAR_MAP = {"true": "SUPPORT", "mostly-true": "SUPPORT", "false": "CONTRADICT", "pants-fire": "CONTRADICT",
            "half-true": "NEUTRAL", "barely-true": "NEUTRAL"}
FEVER_MAP = {"SUPPORTS": "SUPPORT", "REFUTES": "CONTRADICT", "NOT ENOUGH INFO": "NEUTRAL"}


def load_fever(path):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("label") in FEVER_MAP:
                yield {"claim": r["claim"], "gold": FEVER_MAP[r["label"]], "orig": r["label"]}


def load_liar(path):
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) > 2 and row[1] in LIAR_MAP:
                yield {"claim": row[2], "gold": LIAR_MAP[row[1]], "orig": row[1]}


def load_fever_demo(path):
    """Curated FEVER dev claims of the demo corpus (data/fever/demo_claims.json)."""
    with open(path) as f:
        for c in json.load(f)["claims"]:
            yield {"claim": c["claim"], "gold": FEVER_MAP[c["label"]], "orig": c["label"], "gold_pages": c["gold_pages"]}


def load_fakenewsnet(path):
    for fn in sorted(glob.glob(os.path.join(path, "*.csv"))):
        lab = "CONTRADICT" if "fake" in os.path.basename(fn) else "SUPPORT"
        with open(fn, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("title"):
                    yield {"claim": r["title"], "gold": lab, "orig": "fake" if lab == "CONTRADICT" else "real"}


def gold_rank(res, gold_pages):
    """1-based rank of the first kept cluster containing a gold page, or None."""
    pages = set(gold_pages)
    for e in res["evidence"]:
        ids = {m["url"].rsplit("/", 1)[-1].replace("(", "-LRB-").replace(")", "-RRB-") for m in e["members"]}
        if ids & pages:
            return e["rank"]
    return None


def prf(gold, pred, labels):
    out, f1s = {}, []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        out[c] = {"precision": P, "recall": R, "f1": F, "support": sum(g == c for g in gold)}
        f1s.append(F)
    acc = sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold))
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "per_class": out,
            "confusion": {g: dict(Counter(p for gg, p in zip(gold, pred) if gg == g)) for g in labels}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["fever", "fever-demo", "liar", "fakenewsnet"])
    ap.add_argument("--path", default=None, help="dataset file; fever-demo defaults to data/fever/demo_claims.json")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    loader = {"fever": load_fever, "fever-demo": load_fever_demo, "liar": load_liar,
              "fakenewsnet": load_fakenewsnet}[a.dataset]
    if a.path is None:
        if a.dataset != "fever-demo":
            ap.error("--path is required for this dataset")
        a.path = FEVER_CLAIMS_PATH
    items = list(loader(a.path))
    random.Random(a.seed).shuffle(items)
    items = items[: a.limit]
    demo = a.dataset == "fever-demo"
    store = Store(FEVER_DB_PATH, FEVER_EMB_PATH) if demo else Store()
    defaults = CORPUS_DEFAULTS["fever" if demo else "live"]
    titles = [normalise_headline(r["title"]) for r in store.db.execute("SELECT title FROM articles ORDER BY id")]

    modes = ["independence_log", "one_vote", "naive_count", "filter_off"]
    preds = {m: [] for m in modes}
    verdicts = {m: [] for m in modes}
    gold, orig, lat, ranks = [], [], [], []
    for i, it in enumerate(items):
        # Leakage control: never let a claim retrieve its own source article.
        key = normalise_headline(it["claim"])
        mask = np.array([t == key for t in titles], dtype=np.uint8)
        t0 = time.perf_counter()
        # The demo corpus is Wikipedia: the claim is never an article title, so no leakage mask is needed.
        res = pipeline.verify(store, it["claim"], None if demo else {"exclude_mask": mask}, defaults)
        if it.get("gold_pages"):
            ranks.append(gold_rank(res, it["gold_pages"]))
        lat.append(time.perf_counter() - t0)
        for m in modes:
            v = res["ablations"][m]["verdict"]
            verdicts[m].append(v)
            preds[m].append(VERDICT_TO_LABEL[v])
        gold.append(it["gold"])
        orig.append(it["orig"])
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(items)}", flush=True)

    report = {"dataset": a.dataset, "n": len(items), "corpus_size": store.size,
              "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "p95_latency_ms": float(np.percentile(lat, 95) * 1e3), "median_latency_ms": float(np.median(lat) * 1e3),
              "recall_at_k": ({f"@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in (1, 5, 10)}
                              if ranks else "undefined: gold evidence is not in the live news corpus" if a.dataset != "fakenewsnet"
                              else "requires FakeNewsNet source-URL crawl"),
              "params": pipeline.params_from(None, defaults), "modes": {}}
    for m in modes:
        r = prf(gold, preds[m], LABELS)
        r["verdict_counts"] = dict(Counter(verdicts[m]))
        if a.dataset == "liar":
            two = [k for k, o in enumerate(orig) if LIAR_MAP[o] != "NEUTRAL"]
            mid = [k for k, o in enumerate(orig) if LIAR_MAP[o] == "NEUTRAL"]
            r["two_sided_band"] = prf([gold[k] for k in two], [preds[m][k] for k in two], LABELS)["accuracy"]
            r["middle_band_verdicts"] = dict(Counter(verdicts[m][k] for k in mid))
        if a.dataset in ("fever", "fever-demo"):
            r["nei_band_verdicts"] = dict(Counter(verdicts[m][k] for k, o in enumerate(orig) if o == "NOT ENOUGH INFO"))
        if a.dataset == "fakenewsnet":
            g2 = ["SUPPORT" if g == "SUPPORT" else "NOT_SUPPORTED" for g in gold]
            p2 = ["SUPPORT" if p == "SUPPORT" else "NOT_SUPPORTED" for p in preds[m]]
            r["merged_not_supported"] = prf(g2, p2, ["SUPPORT", "NOT_SUPPORTED"])
        report["modes"][m] = r

    path = os.path.join(DATA_DIR, "eval_results.json")
    allres = {}
    if os.path.exists(path):
        with open(path) as f:
            allres = json.load(f)
    allres[a.dataset] = report
    with open(path, "w") as f:
        json.dump(allres, f, indent=1)
    print(json.dumps({m: {"acc": report["modes"][m]["accuracy"], "macroF1": report["modes"][m]["macro_f1"]}
                      for m in modes}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
