"""Held-out tuning split for the FEVER demo, disjoint from the 179 demo (test) claims.

    .venv/bin/python -m scripts.fever_tune_split        -> data/fever/tune_claims.json

Every design choice (NLI model, sentence-level evidence, thresholds) is made on these claims; the demo
claims are scored once afterwards. Selection rules match the demo: SUPPORTS / REFUTES need one evidence
set that is a single sentence in a page's first 4 sentences, and that page must already be in the demo
corpus (the corpus's hard-distractor pages are exactly other dev claims' evidence pages). Claims whose
gold page is also a test claim's gold page are excluded, so no page is shared between the splits.
NOT ENOUGH INFO claims are sampled at random from the remaining dev claims.
"""
import json
import os
import random
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import DATA_DIR, FEVER_CLAIMS_PATH, FEVER_DB_PATH, FEVER_DIR  # noqa: E402
from scripts.build_fever_demo import GOLD_MAX_LINE, title_text  # noqa: E402

PER_LABEL = 60
SEED = 7


def main():
    test = json.load(open(FEVER_CLAIMS_PATH))["claims"]
    test_ids = {c["id"] for c in test}
    test_pages = {p for c in test for p in c["gold_pages"]}
    con = sqlite3.connect(FEVER_DB_PATH)
    corpus = {k[len("fever|"):]: s for k, s in con.execute("SELECT dedup_key, summary FROM articles")}

    dev = [json.loads(l) for l in open(os.path.join(DATA_DIR, "datasets", "fever", "shared_task_dev.jsonl"))]
    pools = defaultdict(list)
    for c in dev:
        if c["id"] in test_ids:
            continue
        if c["label"] == "NOT ENOUGH INFO":
            pools[c["label"]].append((c, None))
            continue
        for es in c["evidence"]:
            if len(es) == 1 and es[0][2] and es[0][3] is not None and es[0][3] <= GOLD_MAX_LINE:
                page = es[0][2]
                if page in corpus and page not in test_pages:
                    pools[c["label"]].append((c, es[0]))
                    break

    rng = random.Random(SEED)
    out, used = [], set()
    for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
        pool = pools[label][:]
        rng.shuffle(pool)
        n = 0
        for c, ev in pool:
            if n >= PER_LABEL:
                break
            if ev and ev[2] in used:
                continue
            gold = []
            if ev:
                used.add(ev[2])
                gold = [{"page": title_text(ev[2]), "page_id": ev[2], "line": ev[3], "sentence": None}]
            out.append({"id": c["id"], "claim": c["claim"], "label": label, "evidence": gold,
                        "gold_pages": [g["page_id"] for g in gold]})
            n += 1
    rng.shuffle(out)
    meta = {"split": "tune", "claims": len(out), "seed": SEED,
            "per_label": {l: sum(c["label"] == l for c in out) for l in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")},
            "disjoint_from": "demo_claims.json (no shared claim ids or gold pages)"}
    path = os.path.join(FEVER_DIR, "tune_claims.json")
    with open(path, "w") as f:
        json.dump({"meta": meta, "claims": out}, f, indent=1)
    print(meta, "->", path)


if __name__ == "__main__":
    main()
