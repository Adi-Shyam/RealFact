"""Choose the FEVER-demo NLI setup on the held-out tuning split (scripts/fever_tune_split.py).

    .venv/bin/python -m scripts.fever_tune [--split tune|demo] [--models ...]

Retrieval and clustering run once per claim (min_rel 0.60, K = 10); NLI outputs are cached per model and
evidence granularity; aggregation thresholds are then swept offline. Leader clustering is prefix-stable
(candidates are processed best-first), so the clusters kept at a higher relevance floor are exactly the
clusters whose representative clears it.

Evidence granularity:
  lead      premise = page title + lead paragraph (current pipeline)
  sentence  premise = page title + one sentence; a cluster takes the sentence whose top polar probability
            (support or contradict) is highest — FEVER systems select sentences, not paragraphs.
"""
import argparse
import itertools
import json
import os
import pickle
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend  # noqa: E402,F401
from backend import models, pipeline  # noqa: E402
from backend.config import FEVER_CLAIMS_PATH, FEVER_DB_PATH, FEVER_DIR, FEVER_EMB_PATH  # noqa: E402
from backend.engine import omp_engine  # noqa: E402
from backend.store import Store  # noqa: E402

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
GOLD = {"SUPPORTS": "SUPPORT", "REFUTES": "CONTRADICT", "NOT ENOUGH INFO": "NEUTRAL"}
VMAP = {"Supported": "SUPPORT", "Contradicted": "CONTRADICT", "Unverified": "NEUTRAL", "Conflicting Evidence": "NEUTRAL"}


def sentences(text):
    return [s for s in SENT_RE.split(text) if len(s) > 3] or [text]


def retrieve(store, claims, K=10, floor=0.60, tau=0.9):
    eng = omp_engine()
    out = []
    for c in claims:
        q = models.embed_claim(c["claim"])
        idx, sc, _ = eng.topm(store.emb, q, 2000, 0, "guided")
        keep = sc >= floor
        idx, sc = idx[keep], sc[keep]
        leaders, assign, _, _ = eng.cluster(np.ascontiguousarray(store.emb[idx]), tau, K)
        arts = store.articles([int(idx[l]) for l in leaders])
        clusters = []
        for ci, l in enumerate(leaders):
            a = arts[int(idx[l])]
            clusters.append({"score": float(sc[l]), "n": int((assign == ci).sum()), "title": a["title"],
                             "summary": a["summary"], "page": a["url"].rsplit("/", 1)[-1]})
        out.append(clusters)
    return out


def nli(claims, ev, model, mode):
    """Per claim, per cluster: (sigma, p, probs)."""
    pairs, owner = [], []
    for ci, (c, clusters) in enumerate(zip(claims, ev)):
        for k, cl in enumerate(clusters):
            prem = [f"{cl['title']}. {cl['summary']}"] if mode == "lead" else [f"{cl['title']}. {s}" for s in sentences(cl["summary"])]
            for p in prem:
                pairs.append((c["claim"], p))
                owner.append((ci, k))
    res = []
    B = 64
    for i in range(0, len(pairs), B):
        chunk = pairs[i:i + B]
        res += models.classify_pairs(chunk, model)
    best = {}
    for (ci, k), r in zip(owner, res):
        sigma, label, p, probs = r
        polar = max(probs["SUPPORT"], probs["CONTRADICT"])
        if (ci, k) not in best or polar > best[(ci, k)][0]:
            best[(ci, k)] = (polar, r)
    return [[best[(ci, k)][1] for k in range(len(clusters))] for ci, clusters in enumerate(ev)]


def score(claims, ev, st, theta, delta, floor, K, lam=0.5):
    y, p = [], []
    for c, clusters, stances in zip(claims, ev, st):
        items = []
        for cl, s in zip(clusters, stances):
            if cl["score"] < floor or len(items) >= K:
                continue
            sigma, label, prob, probs = s
            items.append({"w": 1 + lam * np.log2(cl["n"]), "p": prob, "sigma": sigma})
        v = pipeline.aggregate(items, theta, delta)["verdict"]
        y.append(GOLD[c["label"]])
        p.append(VMAP[v])
    acc = float(np.mean([a == b for a, b in zip(y, p)]))
    f1 = []
    for L in ("SUPPORT", "CONTRADICT", "NEUTRAL"):
        tp = sum(a == L == b for a, b in zip(y, p))
        fp = sum(b == L != a for a, b in zip(y, p))
        fn = sum(a == L != b for a, b in zip(y, p))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return acc, float(np.mean(f1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune")
    ap.add_argument("--models", nargs="+", default=["MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                                                    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"])
    ap.add_argument("--modes", nargs="+", default=["lead", "sentence"])
    a = ap.parse_args()
    path = os.path.join(FEVER_DIR, "tune_claims.json") if a.split == "tune" else FEVER_CLAIMS_PATH
    claims = json.load(open(path))["claims"]
    store = Store(FEVER_DB_PATH, FEVER_EMB_PATH)
    cache = os.path.join(FEVER_DIR, f".tune_cache_{a.split}.pkl")
    C = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    if "ev" not in C:
        C["ev"] = retrieve(store, claims)
    for m, mode in itertools.product(a.models, a.modes):
        if (m, mode) not in C:
            t = time.time()
            C[(m, mode)] = nli(claims, C["ev"], m, mode)
            C[("secs", m, mode)] = time.time() - t
            pickle.dump(C, open(cache, "wb"))
        print(f"{m.split('/')[-1]:45s} {mode:8s} NLI {C[('secs', m, mode)]:.0f}s for {len(claims)} claims", flush=True)

    grid = list(itertools.product((0.4, 0.5, 0.6, 0.7, 0.8), (0.15, 0.25), (0.60, 0.64, 0.68, 0.72), (1, 3, 5, 10)))
    rows = []
    for m, mode in itertools.product(a.models, a.modes):
        st = C[(m, mode)]
        base = score(claims, C["ev"], st, 0.6, 0.25, 0.68, 10)
        best = max(((score(claims, C["ev"], st, *g), g) for g in grid), key=lambda r: (r[0][0], r[0][1]))
        rows.append({"model": m, "mode": mode, "default_acc": base[0], "default_f1": base[1],
                     "best_acc": best[0][0], "best_f1": best[0][1],
                     "best": dict(zip(("theta", "delta", "min_rel", "K"), best[1]))})
        print(f"{m.split('/')[-1]:45s} {mode:8s} default acc {base[0]:.3f}  best acc {best[0][0]:.3f} (F1 {best[0][1]:.3f}) at {rows[-1]['best']}")
    json.dump(rows, open(os.path.join(FEVER_DIR, f"tune_results_{a.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
