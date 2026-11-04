"""Phase 2: verify a claim (methodology §II-B .. §II-D).

embed claim -> OpenMP exact scan top-M -> cluster near-duplicates at tau over the WIDE pool
-> keep top-K clusters -> NLI on one representative per cluster -> independence-weighted
aggregation -> four-way cited verdict.

Ablations from §III-A are computed from the same retrieval so the UI can show them side by side.
"""
import math
import time

import numpy as np

from . import models
from .config import DEFAULTS
from .engine import omp_engine, serial_engine


def params_from(req, overrides=None):
    p = dict(DEFAULTS)
    p.update(overrides or {})
    for k, v in (req or {}).items():
        if k in p and v is not None and v != "":
            t = type(DEFAULTS[k])
            p[k] = int(float(v)) if t is int else t(v)  # "10.0" from a number input is a valid K
    p["K"] = max(1, p["K"])
    p["M"] = max(p["K"], p["M"])
    return p


GRAIN_ROWS = 400        # below this many rows per thread, fork/join + heap setup outweighs the scan
BANDWIDTH_THREADS = 8   # the scan saturates memory bandwidth by ~4-8 threads on the M3 Pro (benchmark)


def auto_threads(N, max_threads):
    """Thread count for one query: enough rows per thread to amortise the parallel region, and no
    more threads than the memory system can feed."""
    return max(1, min(max_threads, BANDWIDTH_THREADS, -(-N // GRAIN_ROWS)))


def greedy_cluster(V, tau, max_clusters):
    """Single-pass leader clustering in score order (NumPy reference; the pipeline runs the same
    algorithm in the C++ engine, rf_cluster, and tests check the two agree exactly).

    V: candidate vectors sorted best-first. A candidate joins its most similar leader if
    cosine > tau; otherwise it founds a new cluster while fewer than max_clusters exist.
    Returns (leaders, assign) where assign[j] is a cluster number or -1 (outside the top clusters).
    """
    m = V.shape[0]
    assign = np.full(m, -1, dtype=np.int64)
    leaders = []
    j = 0
    while j < m and len(leaders) < max_clusters:
        if leaders:
            sims = V[leaders] @ V[j]
            b = int(sims.argmax())
            if sims[b] > tau:
                assign[j] = b
                j += 1
                continue
        leaders.append(j)
        assign[j] = len(leaders) - 1
        j += 1
    if j < m and leaders:  # leaders are full: vectorised assignment of the remaining pool
        sims = V[j:] @ V[leaders].T
        best = sims.argmax(axis=1)
        ok = sims[np.arange(sims.shape[0]), best] > tau
        assign[j:][ok] = best[ok]
    return leaders, assign


def aggregate(items, theta, delta):
    """items: list of dicts with w, p, sigma. Implements Eq. (3) and the verdict table."""
    tot = sum(it["w"] * it["p"] for it in items)
    signed = sum(it["w"] * it["p"] * it["sigma"] for it in items)
    polar = [it for it in items if it["sigma"] != 0]
    polar_mass = sum(it["w"] * it["p"] for it in polar)
    sup_mass = sum(it["w"] * it["p"] for it in polar if it["sigma"] == 1)
    S = signed / tot if tot > 0 else 0.0
    D = sup_mass / polar_mass if polar_mass > 0 else None

    if polar_mass < theta:
        verdict = "Unverified"
    elif delta < D < 1 - delta:
        verdict = "Conflicting Evidence"
    elif S >= 0:
        verdict = "Supported"
    else:
        verdict = "Contradicted"
    return {
        "verdict": verdict, "S": S, "D": D, "polar_mass": polar_mass,
        "support_mass": sup_mass, "contradict_mass": polar_mass - sup_mass,
        "n_evidence": len(items), "n_votes": sum(it["w"] for it in items),
    }


def _article_view(a, score):
    return {
        "id": a["id"], "title": a["title"], "summary": a["summary"], "url": a["url"],
        "domain": a["domain"], "published_at": a["published_at"], "score": float(score),
        "simulated": bool(a.get("simulated")), "origin": a.get("origin", "live"),
    }


def verify(store, claim, req=None, defaults=None):
    p = params_from(req, defaults)
    timings = {}
    t_all = time.perf_counter()
    emb = store.emb
    N = emb.shape[0]
    if N == 0:
        raise ValueError("The evidence corpus is empty. Run ingestion first (Corpus tab).")

    t = time.perf_counter()
    q = models.embed_claim(claim)
    timings["embed_claim"] = time.perf_counter() - t

    # Syndication stress test: republish one real article as n near-identical copies under
    # simulated outlets, so the scan really retrieves them. Copies never touch the stored corpus.
    synthetic = {}
    syn_id, syn_n = (req or {}).get("syndicate_id"), int((req or {}).get("syndicate_n") or 0)
    if syn_id is not None and 0 <= int(syn_id) < N and syn_n > 0:
        syn_id = int(syn_id)
        src = store.articles([syn_id])[syn_id]
        rng = np.random.default_rng(syn_id)
        copies = emb[syn_id] + rng.normal(0, 0.006, (syn_n, emb.shape[1])).astype(np.float32)
        copies /= np.linalg.norm(copies, axis=1, keepdims=True)
        emb = np.ascontiguousarray(np.vstack([emb, copies]))
        for j in range(syn_n):
            synthetic[N + j] = {**src, "id": N + j, "domain": f"wire-copy-{j + 1}.simulated", "simulated": True}

    # --- OpenMP exact scan, top-M ---------------------------------------------------------
    eng = omp_engine()
    threads = p["threads"] or auto_threads(emb.shape[0], eng.max_threads)
    exclude = (req or {}).get("exclude_mask")  # leakage control (§I-C), used by the eval harness
    if exclude is not None and len(exclude) < emb.shape[0]:
        exclude = np.concatenate([exclude, np.zeros(emb.shape[0] - len(exclude), dtype=np.uint8)])
    idx, scores, kernel_s = eng.topm(emb, q, p["M"], threads, p["schedule"], exclude=exclude)
    timings["retrieval_kernel"] = kernel_s
    # Same query through the serial reference build: correctness assertion + per-query speedup.
    s_idx, _, serial_s = serial_engine().topm(emb, q, p["M"], exclude=exclude)
    ids_match = bool(np.array_equal(idx, s_idx))

    keep = scores >= p["min_rel"]
    cand_idx, cand_scores = idx[keep], scores[keep]

    # --- Retrieve-wide clustering ---------------------------------------------------------
    t = time.perf_counter()
    V = np.ascontiguousarray(emb[cand_idx])
    leaders, assign, cluster_s, cluster_phases = eng.cluster(V, p["tau"], p["K"], threads, p["schedule"])
    clusters = []
    for c, lj in enumerate(leaders):
        members = np.nonzero(assign == c)[0]
        clusters.append({"rep": int(lj), "members": members.tolist()})
    # Ablation: cluster AFTER truncating to K (the self-defeating order, §II-C).
    trunc_leaders, trunc_assign, _, _ = eng.cluster(V[: p["K"]], p["tau"], p["K"], 1)
    timings["clustering"] = time.perf_counter() - t

    # --- NLI on one representative per cluster (+ raw top-K for the filter-off ablation) ----
    rep_rows = [int(cand_idx[c["rep"]]) for c in clusters]
    raw_rows = [int(i) for i in cand_idx[: p["K"]]]
    nli_rows = list(dict.fromkeys(rep_rows + raw_rows))
    member_rows = {int(cand_idx[j]) for c in clusters for j in c["members"]}
    wanted = set(nli_rows) | member_rows
    arts = store.articles(wanted - set(synthetic))
    arts.update({r: synthetic[r] for r in wanted if r in synthetic})
    t = time.perf_counter()
    stances = models.classify(claim, [f"{arts[r]['title']}. {arts[r]['summary']}" for r in nli_rows], p["nli_model"])
    timings["nli"] = time.perf_counter() - t
    stance_of = dict(zip(nli_rows, stances))
    score_of = {int(i): float(s) for i, s in zip(cand_idx, cand_scores)}

    # --- Build evidence + aggregate under every mode -------------------------------------------
    t = time.perf_counter()
    evidence = []
    for rank, c in enumerate(clusters):
        row = int(cand_idx[c["rep"]])
        sigma, label, prob, probs = stance_of[row]
        n_c = len(c["members"])
        members = [_article_view(arts[int(cand_idx[j])], score_of[int(cand_idx[j])]) for j in c["members"]]
        evidence.append({
            "rank": rank + 1, "rep": _article_view(arts[row], score_of[row]), "n_c": n_c,
            "domains": sorted({m["domain"] for m in members}), "members": members,
            "sigma": sigma, "label": label, "p": prob, "probs": probs,
            "w_log": 1 + p["lam"] * math.log2(n_c), "w_one": 1.0, "w_naive": float(n_c),
        })

    def mode(weight_key):
        return aggregate([{"w": e[weight_key], "p": e["p"], "sigma": e["sigma"]} for e in evidence],
                         p["theta"], p["delta"])

    raw_items = []
    for r in raw_rows:
        sigma, label, prob, _ = stance_of[r]
        raw_items.append({"w": 1.0, "p": prob, "sigma": sigma})
    ablations = {
        "independence_log": {**mode("w_log"), "label": f"RealFact: log-weighted (λ={p['lam']})"},
        "one_vote": {**mode("w_one"), "label": "One vote per cluster (λ = 0)"},
        "naive_count": {**mode("w_naive"), "label": "Naive duplicate counting (every copy = 1 vote)"},
        "filter_off": {**aggregate(raw_items, p["theta"], p["delta"]),
                       "label": "Filter off: raw top-K articles (typical retrieve-then-classify)"},
    }
    final = ablations["independence_log"]
    for e in evidence:
        e["contribution"] = e["w_log"] * e["p"] * e["sigma"]
    timings["aggregation"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t_all

    copies_in_pool = int((assign >= 0).sum())
    N = emb.shape[0]
    return {
        "claim": claim,
        "params": p,
        "syndication_test": {"article_id": syn_id, "copies": syn_n} if synthetic else None,
        "verdict": final["verdict"],
        "S": final["S"], "D": final["D"],
        "aggregate": final,
        "evidence": evidence,
        "ablations": ablations,
        "retrieve_wide": {
            "corpus_size": N,
            "M_requested": p["M"], "M_returned": int(len(idx)),
            "above_relevance_floor": int(len(cand_idx)),
            "clusters_kept": len(clusters),
            "articles_in_kept_clusters": copies_in_pool,
            "duplicates_collapsed": copies_in_pool - len(clusters),
            "truncate_then_cluster_independent": len(trunc_leaders),
            "truncate_then_cluster_slots_wasted": min(p["K"], len(cand_idx)) - len(trunc_leaders),
        },
        "parallel": {
            "threads": threads, "threads_auto": not p["threads"], "schedule": p["schedule"],
            "kernel_ms": kernel_s * 1e3, "serial_ms": serial_s * 1e3,
            "speedup": serial_s / kernel_s if kernel_s > 0 else None,
            "ids_match_serial": ids_match,
            "cluster_ms": cluster_s * 1e3, "cluster_pool": int(len(cand_idx)),
            "cluster_leader_ms": float(cluster_phases[0]) * 1e3, "cluster_assign_ms": float(cluster_phases[1]) * 1e3,
        },
        "timings_ms": {k: v * 1e3 for k, v in timings.items()},
    }
