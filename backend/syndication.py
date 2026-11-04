"""Corpus-wide syndication map: every pair of articles compared once (the OpenMP all-pairs kernel,
rf_dup_pairs), linked when cos > tau, and grouped into connected components.

This is the second parallel kernel. The retrieval scan is memory-bandwidth bound (each 1.5 KB row
is used for one dot product); the all-pairs kernel reuses each row against thousands of partners,
so it is compute bound and scales further. Its triangular iteration space also makes the choice of
OpenMP schedule visible, which the scan's uniform rows do not.
"""
import time
from collections import Counter

import numpy as np

from .bench import _summ, platform_info, synthetic_corpus
from .engine import omp_engine, serial_engine

SCHED_VARIANTS = [("static", 0, "static"), ("static", 1, "static,1"), ("dynamic", 0, "dynamic"), ("guided", 0, "guided")]


def groups_summary(store, res, tau, top=15):
    labels = res["labels"]
    N = len(labels)
    sizes = np.bincount(labels, minlength=N)
    roots = np.nonzero(sizes >= 2)[0]
    roots = roots[np.argsort(-sizes[roots], kind="stable")]
    in_groups = int(sizes[roots].sum())
    show = roots[:top]
    member_rows = [int(i) for r in show for i in np.nonzero(labels == r)[0]]
    arts = store.articles(member_rows)
    groups = []
    for r in show:
        rows = np.nonzero(labels == r)[0]
        members = [{"id": int(i), "title": arts[int(i)]["title"], "domain": arts[int(i)]["domain"],
                    "url": arts[int(i)]["url"]} for i in rows]
        groups.append({"size": int(len(rows)), "outlets": len({m["domain"] for m in members}), "members": members})
    cross = sum(1 for g in groups if g["outlets"] > 1)
    return {
        "N": N, "tau": tau, "pairs": N * (N - 1) // 2, "edges": res["n_edges"],
        "groups": int(len(roots)), "articles_in_groups": in_groups,
        "share_with_duplicate": in_groups / N if N else 0.0,
        "largest": int(sizes.max()) if N else 0,
        "size_hist": {str(k): v for k, v in sorted(Counter(int(x) for x in sizes[roots]).items())},
        "top_groups": groups, "top_cross_outlet": cross,
    }


def run_map(store, tau=0.9, threads=0, schedule="dynamic"):
    """Syndication groups of the live corpus, with the OpenMP run checked against the serial build."""
    E = store.emb
    eng, ser = omp_engine(), serial_engine()
    eng.dup_pairs(E, tau, threads, schedule)  # warm-up: wakes the OpenMP thread pool and the caches
    r = eng.dup_pairs(E, tau, threads, schedule)
    ser.dup_pairs(E, tau)
    s = ser.dup_pairs(E, tau)
    same = bool(np.array_equal(r["labels"], s["labels"]) and np.array_equal(r["edges_i"], s["edges_i"])
                and np.array_equal(r["edges_j"], s["edges_j"]))
    out = groups_summary(store, r, tau)
    ts = r["thread_secs"]
    out["parallel"] = {
        "threads": r["threads"], "schedule": schedule, "omp_ms": r["seconds"] * 1e3, "serial_ms": s["seconds"] * 1e3,
        "speedup": s["seconds"] / r["seconds"] if r["seconds"] > 0 else None,
        "scan_ms": r["scan_seconds"] * 1e3, "merge_ms": r["merge_seconds"] * 1e3,
        "identical_to_serial": same, "imbalance": float(ts.max() / ts.mean()) if ts.mean() > 0 else None,
        "thread_secs": ts.tolist(), "thread_pairs": r["thread_pairs"].tolist(),
        "gflops": 2 * 384 * out["pairs"] / r["scan_seconds"] / 1e9 if r["scan_seconds"] > 0 else None,
    }
    return out


def _time(fn, reps):
    fn()  # warm-up
    return [fn() for _ in range(reps)]


def sweep(live_emb, threads=(1, 2, 4, 8, 11), sizes=("live", 10000), tau=0.9, reps=5, progress=None):
    """Schedule x threads sweep of the all-pairs kernel. Speedup is against the non-OpenMP build."""
    eng, ser = omp_engine(), serial_engine()
    threads = [t for t in threads if t <= eng.max_threads] or [1]
    results = []
    total = len(sizes) * (1 + len(threads) * len(SCHED_VARIANTS))
    done = 0
    for size in sizes:
        E = live_emb if size == "live" else synthetic_corpus(int(size))
        N = E.shape[0]
        label = f"live corpus ({N:,})" if size == "live" else f"synthetic {N:,}"
        ref = ser.dup_pairs(E, tau)
        serial_ts = _time(lambda: ser.dup_pairs(E, tau)["seconds"], reps)
        done += 1
        if progress:
            progress(done / total, f"{label}: serial")
        serial_med = float(np.median(serial_ts))
        configs = []
        for sched, chunk, name in SCHED_VARIANTS:
            for t in threads:
                last = {}

                def one():
                    last["r"] = eng.dup_pairs(E, tau, t, sched, chunk)
                    return last["r"]["seconds"]

                ts = _time(one, reps)
                r = last["r"]
                med = float(np.median(ts))
                tsec = r["thread_secs"]
                configs.append({
                    "schedule": name, "threads": t, **_summ(ts), "speedup": serial_med / med,
                    "efficiency": serial_med / med / t,
                    "imbalance": float(tsec.max() / tsec.mean()) if tsec.mean() > 0 else None,
                    "merge_ms": r["merge_seconds"] * 1e3,
                    "identical_to_serial": bool(np.array_equal(r["labels"], ref["labels"])
                                                and r["n_edges"] == ref["n_edges"]),
                })
                done += 1
                if progress:
                    progress(done / total, f"{label}: {name} {t}T")
        pairs = N * (N - 1) // 2
        results.append({"label": label, "N": N, "pairs": pairs, "edges": ref["n_edges"],
                        "serial": _summ(serial_ts), "serial_gflops": 2 * 384 * pairs / serial_med / 1e9,
                        "configs": configs})
    return {"tau": tau, "reps": reps, "threads": threads, "sizes": results, "cluster": cluster_sweep(threads, reps),
            "platform": platform_info(), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def syndicated_pool(n_stories=2000, copies_max=8, noise=0.01, seed=5):
    """A retrieved pool shaped like real syndication: stories with 1..copies_max near-identical copies."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_stories, 384)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    rows = []
    for c in range(n_stories):
        for _ in range(int(rng.integers(1, copies_max + 1))):
            v = centres[c] + rng.normal(0, noise, 384).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
    rows = np.array(rows, np.float32)
    return np.ascontiguousarray(rows[rng.permutation(len(rows))])


def cluster_sweep(threads, reps=7, Ks=(10, 100, 1000), tau=0.9):
    """Query-time clustering kernel on a large syndicated pool: serial build, OpenMP, and the NumPy reference
    (whose assignment step runs on Apple's Accelerate BLAS)."""
    from .pipeline import greedy_cluster
    eng, ser = omp_engine(), serial_engine()
    V = syndicated_pool()
    out = []
    for K in Ks:
        ref_l, ref_a = greedy_cluster(V, tau, K)
        serial_ts = _time(lambda: ser.cluster(V, tau, K)[2], reps)

        def npy():
            t = time.perf_counter()
            greedy_cluster(V, tau, K)
            return time.perf_counter() - t

        numpy_ts = _time(npy, reps)
        rows = []
        for t in threads:
            ts = _time(lambda: eng.cluster(V, tau, K, t, "guided")[2], reps)
            l, a, _, _ = eng.cluster(V, tau, K, t, "guided")
            rows.append({"threads": t, **_summ(ts), "speedup": float(np.median(serial_ts) / np.median(ts)),
                         "vs_numpy": float(np.median(numpy_ts) / np.median(ts)),
                         "identical_to_reference": bool(l == ref_l and np.array_equal(a, ref_a))})
        out.append({"K": K, "m": int(len(V)), "serial": _summ(serial_ts), "numpy": _summ(numpy_ts), "configs": rows})
    return out
