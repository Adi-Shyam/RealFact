"""Benchmarking arm (timing only, no accuracy path) — methodology §III-A/B.

schedule {static, dynamic, guided} x threads {1,2,4,8,16} x corpus size, 2 warm-ups + 11 timed runs,
median + IQR, p95, speedup vs a NON-OpenMP serial build, efficiency, throughput, CPU utilisation,
bit-exact ID check against serial, and a Wilcoxon signed-rank test on paired per-query latencies.

Usable from the UI (/api/bench) or the command line:
    .venv/bin/python -m backend.bench --sizes 10000 50000 100000 500000
"""
import argparse
import json
import math
import os
import platform
import resource
import subprocess
import time

import numpy as np

from .config import DIM, DATA_DIR
from .engine import omp_engine, serial_engine

_synthetic = {}


def sysctl(name):
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def platform_info():
    info = {
        "machine": platform.machine(), "os": platform.platform(),
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "physical_cores": sysctl("hw.physicalcpu"), "logical_cores": sysctl("hw.logicalcpu"),
        "perf_cores": sysctl("hw.perflevel0.physicalcpu"), "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
        "l2_perf_bytes": sysctl("hw.perflevel0.l2cachesize"), "l2_eff_bytes": sysctl("hw.perflevel1.l2cachesize"),
        "l1d_bytes": sysctl("hw.l1dcachesize"), "mem_bytes": sysctl("hw.memsize"),
        "omp_max_threads": omp_engine().max_threads,
        "compiler": subprocess.run(["clang++", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "flags": "-O3 -std=c++17 -ffp-contract=off; serial build compiled without -fopenmp",
        "openmp_runtime": "LLVM libomp 19.1.5",
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", "unset"),
        "OMP_PLACES": os.environ.get("OMP_PLACES", "unset"),
        "pinning_note": "macOS exposes no thread-affinity API, so OMP_PROC_BIND/OMP_PLACES are not honoured; "
                        "the OS scheduler places threads across performance and efficiency cores.",
    }
    return info


def synthetic_corpus(n, seed=7):
    if n not in _synthetic:
        rng = np.random.default_rng(seed)
        E = rng.standard_normal((n, DIM), dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        _synthetic[n] = np.ascontiguousarray(E)
    return _synthetic[n]


def _rank_abs(d):
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def wilcoxon(x, y):
    """Two-sided Wilcoxon signed-rank test. Exact null distribution (via DP over doubled ranks) for n <= 30."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "W": None, "p": 1.0, "effect_r": 0.0}
    r = _rank_abs(d)
    w_plus = float(r[d > 0].sum())
    w_minus = float(r[d < 0].sum())
    T = min(w_plus, w_minus)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (T - mu) / sd if sd > 0 else 0.0
    if n <= 30:
        r2 = np.rint(r * 2).astype(int)
        total = int(r2.sum())
        dist = np.zeros(total + 1)
        dist[0] = 1.0
        for v in r2:
            dist[v:] = dist[v:] + dist[: total + 1 - v].copy()
        dist /= dist.sum()
        p = min(1.0, 2 * dist[: int(round(T * 2)) + 1].sum())
    else:
        p = min(1.0, math.erfc(abs(z) / math.sqrt(2)))
    return {"n": n, "W": T, "p": p, "z": z, "effect_r": abs(z) / math.sqrt(n)}


def _summ(ts):
    ts = np.asarray(ts)
    q1, med, q3, p95 = np.percentile(ts, [25, 50, 75, 95])
    return {"median_ms": med * 1e3, "iqr_ms": (q3 - q1) * 1e3, "p95_ms": p95 * 1e3, "runs_ms": (ts * 1e3).tolist()}


def time_config(eng, E, queries, M, threads, schedule, chunk, warmup):
    for w in range(warmup):
        eng.topm(E, queries[w % len(queries)], M, threads, schedule, chunk)
    times, ids = [], []
    ru0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    for q in queries:
        idx, _, secs = eng.topm(E, q, M, threads, schedule, chunk)
        times.append(secs)
        ids.append(idx)
    wall = time.perf_counter() - w0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    return times, ids, cpu / wall if wall > 0 else 0.0


def karp_flatt(S, p):
    """Experimentally determined serial fraction e = (1/S - 1/p) / (1 - 1/p). Rising e with p means
    overhead that grows with p (sync, bandwidth contention); flat e means a true serial fraction."""
    return None if p <= 1 or S <= 0 else (1 / S - 1 / p) / (1 - 1 / p)


def _median_time(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    return float(np.median([fn() for _ in range(reps)]))


def load_balance(E, M, p, configs, reps=7):
    """Per-thread busy time inside the worksharing loop, per schedule. Median run by kernel time."""
    omp = omp_engine()
    q = E[3]
    out = []
    for sch, ch in configs:
        runs = []
        for _ in range(reps + 2):
            _, _, secs, busy, rows = omp.topm_stats(E, q, M, p, sch, ch)
            runs.append((secs, busy, rows))
        runs = sorted(runs[2:], key=lambda r: r[0])
        secs, busy, rows = runs[len(runs) // 2]
        out.append({
            "schedule": sch, "chunk": ch, "threads": p, "kernel_ms": secs * 1e3,
            "busy_ms": (busy * 1e3).tolist(), "rows": rows.tolist(),
            "imbalance": float(busy.max() / busy.mean()) if busy.mean() > 0 else None,
            "idle_pct": float(100 * (1 - busy.mean() / busy.max())) if busy.max() > 0 else None,
        })
    return out


def chunk_sweep(E, M, p, chunks=(1, 16, 256, 4096, 32768), reps=7, warmup=2):
    omp = omp_engine()
    q = E[4]
    rows = [{"schedule": "static", "chunk": 0,
             "median_ms": _median_time(lambda: omp.topm(E, q, M, p, "static")[2], reps, warmup) * 1e3}]
    for sch in ("dynamic", "guided"):
        for ch in chunks:
            rows.append({"schedule": sch, "chunk": ch,
                         "median_ms": _median_time(lambda: omp.topm(E, q, M, p, sch, ch)[2], reps, warmup) * 1e3})
    return rows


def weak_scaling(E, M, threads, n0, schedules=("static", "guided"), reps=7, warmup=2):
    """Gustafson-style: fixed work per thread (n0 rows). Weak efficiency = T(1, n0) / T(p, p n0)."""
    omp = omp_engine()
    out = []
    for sch in schedules:
        pts = []
        base = None
        for p in threads:
            n = p * n0
            if n > E.shape[0]:
                break
            sub = E[:n]
            t = _median_time(lambda: omp.topm(sub, sub[0], M, p, sch)[2], reps, warmup)
            base = base or t
            pts.append({"threads": p, "N": n, "median_ms": t * 1e3, "weak_efficiency": base / t})
        out.append({"schedule": sch, "n_per_thread": n0, "points": pts})
    return out


def batched(E, p, Qs=(1, 4, 8, 16, 32), Ms=(100, 10000), reps=7, warmup=2):
    """Multi-query blocking: each row is read once for Q queries -> arithmetic intensity 0.5 Q flop/byte."""
    omp = omp_engine()
    rng = np.random.default_rng(5)
    out = []
    for M in Ms:
        for Q in Qs:
            Qm = E[rng.choice(E.shape[0], Q, replace=False)]
            t = _median_time(lambda: omp.topm_batch(E, Qm, M, p, "guided")[2], reps, warmup)
            out.append({"M": M, "Q": Q, "median_ms": t * 1e3, "ms_per_query": t * 1e3 / Q,
                        "throughput_qps": Q / t, "flop_per_byte": 0.5 * Q,
                        "gflops": 2 * E.shape[0] * E.shape[1] * Q / t / 1e9,
                        "gbytes_per_s": E.nbytes / t / 1e9})
    return out


def run(sizes=(10000, 50000, 100000), threads=(1, 2, 4, 8, 16), schedules=("static", "dynamic", "guided"),
        chunks=(0,), reps=11, warmup=2, M=10000, real_corpus=None, progress=None, extras=True):
    omp, ser = omp_engine(), serial_engine()
    rng = np.random.default_rng(11)
    out = {"platform": platform_info(), "params": {
        "sizes": list(sizes), "threads": list(threads), "schedules": list(schedules), "chunks": list(chunks),
        "reps": reps, "warmup": warmup, "M": M}, "sizes": []}
    jobs = len(sizes) * (2 + len(schedules) * len(threads) * len(chunks))
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        if progress:
            progress(done / jobs, msg)

    for n in sizes:
        if n == "live":
            if real_corpus is None or real_corpus.shape[0] < 2:
                continue
            E, label = real_corpus, f"live corpus ({real_corpus.shape[0]:,})"
        else:
            E, label = synthetic_corpus(int(n)), f"synthetic {int(n):,}"
        N = E.shape[0]
        m = min(M, N)
        queries = [E[i] for i in rng.choice(N, size=reps, replace=N < reps)]
        ws = N * DIM * 4

        t_ser, ids_ser, util_ser = time_config(ser, E, queries, m, 1, "static", 0, warmup)
        tick(f"{label}: serial T1")
        t_omp1, _, _ = time_config(omp, E, queries, m, 1, "static", 0, warmup)
        tick(f"{label}: OpenMP 1-thread")
        T1 = float(np.median(t_ser))
        entry = {
            "label": label, "N": N, "M": m, "working_set_bytes": ws,
            "serial": _summ(t_ser), "omp1": _summ(t_omp1),
            "openmp_entry_cost_ms": (float(np.median(t_omp1)) - T1) * 1e3,
            "configs": [], "wilcoxon": [],
        }
        per_cfg = {}
        for sch in schedules:
            for ch in (chunks if sch != "static" else (0,)):
                for p in threads:
                    ts, ids, util = time_config(omp, E, queries, m, p, sch, ch, warmup)
                    med = float(np.median(ts))
                    match = all(np.array_equal(a, b) for a, b in zip(ids, ids_ser))
                    entry["configs"].append({
                        "schedule": sch, "chunk": ch, "threads": p, **_summ(ts),
                        "speedup": T1 / med, "efficiency": T1 / med / p,
                        "karp_flatt": karp_flatt(T1 / med, p),
                        "throughput_qps": 1.0 / med, "cpu_util": util / p,
                        "ids_match_serial": bool(match),
                        "oversubscribed": p > int(out["platform"]["physical_cores"] or p),
                    })
                    per_cfg[(sch, ch, p)] = ts
                    tick(f"{label}: {sch}{'/' + str(ch) if ch else ''} x{p}")
                if sch == "static":
                    break
        pmax = max(threads)
        base = per_cfg.get(("static", 0, pmax))
        for sch in schedules:
            if sch == "static" or base is None:
                continue
            other = per_cfg.get((sch, chunks[0], pmax))
            if other is not None:
                entry["wilcoxon"].append({"a": f"static x{pmax}", "b": f"{sch} x{pmax}", **wilcoxon(base, other)})
        out["sizes"].append(entry)

    synth = [n for n in sizes if n != "live"]
    if extras and synth:
        nbig = int(max(synth))
        E = synthetic_corpus(nbig)
        m = min(M, nbig)
        cores = int(out["platform"]["physical_cores"] or max(threads))
        if progress:
            progress(0.97, "load balance per thread")
        out["load_balance"] = load_balance(
            E, m, cores, [("static", 0), ("dynamic", 0), ("dynamic", 4096), ("guided", 0)])
        if progress:
            progress(0.98, "chunk-size sweep")
        out["chunk_sweep"] = {"N": nbig, "threads": cores, "rows": chunk_sweep(E, m, cores)}
        if progress:
            progress(0.99, "weak scaling + batched queries")
        n0 = nbig // max(threads)
        out["weak_scaling"] = weak_scaling(E, m, threads, n0)
        out["batched"] = {"N": nbig, "threads": max(threads), "rows": batched(E, max(threads))}
    out["kernel_version"] = "v2: 16-way ILP dot, register-blocked multi-query"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", type=int, default=[10000, 50000, 100000, 500000])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--chunks", nargs="+", type=int, default=[0])
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--M", type=int, default=10000)
    a = ap.parse_args()
    res = run(a.sizes, a.threads, chunks=a.chunks, reps=a.reps, M=a.M,
              progress=lambda f, m: print(f"[{f:5.0%}] {m}", flush=True))
    path = os.path.join(DATA_DIR, "bench_cli.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)
