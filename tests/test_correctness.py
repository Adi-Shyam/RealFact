"""Correctness suite for RealFact.

    .venv/bin/python -m tests.test_correctness        # CLI
    POST /api/selftest                                # same suite, shown on the Benchmark tab

Engine checks compare against a NumPy reference and against the serial build; pipeline checks
re-derive the methodology's equations by hand.
"""
import itertools
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend  # noqa: E402,F401  (loads the engine before torch)
from backend import bench, pipeline  # noqa: E402
from backend.engine import omp_engine, serial_engine  # noqa: E402
from backend.ingest import dedup_key, is_english  # noqa: E402

TOL = 1e-5
SCHEDULES = ("static", "dynamic", "guided")
_tests = []


def test(fn):
    _tests.append(fn)
    return fn


def unit(n, d, seed):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((n, d), dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return np.ascontiguousarray(E)


def check_against_reference(E, q, M, idx, scores, exclude=None):
    """Scores match NumPy; the returned set is the true top-M (up to float ties at the boundary);
    order is best-first with ties broken by lower index."""
    ref = (E.astype(np.float64) @ q.astype(np.float64))
    if exclude is not None:
        ref[exclude.astype(bool)] = -np.inf
    valid = np.isfinite(ref).sum()
    m = min(M, valid)
    assert len(idx) == m, f"expected {m} hits, got {len(idx)}"
    assert np.allclose(scores, ref[idx], atol=TOL), "scores differ from reference"
    kth = np.sort(ref)[::-1][m - 1] if m else np.inf
    must = set(np.nonzero(ref > kth + TOL)[0])
    assert must <= set(idx.tolist()), "a clearly-better row is missing from top-M"
    assert all(ref[i] >= kth - TOL for i in idx), "a clearly-worse row is in top-M"
    for a, b, ia, ib in zip(scores, scores[1:], idx, idx[1:]):
        assert a > b or (a == b and ia < ib), "order is not best-first with index tie-break"
    if exclude is not None:
        assert not exclude[idx].any(), "an excluded row was returned"


@test
def engine_matches_numpy_all_configs():
    """Every schedule x threads x chunk returns the exact top-M (vs float64 NumPy)."""
    E = unit(20000, 384, 1)
    q = E[123]
    n = 0
    for sch, p, ch in itertools.product(SCHEDULES, (1, 2, 3, 8, 16), (0, 7, 1000)):
        idx, sc, _ = omp_engine().topm(E, q, 500, p, sch, ch)
        check_against_reference(E, q, 500, idx, sc)
        n += 1
    return f"{n} configurations"


@test
def parallel_bit_identical_to_serial():
    """IDs and scores from every parallel configuration equal the non-OpenMP build bit for bit."""
    E = unit(50000, 384, 2)
    n = 0
    for qi in (0, 777, 49999):
        s_idx, s_sc, _ = serial_engine().topm(E, E[qi], 10000)
        for sch, p in itertools.product(SCHEDULES, (2, 4, 11, 16)):
            idx, sc, _ = omp_engine().topm(E, E[qi], 10000, p, sch)
            assert np.array_equal(idx, s_idx) and np.array_equal(sc, s_sc), f"{sch} x{p} differs"
            n += 1
    return f"{n} runs, M = 10,000"


@test
def exact_ties_break_by_index():
    """Duplicated rows score identically; they must come back in ascending index order."""
    base = unit(50, 384, 3)
    E = np.ascontiguousarray(np.vstack([base] * 40))  # 2000 rows, every vector 40 times
    q = base[7]
    for sch, p in itertools.product(SCHEDULES, (1, 4, 16)):
        idx, sc, _ = omp_engine().topm(E, q, 40, p, sch, 3)
        assert np.all(sc == sc[0]), "copies did not tie"
        assert idx.tolist() == [7 + 50 * k for k in range(40)], f"{sch} x{p} tie order wrong"
    return "40-way tie, 9 configurations"


@test
def edge_cases():
    """M > N, M = 1, N = 1, N = 0, threads > N, d not a multiple of the vector width."""
    E = unit(10, 384, 4)
    idx, sc, _ = omp_engine().topm(E, E[0], 100, 16, "guided")
    check_against_reference(E, E[0], 100, idx, sc)
    idx, sc, _ = omp_engine().topm(E, E[3], 1, 8)
    assert idx.tolist() == [3]
    one = unit(1, 384, 5)
    assert omp_engine().topm(one, one[0], 10, 4)[0].tolist() == [0]
    empty = np.zeros((0, 384), np.float32)
    assert len(omp_engine().topm(empty, np.ones(384, np.float32), 10, 4)[0]) == 0
    for d in (1, 3, 17, 383, 385):
        Ed = unit(3000, d, d)
        idx, sc, _ = omp_engine().topm(Ed, Ed[1], 50, 4, "dynamic", 5)
        check_against_reference(Ed, Ed[1], 50, idx, sc)
        s_idx, s_sc, _ = serial_engine().topm(Ed, Ed[1], 50)
        assert np.array_equal(idx, s_idx) and np.array_equal(sc, s_sc)
    return "M>N, M=1, N=1, N=0, d in {1,3,17,383,385}"


@test
def leakage_mask_excludes_rows():
    """Masked rows are never returned, and the result equals the top-M of the unmasked rows."""
    E = unit(8000, 384, 6)
    q = E[42]
    mask = np.zeros(8000, np.uint8)
    mask[[42, 43, 100]] = 1
    mask[5000:6000] = 1
    for sch, p in itertools.product(SCHEDULES, (1, 8)):
        idx, sc, _ = omp_engine().topm(E, q, 300, p, sch, exclude=mask)
        check_against_reference(E, q, 300, idx, sc, exclude=mask)
    return "1,003 rows masked"


@test
def batched_kernel_equals_single_query():
    """The multi-query kernel returns, for every query, exactly what the single-query kernel returns."""
    eng = omp_engine()
    if not hasattr(eng, "topm_batch"):
        return "skipped (no batch kernel)"
    E = unit(30000, 384, 7)
    Q = np.ascontiguousarray(E[[1, 99, 5000, 29999, 12, 13, 14]])
    for sch, p in itertools.product(SCHEDULES, (1, 6, 16)):
        bidx, bsc, _ = eng.topm_batch(E, Q, 700, p, sch)
        for j in range(len(Q)):
            idx, sc, _ = serial_engine().topm(E, Q[j], 700)
            assert np.array_equal(bidx[j], idx) and np.array_equal(bsc[j], sc), f"query {j} {sch} x{p}"
    return f"{len(Q)} queries x 9 configurations"


@test
def thread_stats_account_for_every_row():
    """Per-thread row counts sum to N under every schedule (no row skipped or scanned twice)."""
    eng = omp_engine()
    if not hasattr(eng, "topm_stats"):
        return "skipped (no stats)"
    E = unit(40000, 384, 8)
    for sch, p, ch in itertools.product(SCHEDULES, (1, 3, 11), (0, 64)):
        _, _, _, secs, rows = eng.topm_stats(E, E[0], 100, p, sch, ch)
        assert rows.sum() == 40000 and len(rows) == p and (secs >= 0).all(), f"{sch} x{p} rows {rows.sum()}"
    return "18 configurations"


@test
def aggregation_equations():
    """S, D and the verdict table (methodology Eq. 3) on hand-computed cases."""
    agg = pipeline.aggregate
    # One polar cluster below theta -> Unverified.
    r = agg([{"w": 1, "p": 0.5, "sigma": 1}, {"w": 1, "p": 0.9, "sigma": 0}], theta=0.6, delta=0.25)
    assert r["verdict"] == "Unverified" and abs(r["polar_mass"] - 0.5) < 1e-12
    # Balanced support / contradiction -> Conflicting.
    r = agg([{"w": 1, "p": 0.9, "sigma": 1}, {"w": 1, "p": 0.8, "sigma": -1}], 0.6, 0.25)
    assert r["verdict"] == "Conflicting Evidence" and abs(r["D"] - 0.9 / 1.7) < 1e-12
    # Clear support.
    r = agg([{"w": 1, "p": 0.9, "sigma": 1}, {"w": 1, "p": 0.9, "sigma": 0}], 0.6, 0.25)
    assert r["verdict"] == "Supported" and abs(r["S"] - 0.5) < 1e-12 and r["D"] == 1.0
    # Clear contradiction.
    r = agg([{"w": 2, "p": 0.8, "sigma": -1}], 0.6, 0.25)
    assert r["verdict"] == "Contradicted" and r["S"] == -1.0
    # Report case study (Kuwait tanker): SUPPORT .97 vs an 11-copy CONTRADICT cluster at .80.
    w11 = 1 + 0.5 * math.log2(11)
    items = [{"w": 1, "p": 0.97, "sigma": 1}, {"w": 1, "p": 0.97, "sigma": 0},
             {"w": 1, "p": 0.93, "sigma": 0}, {"w": w11, "p": 0.80, "sigma": -1}]
    r = agg(items, 0.6, 0.25)
    assert r["verdict"] == "Conflicting Evidence" and abs(r["D"] - 0.31) < 0.005
    naive = [dict(it, w=11) if it["w"] == w11 else it for it in items]
    assert agg(naive, 0.6, 0.25)["verdict"] == "Contradicted"
    return "5 cases incl. the report's syndication example"


@test
def retrieve_wide_clustering():
    """Leader clustering groups copies, respects K, and leaves stories beyond K unassigned."""
    rng = np.random.default_rng(9)
    centres = unit(6, 384, 10)
    rows, truth = [], []
    for c, size in enumerate([5, 1, 3, 1, 2, 4]):
        for _ in range(size):
            v = centres[c] + rng.normal(0, 0.004, 384).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
            truth.append(c)
    V = np.array(rows, np.float32)
    leaders, assign = pipeline.greedy_cluster(V, tau=0.9, max_clusters=4)
    assert len(leaders) == 4
    sizes = sorted(int((assign == k).sum()) for k in range(4))
    assert sizes == [1, 1, 3, 5], sizes
    assert (assign == -1).sum() == 6, "stories beyond K should be unassigned"
    assert abs(1 + 0.5 * math.log2(1) - 1) < 1e-12
    return "6 stories, K = 4"


def _syndicated_pool(n_stories, copies_max, noise, seed):
    """Best-first pool of near-duplicate groups (copies of one story differ by small noise)."""
    rng = np.random.default_rng(seed)
    centres = unit(n_stories, 384, seed + 1)
    rows = []
    for c in range(n_stories):
        for _ in range(int(rng.integers(1, copies_max + 1))):
            v = centres[c] + rng.normal(0, noise, 384).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
    rows = np.array(rows, np.float32)
    return np.ascontiguousarray(rows[rng.permutation(len(rows))])


@test
def cluster_kernel_equals_reference():
    """OpenMP leader clustering = NumPy reference = serial build, for every schedule and block size."""
    cases = 0
    for seed, (stories, cmax, noise) in enumerate([(40, 6, 0.004), (300, 12, 0.01), (900, 4, 0.02)]):
        V = _syndicated_pool(stories, cmax, noise, 30 + seed)
        for tau, K in [(0.9, 10), (0.8, 50), (0.95, 500)]:
            ref_l, ref_a = pipeline.greedy_cluster(V, tau, K)
            s_l, s_a, _, _ = serial_engine().cluster(V, tau, K)
            assert s_l == ref_l and np.array_equal(s_a, ref_a), f"serial != reference (tau={tau}, K={K})"
            for sched, t, block in itertools.product(SCHEDULES, (2, 8), (1, 37, 256)):
                l, a, _, _ = omp_engine().cluster(V, tau, K, t, sched, block=block)
                assert l == ref_l and np.array_equal(a, ref_a), f"{sched} {t}T block {block} differs"
                cases += 1
    empty_l, empty_a, _, _ = omp_engine().cluster(np.zeros((0, 384), np.float32), 0.9, 10)
    assert empty_l == [] and len(empty_a) == 0
    return f"{cases} configurations identical"


@test
def all_pairs_kernel_equals_numpy():
    """All-pairs duplicate kernel: same edges as a NumPy Gram matrix, same groups as BFS, every schedule."""
    E = _syndicated_pool(150, 5, 0.01, 77)
    N = E.shape[0]
    tau = 0.9
    G = E.astype(np.float64) @ E.astype(np.float64).T
    iu = np.triu_indices(N, 1)
    margin = np.abs(G[iu] - tau) > 1e-4  # pairs this close to tau may round either way in float32
    want = set(zip(iu[0][(G[iu] > tau) & margin].tolist(), iu[1][(G[iu] > tau) & margin].tolist()))
    ref = serial_engine().dup_pairs(E, tau)
    got = set(zip(ref["edges_i"].tolist(), ref["edges_j"].tolist()))
    assert want <= got and len(got - want) <= int((~margin).sum()), "edge set differs from NumPy"
    # connected components by BFS over the kernel's own edges
    adj = {i: [] for i in range(N)}
    for i, j in got:
        adj[i].append(j)
        adj[j].append(i)
    label = [-1] * N
    for s0 in range(N):
        if label[s0] < 0:
            stack, comp = [s0], []
            label[s0] = s0
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if label[y] < 0:
                        label[y] = s0
                        stack.append(y)
    assert np.array_equal(ref["labels"], np.array(label)), "components differ from BFS"
    for sched, chunk, t in [("static", 0, 4), ("static", 1, 8), ("dynamic", 0, 8), ("guided", 0, 3)]:
        r = omp_engine().dup_pairs(E, tau, t, sched, chunk)
        assert np.array_equal(r["labels"], ref["labels"]) and np.array_equal(r["edges_i"], ref["edges_i"]) \
            and np.array_equal(r["edges_j"], ref["edges_j"]) and np.array_equal(r["edges_s"], ref["edges_s"])
        assert int(r["thread_pairs"].sum()) == N * (N - 1) // 2, "pairs not covered exactly once"
    return f"N = {N}, {len(got)} edges, {len(set(label))} groups"


@test
def wilcoxon_matches_enumeration():
    """Exact Wilcoxon p-value equals brute-force enumeration of all 2^n sign assignments."""
    rng = np.random.default_rng(11)
    for n in (6, 8, 10):
        for _ in range(5):
            x, y = rng.normal(size=n), rng.normal(size=n)
            got = bench.wilcoxon(x, y)
            d = x - y
            r = bench._rank_abs(d)
            T = min(r[d > 0].sum(), r[d < 0].sum())
            sums = [sum(rk for rk, s in zip(r, signs) if s) for signs in itertools.product((0, 1), repeat=n)]
            p = min(1.0, 2 * np.mean([s <= T + 1e-9 for s in sums]))
            assert abs(got["p"] - p) < 1e-12, (n, got["p"], p)
    assert abs(bench.wilcoxon(np.ones(11) * 2, np.ones(11))["p"] - 2 / 2048) < 1e-15
    return "15 random cases + the n = 11 floor"


@test
def ingest_normalisation():
    """Exact-duplicate key ignores case and punctuation but not the outlet; English filter works."""
    assert dedup_key("bbc.co.uk", "Earth is Round!") == dedup_key("bbc.co.uk", "earth is round")
    assert dedup_key("bbc.co.uk", "Earth is round") != dedup_key("npr.org", "Earth is round")
    assert is_english("Ukraine says 13 killed") and not is_english("Украина сообщает")
    return "3 cases"


@test
def eval_label_maps():
    """LIAR and FEVER mappings cover every label exactly as in methodology §I-B."""
    from eval.run_eval import FEVER_MAP, LIAR_MAP, VERDICT_TO_LABEL
    assert set(LIAR_MAP) == {"true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"}
    assert LIAR_MAP["half-true"] == LIAR_MAP["barely-true"] == "NEUTRAL"
    assert FEVER_MAP["NOT ENOUGH INFO"] == "NEUTRAL"
    assert VERDICT_TO_LABEL["Conflicting Evidence"] == VERDICT_TO_LABEL["Unverified"] == "NEUTRAL"
    return "LIAR 6-way, FEVER 3-way, verdict map"


def run_all():
    results = []
    for fn in _tests:
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            ok = not str(detail).startswith("skipped")
            results.append({"name": fn.__name__, "doc": fn.__doc__, "ok": True, "skipped": not ok,
                            "detail": detail, "ms": (time.perf_counter() - t0) * 1e3})
        except Exception as e:
            results.append({"name": fn.__name__, "doc": fn.__doc__, "ok": False, "skipped": False,
                            "detail": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(),
                            "ms": (time.perf_counter() - t0) * 1e3})
    return results


if __name__ == "__main__":
    res = run_all()
    for r in res:
        tag = "SKIP" if r["skipped"] else ("PASS" if r["ok"] else "FAIL")
        print(f"[{tag}] {r['name']:<40} {r['ms']:7.0f} ms  {r['detail']}")
        if not r["ok"]:
            print(r["trace"])
    failed = sum(not r["ok"] for r in res)
    print(f"\n{len(res) - failed}/{len(res)} passed")
    sys.exit(1 if failed else 0)
