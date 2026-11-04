"""ctypes bridge to the C++ retrieval engine (OpenMP build + serial reference build)."""
import ctypes
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = "dylib" if sys.platform == "darwin" else "so"
SCHEDULES = {"static": 0, "dynamic": 1, "guided": 2}


class Engine:
    def __init__(self, path):
        lib = ctypes.CDLL(path)
        lib.rf_topm.restype = ctypes.c_double
        lib.rf_topm.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
        ]
        lib.rf_topm_stats.restype = ctypes.c_double
        lib.rf_topm_stats.argtypes = lib.rf_topm.argtypes + [ctypes.c_void_p, ctypes.c_void_p]
        lib.rf_topm_batch.restype = ctypes.c_double
        lib.rf_topm_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.rf_cluster.restype = ctypes.c_double
        lib.rf_cluster.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.rf_dup_pairs.restype = ctypes.c_double
        lib.rf_dup_pairs.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.rf_has_openmp.restype = ctypes.c_int
        lib.rf_max_threads.restype = ctypes.c_int
        self.lib = lib
        self.has_openmp = bool(lib.rf_has_openmp())
        self.max_threads = lib.rf_max_threads()

    def topm(self, emb, q, M, threads=0, schedule="static", chunk=0, exclude=None):
        """Exact top-M by dot product. Returns (indices, scores, kernel_seconds)."""
        assert emb.dtype == np.float32 and emb.flags["C_CONTIGUOUS"]
        q = np.ascontiguousarray(q, dtype=np.float32)
        N, d = emb.shape
        M = max(1, min(M, N))
        out_idx = np.empty(M, dtype=np.int64)
        out_score = np.empty(M, dtype=np.float32)
        count = ctypes.c_int(0)
        excl_ptr = None
        if exclude is not None:
            exclude = np.ascontiguousarray(exclude, dtype=np.uint8)
            excl_ptr = exclude.ctypes.data
        secs = self.lib.rf_topm(
            emb.ctypes.data, N, d, q.ctypes.data, M,
            threads, SCHEDULES[schedule], chunk, excl_ptr,
            out_idx.ctypes.data, out_score.ctypes.data, ctypes.byref(count),
        )
        n = count.value
        return out_idx[:n], out_score[:n], secs

    def _threads(self, threads):
        return (threads if threads > 0 else self.max_threads) if self.has_openmp else 1

    def topm_stats(self, emb, q, M, threads=0, schedule="static", chunk=0, exclude=None):
        """As topm, plus per-thread busy seconds and rows taken (load-balance analysis)."""
        assert emb.dtype == np.float32 and emb.flags["C_CONTIGUOUS"]
        q = np.ascontiguousarray(q, dtype=np.float32)
        N, d = emb.shape
        M = max(1, min(M, N))
        P = self._threads(threads)
        out_idx = np.empty(M, dtype=np.int64)
        out_score = np.empty(M, dtype=np.float32)
        secs_t = np.zeros(P, dtype=np.float64)
        rows_t = np.zeros(P, dtype=np.int64)
        count = ctypes.c_int(0)
        excl_ptr = np.ascontiguousarray(exclude, dtype=np.uint8).ctypes.data if exclude is not None else None
        secs = self.lib.rf_topm_stats(
            emb.ctypes.data, N, d, q.ctypes.data, M, P, SCHEDULES[schedule], chunk, excl_ptr,
            out_idx.ctypes.data, out_score.ctypes.data, ctypes.byref(count),
            secs_t.ctypes.data, rows_t.ctypes.data,
        )
        n = count.value
        return out_idx[:n], out_score[:n], secs, secs_t, rows_t

    def topm_batch(self, emb, Q, M, threads=0, schedule="static", chunk=0, exclude=None):
        """Q queries in one pass over the matrix. Returns (list of idx, list of scores, seconds)."""
        assert emb.dtype == np.float32 and emb.flags["C_CONTIGUOUS"]
        Q = np.ascontiguousarray(np.atleast_2d(Q), dtype=np.float32)
        N, d = emb.shape
        nq = Q.shape[0]
        M = max(1, min(M, N))
        out_idx = np.empty((nq, M), dtype=np.int64)
        out_score = np.empty((nq, M), dtype=np.float32)
        counts = np.zeros(nq, dtype=np.int32)
        excl_ptr = np.ascontiguousarray(exclude, dtype=np.uint8).ctypes.data if exclude is not None else None
        secs = self.lib.rf_topm_batch(
            emb.ctypes.data, N, d, Q.ctypes.data, nq, M, threads, SCHEDULES[schedule], chunk, excl_ptr,
            out_idx.ctypes.data, out_score.ctypes.data, counts.ctypes.data,
        )
        return ([out_idx[j, :counts[j]] for j in range(nq)],
                [out_score[j, :counts[j]] for j in range(nq)], secs)

    def cluster(self, V, tau, K, threads=0, schedule="static", chunk=0, block=256):
        """Leader clustering of a best-first pool. Returns (leaders list, assign array, seconds, phase seconds)."""
        V = np.ascontiguousarray(V, dtype=np.float32)
        m, d = V.shape if V.ndim == 2 else (0, 384)
        K = max(1, int(K))
        leaders = np.empty(K, dtype=np.int64)
        assign = np.full(max(m, 1), -1, dtype=np.int64)
        phases = np.zeros(2, dtype=np.float64)
        n = ctypes.c_int(0)
        if m == 0:
            return [], assign[:0], 0.0, phases
        secs = self.lib.rf_cluster(V.ctypes.data, m, d, float(tau), K, self._threads(threads), SCHEDULES[schedule],
                                   chunk, block, leaders.ctypes.data, ctypes.byref(n), assign.ctypes.data,
                                   phases.ctypes.data)
        return leaders[: n.value].tolist(), assign[:m], secs, phases

    def dup_pairs(self, E, tau, threads=0, schedule="static", chunk=0, max_edges=200_000):
        """All pairs i < j with cos > tau + connected components. Returns a dict."""
        assert E.dtype == np.float32 and E.flags["C_CONTIGUOUS"]
        N, d = E.shape
        P = self._threads(threads)
        labels = np.empty(N, dtype=np.int64)
        ei = np.empty(max_edges, dtype=np.int64)
        ej = np.empty(max_edges, dtype=np.int64)
        es = np.empty(max_edges, dtype=np.float32)
        ne = ctypes.c_int64(0)
        phases = np.zeros(2, dtype=np.float64)
        secs_t = np.zeros(P, dtype=np.float64)
        pairs_t = np.zeros(P, dtype=np.int64)
        secs = self.lib.rf_dup_pairs(E.ctypes.data, N, d, float(tau), P, SCHEDULES[schedule], chunk,
                                     labels.ctypes.data, ei.ctypes.data, ej.ctypes.data, es.ctypes.data, max_edges,
                                     ctypes.byref(ne), phases.ctypes.data, secs_t.ctypes.data, pairs_t.ctypes.data)
        k = min(ne.value, max_edges)
        return {"labels": labels, "edges_i": ei[:k], "edges_j": ej[:k], "edges_s": es[:k], "n_edges": ne.value,
                "seconds": secs, "scan_seconds": float(phases[0]), "merge_seconds": float(phases[1]),
                "thread_secs": secs_t, "thread_pairs": pairs_t, "threads": P}


_omp = _serial = None


def omp_engine():
    global _omp
    if _omp is None:
        _omp = Engine(os.path.join(ROOT, "build", f"librealfact_omp.{EXT}"))
    return _omp


def serial_engine():
    global _serial
    if _serial is None:
        _serial = Engine(os.path.join(ROOT, "build", f"librealfact_serial.{EXT}"))
    return _serial
