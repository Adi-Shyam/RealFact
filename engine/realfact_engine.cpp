// RealFact retrieval engine: exact top-M similarity scan over a cached
// N x d row-major matrix of unit-normalised embeddings (methodology §II-B, §II-E).
//
// Built twice from this one file:
//   librealfact_omp.dylib    -fopenmp   -> parallel kernel, schedule(runtime)
//   librealfact_serial.dylib (no OpenMP) -> serial reference T1, no runtime overhead
//
// Because the per-row dot product is the same code in both builds and ties are
// broken by index, every configuration must return bit-identical IDs.
//
// Exports:
//   rf_topm        one query                        (verification path)
//   rf_topm_stats  one query + per-thread busy time and row counts (load-balance analysis)
//   rf_topm_batch  Q queries in one pass over the matrix (cache blocking: each row is read
//                  from memory once and scored against all Q queries while it is in L1)
//   rf_cluster     query-time leader clustering of the retrieved pool (§II-C), parallelised
//                  with block speculation; identical output to the sequential algorithm
//   rf_dup_pairs   corpus-wide all-pairs near-duplicate detection + connected components
//                  (syndication groups); triangular O(N^2 d) workload, compute-bound

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Hit {
    float score;
    int64_t idx;
};

// "a is better than b": higher score wins, equal scores -> lower index wins.
inline bool better(const Hit& a, const Hit& b) {
    return a.score > b.score || (a.score == b.score && a.idx < b.idx);
}

// std::*_heap with this comparator keeps the *worst* kept hit at the front.
struct WorseFirst {
    bool operator()(const Hit& a, const Hit& b) const { return better(a, b); }
};

// Thread-local top-M min-heap, padded to a cache line so heap metadata of
// neighbouring threads never shares a line (no false sharing).
struct alignas(64) LocalHeap {
    std::vector<Hit> h;
    char pad[64 - sizeof(std::vector<Hit>) % 64];

    void push(const Hit& x, size_t M) {
        if (h.size() < M) {
            h.push_back(x);
            std::push_heap(h.begin(), h.end(), WorseFirst());
        } else if (better(x, h.front())) {
            std::pop_heap(h.begin(), h.end(), WorseFirst());
            h.back() = x;
            std::push_heap(h.begin(), h.end(), WorseFirst());
        }
    }
};

// Per-thread instrumentation, also cache-line padded.
struct alignas(64) ThreadStat {
    double secs = 0;
    int64_t rows = 0;
};

// Dot product with 16 independent partial sums.
//
// A single accumulator makes every vector add wait for the previous one (a loop-carried
// dependency of 96 adds per 384-d row), so one core runs at FP-add latency, not throughput.
// 16 scalar sums = 4 independent 4-lane NEON/SSE registers, which hides that latency (ILP).
// The summation order is written out explicitly, so the compiler needs no -ffast-math
// reassociation and the result is bit-identical in the serial and OpenMP builds.
inline float dot(const float* __restrict a, const float* __restrict b, int d) {
    float acc[16] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int k = 0;
    for (; k + 16 <= d; k += 16)
        for (int j = 0; j < 16; ++j) acc[j] += a[k + j] * b[k + j];
    for (int j = 0; k < d; ++k, ++j) acc[j] += a[k] * b[k];
    for (int s = 8; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) acc[j] += acc[j + s];
    return acc[0];
}

// Four queries against one row: each row element is loaded once and used four times
// (register blocking). Each query keeps exactly the partial-sum order of dot(), so the
// result for every query is bit-identical to dot(q_j, row).
inline void dot4(const float* __restrict q0, const float* __restrict q1, const float* __restrict q2,
                 const float* __restrict q3, const float* __restrict b, int d, float* out) {
    float a0[16] = {0}, a1[16] = {0}, a2[16] = {0}, a3[16] = {0};
    int k = 0;
    for (; k + 16 <= d; k += 16)
        for (int j = 0; j < 16; ++j) {
            const float x = b[k + j];
            a0[j] += q0[k + j] * x;
            a1[j] += q1[k + j] * x;
            a2[j] += q2[k + j] * x;
            a3[j] += q3[k + j] * x;
        }
    for (int j = 0; k < d; ++k, ++j) {
        const float x = b[k];
        a0[j] += q0[k] * x;
        a1[j] += q1[k] * x;
        a2[j] += q2[k] * x;
        a3[j] += q3[k] * x;
    }
    for (int s = 8; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) {
            a0[j] += a0[j + s];
            a1[j] += a1[j + s];
            a2[j] += a2[j + s];
            a3[j] += a3[j + s];
        }
    out[0] = a0[0];
    out[1] = a1[0];
    out[2] = a2[0];
    out[3] = a3[0];
}

// Score one row against all Q queries: blocks of four through dot4, the rest through dot.
inline void score_row(const float* Qm, int Q, const float* row, int d, int64_t i, LocalHeap* heaps,
                      size_t Mz) {
    int j = 0;
    float s[4];
    for (; j + 4 <= Q; j += 4) {
        const float* q = Qm + (int64_t)j * d;
        dot4(q, q + d, q + 2 * d, q + 3 * d, row, d, s);
        for (int u = 0; u < 4; ++u) heaps[j + u].push(Hit{s[u], i}, Mz);
    }
    for (; j < Q; ++j) heaps[j].push(Hit{dot(Qm + (int64_t)j * d, row, d), i}, Mz);
}

int set_schedule(int threads, int schedule, int chunk) {
#ifdef _OPENMP
    if (threads < 1) threads = omp_get_max_threads();
    omp_sched_t kind = schedule == 1   ? omp_sched_dynamic
                       : schedule == 2 ? omp_sched_guided
                                       : omp_sched_static;
    omp_set_schedule(kind, chunk > 0 ? chunk : 0);
    return threads;
#else
    (void)schedule;
    (void)chunk;
    (void)threads;
    return 1;
#endif
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Serial reduction: P local heaps -> global top-M, independent of N. Methodology §II-E
// specifies a heap merge, O(P M log M); concatenate + nth_element does the same job in
// O(P M + M log M). `better` is a strict total order (indices are unique), so the
// result is identical either way.
int merge(std::vector<LocalHeap*>& heaps, size_t Mz, int64_t* out_idx, float* out_score) {
    std::vector<Hit> all;
    size_t total = 0;
    for (auto* lh : heaps) total += lh->h.size();
    all.reserve(total);
    for (auto* lh : heaps) all.insert(all.end(), lh->h.begin(), lh->h.end());
    size_t keep = std::min(Mz, all.size());
    std::nth_element(all.begin(), all.begin() + keep, all.end(), better);
    all.resize(keep);
    std::sort(all.begin(), all.end(), better);
    int n = (int)all.size();
    for (int j = 0; j < n; ++j) {
        out_idx[j] = all[j].idx;
        out_score[j] = all[j].score;
    }
    return n;
}

// Core scan for Q queries. heaps[t * Q + j] is thread t's heap for query j.
void scan(const float* emb, int64_t N, int d, const float* Qm, int Q, size_t Mz, int threads,
          const uint8_t* exclude, std::vector<LocalHeap>& heaps, std::vector<ThreadStat>& stats) {
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        LocalHeap* mine = &heaps[(size_t)t * Q];
        int64_t rows = 0;
        double t0 = omp_get_wtime();
#pragma omp for schedule(runtime) nowait
        for (int64_t i = 0; i < N; ++i) {
            ++rows;
            if (exclude && exclude[i]) continue;
            const float* row = emb + i * (int64_t)d;
            score_row(Qm, Q, row, d, i, mine, Mz);
        }
        stats[t].secs = omp_get_wtime() - t0;
        stats[t].rows = rows;
    }
#else
    (void)threads;
    double t0 = now();
    for (int64_t i = 0; i < N; ++i) {
        if (exclude && exclude[i]) continue;
        const float* row = emb + i * (int64_t)d;
        score_row(Qm, Q, row, d, i, heaps.data(), Mz);
    }
    stats[0].secs = now() - t0;
    stats[0].rows = N;
#endif
}

double topm_impl(const float* emb, int64_t N, int d, const float* Qm, int Q, int M, int threads,
                 int schedule, int chunk, const uint8_t* exclude, int64_t* out_idx, float* out_score,
                 int* out_count, double* thread_secs, int64_t* thread_rows) {
    double t0 = now();
    const size_t Mz = (size_t)std::max(1, M);
    threads = set_schedule(threads, schedule, chunk);

    std::vector<LocalHeap> heaps((size_t)threads * Q);
    for (auto& lh : heaps) lh.h.reserve(std::min<size_t>(Mz, (size_t)N));
    std::vector<ThreadStat> stats(threads);

    scan(emb, N, d, Qm, Q, Mz, threads, exclude, heaps, stats);

    for (int j = 0; j < Q; ++j) {
        std::vector<LocalHeap*> col;
        for (int t = 0; t < threads; ++t) col.push_back(&heaps[(size_t)t * Q + j]);
        out_count[j] = merge(col, Mz, out_idx + (int64_t)j * Mz, out_score + (int64_t)j * Mz);
    }
    if (thread_secs)
        for (int t = 0; t < threads; ++t) thread_secs[t] = stats[t].secs;
    if (thread_rows)
        for (int t = 0; t < threads; ++t) thread_rows[t] = stats[t].rows;
    return now() - t0;
}

// ---------------------------------------------------------------------------------------------
// Query-time clustering (methodology §II-C)
//
// Sequential definition (pipeline.greedy_cluster): walk candidates best-first; a candidate joins
// its most similar leader if cos > tau, otherwise founds a new cluster while fewer than K exist.
// Once K leaders exist, every remaining candidate is assigned independently.
//
// The leader phase has a loop-carried dependency (candidate j's decision depends on every leader
// founded before it), so it cannot be a plain parallel loop. We use block speculation: for a
// block of B candidates, the similarity to the leaders that exist at block start is computed in
// parallel; then a short sequential walk compares each candidate only against leaders founded
// inside the block and makes the decision. Leaders are compared in founding order with a strict
// '>' everywhere, so the arg-max (first maximum) and the result are exactly the sequential ones.
// The assignment phase is a regular parallel loop.
// ---------------------------------------------------------------------------------------------
#ifdef _OPENMP
const int CLUSTER_MIN_WORK = 4096;  // dot products below which a parallel region costs more than it saves
#endif

// Leaders are scored four at a time with dot4 (the candidate row is loaded once per four leaders);
// dot4 is bit-identical to dot, and leaders are still compared in founding order with strict '>'.
inline void best_leader(const float* V, int d, int64_t i, const std::vector<int64_t>& L, size_t from,
                        size_t to, float& best, int& arg) {
    const float* vi = V + i * (int64_t)d;
    size_t u = from;
    float s[4];
    for (; u + 4 <= to; u += 4) {
        dot4(V + L[u] * (int64_t)d, V + L[u + 1] * (int64_t)d, V + L[u + 2] * (int64_t)d,
             V + L[u + 3] * (int64_t)d, vi, d, s);
        for (int k = 0; k < 4; ++k)
            if (s[k] > best) {
                best = s[k];
                arg = (int)(u + k);
            }
    }
    for (; u < to; ++u) {
        const float s = dot(vi, V + L[u] * (int64_t)d, d);
        if (s > best) {
            best = s;
            arg = (int)u;
        }
    }
}

double cluster_impl(const float* V, int64_t m, int d, float tau, int K, int threads, int schedule,
                    int chunk, int block, int64_t* out_leaders, int* out_nleaders, int64_t* out_assign,
                    double* phase_secs) {
    double t0 = now();
    threads = set_schedule(threads, schedule, chunk);
    if (block < 1) block = 256;
    std::vector<int64_t> L;
    L.reserve(std::max(K, 1));
    std::vector<float> best(m);
    std::vector<int> arg(m);
    for (int64_t i = 0; i < m; ++i) out_assign[i] = -1;

    int64_t j = 0;
    while (j < m && (int)L.size() < K) {
        const int64_t e = std::min<int64_t>(m, j + block);
        const size_t n0 = L.size();
        // Parallel part: similarity of each block candidate to the leaders existing at block start.
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) num_threads(threads) if ((e - j) * (int64_t)n0 >= CLUSTER_MIN_WORK)
#endif
        for (int64_t i = j; i < e; ++i) {
            float b = -2.f;
            int a = -1;
            best_leader(V, d, i, L, 0, n0, b, a);
            best[i] = b;
            arg[i] = a;
        }
        // Sequential walk: only leaders founded inside this block are left to compare.
        int64_t i = j;
        for (; i < e; ++i) {
            if ((int)L.size() >= K) break;
            float b = best[i];
            int a = arg[i];
            best_leader(V, d, i, L, n0, L.size(), b, a);
            if (a >= 0 && b > tau) {
                out_assign[i] = a;
            } else {
                L.push_back(i);
                out_assign[i] = (int64_t)L.size() - 1;
            }
        }
        j = i;
    }
    double t1 = now();
    // Assignment phase: K leaders fixed, candidates are independent.
    if (j < m && !L.empty()) {
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime) num_threads(threads) if ((m - j) * (int64_t)L.size() >= CLUSTER_MIN_WORK)
#endif
        for (int64_t i = j; i < m; ++i) {
            float b = -2.f;
            int a = -1;
            best_leader(V, d, i, L, 0, L.size(), b, a);
            out_assign[i] = b > tau ? a : -1;
        }
    }
    double t2 = now();
    for (size_t u = 0; u < L.size(); ++u) out_leaders[u] = L[u];
    *out_nleaders = (int)L.size();
    if (phase_secs) {
        phase_secs[0] = t1 - t0;
        phase_secs[1] = t2 - t1;
    }
    return t2 - t0;
}

// ---------------------------------------------------------------------------------------------
// Corpus-wide near-duplicate detection (syndication groups)
//
// All pairs i < j with cos(e_i, e_j) > tau, then connected components. The iteration space is a
// triangle: row block b has ~(N - 4b) partners, so the first blocks carry far more work than the
// last. That makes this the textbook case for OpenMP scheduling: schedule(static) with the default
// chunk gives thread 0 the heaviest contiguous range, while static,1 (cyclic), dynamic and guided
// spread it. Rows are taken four at a time and every partner row is loaded once and used four
// times (dot4 register blocking), so the kernel is compute-bound, unlike the bandwidth-bound scan.
// Edges go to thread-local buffers (no locks), are concatenated and sorted, so the output is
// deterministic. Union-find is serial and timed separately (the Amdahl term).
// ---------------------------------------------------------------------------------------------
struct Edge {
    int64_t i, j;
    float s;
};

struct alignas(64) LocalEdges {
    std::vector<Edge> e;
    double secs = 0;
    int64_t pairs = 0;
    char pad[64];
};

int64_t uf_find(std::vector<int64_t>& p, int64_t x) {
    while (p[x] != x) {
        p[x] = p[p[x]];
        x = p[x];
    }
    return x;
}

double dup_pairs_impl(const float* E, int64_t N, int d, float tau, int threads, int schedule, int chunk,
                      int64_t* out_labels, int64_t* out_ei, int64_t* out_ej, float* out_es, int64_t max_edges,
                      int64_t* out_nedges, double* phase_secs, double* thread_secs, int64_t* thread_pairs) {
    double t0 = now();
    threads = set_schedule(threads, schedule, chunk);
    const int64_t nb = (N + 3) / 4;
    std::vector<LocalEdges> local(threads);

    auto do_block = [&](int64_t b, LocalEdges& mine) {
        const int64_t i0 = 4 * b, i1 = std::min<int64_t>(N, i0 + 4);
        // pairs inside the block
        for (int64_t a = i0; a < i1; ++a)
            for (int64_t c = a + 1; c < i1; ++c) {
                const float s = dot(E + a * d, E + c * d, d);
                if (s > tau) mine.e.push_back(Edge{a, c, s});
            }
        if (i1 - i0 == 4) {
            const float *r0 = E + i0 * d, *r1 = r0 + d, *r2 = r1 + d, *r3 = r2 + d;
            float s[4];
            for (int64_t c = i1; c < N; ++c) {
                dot4(r0, r1, r2, r3, E + c * d, d, s);
                for (int u = 0; u < 4; ++u)
                    if (s[u] > tau) mine.e.push_back(Edge{i0 + u, c, s[u]});
            }
        } else {
            for (int64_t a = i0; a < i1; ++a)
                for (int64_t c = i1; c < N; ++c) {
                    const float s = dot(E + a * d, E + c * d, d);
                    if (s > tau) mine.e.push_back(Edge{a, c, s});
                }
        }
        const int64_t rows = i1 - i0;
        mine.pairs += rows * (N - i1) + rows * (rows - 1) / 2;
    };

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
    {
        LocalEdges& mine = local[omp_get_thread_num()];
        double ts = omp_get_wtime();
#pragma omp for schedule(runtime) nowait
        for (int64_t b = 0; b < nb; ++b) do_block(b, mine);
        mine.secs = omp_get_wtime() - ts;
    }
#else
    {
        double ts = now();
        for (int64_t b = 0; b < nb; ++b) do_block(b, local[0]);
        local[0].secs = now() - ts;
    }
#endif
    double t1 = now();

    // Serial merge + union-find (smaller index becomes the root -> label = min index of the group).
    std::vector<Edge> all;
    size_t total = 0;
    for (auto& l : local) total += l.e.size();
    all.reserve(total);
    for (auto& l : local) all.insert(all.end(), l.e.begin(), l.e.end());
    std::sort(all.begin(), all.end(), [](const Edge& a, const Edge& b) { return a.i < b.i || (a.i == b.i && a.j < b.j); });
    std::vector<int64_t> parent(N);
    for (int64_t i = 0; i < N; ++i) parent[i] = i;
    for (auto& e : all) {
        int64_t a = uf_find(parent, e.i), b = uf_find(parent, e.j);
        if (a != b) {
            if (a < b) parent[b] = a;
            else parent[a] = b;
        }
    }
    for (int64_t i = 0; i < N; ++i) out_labels[i] = uf_find(parent, i);
    const int64_t keep = std::min<int64_t>((int64_t)all.size(), max_edges);
    for (int64_t k = 0; k < keep; ++k) {
        out_ei[k] = all[k].i;
        out_ej[k] = all[k].j;
        out_es[k] = all[k].s;
    }
    *out_nedges = (int64_t)all.size();
    double t2 = now();
    if (phase_secs) {
        phase_secs[0] = t1 - t0;
        phase_secs[1] = t2 - t1;
    }
    for (int t = 0; t < threads; ++t) {
        if (thread_secs) thread_secs[t] = local[t].secs;
        if (thread_pairs) thread_pairs[t] = local[t].pairs;
    }
    return t2 - t0;
}

}  // namespace

extern "C" {

int rf_has_openmp() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int rf_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// schedule: 0 = static, 1 = dynamic, 2 = guided. chunk <= 0 -> runtime default.
// exclude: optional N-byte mask, non-zero rows are skipped (leakage control, §I-C).
// Writes min(M, N) hits sorted best-first; returns kernel wall time in seconds.
double rf_topm(const float* emb, int64_t N, int d, const float* q, int M, int threads, int schedule,
               int chunk, const uint8_t* exclude, int64_t* out_idx, float* out_score, int* out_count) {
    return topm_impl(emb, N, d, q, 1, M, threads, schedule, chunk, exclude, out_idx, out_score,
                     out_count, nullptr, nullptr);
}

// As rf_topm, plus per-thread busy time (seconds inside the worksharing loop) and rows taken.
// thread_secs / thread_rows must hold `threads` entries.
double rf_topm_stats(const float* emb, int64_t N, int d, const float* q, int M, int threads,
                     int schedule, int chunk, const uint8_t* exclude, int64_t* out_idx,
                     float* out_score, int* out_count, double* thread_secs, int64_t* thread_rows) {
    return topm_impl(emb, N, d, q, 1, M, threads, schedule, chunk, exclude, out_idx, out_score,
                     out_count, thread_secs, thread_rows);
}

// Q queries (row-major Q x d) in one pass. out_idx / out_score are Q x M, out_count is Q.
double rf_topm_batch(const float* emb, int64_t N, int d, const float* Qm, int Q, int M, int threads,
                     int schedule, int chunk, const uint8_t* exclude, int64_t* out_idx,
                     float* out_score, int* out_count) {
    return topm_impl(emb, N, d, Qm, Q, M, threads, schedule, chunk, exclude, out_idx, out_score,
                     out_count, nullptr, nullptr);
}

// Leader clustering of a best-first candidate pool V (m x d). Writes up to K leader rows,
// the number of leaders, and assign[m] (cluster number or -1). phase_secs[2]: leader phase,
// assignment phase. block <= 0 -> 256.
double rf_cluster(const float* V, int64_t m, int d, float tau, int K, int threads, int schedule, int chunk,
                  int block, int64_t* out_leaders, int* out_nleaders, int64_t* out_assign, double* phase_secs) {
    return cluster_impl(V, m, d, tau, K, threads, schedule, chunk, block, out_leaders, out_nleaders,
                        out_assign, phase_secs);
}

// All pairs i < j of E (N x d) with cos > tau. labels[N] = smallest row index of the connected
// component. Up to max_edges edges (sorted by i, j) are written; *out_nedges is the total.
// phase_secs[2]: parallel pair scan, serial merge + union-find. thread_secs / thread_pairs hold
// `threads` entries (busy time and pairs evaluated per thread).
double rf_dup_pairs(const float* E, int64_t N, int d, float tau, int threads, int schedule, int chunk,
                    int64_t* out_labels, int64_t* out_ei, int64_t* out_ej, float* out_es, int64_t max_edges,
                    int64_t* out_nedges, double* phase_secs, double* thread_secs, int64_t* thread_pairs) {
    return dup_pairs_impl(E, N, d, tau, threads, schedule, chunk, out_labels, out_ei, out_ej, out_es,
                          max_edges, out_nedges, phase_secs, thread_secs, thread_pairs);
}

}  // extern "C"
