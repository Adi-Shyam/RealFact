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

}  // extern "C"
