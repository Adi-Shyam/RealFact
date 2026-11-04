# RealFact: OpenMP-Accelerated News Verification

IT 301 Parallel Computing, NITK Surathkal. Team: Adithya Shyam D (241IT005), Krishna (241IT039), P Suman (241IT049), Saurabh (241IT072).

Reference paper: Guo, Schlichtkrull and Vlachos, *A Survey on Automated Fact-Checking*, TACL 10 (2022), 178–206.

This is a working prototype of the system described in `RealFact_1.pdf`, `realfact_methodology.pdf` and `RealFact_Methodology_Presentation.pptx`.

## Run

```bash
./run.sh            # builds the engine if needed, starts http://127.0.0.1:5050
```

On first start it downloads BGE-small-en-v1.5 and the NLI model from Hugging Face, ingests the live RSS feeds, and then refreshes them every 20 minutes. Set `REALFACT_NO_REFRESH=1` to turn the refresh off.

Setup from scratch on macOS (Apple clang ships no OpenMP runtime, so a prebuilt LLVM libomp goes in `third_party/libomp`):

```bash
python3 -m venv .venv && .venv/bin/pip install torch sentence-transformers transformers flask feedparser numpy sentencepiece protobuf
mkdir -p third_party/libomp && curl -L https://mac.r-project.org/openmp/openmp-19.1.5-darwin20-Release.tar.gz | tar -xz -C third_party/libomp --strip-components=2
make
```

On Linux with GCC: `make OMP_CFLAGS=-fopenmp OMP_LIBS=-fopenmp EXT=so SHARED=-shared`.

## Layout

| Path | What it is |
|---|---|
| `engine/realfact_engine.cpp` | Exact top-M scan. `#pragma omp parallel for schedule(runtime)`, cache-line-padded thread-local heaps, deterministic tie-break, leakage `exclude` mask. Built twice: `build/librealfact_omp` and `build/librealfact_serial` (no OpenMP, the T₁ reference). |
| `backend/ingest.py` | Phase 1: RSS → clean → English filter → exact-duplicate hash on (domain, headline) → BGE embed → append to the contiguous N×384 matrix. |
| `backend/pipeline.py` | Phase 2: embed claim → OpenMP top-M → retrieve-wide clustering at τ → top-K clusters → NLI per representative → `w_c = 1 + λ log₂ n_c` → S, D → four-way verdict. Also computes the ablations (λ=0, naive counting, filter off, truncate-then-cluster) and the syndication stress test. |
| `backend/bench.py` | Benchmarking arm: schedules × threads × corpus size, 2 warm-ups + 11 runs, median/IQR/p95, speedup vs the serial build, efficiency, throughput, CPU utilisation, bit-exact ID check, exact Wilcoxon signed-rank. Also runs from the CLI: `.venv/bin/python -m backend.bench`. |
| `eval/run_eval.py` | FEVER / LIAR / FakeNewsNet harness with the label mapping from §I-B and leakage control. |
| `web/` | The UI: Verify, What we learned (survey), Existing vs RealFact, Methodology, Parallel benchmark, Live corpus, Evaluation. |

## Web search and verification history

Both features were ported from the team's alternate prototype.

- **Search the web first** (Verify tab, live news, on by default): before verifying, RealFact turns the claim into a keyword query and fetches up to 12 Google News RSS search results and the lead paragraphs of up to 3 Wikipedia pages (official MediaWiki API). New items are de-duplicated on (domain, headline), embedded and added to the live corpus with `origin = "web"`. They then go through the same OpenMP scan, duplicate clustering, NLI and independence weighting as RSS articles. Google News often returns several outlets carrying one wire story, and those collapse into one cluster. Typical cost is 2–3 s. Search-result HTML scraping is deliberately not used, and the Google News feed is meant for personal, non-commercial use, which covers this academic demo. Code: `backend/websearch.py`.
- **History tab**: every verdict (live or FEVER) is saved with its full result in `data/history.sqlite` (git-ignored). You can search past checks, open one exactly as it was shown, re-run it against today's corpus, or delete it. Syndication stress-test runs aren't saved. API: `GET /api/history?q=`, `GET /api/history/<id>`, `DELETE /api/history/<id>`, `DELETE /api/history`. Code: `backend/history.py`.

## FEVER demo mode

The Verify tab has an **Evidence corpus** switch. **FEVER demo (Wikipedia 2017)** verifies a fixed list of FEVER dev claims against a 25,324-page subset of the FEVER Wikipedia dump: the gold pages for the claims, 2,723 pages cited by other FEVER claims (hard distractors), and 22,476 random pages. Only the listed claims can be verified, and each result is shown next to FEVER's gold label and annotated evidence.

```bash
# one-time: download the FEVER data (4 MB + 1.7 GB) into data/datasets/fever/
curl -L -o data/datasets/fever/shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
curl -L -o data/datasets/fever/wiki-pages.zip https://fever.ai/download/fever/wiki-pages.zip
.venv/bin/python -m scripts.build_fever_demo          # ~2 minutes: scan the dump, embed on the Apple GPU
.venv/bin/python -m eval.run_eval --dataset fever-demo
```

- **Claim selection:** 60 SUPPORTS, 59 REFUTES and 60 NOT ENOUGH INFO claims. SUPPORTS and REFUTES claims are kept only when their evidence is one sentence within the first 4 sentences of one page, because the index embeds each page's lead.
- **NLI model:** the FEVER corpus uses `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, which is trained on FEVER's train split; the demo claims come from dev. Live news keeps the smaller `nli-deberta-v3-xsmall`.
- **Results on the 179 demo claims:** 74.3% accuracy (macro F1 0.741), versus 58.7% with the xsmall model. The gold page was retrieved within the top cluster for 89.1% of verifiable claims, and within the top 5 for 95.0%.
- **Scope:** this is a demo slice selected to suit a lead-only index, so it is not comparable to published full-FEVER scores.

## Second parallel kernel: all-pairs syndication detection and clustering

The retrieval scan is memory-bandwidth bound: each 1.5 KB row is read once for one dot product. Two more stages now run in the C++ engine with OpenMP (`engine/realfact_engine.cpp`):

- **`rf_dup_pairs`, corpus-wide syndication map** (Live corpus tab → *Find syndicated stories*). Every pair of articles is compared once (N(N−1)/2 pairs, 6.6M for the live corpus), pairs with cos > τ are linked, and union-find groups them.
  - Rows are processed four at a time against each partner row (`dot4` register blocking), so the kernel is compute bound. It runs at about 48 GFLOP/s serial and about 270 GFLOP/s on 11 threads.
  - Edges go to per-thread buffers with no locks, then are sorted, so output is identical to the serial build.
  - The iteration space is a triangle, which makes the schedule matter. At 11 threads on the live corpus: static ×5.0 (busiest thread does 1.9× the average work), static,1 ×5.6, guided ×6.7, dynamic ×6.9.
  - The ceiling is the hardware, not Amdahl's law (the serial merge and union-find take under 1% of the time). Five performance-core threads give ×4.6. The 6 efficiency cores run at about 40% of a performance core's speed, so 11 threads are worth about 7 performance cores.
  - On today's corpus, 13% of articles have a near-copy, and the largest group is one story carried as 23 articles across 10 outlets.
- **`rf_cluster`, query-time leader clustering** (§II-C) now runs in the engine instead of NumPy.
  - Choosing leaders is sequential by definition. The kernel uses block speculation: similarities to existing leaders are computed in parallel, then a short sequential pass compares each candidate only with leaders founded inside its block. Output is exactly the sequential algorithm's.
  - On a 9K-candidate pool at K = 10: ×5.8 vs serial, ×3.2 vs NumPy. At K = 1000: ×2.5 vs NumPy. At K = 100, NumPy is faster (×0.75), because its assignment step runs on Apple's Accelerate BLAS, which uses the M3's matrix unit.
  - On the live corpus the pool above the relevance floor is usually a few dozen articles, so this stage takes microseconds and parallel regions are skipped below 4,096 dot products.
- The **Parallel benchmark** tab has a *Run kernel-2 benchmark* button (schedules × threads on the live corpus and 10K synthetic vectors, plus the clustering sweep). Results are saved to `data/allpairs_last.json`. API: `POST /api/syndication`, `POST /api/syndication/bench`, `GET /api/syndication/bench/last`.

## Correctness and performance work

- `tests/test_correctness.py` holds 14 checks: engine vs NumPy across 45 configurations, bit-exactness vs the serial build, exact ties, edge cases, the leakage mask, batched == single, per-thread row accounting, the aggregation equations, clustering, the C++ clustering kernel vs the NumPy reference (162 configurations), the all-pairs kernel vs a NumPy Gram matrix and BFS components, exact Wilcoxon vs enumeration, ingestion and label maps. Run it with `.venv/bin/python -m tests.test_correctness` or the **Run self-test** button on the Benchmark tab.
- **Kernel v2:** a 16-way ILP dot product (four independent vector accumulators, explicit summation order, still bit-exact) makes the serial scan 2.6× faster and reaches the ~96 GB/s memory ceiling with 4–8 threads.
- **New engine entry points:** `rf_topm_stats` returns per-thread busy time and row counts (load balance). `rf_topm_batch` scores Q queries per row load, register-blocked 4 at a time (roofline; 5× throughput at M = 100).
- **New benchmark analyses:** Karp–Flatt serial fraction, per-thread load balance, chunk-size sweep, weak scaling, and batched throughput with a roofline plot. The v1 results are kept in `data/bench_v1_baseline.json` for comparison.
- **Verification defaults:** an adaptive thread count `min(cores, 8, ceil(N/400))`, and the guided schedule.

## Notes and deviations

- **Relevance floor (`min_rel`, default 0.68).** NLI models label off-topic premises as "contradiction" (we checked this with two models). Candidates below the floor are not treated as evidence. It is adjustable in the UI.
- **Heap merge.** The merge uses concatenate + `nth_element` (O(PM + M log M)) instead of a heap merge (O(PM log M)). The output is identical and the serial fraction is smaller.
- **Thread pinning.** macOS exposes no thread-affinity API, so `OMP_PROC_BIND` / `OMP_PLACES` have no effect. The M3 Pro mixes performance and efficiency cores, which is why `guided` can beat `static` here. The benchmark tab explains this.
- **RSS gives headline + lead only.** We never scrape article pages. BGE truncates at 512 tokens anyway, so an article is represented by its lead (methodology §I-C).
- **Benchmarks are not bundled.** FEVER, LIAR and FakeNewsNet must be downloaded to `data/datasets/`. FEVER-style accuracy also needs the FEVER Wikipedia dump indexed as the evidence pool.

## Report

`report/realfact_report.pdf` is built from `report/realfact_report.tex` with `make report`. This uses the bundled Tectonic engine in `third_party/tectonic`, which downloads LaTeX packages on its first run.
