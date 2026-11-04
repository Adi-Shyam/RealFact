"""RealFact web server.  Run:  .venv/bin/python -m backend.app  ->  http://127.0.0.1:5050"""
import json
import os
import threading
import time
import traceback
import uuid

from flask import Flask, jsonify, request, send_from_directory

from . import bench, pipeline, websearch
from .config import (CORPUS_DEFAULTS, DATA_DIR, DEFAULTS, EMBED_MODEL, FEEDS, FEVER_CLAIMS_PATH, FEVER_DB_PATH,
                     FEVER_EMB_PATH, NLI_MODEL, ROOT)
from .engine import omp_engine
from .history import History
from .ingest import ingest
from .store import Store

WEB = os.path.join(ROOT, "web")
BENCH_PATH = os.path.join(DATA_DIR, "bench_last.json")
EVAL_PATH = os.path.join(DATA_DIR, "eval_results.json")
REFRESH_SECONDS = int(os.environ.get("REALFACT_REFRESH", 20 * 60))

app = Flask(__name__, static_folder=None)


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp

store = Store()
history = History()
jobs = {}
_fever = {"store": None, "claims": None, "mtime": None}


def fever():
    """The FEVER demo corpus, loaded lazily and reloaded if scripts/build_fever_demo.py rebuilt it."""
    if not os.path.exists(FEVER_CLAIMS_PATH):
        return None
    mtime = os.path.getmtime(FEVER_CLAIMS_PATH)
    if _fever["mtime"] != mtime:
        with open(FEVER_CLAIMS_PATH) as f:
            data = json.load(f)
        _fever.update(store=Store(FEVER_DB_PATH, FEVER_EMB_PATH), claims=data, mtime=mtime,
                      by_text={c["claim"]: c for c in data["claims"]})
    return _fever


def fever_gold(result, gold):
    """Attach the FEVER gold label and say whether retrieval found the gold page, and at which cluster rank."""
    pages = set(gold["gold_pages"])
    hit_rank = None
    for e in result["evidence"]:
        urls = {m["url"].rsplit("/", 1)[-1] for m in e["members"]}
        ids = {u.replace("(", "-LRB-").replace(")", "-RRB-") for u in urls}
        if ids & pages:
            hit_rank = e["rank"]
            break
    mapped = {"Supported": "SUPPORTS", "Contradicted": "REFUTES",
              "Unverified": "NOT ENOUGH INFO", "Conflicting Evidence": "NOT ENOUGH INFO"}[result["verdict"]]
    return {"label": gold["label"], "evidence": gold["evidence"], "gold_pages": gold["gold_pages"],
            "gold_retrieved_rank": hit_rank, "system_label": mapped, "correct": mapped == gold["label"]}
ingest_lock = threading.Lock()


def start_job(kind, fn):
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "state": "running", "progress": 0.0, "message": "starting", "result": None}
    jobs[jid] = job

    def progress(frac, msg):
        job["progress"], job["message"] = frac, msg

    def runner():
        try:
            job["result"] = fn(progress)
            job["state"] = "done"
            job["progress"] = 1.0
        except Exception as e:
            traceback.print_exc()
            job["state"], job["message"] = "error", str(e)

    threading.Thread(target=runner, daemon=True).start()
    return job


def run_ingest(progress=None):
    with ingest_lock:
        if progress:
            progress(0.1, f"fetching {len(FEEDS)} feeds")
        return ingest(store)


def refresher():
    while True:
        try:
            run_ingest()
        except Exception:
            traceback.print_exc()
        time.sleep(REFRESH_SECONDS)


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/web/<path:p>")
def web(p):
    return send_from_directory(WEB, p)


@app.get("/api/status")
def status():
    eng = omp_engine()
    fv = fever()
    return jsonify({
        "corpus_size": store.size, "defaults": DEFAULTS, "corpus_defaults": CORPUS_DEFAULTS,
        "fever": {"available": bool(fv), "size": fv["store"].size if fv else 0,
                  "meta": fv["claims"]["meta"] if fv else None}, "embed_model": EMBED_MODEL, "nli_model": NLI_MODEL,
        "openmp": eng.has_openmp, "max_threads": eng.max_threads, "feeds": len(FEEDS),
        "ingesting": ingest_lock.locked(), "refresh_minutes": REFRESH_SECONDS / 60,
        "platform": bench.platform_info(),
    })


@app.post("/api/verify")
def verify():
    body = request.get_json(force=True)
    claim = (body.get("claim") or "").strip()
    corpus = body.get("corpus", "live")
    if not claim:
        return jsonify({"error": "Enter a claim to verify."}), 400
    try:
        if corpus == "fever":
            fv = fever()
            if not fv:
                return jsonify({"error": "The FEVER demo corpus is not built. Run: .venv/bin/python -m scripts.build_fever_demo"}), 400
            gold = fv["by_text"].get(claim)
            if gold is None:
                return jsonify({"error": "The FEVER demo only verifies the listed claims. Pick one from the list."}), 400
            res = pipeline.verify(fv["store"], claim, body.get("params"), CORPUS_DEFAULTS["fever"])
            res["corpus"] = "fever"
            res["fever"] = fever_gold(res, gold)
            res["history_id"] = history.add(res)
            return jsonify(res)
        web = None
        if body.get("web"):
            web = websearch.fetch_for_claim(store, claim)
        res = pipeline.verify(store, claim, body.get("params"), CORPUS_DEFAULTS["live"])
        res["corpus"] = "live"
        if web:
            res["web"] = web
            res["timings_ms"]["web_search"] = web["seconds"] * 1e3
        res["history_id"] = history.add(res)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/history")
def history_list():
    return jsonify(history.list(request.args.get("q", "").strip(), request.args.get("limit", 200)))


@app.get("/api/history/<int:hid>")
def history_get(hid):
    h = history.get(hid)
    return (jsonify(h), 200) if h else (jsonify({"error": "no such history entry"}), 404)


@app.delete("/api/history/<int:hid>")
def history_delete(hid):
    return jsonify({"deleted": history.delete(hid)})


@app.delete("/api/history")
def history_clear():
    return jsonify({"deleted": history.delete()})


@app.get("/api/fever/claims")
def fever_claims():
    fv = fever()
    if not fv:
        return jsonify(None)
    return jsonify(fv["claims"])


@app.post("/api/ingest")
def ingest_now():
    return jsonify(start_job("ingest", run_ingest))


@app.get("/api/corpus")
def corpus():
    return jsonify(store.stats())


@app.post("/api/bench")
def bench_start():
    b = request.get_json(force=True) or {}
    sizes = [s if s == "live" else int(s) for s in b.get("sizes", ["live", 10000, 50000, 100000])]
    threads = [int(t) for t in b.get("threads", [1, 2, 4, 8, 16])]
    chunks = [int(c) for c in b.get("chunks", [0])]
    reps = int(b.get("reps", 11))
    M = int(b.get("M", 10000))

    def fn(progress):
        progress(0.0, "waiting for ingestion to finish" if ingest_lock.locked() else "starting")
        with ingest_lock:  # never time the kernel while feeds are being embedded
            return _bench(progress)

    def _bench(progress):
        res = bench.run(sizes, threads, chunks=chunks, reps=reps, M=M, real_corpus=store.emb, progress=progress)
        res["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(BENCH_PATH, "w") as f:
            json.dump(res, f)
        return res

    return jsonify(start_job("bench", fn))


@app.get("/api/bench/last")
def bench_last():
    if not os.path.exists(BENCH_PATH):
        return jsonify(None)
    with open(BENCH_PATH) as f:
        return app.response_class(f.read(), mimetype="application/json")


@app.get("/api/bench/baseline")
def bench_baseline():
    path = os.path.join(DATA_DIR, "bench_v1_baseline.json")
    if not os.path.exists(path):
        return jsonify(None)
    with open(path) as f:
        return app.response_class(f.read(), mimetype="application/json")


@app.post("/api/selftest")
def selftest():
    from tests.test_correctness import run_all
    with ingest_lock:
        res = run_all()
    for r in res:
        r.pop("trace", None) if r["ok"] else None
    return jsonify({"results": res, "passed": sum(r["ok"] for r in res), "total": len(res)})


@app.get("/api/eval/last")
def eval_last():
    if not os.path.exists(EVAL_PATH):
        return jsonify(None)
    with open(EVAL_PATH) as f:
        return app.response_class(f.read(), mimetype="application/json")


@app.get("/api/jobs/<jid>")
def job(jid):
    j = jobs.get(jid)
    return (jsonify(j), 200) if j else (jsonify({"error": "no such job"}), 404)


def warm_models():
    from . import models
    models.embed_claim("warm up")
    models.classify("warm up", ["warm up"])
    if fever():
        models.classify("warm up", ["warm up"], CORPUS_DEFAULTS["fever"]["nli_model"])


def main():
    threading.Thread(target=warm_models, daemon=True).start()
    if os.environ.get("REALFACT_NO_REFRESH") != "1":
        threading.Thread(target=refresher, daemon=True).start()
    port = int(os.environ.get("PORT", 5050))
    print(f"RealFact on http://127.0.0.1:{port}  (corpus: {store.size} articles)")
    app.run(host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    main()
