"""Build the FEVER demo corpus: a fixed subset of the FEVER 2017 Wikipedia dump plus curated dev claims.

    .venv/bin/python -m scripts.build_fever_demo                  # defaults: 60 claims per label, 25,000 pages
    .venv/bin/python -m scripts.build_fever_demo --per-label 40 --pages 15000

Inputs (download once into data/datasets/fever/):
    https://fever.ai/download/fever/shared_task_dev.jsonl
    https://fever.ai/download/fever/wiki-pages.zip

Selection, stated so results can be read honestly:
  * SUPPORTS / REFUTES claims are kept only when one evidence set is a single sentence within the first
    LEAD_LINES sentences of one page. The index embeds each page's lead, so this is the slice of FEVER
    a lead index can answer; claims needing deep or multi-page evidence are left out of the demo.
  * NOT ENOUGH INFO claims are sampled at random (they have no evidence pages).
  * Pages = gold pages of the chosen claims + pages cited as evidence by *other* dev claims (topical,
    hard distractors) + a deterministic random sample of the rest of the dump, up to --pages.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend  # noqa: E402,F401  (engine before torch)
from backend.config import DATA_DIR, FEVER_CLAIMS_PATH, FEVER_DB_PATH, FEVER_DIR, FEVER_EMB_PATH  # noqa: E402
from backend.store import Store  # noqa: E402

SRC = os.path.join(DATA_DIR, "datasets", "fever")
LEAD_LINES = 5          # sentences per page that go into the embedded lead
GOLD_MAX_LINE = 3       # demo SUPPORTS/REFUTES evidence must sit in sentences 0..3

PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}",
       "-COLON-": ":", "``": '"', "''": '"', "--": "–"}


def detok(s):
    for k, v in PTB.items():
        s = s.replace(k, v)
    for a, b in ((" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"), (" !", "!"), (" ?", "?"), ("( ", "("),
                 (" )", ")"), (" 's", "'s"), (" n't", "n't"), (' " ', ' "'), ("[ ", "["), (" ]", "]")):
        s = s.replace(a, b)
    return " ".join(s.split())


def title_text(page_id):
    return detok(page_id.replace("_", " "))


def page_lines(lines_field):
    out = {}
    for row in lines_field.split("\n"):
        parts = row.split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
            out[int(parts[0])] = detok(parts[1])
    return out


def pick_claims(dev, per_label, rng):
    by_label = defaultdict(list)
    for c in dev:
        if c["label"] == "NOT ENOUGH INFO":
            by_label["NOT ENOUGH INFO"].append(c)
            continue
        singles = [es[0] for es in c["evidence"] if len(es) == 1 and es[0][2] and es[0][3] is not None]
        lead = [ev for ev in singles if ev[3] <= GOLD_MAX_LINE]
        if lead:
            c["_gold"] = lead[0]
            by_label[c["label"]].append(c)
    chosen, used_pages = [], set()
    for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
        pool = by_label[label][:]
        rng.shuffle(pool)
        n = 0
        for c in pool:
            if n >= per_label:
                break
            page = c.get("_gold", [None, None, None])[2]
            if page and page in used_pages:   # one claim per gold page keeps the demo varied
                continue
            if page:
                used_pages.add(page)
            chosen.append(c)
            n += 1
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-label", type=int, default=60)
    ap.add_argument("--pages", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.time()

    dev = [json.loads(l) for l in open(os.path.join(SRC, "shared_task_dev.jsonl"))]
    claims = pick_claims(dev, a.per_label, rng)
    gold_pages = set()
    for c in claims:
        for es in c["evidence"]:
            for ev in es:
                if ev[2]:
                    gold_pages.add(ev[2])
    chosen_ids = {c["id"] for c in claims}
    hard = set()
    for c in dev:
        if c["id"] in chosen_ids:
            continue
        for es in c["evidence"]:
            for ev in es:
                if ev[2]:
                    hard.add(ev[2])
    hard -= gold_pages
    budget_hard = max(0, (a.pages - len(gold_pages)) * 2 // 3)
    hard = set(rng.sample(sorted(hard), min(len(hard), budget_hard)))
    n_random = max(0, a.pages - len(gold_pages) - len(hard))
    p_random = n_random / 5.3e6
    print(f"claims {len(claims)}  gold pages {len(gold_pages)}  hard distractors {len(hard)}  random target {n_random}")

    def is_random(pid):
        h = int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return h < p_random

    pages = {}
    with zipfile.ZipFile(os.path.join(SRC, "wiki-pages.zip")) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".jsonl") and not n.startswith("__MACOSX"))
        for k, name in enumerate(names):
            with z.open(name) as f:
                for raw in f:
                    # cheap pre-filter on the id before parsing the whole JSON line
                    i = raw.find(b'"id": "') + 7
                    pid = raw[i:raw.find(b'"', i)].decode("utf-8", "replace") if i > 6 else ""
                    if not pid:
                        continue
                    if pid in gold_pages or pid in hard or is_random(pid):
                        d = json.loads(raw)
                        lines = page_lines(d["lines"])
                        if lines:
                            pages[d["id"]] = lines
            if (k + 1) % 10 == 0:
                print(f"  scanned {k + 1}/{len(names)} files, kept {len(pages):,} pages  ({time.time() - t0:.0f}s)", flush=True)

    missing = gold_pages - set(pages)
    claims = [c for c in claims if not any(ev[2] in missing for es in c["evidence"] for ev in es if ev[2])]
    print(f"kept {len(pages):,} pages; dropped {len(missing)} gold pages not in dump; {len(claims)} claims remain")

    # --- build the store --------------------------------------------------------------------------
    os.makedirs(FEVER_DIR, exist_ok=True)
    for p in (FEVER_DB_PATH, FEVER_EMB_PATH):
        if os.path.exists(p):
            os.remove(p)
    store = Store(FEVER_DB_PATH, FEVER_EMB_PATH)
    ids = sorted(pages)
    rows, texts = [], []
    for pid in ids:
        lines = pages[pid]
        lead = " ".join(lines[i] for i in sorted(lines)[:LEAD_LINES])
        title = title_text(pid)
        rows.append({"dedup_key": "fever|" + pid, "domain": "en.wikipedia.org", "title": title, "summary": lead,
                     "url": "https://en.wikipedia.org/wiki/" + pid.replace("-LRB-", "(").replace("-RRB-", ")"),
                     "published_at": "2017-06-01T00:00:00+00:00", "fetched_at": "FEVER 2017 dump", "origin": "fever"})
        texts.append(f"{title}. {lead}")

    from sentence_transformers import SentenceTransformer
    from backend.config import EMBED_MODEL
    model = SentenceTransformer(EMBED_MODEL, device=a.device)
    model.max_seq_length = 512
    t1 = time.time()
    vecs = model.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True).astype(np.float32)
    print(f"embedded {len(texts):,} pages in {time.time() - t1:.0f}s on {a.device}")
    store.add(rows, vecs)

    # --- demo claim list with gold evidence for display ---------------------------------------------
    out = []
    for c in claims:
        gold = []
        for es in c["evidence"]:
            for ev in es:
                if ev[2] and ev[2] in pages and ev[3] in pages[ev[2]]:
                    g = {"page": title_text(ev[2]), "page_id": ev[2], "line": ev[3], "sentence": pages[ev[2]][ev[3]]}
                    if g not in gold:
                        gold.append(g)
        out.append({"id": c["id"], "claim": c["claim"], "label": c["label"], "evidence": gold[:3],
                    "gold_pages": sorted({g["page_id"] for g in gold})})
    rng.shuffle(out)
    meta = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "pages": len(ids), "claims": len(out),
            "per_label": {l: sum(c["label"] == l for c in out) for l in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")},
            "gold_pages": len(gold_pages - missing), "hard_distractors": len(hard & set(pages)),
            "random_pages": len(set(pages) - gold_pages - hard), "lead_lines": LEAD_LINES,
            "gold_max_line": GOLD_MAX_LINE, "seed": a.seed, "source": "FEVER shared_task_dev + wiki-pages (2017)"}
    with open(FEVER_CLAIMS_PATH, "w") as f:
        json.dump({"meta": meta, "claims": out}, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
