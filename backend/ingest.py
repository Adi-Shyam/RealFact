"""Phase 1: build the evidence index from permitted RSS feeds (methodology §I-C).

clean -> English filter -> exact-duplicate hash on (domain, normalised headline) -> BGE embed -> append.
"""
import hashlib
import html
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import feedparser

from .config import FEEDS
from .models import embed_articles

UA = "RealFact/1.0 (academic news-verification prototype; IT301 NITK)"
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
BOILER_RE = re.compile(r"(continue reading\.*|read more\.*|the post .* appeared first on .*)$", re.I)


def clean(text):
    text = html.unescape(TAG_RE.sub(" ", text or ""))
    text = WS_RE.sub(" ", text).strip()
    return BOILER_RE.sub("", text).strip()


def normalise_headline(t):
    return WS_RE.sub(" ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def dedup_key(domain, title):
    return hashlib.sha1(f"{domain}|{normalise_headline(title)}".encode()).hexdigest()


def is_english(text):
    if not text:
        return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in text)
    letters = sum(c.isalpha() for c in text)
    return letters > 0 and ascii_letters / letters > 0.9


def _fetch(feed):
    domain, url = feed
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=12) as r:
            parsed = feedparser.parse(r.read())
        return domain, parsed.entries, None
    except Exception as e:  # one dead feed must not stop ingestion
        return domain, [], f"{url}: {e}"


def _published(entry):
    for k in ("published_parsed", "updated_parsed"):
        if entry.get(k):
            return datetime(*entry[k][:6], tzinfo=timezone.utc).isoformat()
    return None


def ingest(store, feeds=FEEDS):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_fetch, feeds))

    known = store.existing_keys()
    now = datetime.now(timezone.utc).isoformat()
    rows, seen, exact_dups, errors = [], 0, 0, []
    for domain, entries, err in results:
        if err:
            errors.append(err)
        for e in entries:
            title = clean(e.get("title", ""))
            summary = clean(e.get("summary", "") or e.get("description", ""))
            link = e.get("link", "")
            if not title or not link or not is_english(title + " " + summary):
                continue
            seen += 1
            key = dedup_key(domain, title)
            if key in known:
                exact_dups += 1
                continue
            known.add(key)
            rows.append({
                "dedup_key": key, "domain": domain, "title": title, "summary": summary[:2000],
                "url": link, "published_at": _published(e), "fetched_at": now,
            })

    vectors = embed_articles([f"{r['title']}. {r['summary']}" for r in rows])
    added = store.add(rows, vectors)
    stats = {
        "at": now, "feeds_ok": sum(1 for r in results if not r[2]), "feeds_failed": len(errors),
        "seen": seen, "exact_dups": exact_dups, "added": added, "seconds": round(time.time() - t0, 2),
    }
    store.log_ingest(**stats)
    stats["errors"] = errors
    stats["total"] = store.size
    return stats
