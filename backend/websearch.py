"""On-demand web evidence for a claim (ported from the team's alternate prototype).

Before verifying, fetch fresh evidence for this specific claim and add it to the live corpus, so it passes through
the same OpenMP scan, duplicate clustering, NLI and independence weighting as RSS articles:

- Google News RSS search: recent headlines from many outlets. Its results often include syndicated copies of one
  wire story, which is exactly what the clustering stage is for.
- Wikipedia (official MediaWiki API): lead paragraphs of matching pages, for background facts that news does not repeat.

Items are de-duplicated on (domain, normalised headline) like RSS ingestion, tagged origin = "web", and kept.
Scraping search-result HTML (the alternate prototype's DuckDuckGo path) is deliberately not used.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import feedparser

from .ingest import UA, _published, clean, dedup_key, is_english
from .models import embed_articles

LEAD_RE = re.compile(r"^(is it true that|fact check:?|verify( that| if)?:?|check if|tell me if|breaking:?)\s*", re.I)
AUX_RE = re.compile(r"^(is|are|was|were|has|have|had|did|does|do|can|could|will|would|should)\s+", re.I)
PUNCT_RE = re.compile(r"[?!\"“”‘’;:,()]")
SOURCE_SUFFIX_RE = re.compile(r"\s+[-–|]\s+[^-–|]{2,60}$")


def search_query(claim):
    """Keyword query for search engines: strip fact-check prefixes, a leading auxiliary verb and punctuation."""
    q = AUX_RE.sub("", LEAD_RE.sub("", claim.strip()))
    q = PUNCT_RE.sub(" ", q).replace("'s ", " ")
    words = [w for w in q.split() if len(w) > 1 or w.isupper() or w.isdigit()]
    return " ".join(words[:12])


def _get(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def google_news(query, limit=12):
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    out = []
    for e in feedparser.parse(_get(url)).entries[:limit]:
        src = e.get("source") or {}
        domain = urllib.parse.urlparse(src.get("href", "")).netloc.replace("www.", "") or "news.google.com"
        outlet = src.get("title", "")
        title = clean(e.get("title", ""))
        if outlet and title.endswith(" - " + outlet):  # Google appends " - Outlet" to every headline
            title = title[: -len(outlet) - 3]
        elif not outlet:
            title = SOURCE_SUFFIX_RE.sub("", title)
        summary = clean(e.get("summary", ""))
        if summary.startswith(title):  # the summary is usually just the headline + outlet again
            summary = ""
        out.append({"source": "Google News", "domain": domain, "title": title, "summary": summary,
                    "url": e.get("link", ""), "published_at": _published(e)})
    return out


def wikipedia(query, limit=3):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": limit,
        "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exsentences": 5, "exlimit": limit,
        "inprop": "url", "format": "json", "formatversion": 2})
    pages = json.loads(_get(url)).get("query", {}).get("pages", [])
    pages.sort(key=lambda p: p.get("index", 99))
    return [{"source": "Wikipedia", "domain": "en.wikipedia.org", "title": p["title"],
             "summary": clean(p.get("extract", "")), "url": p.get("fullurl", ""), "published_at": None}
            for p in pages if p.get("extract")]


def fetch_for_claim(store, claim, news_limit=12, wiki_limit=3):
    """Search the web for the claim, add new items to the store, and report what was found."""
    t0 = time.time()
    query = search_query(claim) or claim
    errors, found = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {"Google News": pool.submit(google_news, query, news_limit),
                "Wikipedia": pool.submit(wikipedia, query, wiki_limit)}
        for name, f in futs.items():
            try:
                found += f.result()
            except Exception as e:  # one source failing must not block verification
                errors.append(f"{name}: {e}")

    now = datetime.now(timezone.utc).isoformat()
    items, rows = [], []
    with store.lock:  # RSS ingestion may run concurrently; check keys and append atomically
        known = store.existing_keys()
        for it in found:
            if not it["title"] or not it["url"] or not is_english(it["title"] + " " + it["summary"]):
                continue
            key = dedup_key(it["domain"], it["title"])
            is_new = key not in known
            items.append({**it, "new": is_new})
            if is_new:
                known.add(key)
                rows.append({"dedup_key": key, "domain": it["domain"], "title": it["title"],
                             "summary": it["summary"][:2000], "url": it["url"], "published_at": it["published_at"],
                             "fetched_at": now, "origin": "web"})
        if rows:
            store.add(rows, embed_articles([f"{r['title']}. {r['summary']}" for r in rows]))
    return {"query": query, "found": len(items), "added": len(rows), "items": items, "errors": errors,
            "seconds": round(time.time() - t0, 2)}
