"""Defaults for every tunable named in the methodology. All are overridable per request from the UI."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "corpus.sqlite")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.f32")

# FEVER demo corpus: a fixed subset of the FEVER 2017 Wikipedia dump plus curated FEVER dev claims
# (built by scripts/build_fever_demo.py).
FEVER_DIR = os.path.join(DATA_DIR, "fever")
FEVER_DB_PATH = os.path.join(FEVER_DIR, "corpus.sqlite")
FEVER_EMB_PATH = os.path.join(FEVER_DIR, "embeddings.f32")
FEVER_CLAIMS_PATH = os.path.join(FEVER_DIR, "demo_claims.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # d = 384, 512-token limit (§I-C)
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"  # SUPPORT / CONTRADICT / NEUTRAL head
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DIM = 384

DEFAULTS = {
    "M": 10000,        # retrieve-wide pool, M >> K (§II-C)
    "K": 10,           # clusters kept for NLI
    "tau": 0.90,       # near-duplicate threshold on cosine similarity
    "lam": 0.5,        # w_c = 1 + lam * log2(n_c)   (Eq. 2)
    "theta": 0.6,      # evidence sufficiency on sum_{sigma!=0} w_c p_c
    "delta": 0.25,     # disagreement band on D
    "min_rel": 0.68,   # relevance floor: candidates below this cosine are not evidence
    "threads": 0,      # 0 -> auto: grain-size and bandwidth-aware (pipeline.auto_threads)
    "schedule": "guided",  # measured best on heterogeneous P/E cores (Parallel benchmark tab)
    "nli_model": NLI_MODEL,
}

# Per-corpus overrides of DEFAULTS. Similarity scales differ between news leads and Wikipedia leads,
# so the relevance floor is calibrated per corpus (see scripts/build_fever_demo.py and the eval).
CORPUS_DEFAULTS = {
    "live": {},
    # FEVER-trained NLI (trained on FEVER's train split; the demo claims come from dev). On the 179 demo
    # claims it scores 74.3% vs 58.7% for the xsmall model, at 167 vs 65 ms per claim.
    "fever": {"min_rel": 0.68, "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"},
}

# Permitted public RSS feeds (headline + lead only; we never scrape article pages).
FEEDS = [
    ("bbc.co.uk", "https://feeds.bbci.co.uk/news/rss.xml"),
    ("bbc.co.uk", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("bbc.co.uk", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("bbc.co.uk", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
    ("bbc.co.uk", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("bbc.co.uk", "https://feeds.bbci.co.uk/news/health/rss.xml"),
    ("theguardian.com", "https://www.theguardian.com/world/rss"),
    ("theguardian.com", "https://www.theguardian.com/science/rss"),
    ("theguardian.com", "https://www.theguardian.com/technology/rss"),
    ("theguardian.com", "https://www.theguardian.com/business/rss"),
    ("theguardian.com", "https://www.theguardian.com/environment/rss"),
    ("npr.org", "https://feeds.npr.org/1001/rss.xml"),
    ("npr.org", "https://feeds.npr.org/1004/rss.xml"),
    ("npr.org", "https://feeds.npr.org/1019/rss.xml"),
    ("npr.org", "https://feeds.npr.org/1128/rss.xml"),
    ("aljazeera.com", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("nytimes.com", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    ("nytimes.com", "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml"),
    ("nytimes.com", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"),
    ("nytimes.com", "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml"),
    ("cbsnews.com", "https://www.cbsnews.com/latest/rss/main"),
    ("cbsnews.com", "https://www.cbsnews.com/latest/rss/world"),
    ("abcnews.go.com", "https://abcnews.go.com/abcnews/topstories"),
    ("abcnews.go.com", "https://abcnews.go.com/abcnews/internationalheadlines"),
    ("dw.com", "https://rss.dw.com/rdf/rss-en-all"),
    ("france24.com", "https://www.france24.com/en/rss"),
    ("skynews.com", "https://feeds.skynews.com/feeds/rss/world.xml"),
    ("cnbc.com", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("cnbc.com", "https://www.cnbc.com/id/100727362/device/rss/rss.html"),
    ("thehindu.com", "https://www.thehindu.com/news/international/feeder/default.rss"),
    ("thehindu.com", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("indianexpress.com", "https://indianexpress.com/section/world/feed/"),
    ("indianexpress.com", "https://indianexpress.com/section/india/feed/"),
    ("timesofindia.com", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("independent.co.uk", "https://www.independent.co.uk/news/world/rss"),
    ("euronews.com", "https://www.euronews.com/rss"),
    ("theverge.com", "https://www.theverge.com/rss/index.xml"),
    ("arstechnica.com", "https://feeds.arstechnica.com/arstechnica/index"),
    ("sciencedaily.com", "https://www.sciencedaily.com/rss/top/science.xml"),
    ("nasa.gov", "https://www.nasa.gov/news-release/feed/"),
    ("who.int", "https://www.who.int/rss-feeds/news-english.xml"),
    ("un.org", "https://news.un.org/feed/subscribe/en/news/all/rss.xml"),
]
