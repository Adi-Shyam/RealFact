"""BGE-small-en-v1.5 sentence embeddings and a 3-way NLI cross-encoder. Both loaded lazily, once."""
import os
import threading

import numpy as np

from .config import EMBED_MODEL, NLI_MODEL, BGE_QUERY_PREFIX

_lock = threading.Lock()
_embedder = None
_nli = {}  # model name -> (CrossEncoder, labels)


def embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
            _embedder.max_seq_length = 512
    return _embedder


def embed_articles(texts, batch_size=64):
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    v = embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
    return v.astype(np.float32)


def embed_claim(claim):
    # BGE recommends an instruction prefix on short queries for retrieval.
    v = embedder().encode([BGE_QUERY_PREFIX + claim], normalize_embeddings=True, convert_to_numpy=True)
    return v[0].astype(np.float32)


NLI_DEVICE = os.environ.get("REALFACT_NLI_DEVICE", "cpu")  # "mps" uses the Apple GPU


def nli_model(name=NLI_MODEL):
    with _lock:
        if name not in _nli:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(name, device=NLI_DEVICE)
            id2label = model.config.id2label if hasattr(model, "config") else model.model.config.id2label
            _nli[name] = (model, [id2label[i].lower() for i in range(len(id2label))])
    return _nli[name]


def classify(claim, premises, model_name=NLI_MODEL):
    """NLI(premise=evidence, hypothesis=claim) -> list of (stance sigma in {+1,-1,0}, label, p, probs)."""
    if not premises:
        return []
    return classify_pairs([(claim, p) for p in premises], model_name)


def classify_pairs(pairs, model_name=NLI_MODEL):
    """pairs: list of (claim, premise). NLI(premise=evidence, hypothesis=claim) for each pair."""
    if not pairs:
        return []
    model, labels = nli_model(model_name)
    logits = model.predict([(p, c) for c, p in pairs], convert_to_numpy=True, show_progress_bar=False, batch_size=8)
    logits = np.atleast_2d(logits)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    out = []
    for row in probs:
        j = int(row.argmax())
        lab = labels[j]
        if lab.startswith("entail"):
            sigma, name = 1, "SUPPORT"
        elif lab.startswith("contra"):
            sigma, name = -1, "CONTRADICT"
        else:
            sigma, name = 0, "NEUTRAL"
        named = {}
        for k, l in enumerate(labels):
            key = "SUPPORT" if l.startswith("entail") else "CONTRADICT" if l.startswith("contra") else "NEUTRAL"
            named[key] = float(row[k])
        out.append((sigma, name, float(row[j]), named))
    return out
