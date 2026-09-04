"""Keyless paper fetching: arXiv API (Atom), driven by a research FIELD.

The field string (e.g. "ai agent self improvement") is expanded into a few
disjunctive arXiv queries (whole-phrase + 2-grams + hyphenated key pair,
each ANDed with the configured categories). Explicit overrides:
`arxiv.queries` in config.json.

Item schema (news.py-analogous):
  {arxiv_id, title, url, published, authors, category, abstract, source}
"""
import re
from datetime import datetime, timedelta, timezone

import requests

API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"


def _stop(field):
    words = [w for w in re.sub(r"[^a-z0-9 ]+", " ", field.lower()).split() if len(w) > 2]
    return words


def derive_queries(field, categories, max_queries=4):
    """Field string -> arXiv search_query strings (disjunctive, category-bounded)."""
    cats = " OR ".join(f"cat:{c}" for c in categories)
    words = _stop(field)
    phrases = []
    if len(" ".join(words)) > 4:
        phrases.append(" ".join(words))                      # whole field
    pairs = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    pairs += [f"{words[i]}-{words[i+1]}" for i in range(len(words) - 1)]  # hyphenated
    for p in pairs:
        if p not in phrases:
            phrases.append(p)
    phrases = phrases[:max_queries] or [field]
    out = []
    for p in phrases:
        out.append(f'(ti:"{p}" OR abs:"{p}") AND ({cats})')
    return out


def _arxiv_id(raw):
    m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})(v\d+)?$", raw or "")
    return m.group(1) if m else raw


def search(query, n=15, max_age_days=30):
    """One arXiv search query -> recent papers (newest first)."""
    try:
        r = requests.get(API, params={
            "search_query": query, "start": 0, "max_results": n,
            "sortBy": "submittedDate", "sortOrder": "descending"}, timeout=45)
        r.raise_for_status()
    except Exception:
        return []
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    items = []
    for e in root.iter(f"{ATOM}entry"):
        pub = (e.findtext(f"{ATOM}published") or "")[:10]
        try:
            if datetime.strptime(pub, "%Y-%m-%d").replace(tzinfo=timezone.utc) < cutoff:
                continue
        except ValueError:
            pass
        authors = [a.findtext(f"{ATOM}name") for a in e.iter(f"{ATOM}author")]
        pc = e.find(f"{ARX}primary_category")
        items.append({
            "arxiv_id": _arxiv_id(e.findtext(f"{ATOM}id")),
            "title": re.sub(r"\s+", " ", (e.findtext(f"{ATOM}title") or "")).strip(),
            "url": f"https://arxiv.org/abs/{_arxiv_id(e.findtext(f'{ATOM}id'))}",
            "published": pub,
            "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
            "category": pc.get("term", "?") if pc is not None else "?",
            "abstract": re.sub(r"\s+", " ", (e.findtext(f"{ATOM}summary") or "")).strip(),
        })
    for it in items:
        it["source"] = "arxiv"
    return items[:n]


def fetch_all(cfg):
    """All configured field queries -> flat, arxiv_id-deduped list of papers."""
    acfg = cfg.get("arxiv", {})
    field = cfg.get("field", "")
    cats = acfg.get("categories", ["cs.AI", "cs.LG", "cs.CL", "cs.MA"])
    queries = acfg.get("queries") or derive_queries(field, cats)
    per = acfg.get("per_query", 15)
    age = acfg.get("max_age_days", 30)
    out = []
    for q in queries:
        batch = search(q, per, age)
        for p in batch:
            p["query"] = q
        out += batch
        import time
        time.sleep(2)  # arXiv API politeness
    seen, dedup = set(), []
    for p in out:
        if p["arxiv_id"] and p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            dedup.append(p)
    return dedup


def keyword_search(terms, n=8, max_age_days=365):
    """Saturation / related-work probe: free-form terms across the field's
    categories (also used by the grill stage's fact-finding)."""
    cats = " OR ".join(f"cat:{c}" for c in ["cs.AI", "cs.LG", "cs.CL", "cs.MA"])
    q = f'(ti:"{terms}" OR abs:"{terms}") AND ({cats})'
    return search(q, n=n, max_age_days=max_age_days)
