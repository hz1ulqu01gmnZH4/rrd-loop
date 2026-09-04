"""Duplicate-avoidance gate for research requirements (cheap -> expensive).

L1  string dedupe on normalized title            (state.py, runs in cycle)
L2  wiki registry grep, token overlap, no LLM    (this file)
L3  knowledge-graph probe (kg q --domain rrd-loop) (this file, subprocess)
L4  LLM gate verifier, independent call, temp 0  (this file)

Policy: conservative. When in doubt, DUPLICATE.
On gate PASS the requirement is registered back into the wiki registry and
kg so later sessions/restarts know it.
"""
import os
import re
import subprocess
import time

KG = os.path.expanduser("~/kg/kg.py")
REGISTRY = os.path.expanduser("~/wiki/projects/rrd-loop-registry.md")

REGISTRY_HEADER = """---
title: "rrd-loop research requirement registry"
category: project
created: %s
updated: %s
sources:
  - type: note
    note: "auto-maintained by ~/rrd-loop (gate.py) - one line per evaluated research requirement; grep-able dedupe layer, do not hand-edit the table"
provenance: extracted
tags:
  - rrd-loop
  - registry
---

# rrd-loop research requirement registry

One line per evaluated research requirement, appended by the loop. This table
is dedupe layer L2: new candidates are token-matched against the **title**
column before the LLM gate (layer L4) decides. `dup of ...` marks entries
the gate flagged.

| ts | title | verdict | score | rrd | source |
|----|-------|---------|-------|-----|--------|
"""


def _ensure_registry():
    if not os.path.exists(REGISTRY):
        os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
        t = time.strftime("%Y-%m-%d")
        with open(REGISTRY, "w") as f:
            f.write(REGISTRY_HEADER % (t, t))


def _tokens(s):
    return {t for t in re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split()
            if len(t) > 3}


# ---------------------------------------------------------------------------
# L2: wiki registry grep (no LLM)
# ---------------------------------------------------------------------------
def registry_candidates(req, min_overlap=2):
    cand_t = _tokens(req.get("title", "")) | _tokens(req.get("one_liner", ""))
    out, seen = [], set()
    if os.path.exists(REGISTRY):
        for line in open(REGISTRY):
            m = re.match(r"^\|\s*[\d\- :]+\|\s*([^|]+)\|", line)
            if not m:
                continue
            title = m.group(1).strip()
            if title in seen:
                continue
            overlap = cand_t & _tokens(title)
            if len(overlap) >= min_overlap:
                seen.add(title)
                out.append({"title": title, "one_liner": "", "found_in": "wiki-registry",
                            "match": sorted(overlap)[:8]})
    return out[:10]


# ---------------------------------------------------------------------------
# L3: knowledge-graph probe (persistent across restarts/sessions)
# ---------------------------------------------------------------------------
def kg_candidates(req, limit=5):
    terms = (req.get("title", "") + " " + (req.get("one_liner") or ""))[:120]
    try:
        r = subprocess.run(
            ["python3", KG, "q", terms, "--domain", "rrd-loop",
             "--limit", str(limit), "--expand", "0"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    lines = r.stdout.splitlines()
    out = []
    for i, line in enumerate(lines):
        if not re.match(r"^n\d+\s", line):
            continue
        content = lines[i + 1].strip() if i + 1 < len(lines) else ""
        m = re.match(r"Requirement\s+(.+?):", content)
        out.append({"title": (m.group(1).strip() if m else content[:60]),
                    "one_liner": "", "found_in": "kg", "content": content[:300]})
    return out[:limit]


def candidates(req, state_reqs, cfg=None):
    """Merge L2 + L3 + state (L1 passed already). Dedupe by normalized title."""
    all_c = registry_candidates(req) + kg_candidates(req)
    for o in state_reqs:
        all_c.append({k: o.get(k, "") for k in ("title", "one_liner", "gap")} |
                     {"found_in": "state"})
    seen, out = set(), []
    for c in all_c:
        k = re.sub(r"[^a-z0-9]+", "", c.get("title", "").lower())
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out[:12]


# ---------------------------------------------------------------------------
# L4: LLM gate verifier (independent of the synthesizer: producer != grader)
# ---------------------------------------------------------------------------
SYSTEM_GATE = """You are an independent duplicate-verification gate in a research-requirement pipeline.
You did NOT generate the candidate below. Your only job: decide whether it is already a known requirement.
Same requirement = same research gap AND same core research question, even if wording, framing, model family or evaluation differs
(e.g. 'self-reflection loop' vs 'self-reflective agent', 'verification' vs 'verifier').
A different gap, or a genuinely different research question that merely touches the same topic, is NOT a duplicate.
When in doubt, choose DUPLICATE.
Output schema: {"duplicate": true|false, "duplicate_of": "<exact title of the existing entry matched, or null>", "confidence": 0.0-1.0, "reason": "<one line>"}"""


def llm_gate(llm, req, cands, cfg):
    """Returns {duplicate, duplicate_of, confidence, reason, error?}."""
    if not cands:
        return {"duplicate": False, "duplicate_of": None, "confidence": 1.0,
                "reason": "no known entries to compare"}
    known = "\n".join(
        f'{i}. {c["title"]} | found in {c["found_in"]}'
        + (f' | {c["one_liner"][:120]}' if c.get("one_liner") else "")
        for i, c in enumerate(cands, 1))
    user = f"""Candidate research requirement:
Title: {req.get("title")}
One-liner: {req.get("one_liner")}
Gap: {req.get("gap", "")}
Why now: {req.get("why_now", "")}

Known requirements (wiki registry + knowledge graph + state):
{known}"""
    try:
        out = llm.chat_json(SYSTEM_GATE, user, temperature=0.0)
    except Exception as e:
        # gate unavailable -> do not write (conservative); retry next cycle
        return {"duplicate": False, "duplicate_of": None, "confidence": 0.0,
                "reason": "gate error", "error": str(e)}
    out.setdefault("duplicate", False)
    out.setdefault("duplicate_of", None)
    out.setdefault("confidence", 0.5)
    out.setdefault("reason", "")
    return out


# ---------------------------------------------------------------------------
# Registration: pass -> back into wiki registry + kg (make it a known known)
# ---------------------------------------------------------------------------
def registry_line(item, req, ev, rrd_path, note=""):
    _ensure_registry()
    title = re.sub(r"\|", "/", (req.get("title") or "untitled"))
    rrd = rrd_path or ("—" + (" " + note if note else ""))
    line = (f'| {time.strftime("%Y-%m-%d %H:%M")} | {title} | {ev.get("verdict")} '
            f'| {ev.get("score")} | {rrd} | [{(item.get("title") or "")[:60]}]({item.get("url", "")}) |\n')
    with open(REGISTRY, "a") as f:
        f.write(line)


def kg_register(item, req, ev, rrd_path):
    """Record a written RRD's requirement as a kk fact (only RRD'd reqs)."""
    if not rrd_path:
        return
    content = (f"Requirement {req.get('title')}: {req.get('one_liner','')}; "
               f"verdict {ev.get('verdict')} {ev.get('score')}; RRD {rrd_path}; "
               f"from paper: {(item.get('title') or '')[:80]}")
    try:
        subprocess.run(
            ["python3", KG, "add", content, "--type", "fact", "--quadrant", "kk",
             "--domain", "rrd-loop", "--entity", req.get("title", "requirement")[:60],
             "--confidence", "0.8", "--source", "rrd-loop auto"],
            capture_output=True, text=True, timeout=30, check=False)
    except Exception:
        pass
