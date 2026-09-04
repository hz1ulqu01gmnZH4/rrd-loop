"""GRILL stage: adversarial design-tree interrogation of an RRD candidate,
run as a loop: grill -> improve -> grill -> ... (bounded).

Headless adaptation of the grilling agent-skill family (Matt Pocock's skills
set; see Your-AI-Dept/yaid-hackathon skills/grilling, RobMitt/grill-me-skill),
retargeted at RESEARCH REQUIREMENTS. The LLM interrogates the requirement:

  - map it as a decision tree (hypothesis, method, evals/baselines,
    dataset/compute, timeline, threats to validity, kill criteria)
  - work the frontier in rounds; ANSWER from the evidence, SEARCH (bounded
    auto fact-finding via arXiv keyword search + web search), or ASSUME
    (open assumption with a recommended answer)
  - verdict: dead (unresolvable fatal objection) / wounded / survived

grill_loop(): when a grill does not pass, the LLM IMPROVES the requirement
(resolving objections / assumptions within the same core gap), and the
grill runs again. Bounded by grill.max_iterations (default 2 grill passes).
dead even after the last pass -> the RRD is not written.
"""
import json
import os
import re
import subprocess


SYSTEM_GRILL = """You are the GRILL stage: an adversarial design-tree interrogator for a RESEARCH REQUIREMENT.
You are HEADLESS: no human is available to answer you. Resolve every question yourself.
Method:
1. Map the requirement as a DECISION TREE: the decisions the research program silently relies on (falsifiable hypothesis, method, evals + baselines, dataset + compute access, timeline, threats to validity, kill criteria, what 'done' means).
2. FRONTIER = decisions whose prerequisites are settled by the evidence. Work in at most {max_rounds} rounds: answer each frontier question now; questions depending on open answers wait.
3. For each frontier question choose exactly one:
   - ANSWER: resolve it with the provided evidence.
   - SEARCH: it needs a fact not in the evidence (prior work, benchmark, compute number) -> give a search query (put it in "searches").
   - ASSUME: a pure design decision with no objective answer (baseline choice, subset of tasks, metric) -> register it in "open_assumptions" WITH your recommended answer.
4. VERDICT: dead = at least one fatal objection that the evidence and searches cannot resolve (e.g. the core hypothesis contradicts published results, or the gap is already closed); wounded = open assumptions remain but none fatal; survived = no material open objection.
Find facts yourself (SEARCH); be concrete: name the objection, the target decision, and the evidence.
If verdict is "dead", survival must be below 4.
Output schema:
{{"rounds": int, "decisions": ["settled decision"], "objections": [{{"question": "...", "answer_or_assumption": "...", "severity": "fatal|major|minor", "evidence": "..."}}], "searches": ["query"], "open_assumptions": ["assumption (recommended answer)"], "verdict": "dead|wounded|survived", "survival": 0-10, "reason": "one line"}}"""


SYSTEM_IMPROVE = """You are a research program lead revising a RESEARCH REQUIREMENT after an adversarial grill.
Given the requirement and the grill's objections, revise it so the fatal and major objections are addressed.
Rules:
- Keep the SAME core gap. You may sharpen the falsifiable claim, narrow the setting, switch the method direction, tighten evals/baselines, or split scope -- but do not abandon the gap to escape the objection.
- Resolve as many open assumptions as you can by DECIDING them; list each decision in "decided".
- Never claim evidence you do not have. If an objection rests on a fact you cannot establish, list it in "unresolved" instead of hand-waving.
Output schema:
{"title":"...","one_liner":"...","requirement":"...","gap":"...","why_now":"...","audience":"...","changes":["what changed and why, one line each"],"decided":["assumption you decided and its value"],"unresolved":["objections that remain open"]}"""


def _grill_user(req, ev, related, item, field, facts=None):
    txt = f"""Field: {field}

Research requirement:
{json.dumps(req, ensure_ascii=False)[:1000]}

Evaluation: verdict {ev.get('verdict')} {ev.get('score')}/10, scores {ev.get('scores')}
Rationale: {ev.get('rationale', '')[:300]}

Related-work / saturation scan: {json.dumps(related, ensure_ascii=False)[:1200] if related else '(none)'}

Source paper: {item.get('title')} | {item.get('url')}
{(item.get('abstract') or '')[:400]}"""
    if facts:
        txt += f"\n\nFact-search results (use to resolve SEARCH items):\n{facts[:3000]}"
    return txt


def _facts(searches, cfg):
    """Run the LLM's own SEARCH requests: arXiv keyword search + web search."""
    import paper
    import pipeline
    max_searches = int(cfg.get("grill", {}).get("max_searches", 2))
    out = []
    for q in [s for s in searches if isinstance(s, str)][:max_searches]:
        lines = [f"Query: {q}"]
        for r in paper.keyword_search(q[:60], n=5):
            lines.append(f"  [arxiv {r.get('published','')}] {r.get('title','')} | {r.get('url','')} | {(r.get('abstract') or '')[:160]}")
        web = pipeline._run_search(q, 3, cfg.get("related", {}).get("skill_dir", ""))
        for r in web:
            lines.append(f"  [web] {r.get('title','')} | {r.get('url','')} | {(r.get('snippet') or '')[:160]}")
        out.append("\n".join(lines))
    return "\n".join(out) if out else None


def grill(llm, item, req, ev, related, cfg):
    """Run the grill. Returns parsed grill dict (or None when disabled)."""
    g = cfg.get("grill", {})
    if not g.get("enabled", True):
        return None
    system = SYSTEM_GRILL.format(max_rounds=int(g.get("max_rounds", 2)))
    field = cfg.get("field", "")
    out = llm.chat_json(system, _grill_user(req, ev, related, item, field))
    searches = out.get("searches") or []
    if searches:
        facts = _facts(searches, cfg)
        if facts:
            out2 = llm.chat_json(
                system + "\nResolve the SEARCH items using the facts; keep the same schema.",
                _grill_user(req, ev, related, item, field, facts=facts))
            out2.setdefault("objections", [])
            merged = dict(out2)
            merged["objections"] = out.get("objections", []) + out2.get("objections", [])
            merged["searches_done"] = searches
            out = merged
    out.setdefault("verdict", "wounded")
    out.setdefault("survival", 5)
    out.setdefault("open_assumptions", [])
    out.setdefault("objections", [])
    out.setdefault("rounds", 1)
    out.setdefault("reason", "")
    # sanitize: LLM may emit malformed entries; downstream consumers assume dict/str
    out["objections"] = [o for o in out["objections"] if isinstance(o, dict)]
    out["open_assumptions"] = [a for a in out["open_assumptions"] if isinstance(a, str)]
    try:
        out["survival"] = int(float(out.get("survival", 5)))
    except (TypeError, ValueError):
        out["survival"] = 5
    if out.get("verdict") not in ("dead", "wounded", "survived"):
        out["verdict"] = "wounded"
    return out


def improve(llm, req, ev, rel, gr, cfg):
    """After a non-passing grill: revise the requirement to resolve the
    fatal/major objections and as many open assumptions as possible,
    without drifting from the same core gap."""
    user = f"""Research requirement:
{json.dumps(req, ensure_ascii=False)[:1200]}

Evaluation: {ev.get('verdict')} {ev.get('score')}/10 :: {ev.get('rationale','')[:200]}

Grill verdict: {gr.get('verdict')} {gr.get('survival')}/10 :: {gr.get('reason','')}
Objections:
{json.dumps(gr.get('objections', [])[:8], ensure_ascii=False)[:1500]}
Open assumptions: {json.dumps(gr.get('open_assumptions', [])[:6], ensure_ascii=False)}"""
    try:
        out = llm.chat_json(SYSTEM_IMPROVE, user)
    except Exception:
        return dict(req, _changes=["improve call failed; carried forward"], _unresolved=[])
    merged = dict(req)
    for k in ("title", "one_liner", "requirement", "gap", "why_now", "audience"):
        if out.get(k):
            merged[k] = out[k]
    merged["_changes"] = out.get("changes", [])
    merged["_unresolved"] = out.get("unresolved", [])
    return merged


def grill_loop(llm, item, req, ev, rel, cfg):
    """grill -> improve -> grill -> ... until pass or max_iterations.

    Returns (grill, final_req, history). grill=None when disabled.
    history: one entry per grill pass {iter, verdict, survival, title, improved?}.
    """
    g = cfg.get("grill", {})
    if not g.get("enabled", True):
        return None, req, []
    max_iter = max(1, int(g.get("max_iterations", 2)))
    history = []
    cur = req
    gr = grill(llm, item, cur, ev, rel, cfg)
    if gr is None:
        return None, cur, []
    for i in range(max_iter):
        history.append({"iter": i + 1, "verdict": gr.get("verdict"),
                        "survival": gr.get("survival"),
                        "objections": len(gr.get("objections", [])),
                        "title": cur.get("title")})
        if grill_passes(gr, cfg):
            return gr, cur, history
        if i == max_iter - 1:
            break  # last pass recorded; stop
        improved = improve(llm, cur, ev, rel, gr, cfg)
        gr2 = grill(llm, item, improved, ev, rel, cfg)
        if gr2 is None:  # LLM down mid-loop: stop, keep last grill + unimproved req
            break
        cur, gr = improved, gr2
    return gr, cur, history


def grill_passes(gr, cfg):
    """Gate: does this grilled candidate get an RRD written?"""
    if gr is None:
        return True
    if gr.get("verdict") == "dead":
        return False
    min_survival = float(cfg.get("grill", {}).get("min_survival", 4))
    return (gr.get("survival", 5) >= min_survival
            or cfg.get("grill", {}).get("write_if_wounded", True))


def grill_top_objection(gr):
    for o in gr.get("objections", []):
        if o.get("severity") == "fatal":
            return o
    return (gr.get("objections") or [{}])[0]
