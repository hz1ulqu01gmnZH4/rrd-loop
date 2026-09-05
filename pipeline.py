"""Pipeline stages: synthesize -> evaluate -> related -> grill -> write RRD.

RRD = Research Requirement Document. Each stage = one prompt constant + one
function; tune the prompts here. All LLM stages return parsed JSON
(see llm.LLM.chat_json).
"""
import json
import os
import re
import subprocess
import time

import paper as paper_mod
import requests

FIELD_CFG = "field"  # cfg key: the research field being narrowed to

# ---------------------------------------------------------------------------
# Document language: rrd.language (default "ja"). The final RRD document is
# written in this language; all intermediate stages (synth/evaluate/grill) stay
# in English and their output is translated by the writer. ISO code -> natural
# language name; unknown values pass through as free-form language names.
# ---------------------------------------------------------------------------
_LANG_NAMES = {
    "ja": "Japanese", "ja-jp": "Japanese", "en": "English",
    "zh": "Simplified Chinese", "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish",
}


def _doc_lang(cfg):
    """(html lang attr, prompt language name) for the generated RRD."""
    raw = str(cfg.get("rrd", {}).get("language", "ja")).strip()
    code = raw.lower()
    return code, _LANG_NAMES.get(code, raw)


def _lang_directive(lang):
    """Output-language instruction for the writer prompt ('' for English)."""
    if lang.lower() == "english":
        return ""
    return (f"LANGUAGE: Write the ENTIRE document in {lang} - translate the H1 title, all section headings "
            f"(the names listed below) and every sentence of prose into natural, fluent {lang} suitable for a "
            f"professional document. Keep unchanged: URLs, arXiv identifiers, acronyms, code, and English paper "
            f"titles (a brief parenthetical translation is ok). No mixed-language prose otherwise.")


# ---------------------------------------------------------------------------
# 1. SYNTH: paper (+ field) -> candidate research requirements
# ---------------------------------------------------------------------------
SYSTEM_SYNTH = """You are a research-gap synthesist for the field: <<field>>.
Given ONE paper, synthesize the RESEARCH REQUIREMENTS it surfaces: concrete, falsifiable gaps or underexplored directions a small research program (1-5 people) could commit to.
Rules:
- Grounded in THIS paper only: a failure it reports, an assumption it makes, a negative result, an open question it raises, a method it underexplores. No generic "more work needed on X".
- Each requirement needs a FALSIFIABLE claim, a GAP statement, and WHY NOW (why the window is open).
- At most 2. Return an empty list if this paper surfaces nothing actionable for the field.
Output schema:
{"requirements":[{"title":"max 8 words","one_liner":"a research program that tests <falsifiable claim> in <setting>","requirement":"what the research must demonstrate","gap":"what is missing in prior work","why_now":"why the window is open now","audience":"who in the field will care or adopt"}]}"""


def synth(llm, paper_item, field, max_reqs=2):
    user = f"""Research field: {field}

Paper:
Title: {paper_item['title']}
Authors: {paper_item.get('authors', '')}
Published: {paper_item.get('published', '')}  ({paper_item.get('category', '')})
arXiv: {paper_item.get('url', '')}
Abstract: {paper_item.get('abstract', '')[:1500]}"""
    try:
        out = llm.chat_json(SYSTEM_SYNTH.replace("<<field>>", field), user)
    except Exception:
        return []
    reqs = out.get("requirements", out if isinstance(out, list) else [])
    if not isinstance(reqs, list):
        return []
    reqs = [r for r in reqs if isinstance(r, dict) and r.get("one_liner")]
    return reqs[:max_reqs]


# ---------------------------------------------------------------------------
# 2. EVALUATE: requirement -> scores + verdict (the "urgent or the gap will
#    close itself" gate). PURSUE/WATCH proceed; DROP moves on.
# ---------------------------------------------------------------------------
SYSTEM_EVALUATE = """You are a research program director scoring a RESEARCH REQUIREMENT for a small group.
Field: <<field>>
Score each dimension 0-10 (integers):
- relevance: fit with the field's open problems
- gap_durability: how fast the value of closing this gap decays (10 = obvious next work by many groups in weeks; low = durable open problem)
- impact: field-level significance if solved
- measurability: how cleanly you can define evals, baselines, kill criteria (10 = crisp)
- feasibility: effort/compute for a 1-5 person group with at most one GPU workstation (10 = very hard)
- saturation: how many groups are already working this exact gap (10 = crowded)
Overall score: start from the mean of (relevance, gap_durability, impact, measurability);
subtract 0.4*(feasibility-5); subtract 0.6*(saturation-5); clamp to 0-10.
Verdict rules: PURSUE if score>=7 and saturation<=7; WATCH if 4<=score<7; DROP otherwise.
Output schema:
{"scores":{"relevance":int,"gap_durability":int,"impact":int,"measurability":int,"feasibility":int,"saturation":int},"score":0.0,"verdict":"PURSUE|WATCH|DROP","rationale":"2-3 specific sentences"}"""

_VALID_VERDICTS = {"PURSUE", "WATCH", "DROP"}


def _coerce_eval(raw, cfg):
    """Clamp LLM output; re-derive verdict if the LLM broke its own rubric."""
    s = raw.get("scores", raw) if isinstance(raw, dict) else {}
    num = lambda k: max(0, min(10, int(float(s.get(k, 5)))))
    scores = {k: num(k) for k in ("relevance", "gap_durability", "impact",
                                  "measurability", "feasibility", "saturation")}
    score = ((scores["relevance"] + scores["gap_durability"] + scores["impact"]
              + scores["measurability"]) / 4
             - 0.4 * (scores["feasibility"] - 5)
             - 0.6 * (scores["saturation"] - 5))
    score = round(max(0, min(10, score)), 1)
    ev_cfg = cfg.get("evaluate", {})
    pursue = ev_cfg.get("pursue_threshold", 7)
    watch = ev_cfg.get("watch_threshold", 4)
    if score >= pursue and scores["saturation"] <= 7:
        verdict = "PURSUE"
    elif score >= watch:
        verdict = "WATCH"
    else:
        verdict = "DROP"
    if raw.get("verdict") in _VALID_VERDICTS:
        verdict = raw["verdict"]
        if verdict == "DROP" and score >= pursue:
            verdict = "WATCH"
        if verdict == "PURSUE" and score < watch:
            verdict = "WATCH"
    return {"scores": scores, "score": score, "verdict": verdict,
            "rationale": str(raw.get("rationale", ""))[:500]}


def evaluate(llm, paper_item, req, cfg):
    user = f"""Research field: {cfg.get(FIELD_CFG, '')}

Research requirement:
{str(req).replace('"', "'")[:1500]}

Source paper:
Title: {paper_item['title']}
Abstract: {(paper_item.get('abstract') or '')[:400]}"""
    try:
        raw = llm.chat_json(SYSTEM_EVALUATE.replace("<<field>>", cfg.get(FIELD_CFG, "")), user)
    except Exception:
        raw = {}
    return _coerce_eval(raw, cfg)


# ---------------------------------------------------------------------------
# 3. RELATED: saturation / related-work scan (arXiv keyword search + web),
#    then LLM separates genuinely related work from noise.
# ---------------------------------------------------------------------------
SYSTEM_RELATED = """Research-related-work analyst.
Given a research requirement and raw search results (arXiv + web), separate genuinely related work (overlapping gap, competing method, or a baseline you'd need) from noise (tutorials, other topics, stale surveys without a gap).
For each related item: title, arxiv id or url, relation (overlapping-gap | competing-method | needed-baseline | adjacent).
Saturation: estimate how many INDEPENDENT groups are already working this exact gap (0-10). saturated=true when >=4 groups with <2 years of differentiation.
Find the white space: what has NOT been tried.
Output schema:
{"related":[{"title":"...","ref":"arxiv id or url","relation":"...","note":"..."}],"saturation":int 0-10,"saturated":true|false,"white_space":"the gap no one covers","signal":"strong|mixed|weak"}"""


def _run_search(query, n, skill_dir):
    """Use the local web-search skill (keyless Bing/DDG HTML search)."""
    script = os.path.join(skill_dir, "search.py")
    try:
        r = subprocess.run(["python3", script, query, str(n)],
                           capture_output=True, text=True, timeout=90)
    except Exception:
        return []
    out, cur = [], None
    for line in r.stdout.splitlines():
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            if cur:
                out.append(cur)
            cur = {"title": m.group(1).strip(), "url": "", "snippet": ""}
        elif cur and line.startswith("http"):
            cur["url"] = line.strip().split()[0]
        elif cur and line.strip() and not line.startswith("http"):
            cur["snippet"] += " " + line.strip()
    if cur:
        out.append(cur)
    return out[:n]


def related(llm, paper_item, req, cfg):
    c = cfg.get("related", {})
    if not c.get("enabled", True):
        return None
    terms = re.sub(r"[^a-z0-9 ]+", " ", (req.get("one_liner", "") or req.get("title", ""))
                   .lower())[:80].strip()
    arxiv_res = paper_mod.keyword_search(terms, n=c.get("arxiv_results", 6))
    web_res = _run_search(terms, c.get("web_search_results", 4),
                          c.get("skill_dir", ""))
    raw = "\n".join(
        [f"{i}. [arxiv {p.get('published','')}] {p.get('title','')} | {p.get('url','')} | {(p.get('abstract') or '')[:160]}"
         for i, p in enumerate(arxiv_res, 1)]
        + [f"{len(arxiv_res)+i}. [web] {r.get('title','')} | {r.get('url','')} | {(r.get('snippet') or '')[:160]}"
           for i, r in enumerate(web_res, 1)]
    ) or "(no results)"
    user = f"""Research requirement: {str(req)[:800]}
Search results (arXiv + web):
{raw[:4000]}"""
    try:
        out = llm.chat_json(SYSTEM_RELATED, user)
    except Exception:
        out = {}
    out.setdefault("related", [])
    out.setdefault("saturation", 3)
    out.setdefault("saturated", False)
    out.setdefault("white_space", "")
    out.setdefault("signal", "weak")
    return out


# ---------------------------------------------------------------------------
# 4. WRITE: scored requirement + related scan + grill -> RRD (md -> HTML)
# ---------------------------------------------------------------------------
def _rrd_system(title, field, lang="Japanese"):
    d = _lang_directive(lang)
    lang_block = (d + "\n") if d else ""
    return f"""You are a research program lead writing a RESEARCH REQUIREMENT DOCUMENT (RRD) for a small group (1-5 people) committing to a research direction in the field: {field}.
{lang_block}Write tight markdown, 600-900 words, exactly these sections in this order:
# {title}
## Field & why now
## Prior work & gap   (cite the provided related items; state precisely what is missing)
## Requirement        (the falsifiable claim, one paragraph)
## Milestones & evals (M1-M3, each with a measurable criterion and its baseline)
## Method sketch      (1-3 candidate approaches; which one and why)
## Resources & constraints  (compute, data, access a small group realistically has)
## Saturation check   (concurrent work from the scan; where this sits)
## Grilling (red team)   (verdict + the 2-4 strongest objections with answers/mitigations, and the open assumptions this RRD silently relies on)
## Risks & kill criteria
## Success criteria   (what 90 days of work must produce)
## Related papers     (arXiv links from the scan + the source paper)
Ground every claim in the provided data (scores, related work, grill). No filler, no buzzword soup."""


def write_rrd(llm, paper_item, req, ev, rel, cfg, grill=None, history=None):
    user = f"""Research field: {cfg.get(FIELD_CFG, '')}

Requirement:
{str(req)[:1200]}

Evaluation: score {ev['score']}/10, verdict {ev['verdict']}
Scores: {ev['scores']}
Rationale: {ev['rationale']}

Related-work scan: {str(rel)[:1500] if rel else '(none)'}

Improvement history (grill -> improve -> grill): {json.dumps(history, ensure_ascii=False)[:800] if history else '(none)'}
Revisions to the requirement: changes={json.dumps(req.get('_changes', []), ensure_ascii=False)[:400]} decided={json.dumps(req.get('_decided', []), ensure_ascii=False)[:400]} unresolved={json.dumps(req.get('_unresolved', []), ensure_ascii=False)[:300]}

Grilling (adversarial design-tree check): {_grill_block(grill)}

Source paper:
Title: {paper_item['title']}
arXiv: {paper_item.get('url')}
Abstract: {(paper_item.get('abstract') or '')[:600]}"""
    md = llm.chat(_rrd_system(req.get("title", "Requirement"), cfg.get(FIELD_CFG, ""),
                              _doc_lang(cfg)[1]), user)
    md = md.strip()
    if not md.startswith("# "):
        m = re.search(r"\n# ", md)
        if m:
            md = md[m.start() + 1:]
    if md.startswith("```"):
        md = re.sub(r"^```(?:markdown)?\n?", "", md).rsplit("```", 1)[0]
    return md


def _grill_block(gr, history=None):
    """Compact grill summary for the RRD writer prompt / render meta."""
    if not gr:
        return "(grill stage disabled or unavailable)"
    lines = [f"verdict {gr.get('verdict')} survival {gr.get('survival')}/10 :: {gr.get('reason','')}"]
    if history and len(history) > 1:
        lines.append("rounds: " + " -> ".join(
            f"r{h.get('iter')}: {h.get('verdict')} {h.get('survival')}/10" for h in history))
    for o in (gr.get("objections") or [])[:5]:
        lines.append(f"- [{o.get('severity','?')}] {o.get('question','')} -> {o.get('answer_or_assumption','')[:160]}")
    for a in (gr.get("open_assumptions") or [])[:5]:
        lines.append(f"- open assumption: {a[:160]}")
    return "\n".join(lines)


def save_rrd(cfg, paper_item, req, ev, rel, md, grill=None, history=None):
    import render
    import time as _time
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           cfg.get("rrd", {}).get("dir", "out"))
    os.makedirs(out_dir, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", (req.get("title") or "requirement").lower()).strip("-")[:60]
    path = os.path.join(out_dir, f"rrd-{_time.strftime('%Y%m%d-%H%M')}-{slug}.html")
    n = 2
    while os.path.exists(path):  # minute-resolution collision: suffix -2, -3, ...
        path = os.path.join(out_dir, f"rrd-{_time.strftime('%Y%m%d-%H%M')}-{slug}-{n}.html")
        n += 1
    with open(path, "w") as f:
        f.write(render.render_rrd(req.get("title", "RRD"), ev["verdict"], ev, rel,
                                  paper_item, md, generated=_time.strftime("%Y-%m-%d %H:%M"),
                                  grill=grill, history=history, lang=_doc_lang(cfg)[0]))
    return path


# ---------------------------------------------------------------------------
# Stage runner used by cycle.py
# ---------------------------------------------------------------------------
def process_requirement(llm, paper_item, req, ev, cfg, debug=False, recent_reqs=None):
    """Decide + act on one evaluated requirement.
    Returns (verdict, related, rrd_path, note, grill)."""
    verdict = ev["verdict"]
    rel = None
    if verdict in ("PURSUE", "WATCH") and cfg.get("related", {}).get("enabled", True):
        rel = related(llm, paper_item, req, cfg)
    if debug:
        print(f"    [related] {str(rel)[:300]}")
    saturated = bool(rel and rel.get("saturated"))
    rrd_path = None
    note = ""
    gr = None
    rrd_cfg = cfg.get("rrd", {})
    if verdict == "DROP":
        import gate
        gate.registry_line(paper_item, req, ev, None)
        note = "drop"
    elif saturated and not rrd_cfg.get("write_if_saturated", False):
        note = "saturated-field skip"
        import gate
        gate.registry_line(paper_item, req, ev, None, note=note)
        print(f"    .. saturated field, skipping RRD (white_space: {rel.get('white_space','')[:120]})")
    elif verdict == "WATCH" and not rrd_cfg.get("write_watch", False):
        note = "watch, no-rrd-by-config"
    else:
        # ---- dedupe gate: L2 wiki grep + L3 kg probe + L4 LLM verifier ----
        import gate
        cands = gate.candidates(req, recent_reqs or [])
        gv = gate.llm_gate(llm, req, cands, cfg)
        if debug:
            print(f"    [gate] cands={len(cands)} -> {gv}")
        if gv.get("error"):
            note = f"gate unavailable ({gv['error'][:60]}), deferred to next cycle"
            print(f"    [gate] {note}")
        elif gv.get("duplicate"):
            note = f"dup of '{gv.get('duplicate_of')}'"
            print(f"    [gate] duplicate of '{gv.get('duplicate_of')}' (conf {gv.get('confidence')}): {gv.get('reason','')[:100]}")
            gate.registry_line(paper_item, req, ev, None, note=f"dup of {gv.get('duplicate_of')}")
        else:
            # ---- GRILL loop: grill -> improve -> grill -> ... (bounded) ----
            import grill as grill_mod
            gr, req_final, ghistory = grill_mod.grill_loop(llm, paper_item, req, ev, rel, cfg)
            if debug:
                print(f"    [grill] rounds={len(ghistory)} {json.dumps(gr, ensure_ascii=False)[:300] if gr else 'disabled'}")
            if gr is not None and not grill_mod.grill_passes(gr, cfg):
                top = grill_mod.grill_top_objection(gr)
                note = (f"grilled dead after {len(ghistory)} round(s) "
                        f"(survival {gr.get('survival')}): {(top or {}).get('question', gr.get('reason',''))[:100]}")
                print(f"    [grill] {note}")
                gate.registry_line(paper_item, req_final, ev, None, note=note)
                gr["history"] = ghistory
                gr["_improved"] = req_final.get("_changes", [])
                return verdict, rel, None, note, gr
            md = write_rrd(llm, paper_item, req_final, ev, rel, cfg, grill=gr, history=ghistory)
            rrd_path = save_rrd(cfg, paper_item, req_final, ev, rel, md, grill=gr, history=ghistory)
            note = (f"grill: {gr.get('verdict')} {gr.get('survival')}/10, "
                    f"{len(ghistory)} round(s), {len(gr.get('open_assumptions',[]))} open assumptions") if gr else ""
            gate.registry_line(paper_item, req_final, ev, rrd_path, note=note)
            gate.kg_register(paper_item, req_final, ev, rrd_path)
            gr["history"] = ghistory
            gr["_improved"] = req_final.get("_changes", [])
            print(f"    [grill] {gr.get('verdict')} {gr.get('survival')}/10 ({len(ghistory)} round(s)) -> RRD {rrd_path}")
    return verdict, rel, rrd_path, note, gr
