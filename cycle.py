"""One cycle: search papers -> synthesize requirements -> evaluate ->
related -> grill -> write RRD -> next paper."""
import json
import os
import time

import llm as llm_mod
import paper as paper_mod
import pipeline
import state as state_mod


def _log(log_dir, entry):
    os.makedirs(log_dir, exist_ok=True)
    entry = dict(entry, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(log_dir, "cycles.jsonl"), "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_cycle(cfg, llm, state, debug=False, max_items=None, papers_only=False):
    base = os.path.dirname(os.path.abspath(__file__))
    items = paper_mod.fetch_all(cfg)
    fresh = state.unseen(items)
    print(f"[papers] field='{cfg.get('field','')}' fetched {len(items)} papers, {len(fresh)} new")
    if papers_only:
        for it in fresh[:20]:
            print(f"  - [{it.get('category')}] {it['title'][:100]} ({it.get('published')})")
        return {"fetched": len(items), "new": len(fresh)}

    budget = max_items if max_items is not None else cfg.get("limits", {}).get("max_items_per_cycle", 6)
    batch = fresh[:budget]
    state.mark_seen(batch)  # dedupe for future cycles even if LLM fails

    res = {"fetched": len(items), "new": len(fresh), "processed": len(batch),
           "reqs": 0, "verdicts": {}, "rrds": [], "dupes": [], "notes": []}
    if not batch:
        _log(base, {"new": 0, "note": "nothing new"})
        return res

    recent_reqs = state.recent_reqs(30)  # L1 passed; L4 compares against these
    for p in batch:
        print(f"\n[paper] ({p.get('category')}) {p['title'][:110]} ({p.get('published')})")
        reqs = pipeline.synth(llm, p, cfg.get("field", ""),
                              cfg.get("limits", {}).get("max_reqs_per_paper", 2))
        if debug:
            print(f"    [synth] {json.dumps(reqs)[:400]}")
        if not reqs:
            print("    (no research requirement)")
            continue
        for req in reqs:
            title = req.get("title", "unnamed")
            if state.has_req(title):
                print(f"    [dup] {title} -- skip")
                continue
            ev = pipeline.evaluate(llm, p, req, cfg)
            verdict = ev["verdict"]
            res["verdicts"][verdict] = res["verdicts"].get(verdict, 0) + 1
            res["reqs"] += 1
            print(f"    [req] {title}\n           {verdict} ({ev['score']}/10): {ev['rationale'][:140]}")
            v, rel, rrd_path, note, gr = pipeline.process_requirement(
                llm, p, req, ev, cfg, debug, recent_reqs)
            if note:
                res["notes"].append({"title": title, "note": note})
                if note.startswith("dup"):
                    res["dupes"].append(f"{title} ~ {note}")
            req_id = state.add_req(p["url"], title, req, ev, verdict, rel, gr)
            recent_reqs.append({"title": title, "one_liner": req.get("one_liner", ""),
                                "gap": req.get("gap", "")})
            if rrd_path:
                state.save_rrd(req_id, rrd_path)
                res["rrds"].append(rrd_path)

    _log(base, res)
    print(f"\n[cycle] {res['reqs']} reqs, verdicts={res['verdicts']}, rrds={len(res['rrds'])}, dupes={len(res['dupes'])}")
    return res
