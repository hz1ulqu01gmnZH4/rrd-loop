#!/usr/bin/env python3
"""rrd-loop: papers -> research requirement -> RRD, in a continuous loop.

Usage:
  python3 run.py --once            # one full cycle (search, synthesize, evaluate, RRDs)
  python3 run.py --loop            # run cycles forever, sleep cfg loop.interval_sec
  python3 run.py --loop --cycles 5 --interval 300
  python3 run.py --papers-only     # just fetch+dedupe papers, no LLM
  python3 run.py --status          # show state: counts + recent RRDs
  python3 run.py --once --field "ai agent self improvement" --items 2 --debug
"""
import argparse
import json
import os
import time

import cycle as cycle_mod
import llm as llm_mod
import state as state_mod

BASE = os.path.dirname(os.path.abspath(__file__))


def load_cfg(path=None):
    with open(path or os.path.join(BASE, "config.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="papers -> research requirement -> RRD loop")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="one cycle")
    mode.add_argument("--loop", action="store_true", help="continuous loop")
    mode.add_argument("--papers-only", action="store_true", help="fetch papers, no LLM")
    mode.add_argument("--status", action="store_true", help="show state")
    ap.add_argument("--interval", type=int, help="seconds between cycles (default cfg)")
    ap.add_argument("--cycles", type=int, help="stop after N cycles (with --loop)")
    ap.add_argument("--items", type=int, help="max papers per cycle")
    ap.add_argument("--field", help="research field to narrow to (overrides config)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--config", help="alternate config.json")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    if a.field:
        cfg["field"] = a.field
        cfg.setdefault("arxiv", {})["queries"] = None  # re-derive for the new field

    if a.status:
        st = state_mod.State(os.path.join(BASE, "state.db"))
        print(f"field: {cfg.get('field')}")
        print(json.dumps(st.counts(), indent=2))
        print("recent RRDs:")
        for p in st.recent_rrds(10):
            print(f"  [{p['verdict']}] {p['title']} -> {p['path']} ({p['ts']})")
        return

    st = state_mod.State(os.path.join(BASE, "state.db"))

    if a.papers_only:
        cycle_mod.run_cycle(cfg, None, st, papers_only=True)
        return

    L = llm_mod.LLM(cfg.get("vllm", {}))
    if not L.ping():
        print(f"vllm not reachable at {L.base} -- is `vllm serve` running?")
        return
    print(f"vllm ok: {L.base} model={L.model} | field: {cfg.get('field')}")

    cycles_done = 0
    while True:
        try:
            cycle_mod.run_cycle(cfg, L, st, debug=a.debug, max_items=a.items)
        except llm_mod.LLMError as e:
            print(f"[error] {e}")
        except Exception as e:
            print(f"[error] cycle failed: {e}")
        cycles_done += 1
        if a.once or (a.cycles is not None and cycles_done >= a.cycles):
            return
        sleep_s = a.interval or cfg.get("loop", {}).get("interval_sec", 1200)
        print(f"[loop] sleeping {sleep_s}s ...")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
