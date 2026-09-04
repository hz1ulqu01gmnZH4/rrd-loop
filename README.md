# rrd-loop

Continuous **papers → research requirement → RRD (Research Requirement
Document)** factory, driven by the local vLLM server
(`qwen3.8-27b` on `localhost:8000`). The sibling of
[prd-loop](https://github.com/hz1ulqu01gmnZH4/prd-loop) (that one loops
market scans into PRDs; this one loops the literature into research programs).

```
search recent arXiv papers (keyless arXiv API, narrowed to a FIELD)
  ─> synthesize research requirements (vllm, <=2/paper, paper-grounded gaps)
  ─> evaluate (vllm rubric: relevance / gap durability / impact /
     measurability / feasibility / saturation)
       DROP → next                    PURSUE / WATCH
  ─> related-work & saturation scan (arXiv keyword + keyless web search, vllm)
       saturated? → skip RRD (log white space)
  ─> dedupe gate (4 layers: state.db | wiki-registry grep | kg probe | LLM verifier)
  ─> GRILL (adversarial design-tree check, auto fact-searches, vllm)
       dead? → skip RRD (registry note)
  ─> write RRD (vllm) → out/rrd-*.html
  ─> state dedup → next paper / next cycle
```

**Field narrowing** — the whole loop is scoped to a research field in
`config.json` (`"field": "ai agent self improvement"`). The field string is
expanded into a few disjunctive arXiv queries (whole-phrase + 2-grams +
hyphenated key pair, each ANDed with the configured categories). Narrow or
switch fields at runtime: `python3 run.py --once --field "LLM agent safety"`.
Explicit query overrides live under `arxiv.queries`.

**Evaluate gates the write**: only PURSUE (or optionally WATCH) requirements
become RRDs; DROP/saturated/duplicate/dead-grill candidates cost one LLM call
and get registry lines instead.

## Quick start

```bash
cd ~/rrd-loop
python3 run.py --status                          # state: papers seen, reqs, RRDs
python3 run.py --papers-only                    # fetch+dedupe papers (no LLM)
python3 run.py --once                           # one full cycle
python3 run.py --once --field "ai agent self improvement" --items 2 --debug
nohup python3 -u run.py --loop --interval 1200 > loop.out 2>&1 &   # run forever
```

Cycle results append to `cycles.jsonl`.

## Layout

| file | role |
|---|---|
| `run.py` | CLI (`--once`, `--loop`, `--papers-only`, `--status`, `--items`, `--interval`, `--field`, `--debug`) |
| `cycle.py` | one cycle: fetch → dedupe → per-paper stages → log to `cycles.jsonl` |
| `paper.py` | keyless arXiv API fetcher + field→query derivation + keyword (saturation) search |
| `pipeline.py` | the 4 stages + prompt constants (tune prompts here) |
| `grill.py` | GRILL stage: adversarial design-tree interrogation run as a grill -> improve -> grill loop, auto fact-searches |
| `gate.py` | duplicate-avoidance gate: L2 registry grep, L3 kg probe, L4 LLM verifier, registration |
| `state.py` | SQLite `state.db`: paper dedup (arXiv id + title-norm), req dedup + grill results, RRD paths |
| `llm.py` | vLLM OpenAI-compatible client + robust JSON extraction |
| `render.py` | RRD markdown → HTML |

## Duplicate avoidance (4 layers, cheap -> expensive)

Same architecture as prd-loop, with its own registry
(`~/wiki/projects/rrd-loop-registry.md`) and kg domain (`rrd-loop`):

1. **L1 string** — `state.db` normalized-requirement-title match.
2. **L2 wiki registry grep** — token overlap vs registry titles; no LLM.
3. **L3 knowledge-graph probe** — `kg q --domain rrd-loop`.
4. **L4 LLM gate verifier** — independent call at temp 0, conservative
   "when in doubt → duplicate".

## Grilling (red team)

Same headless adaptation of the grilling agent-skill family as prd-loop,
retargeted at research: the decision tree is hypothesis / method / evals +
baselines / data + compute / timeline / threats to validity / kill criteria.
SEARCH items are answered with arXiv keyword search **and** web search.

The grill runs **as a loop: grill -> improve -> grill -> ...**
(`grill.max_iterations`, default 2 -> up to 2 grill passes when it never
passes early). Each pass that
doesn't pass triggers an IMPROVE round: the LLM revises the requirement
(sharpen the falsifiable claim, narrow the setting, switch method direction,
decide open assumptions) without abandoning the core gap; the revised
requirement is grilled again carrying the mitigations tried so far. It stops
early the moment the grill passes, and stops at the cap otherwise. If a
fatal objection still stands when the loop ends, the grill wins.
`dead` (or, with `grill.write_if_wounded=false`, survival <
`grill.min_survival`, default 4) blocks the write. The HTML meta shows the
round trail (`r1:dead 2/10 -> r2:wounded 6/10`).

## RRD structure

Field & why now · Prior work & gap · Requirement (falsifiable claim) ·
Milestones & evals (M1–M3 with baselines) · Method sketch · Resources &
constraints · Saturation check · Grilling (red team) · Risks & kill criteria ·
Success criteria (90-day) · Related papers.

## Configuration knobs (`config.json`)

- `field` — the research field the loop narrows to (CLI: `--field`).
- `arxiv.categories / queries / per_query / max_age_days` — paper sourcing.
- `evaluate.pursue_threshold / watch_threshold` — verdict gates (7 / 4).
- `rrd.write_watch` — also write RRDs for WATCH.
- `rrd.write_if_saturated` — write RRDs even when the field is saturated.
- `grill.enabled / max_rounds / max_searches / min_survival / write_if_wounded` — the grill stage (section above).
- `grill.max_iterations` (default 2) — grill -> improve -> grill loop bound: total grill passes when the loop never passes early = max_iterations (pass 1 is the unimproved requirement).
- `vllm.*` — endpoint/model for the LLM driver (sends
  `chat_template_kwargs.enable_thinking=false`; drop it if another model
  rejects the param).

## Failure behaviour

- vLLM down → cycle prints error, loop sleeps and retries next interval.
- arXiv API down → that query is skipped (others still run).
- every fetched paper is marked seen **before** LLM work, so a crash can't
  re-process the same paper; requirements are deduped by normalized title.
