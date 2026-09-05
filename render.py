"""Minimal markdown -> HTML rendering for RRDs (no external deps)."""
import html as _html
import re

PAGE = """<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font:16px/1.55 -apple-system,'Segoe UI',Roboto,'Hiragino Kaku Gothic ProN','Hiragino Sans','Noto Sans JP','Yu Gothic','Meiryo',sans-serif;max-width:46rem;margin:3rem auto;padding:0 1rem;color:#1a1a1a}}
h1{{font-size:1.6rem;line-height:1.25;margin-bottom:.4rem}}
h2{{font-size:1.25rem;margin-top:1.7rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}}
h3{{font-size:1.05rem;margin-top:1.2rem}}
.meta{{color:#666;font-size:.85rem;margin-bottom:1.6rem;line-height:1.7}}
.badge{{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.78rem;font-weight:600;margin-right:.4rem}}
.badge.PURSUE{{background:#d9f2d9;color:#0a7226}}
.badge.WATCH{{background:#fff3cd;color:#8a6d00}}
.badge.DROP{{background:#f8d7da;color:#a00}}
ul{{padding-left:1.3rem}}li{{margin:.25rem 0}}
a{{color:#0645ad}}code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:4px;font-size:.9em}}
.scores{{font-size:.9rem}}
</style></head>
<body>
<div class="meta">{meta}</div>
<article>{body}</article>
</body></html>
"""


def _inline(s):
    s = _html.escape(s)
    s = re.sub(r"\[([^\]]+)\]\((https?[^)\s]+)\)",
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def md2html(md):
    lines = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("### "):
            lines.append("<h3>" + _inline(line[4:]) + "</h3>")
        elif line.startswith("## "):
            lines.append("<h2>" + _inline(line[3:]) + "</h2>")
        elif line.startswith("# "):
            lines.append("<h1>" + _inline(line[2:]) + "</h1>")
        elif re.match(r"^\s*[-*]\s+", line):
            lines.append("<li>" + _inline(re.sub(r"^\s*[-*]\s+", "", line)) + "</li>")
        else:
            lines.append("<p>" + _inline(line.strip()) + "</p>")
    out, buf = [], []
    for it in lines:
        if it.startswith("<li>"):
            buf.append(it)
        else:
            if buf:
                out.append("<ul>" + "".join(buf) + "</ul>")
                buf = []
            out.append(it)
    if buf:
        out.append("<ul>" + "".join(buf) + "</ul>")
    return "\n".join(out)


def render_rrd(title, verdict, ev, rel, item, md, generated, grill=None, history=None, lang="ja"):
    s = ev.get("scores", {})
    scores = " ".join(f"{k} <b>{v}</b>" for k, v in s.items())
    rel_names = ", ".join(r.get("title", "?") for r in (rel or {}).get("related", [])[:6]) \
        or "none found"
    meta = (f'<span class="badge {verdict}">{verdict}</span>'
            f'score <b>{ev.get("score")}</b>/10 &middot; {generated} &middot; '
            f'from <a href="{_html.escape(item.get("url", ""))}" target="_blank">{_html.escape(item.get("title", "")[:90])}</a>'
            f'<div class="scores">scores: {scores}</div>'
            f'<div>related: {_html.escape(rel_names)}</div>')
    if grill:
        meta += (f'<div>grill: {_html.escape(str(grill.get("verdict", "?")))} '
                 f'{grill.get("survival", "?")}/10 &middot; '
                 f'{len(grill.get("open_assumptions", []))} open assumptions')
        if history and len(history) > 1:
            meta += f' &middot; {len(history)} grill rounds'
            meta += (' &middot; ' + _html.escape(
                " -> ".join(f"r{h.get('iter')}:{h.get('verdict')} {h.get('survival')}/10" for h in history)))
        meta += '</div>'
    body = md2html(md)
    if body.startswith("<h1>"):
        body = body.split("\n", 1)[1] if "\n" in body else ""  # title already in header
    return PAGE.format(title=_html.escape(title), lang=(lang or "en"), meta=meta, body=body)
