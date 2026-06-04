# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""scorecard — every run mints a scorecard the OS dashboard / website can render.

The operator's ask: "make it so that all runs produce a scorecard I can see or use in the OS or on the
website." So EVERY solve run writes one canonical `scorecard/v1` JSON (always, local + OS-readable) plus
an append-only `index.jsonl`. The watchable ARC-Prize replay URL is attached when a run is posted online
(honest provenance: source-COUPLED deepcopy/teacher vs source-FREE). The substrate emits here; the OS
(codebox-os dashboard) and the website (intuitionlabs.tech) read `scorecards/index.jsonl` + the per-run
cards and render them — the 1→2 pattern (substrate produces the artifact, the frontend draws it).

scorecard/v1 fields:
  schema, run_id, ts, game, method, source_coupled, levels_cleared, won, state, mean_rhae,
  per_level, stall, wall_s, win_phi{max,total,trajectory}, coherence_R, watchable_url, extra(dict)
"""
from __future__ import annotations

import json
import os
import time
import uuid

SCHEMA = "scorecard/v1"
DEFAULT_DIR = os.environ.get("ARC_SCORECARD_DIR", "probes/logs/scorecards")


def mint(result, *, method="general-bfs", source_coupled=True, win_phi=None, coherence_R=None,
         watchable_url=None, scorecard_dir=None, run_id=None, extra=None):
    """Write one scorecard/v1 JSON + append to the index. Returns the card dict. Degrade-closed:
    never raises into a solve. `result` = the solver's result dict (game, levels, won, state,
    mean_rhae, per_level, stall, wall_s)."""
    try:
        d = scorecard_dir or DEFAULT_DIR
        os.makedirs(d, exist_ok=True)
        rid = run_id or uuid.uuid4().hex[:12]
        card = {
            "schema": SCHEMA,
            "run_id": rid,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "game": result.get("game"),
            "method": method,
            # honest provenance: a source-COUPLED run reads env._game (teacher/demonstration) and is NOT
            # a community-board source-free number; a source-FREE run learned by playing.
            "source_coupled": bool(source_coupled),
            "levels_cleared": result.get("levels", 0),
            "won": result.get("won", False),
            "state": result.get("state"),
            "mean_rhae": result.get("mean_rhae"),
            "per_level": result.get("per_level"),
            "stall": result.get("stall"),
            "wall_s": result.get("wall_s"),
            "win_phi": win_phi,                 # {"max":.., "total":.., "trajectory":[..]} φ 0→⅓→⅔→win
            "coherence_R": coherence_R,
            "watchable_url": watchable_url,     # arcprize.org/replay/{guid} when posted online
            "extra": extra or {},
        }
        with open(os.path.join(d, f"{card['game'] or 'run'}-{rid}.json"), "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        with open(os.path.join(d, "index.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(card) + "\n")
        render_html(d)                          # regenerate the viewable page on every mint
        return card
    except Exception:
        return None


def load_index(scorecard_dir=None):
    """Read the scorecard index (newest last) — what the OS/website iterates to render the list."""
    d = scorecard_dir or DEFAULT_DIR
    p = os.path.join(d, "index.jsonl")
    out = []
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return out


def summary(scorecard_dir=None):
    """Roll up the scorecards for an OS/website header: runs, games, levels, source-free vs coupled."""
    cards = load_index(scorecard_dir)
    if not cards:
        return {"runs": 0}
    games = sorted({c.get("game") for c in cards if c.get("game")})
    return {
        "runs": len(cards),
        "games": games,
        "levels_cleared_total": sum(int(c.get("levels_cleared") or 0) for c in cards),
        "wins": sum(1 for c in cards if c.get("won")),
        "source_free_runs": sum(1 for c in cards if not c.get("source_coupled")),
        "source_coupled_runs": sum(1 for c in cards if c.get("source_coupled")),
        "with_watchable_url": sum(1 for c in cards if c.get("watchable_url")),
    }


def _esc(x):
    return (str(x) if x is not None else "—").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_html(scorecard_dir=None):
    """Write a standalone `index.html` the OS dashboard / website can serve (or the operator opens
    directly) — the same cards drawn as a paper/amber table. Substrate→frontend, the 1→2 pattern."""
    try:
        d = scorecard_dir or DEFAULT_DIR
        cards = load_index(d)
        s = summary(d)
        rows = []
        for c in reversed(cards):                # newest first
            wp = c.get("win_phi") or {}
            url = c.get("watchable_url")
            link = f'<a href="{_esc(url)}">watch ▶</a>' if url else "—"
            src = "coupled" if c.get("source_coupled") else "source-free"
            rhae = c.get("mean_rhae")
            rows.append(
                f"<tr><td><b>{_esc(c.get('game'))}</b></td><td>{_esc(c.get('method'))}</td>"
                f"<td class='{src}'>{src}</td><td>{_esc(c.get('levels_cleared'))}</td>"
                f"<td>{'—' if rhae is None else rhae}</td><td>{_esc(c.get('coherence_R'))}</td>"
                f"<td>{_esc(wp.get('max'))}</td><td>{link}</td>"
                f"<td class='ts'>{_esc(c.get('ts'))}</td></tr>")
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Terminals OS — run scorecards</title><style>"
            "body{background:oklch(0.985 0.005 80);color:oklch(0.2 0.01 70);"
            "font:14px/1.6 ui-monospace,'SF Mono',monospace;margin:2.2rem;max-width:1100px}"
            "h1{font-weight:600;font-size:1.15rem;margin:0 0 .2rem}"
            ".sub{color:oklch(0.5 0.02 70);margin-bottom:1.2rem;font-size:.92em}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{text-align:left;padding:.42rem .75rem;border-bottom:1px solid oklch(0.91 0.01 80)}"
            "th{color:oklch(0.45 0.03 60);font-weight:600;border-bottom:2px solid oklch(0.85 0.02 70)}"
            ".coupled{color:oklch(0.56 0.09 40)}.source-free{color:oklch(0.5 0.11 150)}"
            ".ts{color:oklch(0.62 0.01 70);font-size:.84em}a{color:oklch(0.5 0.13 250);text-decoration:none}"
            "</style></head><body><h1>Terminals OS — run scorecards</h1>"
            f"<div class='sub'>{s.get('runs',0)} runs · {s.get('levels_cleared_total',0)} levels cleared · "
            f"{s.get('source_free_runs',0)} source-free / {s.get('source_coupled_runs',0)} coupled (teacher) · "
            f"{s.get('with_watchable_url',0)} watchable</div>"
            "<table><thead><tr><th>game</th><th>method</th><th>provenance</th><th>levels</th>"
            "<th>rhae</th><th>R</th><th>win-φ</th><th>replay</th><th>ts</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></body></html>")
        path = os.path.join(d, "index.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path
    except Exception:
        return None
