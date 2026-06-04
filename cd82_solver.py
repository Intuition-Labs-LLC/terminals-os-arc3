# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""cd82_solver — the PER-TISSUE (cd82-method) solver for cd82, the canvas paint-by-fills matcher.

WHY a per-tissue solver. the general solver downloads + probes cd82 but does NOT auto-clear it
(verified: levels=0, stall=max_levels, 3.5 s). The miss is structural, read from the public source
(`environment_files/cd82/fb555c5d/cd82.py`):

  * The win predicate `wvrremwltt` (L740) is a per-pixel CANVAS-MATCH check: it compares the player
    canvas `xytrjjbyib` (a 10x10 grid, all 0 at level start) against the per-level target sprite
    `eoqnvkspoa-pqwmeN-1` (10x10) at every cell EXCEPT the two diagonals (the X-mask `poqpfcjieu`
    sets i==i and i==9-i False). When `canvas[mask] == target[mask]` it calls `next_level()`. There
    is no boolean done-mask and no `score >= win_score` pair, so the general field-scan finds no
    director → blind BFS over a 6-action space where fills overwrite → exhausts at max_levels.

  * The objective is a LAYERED REGION-PAINT. The canvas is painted by FILLS, each with the current
    color `knqmgavuh`. Two fill families, both read straight off the game's own source:
      - 8 RING fills (`rtjwayrycq`, fired by ACTION5): 4 half-planes (top/bottom rows 0-4 / 5-9,
        left/right cols 0-4 / 5-9) + 4 diagonal triangles (the four corners split on the diagonals).
        Which one fires is set by the ACTIVE-BASKET ring position `xwmfgtlso` (0..7) and its rotation.
      - 4 CORNER fills (`coublenfir`, fired by clicking the `ctwspzkygu` corner sprite via ACTION6):
        3x4 blocks at [0:3,3:7] / [7:10,3:7] / [3:7,0:3] / [3:7,7:10] (top/bottom/left/right), only
        present on levels that carry the corner sprite (L3-L6) and only at even ring positions.
    Color is chosen by clicking a `pqkenviek` palette swatch (ACTION6 → `qbiojckwxl` sets knqmgavuh).

THE TRAP the general solver falls into. The win-φ (off-diagonal match count) is NON-MONOTONE in
RAW ACTION space: because fills overwrite, the path to the target must repaint cells, so a fill that
advances the final picture can momentarily DROP the live match. Best-first over single actions
plateaus (exactly re86's deceptive summed-match). The win is reachable; the per-action director
can't see the path.

WHAT this solver adds (the per-tissue piece). PLAN IN FILL-SPACE, then translate to actions:

  1. Read the 8 ring-fill masks + 4 corner masks ONCE by simulating each on a blank canvas (the
     game's OWN dynamics — a deepcopy world-model, no source re-derivation).
  2. BEAM-SEARCH the (region, color) FILL sequence whose layered composition matches the target
     off-diagonal. The director is the win-φ itself — `(matched, total)` off-diagonal cells, the win
     predicate read at PARTIAL resolution → a monotone (sat, total) the beam descends. Layered
     overwrites are handled by keeping the top-K canvases (beam), so a transient drop is explored.
  3. Translate each planned fill into game actions on the real env: select its color (ACTION6 click
     on the matching palette swatch), navigate the ring to its position (ACTION1-4 along the ring
     graph, shortest path by BFS), then ACTION5 (ring fill) or ACTION6 corner-click (corner fill).
     The win fires mid-plan when the last fill snaps the canvas into match → resolved through the
     harness's `levels_completed` rise.

WHAT IS REUSED FROM THE FOUNDRY LIBRARY (the library-validation):
  * the WIN-φ DIRECTOR read at partial resolution → monotone (sat,total) — re86/ft09's win-φ pattern,
    here the off-diagonal canvas-match count;
  * the deepcopy world-model to read the game's OWN fill dynamics (no source re-derivation of masks);
  * the per-tissue translate-plan-to-actions skeleton (re86 drives sprites to precomputed goals; here
    we drive the canvas to the target via precomputed fills);
  * the trace/v1 Tracer (win-φ improvement → edge mint) and scorecard.mint machinery, verbatim.
  NOT reused: the A* frontier over single actions (cd82's win is non-monotone in action-space — the
  beam-over-fills replaces it) and the click-centroid sampler (cd82's clicks are exact swatch/corner
  coordinates read from the game, not blob centroids).

cd82 is SOURCE-COUPLED (it reads env._game for the basket masks, target sprites, nav graph) — a
demonstration/teacher solver, NOT a source-free community-board number. Marked source_coupled=True.

Run BOUNDED (CPU/venv only, no model):
  ulimit -v 8388608
  timeout 90 /var/home/zero/arc-agi/toolkit/.venv/bin/python cd82_solver.py \
      --max-levels 6 --max-nodes 40000 --timeout 80 --json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import arc_agi
from arc_agi import OperationMode

CANVAS_NAME = "xytrjjbyib"          # the 10x10 player canvas (all 0 at level start)
TARGET_PREFIX = "eoqnvkspoa-"       # the per-level target sprite eoqnvkspoa-pqwmeN-1
CORNER_NAME = "ctwspzkygu"          # the corner-trigger sprite (L3-L6); click fires a 3x4 corner fill
PALETTE_PREFIX = "pqkenviek"        # the color swatches (click sets knqmgavuh)
FILL_ACTION = "ACTION5"             # fires the ring fill (rtjwayrycq)
CLICK_ACTION = "ACTION6"            # complex/click action (swatch select, corner fire)
NAV_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")  # ring navigation (nqhfiooufi)

# the 4 corner-fill regions, read verbatim from coublenfir (iswxsbrge -> canvas slice)
CORNER_SLICES = {0: (0, 3, 3, 7), 4: (7, 10, 3, 7), 6: (3, 7, 0, 3), 2: (3, 7, 7, 10)}


def _win_mask():
    """The win predicate's off-diagonal mask (poqpfcjieu): every 10x10 cell EXCEPT the two diagonals.
    The win compares canvas vs target only on these 80 cells. Read straight from wvrremwltt."""
    m = np.ones((10, 10), dtype=bool)
    for i in range(10):
        m[i, i] = False
        m[i, 9 - i] = False
    return m


WMASK = _win_mask()
TOT = int(WMASK.sum())              # 80 off-diagonal cells


def _game_globals(g):
    """The cd82 module globals (where `sprites` + the level target pixels live). Degrade-closed → None."""
    try:
        return type(g).__init__.__globals__
    except Exception:
        return None


def _canvas(g):
    """The current player-canvas pixels (10x10), or None."""
    try:
        for s in g.current_level.get_sprites():
            if s.name == CANVAS_NAME:
                return s.pixels
    except Exception:
        pass
    return None


def _target(g):
    """The current level's target sprite pixels (eoqnvkspoa-pqwmeN-1), or None."""
    try:
        for s in g.current_level.get_sprites():
            if s.name.startswith(TARGET_PREFIX):
                return s.pixels
    except Exception:
        pass
    return None


def _win_phi(g):
    """(satisfied, total) off-diagonal cells already matching the target = the win predicate read at
    PARTIAL resolution. The monotone (sat,total) win-φ director. None on miss (caller blinds)."""
    canvas = _canvas(g)
    target = _target(g)
    if canvas is None or target is None:
        return None
    try:
        sat = int((canvas[WMASK] == target[WMASK]).sum())
        return sat, TOT
    except Exception:
        return None


def _level_colors(g):
    """The palette colors available this level (from the live pqkenviek swatches), deduped/ordered."""
    cols = []
    try:
        for s in g.current_level.get_sprites():
            if s.name.startswith(PALETTE_PREFIX):
                c = int(s.pixels[2, 2])
                if c not in cols:
                    cols.append(c)
    except Exception:
        pass
    return cols


def _has_corner(g):
    """True if this level carries the corner-trigger sprite (yxjfgsdkm) → corner fills are available."""
    try:
        return bool(getattr(g, "yxjfgsdkm", False)) or any(
            s.name == CORNER_NAME for s in g.current_level.get_sprites())
    except Exception:
        return False


def _ring_masks(env_proto):
    """The 8 ring-fill masks (pos -> 10x10 bool), built by firing each on a blank canvas via the
    game's OWN dynamics (a deepcopy world-model). The masks are level-invariant (the basket geometry
    is fixed), so this is computed once."""
    masks = {}
    for pos in range(8):
        try:
            e = copy.deepcopy(env_proto)
            g = e._game
            A = {a.name: a for a in e.action_space}
            g.xwmfgtlso = pos
            g.knqmgavuh = 7                       # a sentinel paint color not in any target
            g.azhynfjdiz()                        # rebuild the active basket at this ring position
            e.step(A[FILL_ACTION], data=None)     # fire the ring fill (instant in NORMAL mode)
            c = _canvas(g)
            masks[pos] = (c == 7) if c is not None else np.zeros((10, 10), bool)
        except Exception:
            masks[pos] = np.zeros((10, 10), bool)
    return masks


def _corner_masks():
    """The 4 corner-fill masks (even ring pos -> 10x10 bool), from coublenfir's fixed 3x4 slices."""
    out = {}
    for key, (rs, re_, cs, ce) in CORNER_SLICES.items():
        m = np.zeros((10, 10), bool)
        m[rs:re_, cs:ce] = True
        out[key] = m
    return out


def _nav_graph(g):
    """The ring navigation transition table pos -> {act: newpos}, read from the game's OWN nqhfiooufi
    logic (3x3 grid minus center). Returns (nav_fn, shortest_path_fn)."""
    nf = g.nfhykrqjp            # pos -> (r,c)
    fb = g.fbnqejrbl            # (r,c) -> pos

    def nav(pos, act):
        r, c = nf[pos]
        if act == 1:
            nr, nc = max(0, r - 1), c
        elif act == 2:
            nr, nc = min(2, r + 1), c
        elif act == 3:
            nr, nc = r, max(0, c - 1)
        else:
            nr, nc = r, min(2, c + 1)
        if (nr, nc) == (1, 1):     # the forbidden center → no move
            return pos
        return fb.get((nr, nc), pos)

    def path(src, dst):
        if src == dst:
            return []
        q = deque([(src, [])])
        seen = {src}
        while q:
            p, pl = q.popleft()
            for a in (1, 2, 3, 4):
                n = nav(p, a)
                if n == dst:
                    return pl + [a]
                if n not in seen:
                    seen.add(n)
                    q.append((n, pl + [a]))
        return None

    return nav, path


def beam_plan(target, colors, ring_masks, corner_masks, *, allow_corner,
              beam=400, max_fills=8):
    """BEAM-SEARCH the (region, color) FILL sequence whose layered composition matches `target`
    off-diagonal. The director is the win-φ off-diagonal match count. Returns (best_match, plan)
    where plan = [(kind, key, color)] (kind in {'ring','corner'}). Solved iff best_match == TOT."""
    regions = [("ring", p, ring_masks[p]) for p in range(8)]
    if allow_corner:
        regions += [("corner", k, corner_masks[k]) for k in sorted(corner_masks)]
    start = np.zeros((10, 10), int)

    def m(c):
        return int((c[WMASK] == target[WMASK]).sum())

    beam_list = [(start, [])]
    best = (m(start), [])
    for _ in range(max_fills):
        cand = []
        for canvas, plan in beam_list:
            for (kind, key, mask) in regions:
                for col in colors:
                    cc = canvas.copy()
                    cc[mask] = col
                    cand.append((m(cc), cc, plan + [(kind, key, col)]))
        cand.sort(key=lambda x: -x[0])
        new = []
        ns = set()
        for sc, cc, pl in cand:
            kk = cc[WMASK].tobytes()
            if kk in ns:
                continue
            ns.add(kk)
            new.append((cc, pl))
            if sc > best[0]:
                best = (sc, pl)
            if len(new) >= beam:
                break
        beam_list = new
        if best[0] == TOT:
            break
    return best[0], best[1]


def _select_color(env, g, A, col):
    """Select paint color `col` by clicking its palette swatch (ACTION6 → qbiojckwxl). The click
    coordinate is the swatch center mapped to display space (the game's own yrfgxhebei convention)."""
    if int(getattr(g, "knqmgavuh", -999)) == col:
        return True
    try:
        scale, ox, oy = g.camera._calculate_scale_and_offset()
        for s in g.current_level.get_sprites():
            if s.name.startswith(PALETTE_PREFIX) and int(s.pixels[2, 2]) == col:
                x = int((s.x + 2) * scale + ox)
                y = int((s.y + 2) * scale + oy)
                env.step(A[CLICK_ACTION], data={"x": x, "y": y})
                return int(getattr(g, "knqmgavuh", -999)) == col
    except Exception:
        pass
    return False


def _do_fill(env, g, A, path_fn, kind, key, col, plan_actions):
    """Execute one planned fill on the real env, appending the realized actions to `plan_actions`.
    Returns the new ring position. Degrade-closed: a failed sub-step is skipped, not raised."""
    _select_color(env, g, A, col)
    plan_actions.append((CLICK_ACTION, "select", col))
    # navigate the ring to `key` (the fill's ring position / the corner's even position)
    p = path_fn(g.xwmfgtlso, key)
    if p:
        for a in p:
            env.step(A[f"ACTION{a}"], data=None)
            plan_actions.append((f"ACTION{a}", "nav", None))
    last_obs = None
    if kind == "ring":
        last_obs = env.step(A[FILL_ACTION], data=None)
        plan_actions.append((FILL_ACTION, "fill", None))
    else:  # corner fill: click the corner-trigger sprite, then drain its animation
        try:
            ci = g.bmwcxxvjum()
        except Exception:
            ci = []
        if ci:
            last_obs = env.step(A[CLICK_ACTION], data=ci[0].data)
            plan_actions.append((CLICK_ACTION, "corner", None))
        cnt = 0
        while getattr(g, "yfobpcuef", False) and cnt < 60:
            last_obs = env.step(A[FILL_ACTION], data=None)
            cnt += 1
    return g.xwmfgtlso, last_obs


def solve_game(short_id="cd82", *, max_levels=6, max_nodes=40000, timeout=120,
               beam=400, max_fills=8, trace_path=None):
    t0 = time.time()
    deadline = t0 + timeout
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(short_id, scorecard_id=None, save_recording=False)
    if env is None:
        return {"game": short_id, "error": "make_returned_none", "levels": 0, "won": False}
    baseline = []
    for e in arc.get_environments():
        if e.game_id.split("-")[0] == short_id:
            baseline = e.baseline_actions or []
            break
    A = {a.name: a for a in env.action_space}
    obs = env.reset()
    g = env._game
    levels = getattr(obs, "levels_completed", 0) or 0
    state = obs.state.name

    # read the game's OWN fill dynamics + nav graph once (level-invariant)
    ring_masks = _ring_masks(env)
    corner_masks = _corner_masks()
    _, path_fn = _nav_graph(g)

    tracer = None
    if trace_path:
        try:
            from arc_trace import Tracer
            tracer = Tracer(trace_path, world="arc", subject=short_id, method="cd82-fillbeam",
                            clock_every=1, event_delta=0.1)
        except Exception:
            tracer = None

    per_level = []
    stall = None
    win_phi_max, win_phi_total, R_max = 0.0, TOT, 0.0
    win_phi_traj = []

    try:
        for lvl in range(max_levels):
            if time.time() > deadline:
                stall = "timeout"
                break
            base = getattr(obs, "levels_completed", 0) or 0
            target = _target(g)
            if target is None:
                stall = "no_target"
                break
            colors = _level_colors(g) or [0, 15]
            allow_corner = _has_corner(g)
            # plan the fills in fill-space (win-φ-directed beam)
            sc, plan = beam_plan(target, colors, ring_masks, corner_masks,
                                 allow_corner=allow_corner, beam=beam, max_fills=max_fills)
            wp = (sc / TOT) if TOT else 0.0
            win_phi_max = max(win_phi_max, wp)
            win_phi_traj.append(round(wp, 4))
            if sc < TOT:
                stall = "plan_incomplete"   # the beam could not reconstruct this target off-diagonal
                if tracer:
                    try:
                        tracer.emit(step=lvl, action=None, coherence_R=0.0, phi=round(wp, 6),
                                    goal_progress=sc, total=TOT, levels=base, state=state,
                                    won=False, force=True, stall=stall)
                    except Exception:
                        pass
                break
            # execute the plan on the real env (color-select + ring-nav + fill/corner-click)
            level_actions = []
            for (kind, key, col) in plan:
                if time.time() > deadline:
                    break
                _pos, last_obs = _do_fill(env, g, A, path_fn, kind, key, col, level_actions)
                if last_obs is not None:
                    obs = last_obs
                cur = getattr(g, "_score", base)
                if cur > base:
                    break
            # re-read the harness observation by a no-op-safe re-query: use the game score as truth
            cur = getattr(g, "_score", base)
            state = "NOT_FINISHED"
            if cur > base:
                levels = cur
                n_actions = len(level_actions)
                human = baseline[lvl] if lvl < len(baseline) else None
                rhae = round(human / n_actions, 3) if (human and n_actions) else None
                R = max(0.0, min(1.0, pow(2.718281828, -(n_actions / 80.0))))
                R_max = max(R_max, R)
                per_level.append({"level": lvl, "actions": n_actions, "human": human, "rhae": rhae,
                                  "fills": len(plan), "win_phi": round(wp, 4), "R": round(R, 3)})
                if tracer:
                    try:
                        tracer.emit(step=lvl, action="cd82", coherence_R=R, phi=round(wp, 6),
                                    goal_progress=sc, total=TOT, levels=levels, state="WIN",
                                    won=True, force=True, actions=n_actions, rhae=rhae)
                    except Exception:
                        pass
                # the engine raised the game WIN on the FINAL level's clear (next_level → win()) —
                # detect it on the harness obs and stop without trying to advance past the end
                try:
                    if getattr(obs, "state", None) is not None and obs.state.name == "WIN":
                        state = "WIN"
                        break
                except Exception:
                    pass
                # otherwise advance: the engine flips _next_level; pull it through a benign step so
                # the new level's canvas/target load (NORMAL mode applies _next_level on next step)
                try:
                    obs = env.step(A[NAV_ACTIONS[0]], data=None)
                    g = env._game
                    if getattr(obs, "state", None) is not None and obs.state.name == "WIN":
                        state = "WIN"
                        break
                except Exception:
                    pass
            else:
                stall = "exec_no_clear"
                break
    finally:
        if tracer:
            try:
                tracer.close()
            except Exception:
                pass

    eff = [p["rhae"] for p in per_level if p["rhae"] is not None]
    won = levels > 0
    return {
        "game": short_id, "method": "cd82-fillbeam", "levels": levels,
        "won": won, "state": ("WIN" if (won and stall is None) else state),
        "stall": None if (won and not stall) else (stall or "stuck"),
        "n_levels_baseline": len(baseline), "per_level": per_level,
        "mean_rhae": round(sum(eff) / len(eff), 3) if eff else None,
        "win_phi": {"max": round(win_phi_max, 4), "total": win_phi_total, "trajectory": win_phi_traj},
        "coherence_R": round(R_max, 4),
        "wall_s": round(time.time() - t0, 1),
    }


def _alarm(signum, frame):
    raise TimeoutError("hard wall-clock budget hit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, default="cd82")
    ap.add_argument("--max-levels", type=int, default=6)
    ap.add_argument("--max-nodes", type=int, default=int(os.environ.get("ARC_MAX_NODES", "40000")))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("ARC_TIMEOUT", "120")))
    ap.add_argument("--beam", type=int, default=int(os.environ.get("ARC_BEAM", "400")))
    ap.add_argument("--max-fills", type=int, default=8)
    ap.add_argument("--trace", type=str, default=os.environ.get(
        "ARC_TRACE_PATH", "probes/logs/cd82_solver_traces.jsonl"))
    ap.add_argument("--results", type=str, default="probes/logs/cd82_solver_results.jsonl")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(a.timeout + 30)
    try:
        r = solve_game(a.game, max_levels=a.max_levels, max_nodes=a.max_nodes,
                       timeout=a.timeout, beam=a.beam, max_fills=a.max_fills, trace_path=a.trace)
    except TimeoutError:
        r = {"game": a.game, "levels": 0, "won": False, "state": "TIMEOUT", "stall": "hard_timeout"}
    except Exception as e:
        r = {"game": a.game, "levels": 0, "won": False, "state": "ERROR",
             "stall": f"{type(e).__name__}:{e}"[:160]}
    finally:
        signal.alarm(0)

    try:
        os.makedirs(os.path.dirname(a.results), exist_ok=True)
        with open(a.results, "a") as f:
            f.write(json.dumps(r) + "\n")
    except Exception:
        pass
    try:                                          # mint a scorecard the OS/website can see
        import scorecard as _sc
        _sc.mint(r, method="cd82-fillbeam", source_coupled=True,
                 win_phi=r.get("win_phi"), coherence_R=r.get("coherence_R"))
    except Exception:
        pass

    if a.json:
        print(json.dumps(r))
    else:
        print(f"{a.game:8s} levels={r.get('levels', 0)} won={r.get('won')} state={r.get('state')} "
              f"stall={r.get('stall')} rhae={r.get('mean_rhae')} ({r.get('wall_s', '?')}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
