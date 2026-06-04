# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""lp85_solver — the PER-TISSUE solver for lp85, the ring-rotation goal-alignment puzzle.

WHY a per-tissue solver. the general solver downloads lp85 and auto-finds the win predicate
(`khartslnwa`, read by the manual-loop fall-through reader) but gets levels=0 / "exhausted":

  * lp85 is a CLICK game (`available_actions=[6]`). The general candidate sampler clicks the 8
    biggest frame centroids — it cannot reliably hit the *button* sprites that drive the puzzle,
    so most of its branches are no-ops.
  * The win predicate `khartslnwa` (read at partial resolution) is only a 2-GUARD conjunction
    (`for marker in bghvgbtwcb: goal at +1,+1?` ∧ `for marker in fdgmtkfrxl: goal-o at +1,+1?`).
    The manual reader counts SATISFIED GUARD-BLOCKS, so it reads (1,2) at root with a single
    `bghvgbtwcb` marker and stays flat at (1,2) until the exact win — a deceptive director with
    no gradient for best-first to descend.

THE MECHANIC (read from the public source + verified live on the real env).
Every level is one or more CYCLIC RINGS. Each button is tagged `button_<MAP>_<R|L>`; clicking it
rotates the sprites sitting on ring `<MAP>` by one cell forward (R) or backward (L) — the
`chmfaflqhy`/`step` permutation. A `goal` (or `goal-o`) sprite rides the ring; a `bghvgbtwcb`
(or `fdgmtkfrxl`) MARKER is a FIXED slot. The level is won when, for EVERY marker, its matching
goal sprite sits at (marker.x+1, marker.y+1). Each click spends one StepCounter step; running out
of steps = lose. So the objective is: rotate each goal sprite onto its marker's target slot.

THE PER-TISSUE DIRECTOR (the win-φ at FINE resolution). Read straight off the game's OWN state:
for each marker, take target_slot=(marker.x+1, marker.y+1), find the matching goal sprite(s)
(`goal` for `bghvgbtwcb`, `goal-o` for `fdgmtkfrxl`), and take the min grid-MANHATTAN distance
from a goal sprite to the target slot. `sat` = markers at distance 0. The director to MINIMIZE is
`(total_markers - sat)*BIG + sum_min_distances` — a clean monotone gradient toward each win, with
no plateau (unlike the 2-guard predicate count). This is bp35's `_grid_progress` (grid-distance
director) transferred verbatim in shape (player→goal-sprite, gem→marker-target-slot), applied
per-marker with min-distance assignment when a level has several goals.

CANDIDATE MOVES (the click-target fix). Instead of frame centroids, enumerate ONE ACTION6 click
per `button_*` sprite, at the sprite's center grid cell mapped through the camera's grid→display
inverse (built once per level, ~2 ms). Branching = #buttons (2 on L0 … up to 72 on L5), bounded
by max-nodes/max-depth so the click-game frontier can't detonate.

WHAT IS REUSED FROM THE FOUNDRY LIBRARY (the library-validation):
  * the A* `search_to_clear` skeleton (frontier keyed `director*K + depth`, GAME_OVER death-prune,
    dedup on a GAME-STABLE signature, `levels_completed` rise = win) — the bp35/re86 shell;
  * the GAME-STABLE signature `_gcanon`/`gen_sig` from the general solver (sprite identity by
    (type,x,y,rotation), NOT str(address) — the sk48 dedup fix), so deepcopies collapse;
  * the deepcopy world-model `_apply` (the game's OWN `step` on `copy.deepcopy(env)`, exact
    transition, no model) and the scorecard/arc_trace machinery.
  NOT reused: the frame-centroid click sampler (replaced by button-center clicks) and the
  scalar/mask field-scan (lp85's win is the ring-alignment director, computed here directly).

Run BOUNDED (CPU/venv only, NO model):
  ulimit -v 8388608
  timeout 90 /var/home/zero/arc-agi/toolkit/.venv/bin/python lp85_solver.py \
      --max-levels 8 --max-nodes 40000 --timeout 80 --json
"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arc_agi
from arc_agi import OperationMode

# reuse the foundry library: deepcopy world-model step. NOTE on the signature: the general solver's
# gen_sig + heavy_static_skip is NOT reused here — its perturbation-derived skip drops `_levels` /
# `_clean_levels` as "large + root-static" (one click moves only a couple sprites, so the heavy bulk
# reads inert), but lp85 keeps its LIVE sprite positions inside `current_level._sprites` (=
# `_levels[level_index]`). Dropping `_levels` collapses every ring-state to one signature → dedup
# kills the whole frontier → instant "exhausted" (measured: 6 distinct ring states all hashed
# identical). lp85 needs the TIGHT game-stable signature below (the ls20/sk48 pattern): hash the
# sorted positions of the win-relevant (non-button) sprites + level index.
from foundry_core import _apply

# ── lp85 tags (read from the public source) ───────────────────────────────────
MARKER_GOAL = {            # marker tag -> the goal tag whose sprite must sit at (marker.x+1, marker.y+1)
    "bghvgbtwcb": "goal",
    "fdgmtkfrxl": "goal-o",
}
GRID_STEP = 3              # `crxpafuiwp` — rings live on a 3px lattice; one click = one cell = 3px
BIG = 1000                 # per-unsatisfied-marker weight (dominates the distance tiebreak)


def _sprites(g):
    try:
        return list(g.current_level._sprites)
    except Exception:
        return []


def _tag(s):
    return s.tags[0] if getattr(s, "tags", None) else None


def director(g):
    """(sat, total, cost): the per-tissue win-φ at FINE resolution, read off the game's own state.
    For each marker, min grid-manhattan from a matching goal sprite to the marker's target slot
    (marker.x+1, marker.y+1). sat = markers at distance 0; cost = (total-sat)*BIG + sum_min_dist.
    (0, 0, 0) when a level somehow has no markers (degenerate — caller treats as trivially won)."""
    sprites = _sprites(g)
    by_tag = {}
    for s in sprites:
        by_tag.setdefault(_tag(s), []).append((s.x, s.y))
    sat = total = 0
    cost = 0
    for mtag, gtag in MARKER_GOAL.items():
        markers = by_tag.get(mtag, [])
        goals = by_tag.get(gtag, [])
        for (mx, my) in markers:
            total += 1
            slot = (mx + 1, my + 1)
            if not goals:
                cost += BIG * 4                       # no goal sprite present → unreachable-far
                continue
            dmin = min(abs(gx - slot[0]) + abs(gy - slot[1]) for (gx, gy) in goals)
            # grid distance in px → ring-cell steps (3px lattice); the live win check is exact at d==0
            if dmin == 0:
                sat += 1
            else:
                cost += BIG + (dmin // GRID_STEP)     # one BIG per unsatisfied marker + cell-distance
    return (sat, total, cost)


# ── decomposition: the WELDED-ACTUATOR joint planner (the real L5 structure, measured not assumed) ──
# WHAT L5 actually is. Three goals on coupled rings + 36 buttons — but the 36 buttons are physically
# STACKED into only 7 distinct clickable cells: clicking one cell fires EVERY button at it, so one
# click rotates a whole GROUP of rings simultaneously (measured: cell (24,14) welds rings 1-8, (54,14)
# welds 9-16, (39,44) welds 17-24, (12,27)=ABC, (42,27)=DEF, (27,57)=GHI, (50,54)=25/26/27). The
# `on_set_level` sys_click dedup does NOT un-weld them — the engine's step handler rotates on the FIRST
# tag (`"button" in tags[0]`), not `sys_click`, so all stacked buttons fire. This is why the flat
# 36-button director plateaus and why a per-marker "disjoint ring-set" decomposition is UNSOUND (the
# rings can't be actuated independently). The markers are GENUINELY jointly coupled.
#
# WHY a joint plan still exists. Even welded, the reachable orbit of the three goal sprites under the 7
# group-clicks is small (measured: 14,688 distinct goal-configs) and the all-3-satisfied target IS in
# it (shortest plan = 19 clicks, well under the 80-step budget). The flat solver never found it only
# because each engine step is a ~10 ms deepcopy → it cannot explore 14k states in the wall budget. The
# decomposition that works is therefore not "split into independent markers" but "PLAN ON A FAST MODEL":
# extract the exact welded permutation for each group ONCE, BFS over goal-config tuples in pure Python
# (no deepcopy), reconstruct the click sequence, then replay it on the real env. All read from the live
# config + live stacked-button geometry — no per-level hardcoding.


def _ring_step(g):
    """{ring_map: {grid_px_pos: next_grid_px_pos}} under R-rotation, for the CURRENT level. Read from
    the live `g.uopmnplcnv[g.ucybisahh]`; each ring's `qcmzcjocmj` maps ordinal -> (y,x) NamedTuple,
    grid-px = ordinal_pos * GRID_STEP. R only (every lp85 button is `_R`)."""
    cfg = getattr(g, "uopmnplcnv", None)
    name = getattr(g, "ucybisahh", None)
    if not cfg or name not in cfg:
        return {}
    out = {}
    for ring, spec in cfg[name].items():
        try:
            order = spec["qcmzcjocmj"]
            mx = spec["oxbwsencfv"]
        except Exception:
            continue
        if mx <= 1:
            continue
        step = {}
        for n, pos in order.items():
            nn = 1 if n == mx else n + 1
            try:
                step[(pos.x * GRID_STEP, pos.y * GRID_STEP)] = (order[nn].x * GRID_STEP,
                                                                order[nn].y * GRID_STEP)
            except Exception:
                continue
        if step:
            out[ring] = step
    return out


def _welded_groups(g):
    """Group buttons by the CELL they share (the physical stacking). Returns a list of
    (representative_button_grid_cell, [ring_maps]) — one entry per distinct clickable cell, since a
    click at that cell fires every stacked button (rotates all those rings at once). The button cell
    used is the sprite (x, y); the click is later resolved to a display coord at the sprite center."""
    by_cell = {}
    for s in _sprites(g):
        r = _button_ring(_tag(s))
        if r is None:
            continue
        key = (s.x, s.y, getattr(s, "width", 1), getattr(s, "height", 1))
        by_cell.setdefault(key, []).append(r)
    return [((x, y, w, h), rings) for (x, y, w, h), rings in by_cell.items()]


def _apply_group(positions, group_rings, ring_step):
    """Apply ONE welded click to a goal-position tuple: every goal sprite sitting on a cell of any ring
    in `group_rings` advances one cell along that ring (snapshot/simultaneous semantics — moves are
    computed from the CURRENT positions then applied, matching the engine's two-pass step). A goal not
    on any group ring stays put. First ring that owns the cell wins (rings in a group are disjoint in
    cells on lp85, so the order is immaterial)."""
    moved = {}
    for r in group_rings:
        rs = ring_step.get(r, {})
        for p in positions:
            if p in rs and p not in moved:
                moved[p] = rs[p]
    return tuple(moved.get(p, p) for p in positions)


def plan_welded(g, *, max_states=200000):
    """The fast model-only joint planner. Reads the live ring config + welded button groups, then BFS
    over GOAL-CONFIG tuples (no deepcopy) to the all-markers-satisfied target. Returns (group_sequence,
    stats) where group_sequence is a list of group-indices into `groups`, or (None, stats) if the level
    isn't a multi-goal coupling or the target isn't in the reachable orbit within `max_states`. The
    sprite identity is positional (goal sprites are order-stable in `_sprites`), so per-goal targets are
    matched by the win's set membership: the win is `slots ⊆ {goal positions}`."""
    from collections import deque
    ring_step = _ring_step(g)
    if not ring_step:
        return None, {"stop": "no_rings"}
    groups = _welded_groups(g)
    if not groups:
        return None, {"stop": "no_buttons"}
    by_tag = {}
    for s in _sprites(g):
        by_tag.setdefault(_tag(s), []).append((s.x, s.y))
    # the per-marker target slots, by goal tag (bghvgbtwcb->goal, fdgmtkfrxl->goal-o)
    slots = set()
    goal_tags = set()
    for mtag, gtag in MARKER_GOAL.items():
        for (mx, my) in by_tag.get(mtag, []):
            slots.add((mx + 1, my + 1))
            goal_tags.add(gtag)
    if len(slots) < 2:
        return None, {"stop": "single_marker"}      # not a joint coupling → flat search handles it
    start = tuple((s.x, s.y) for s in _sprites(g) if _tag(s) in goal_tags)
    if not start:
        return None, {"stop": "no_goals"}

    def won(positions):
        return slots <= set(positions)

    group_rings = [rings for (_cell, rings) in groups]
    seen = {start: None}
    q = deque([start])
    states = 0
    target = None
    while q:
        cur = q.popleft()
        if won(cur):
            target = cur
            break
        for gi, rings in enumerate(group_rings):
            nxt = _apply_group(cur, rings, ring_step)
            if nxt not in seen:
                seen[nxt] = (cur, gi)
                states += 1
                if states > max_states:
                    return None, {"stop": "state_cap", "states": states}
                q.append(nxt)
    if target is None:
        return None, {"stop": "unreachable", "states": len(seen)}
    seq = []
    c = target
    while seen[c] is not None:
        prev, gi = seen[c]
        seq.append(gi)
        c = prev
    seq.reverse()
    return seq, {"stop": "planned", "states": len(seen), "plan_len": len(seq), "groups": len(groups)}


# ── candidate clicks: one ACTION6 per button sprite, at its center, through the camera inverse ──
def _grid_to_display(cam, dmax=64):
    """Build the grid-cell -> display-coord inverse once (camera.display_to_grid is the only public
    mapping; scanning the 64x64 display space is ~2 ms). First display coord wins per grid cell."""
    inv = {}
    for dx in range(dmax):
        for dy in range(dmax):
            try:
                r = cam.display_to_grid(dx, dy)
            except Exception:
                r = None
            if r and r not in inv:
                inv[r] = (dx, dy)
    return inv


def _button_ring(t):
    """`button_<MAP>_<R|L>` -> `<MAP>` (the ring this button rotates). None if not a 3-part button tag."""
    if not t or not t.startswith("button_"):
        return None
    parts = t.split("_")
    return parts[1] if len(parts) == 3 else None


def button_clicks(g, ring_filter=None):
    """List of (display_x, display_y) — one click per `button_*` sprite, at the sprite's center grid
    cell mapped to a display coordinate. The center is used so the click lands inside the button box;
    falls back to the top-left cell if the center isn't in the inverse map. NOTE on lp85: many buttons
    are STACKED at the same cell, so distinct clicks << #buttons (see `_welded_groups`); the flat
    search dedups the resulting no-different-effect branches. `ring_filter` (a set of ring-map names)
    optionally keeps only buttons that rotate one of those rings — a general capability, unused by the
    current flat path (which clicks all buttons)."""
    cam = getattr(g, "camera", None)
    if cam is None:
        return []
    inv = _grid_to_display(cam)
    out = []
    for s in _sprites(g):
        t = _tag(s)
        if not t or not t.startswith("button_"):
            continue
        if ring_filter is not None and _button_ring(t) not in ring_filter:
            continue
        cx = s.x + getattr(s, "width", 1) // 2
        cy = s.y + getattr(s, "height", 1) // 2
        d = inv.get((cx, cy)) or inv.get((s.x, s.y))
        if d is not None:
            out.append(d)
    return out


def candidate_moves(env, action, ring_filter=None):
    """[(action, {x,y})] — one branch per button click. Click games in lp85 have only ACTION6.
    `ring_filter` (a set of ring-map names) optionally restricts the branching to those rings' buttons
    (general capability; the flat path passes None = all buttons)."""
    g = getattr(env, "_game", None)
    if g is None:
        return []
    return [(action, {"x": dx, "y": dy}) for (dx, dy) in button_clicks(g, ring_filter=ring_filter)]


def lp_sig(env, lv):
    """TIGHT game-stable signature: (level, sorted positions of the win-relevant sprites). Buttons are
    FIXED (never move), so only the markers/goals/tiles carry the ring state. Keying on (tag, x, y)
    of the non-button sprites is injective on the logical ring configuration and stable across
    deepcopies (no memory-address leak — the sk48/ls20 dedup discipline). None on any miss → caller
    skips dedup for that node (degrade-closed)."""
    g = getattr(env, "_game", None)
    if g is None:
        return None
    try:
        items = tuple(sorted((_tag(s), s.x, s.y) for s in _sprites(g)
                             if not (_tag(s) or "").startswith("button_")))
        return (lv, items)
    except Exception:
        return None


def heuristic(env, depth):
    g = getattr(env, "_game", None)
    if g is None:
        return depth
    sat, total, cost = director(g)
    return cost + depth          # cost already weights unsatisfied markers; depth breaks ties


def search_to_clear(env, obs0, base_lv, action, *, max_nodes, max_depth, deadline,
                    tracer=None, level=0):
    """Best-first to the next level-clear (levels_completed rise), guided by the ring-distance
    director, GAME_OVER-pruned, bounded by nodes/depth/wall-clock. Returns (plan, stats)."""
    root = copy.deepcopy(env)
    g0 = root._game
    sat0, total, _ = director(g0)
    counter = 0
    frontier = [(heuristic(root, 0), 0, counter, root, [])]
    seen = set()
    nodes = dead = 0
    best_sat = sat0
    best_cost = director(g0)[2]
    while frontier and nodes < max_nodes:
        if time.time() > deadline:
            return None, {"nodes": nodes, "frontier": len(frontier), "R": 0.0, "dead": dead,
                          "best_sat": best_sat, "total": total, "stop": "timeout"}
        _, _, _, ec, plan = heapq.heappop(frontier)
        if len(plan) >= max_depth:
            continue
        for (a, data) in candidate_moves(ec, action):
            nodes += 1
            if nodes >= max_nodes or time.time() > deadline:
                break
            try:
                ec2, obs2 = _apply(ec, a, data)
            except Exception:
                continue
            lv = getattr(obs2, "levels_completed", 0) or 0
            if lv > base_lv:                                  # level cleared (win fired mid-plan)
                R = max(0.0, min(1.0, pow(2.718281828, -(nodes / 2000.0))))
                return plan + [(a, data)], {"nodes": nodes, "depth": len(plan) + 1,
                                            "frontier": len(frontier), "R": R, "dead": dead,
                                            "best_sat": best_sat, "total": total, "stop": "win"}
            st = getattr(obs2, "state", None)
            if st is not None and getattr(st, "name", "") == "GAME_OVER":
                dead += 1
                continue                                      # out of steps → dead branch
            sig = lp_sig(ec2, lv)
            if sig is not None and sig in seen:
                continue
            if sig is not None:
                seen.add(sig)
            sat, _, cost = director(ec2._game)
            if sat > best_sat or (sat == best_sat and cost < best_cost):
                if sat > best_sat or cost < best_cost:
                    best_sat = max(best_sat, sat)
                    best_cost = min(best_cost, cost)
                    if tracer is not None and total:
                        try:
                            cstate = None
                            try:
                                from arc_algebra import compact_state
                                cstate = compact_state(ec2)
                            except Exception:
                                cstate = None
                            tracer.emit(step=nodes, action="ACTION6", coherence_R=None,
                                        phi=round(sat / total, 6), goal_progress=sat, levels=lv,
                                        state="SEARCHING", cstate=cstate, force=True, total=total,
                                        depth=len(plan) + 1, frontier=len(frontier), level=level,
                                        edge="search")
                        except Exception:
                            pass
            counter += 1
            heapq.heappush(frontier, (heuristic(ec2, len(plan) + 1), len(plan) + 1, counter,
                                      ec2, plan + [(a, data)]))
    stop = "exhausted" if not frontier else "node_budget"
    return None, {"nodes": nodes, "frontier": len(frontier), "R": 0.0, "dead": dead,
                  "best_sat": best_sat, "total": total, "stop": stop}


def _group_click(g, group_cell):
    """Resolve a welded button group's grid cell (x, y, w, h) to a display click coord at the sprite
    center, through the camera inverse (the same mapping `button_clicks` uses). None if uninvertible."""
    cam = getattr(g, "camera", None)
    if cam is None:
        return None
    inv = _grid_to_display(cam)
    x, y, w, h = group_cell
    return inv.get((x + w // 2, y + h // 2)) or inv.get((x, y))


def solve_decomposed(env, base_lv, action, *, max_nodes, max_depth, deadline):
    """The welded-orbit joint planner driver. Plans the whole multi-goal coupling at ONCE on the fast
    pure-Python model (`plan_welded`, no per-step deepcopy), then materializes the group-click sequence
    into real (action, {x,y}) moves and replays it on a working deepcopy to confirm the level clears.
    Returns (plan, stats) — plan is the click list, stats carries `R`/`depth`/`states`. None when the
    level isn't a multi-goal coupling or the welded orbit doesn't contain the win (caller falls through
    to the flat search — degrade-closed). `max_nodes`/`max_depth` are kept in the signature for caller
    symmetry; the joint planner is bounded by its own `max_states` (the orbit is tiny: ~15k for L5)."""
    g0 = env._game
    seq, pst = plan_welded(g0, max_states=max(max_nodes, 200000))
    if seq is None:
        return None, {"stop": "welded_" + pst.get("stop", "fail"), "nodes": pst.get("states", 0)}
    groups = _welded_groups(g0)
    # group index -> display click coord (resolved once on the root game)
    click_for = {}
    for gi, (cell, _rings) in enumerate(groups):
        d = _group_click(g0, cell)
        if d is None:
            return None, {"stop": "welded_uninvertible_click", "nodes": pst.get("states", 0)}
        click_for[gi] = {"x": d[0], "y": d[1]}
    plan = [(action, click_for[gi]) for gi in seq]
    # replay on a deepcopy to CONFIRM the model plan actually clears on the real engine (degrade-closed:
    # if the engine and the model ever diverge, we refuse the plan rather than ship a bad one)
    cur = copy.deepcopy(env)
    last_obs = None
    for (a, data) in plan:
        try:
            cur, last_obs = _apply(cur, a, data)
        except Exception:
            return None, {"stop": "welded_replay_rejected", "nodes": pst.get("states", 0)}
        if getattr(getattr(last_obs, "state", None), "name", "") in ("WIN", "GAME_OVER"):
            break
    lv = getattr(last_obs, "levels_completed", 0) or 0
    if lv <= base_lv:
        return None, {"stop": "welded_no_clear", "nodes": pst.get("states", 0),
                      "plan_len": len(plan), "states": pst.get("states")}
    R = max(0.0, min(1.0, pow(2.718281828, -(pst.get("states", 0) / 8000.0))))
    return plan, {"stop": "win", "nodes": pst.get("states", 0), "depth": len(plan), "R": R,
                  "decomposed": True, "welded": True, "states": pst.get("states"),
                  "groups": pst.get("groups")}


def solve_game(short_id="lp85", *, max_levels=8, max_nodes=40000, max_depth=120, timeout=120,
               trace_path=None):
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
    action = list(env.action_space)[0]      # lp85 is single-action (ACTION6, the click)
    obs = env.reset()
    levels = getattr(obs, "levels_completed", 0) or 0
    state = obs.state.name
    per_level = []
    tracer = None
    if trace_path:
        from arc_trace import Tracer
        tracer = Tracer(trace_path, world="arc", subject=short_id, method="per-tissue-lp85",
                        clock_every=1, event_delta=0.1)
    stall = None
    win_phi_max, win_phi_total, R_max = 0.0, None, 0.0
    try:
        for lvl in range(max_levels):
            base = getattr(obs, "levels_completed", 0) or 0
            if time.time() > deadline:
                stall = "timeout"
                break
            # DECOMPOSE-FIRST on a coupled level. The welded joint planner (`solve_decomposed` →
            # `plan_welded`) plans the whole multi-goal coupling on the fast pure-Python model (no
            # per-step deepcopy), so a level whose goals are JOINTLY coupled through stacked buttons (L5,
            # L7 — measured ≥2 target slots) is solved in one shot. The flat 72-button frontier would
            # burn the whole wall-budget here (each engine step is ~10 ms; the orbit is ~15k states) and
            # never finish. On simple levels (L0-L4: single marker) `plan_welded` returns single_marker →
            # plan stays None → the flat path runs exactly as before (current clears preserved).
            plan = st = None
            plan, st = solve_decomposed(env, base, action, max_nodes=max_nodes,
                                        max_depth=max_depth, deadline=deadline)
            if plan is None:
                plan, st = search_to_clear(env, obs, base, action, max_nodes=max_nodes,
                                           max_depth=max_depth, deadline=deadline, tracer=tracer, level=lvl)
            _bs, _tt = (st.get("best_sat") or 0), st.get("total")
            if _tt and _tt > 0 and (_bs / _tt) > win_phi_max:
                win_phi_max, win_phi_total = _bs / _tt, _tt
            R_max = max(R_max, st.get("R", 0.0) or 0.0)
            if plan is None:
                stall = st.get("stop", "stuck")
                if tracer:
                    tracer.emit(step=lvl, action=None, coherence_R=0.0,
                                goal_progress=st.get("best_sat"), levels=base, state=state,
                                won=False, force=True, stall=stall, nodes=st.get("nodes"),
                                best=st.get("best_sat"), total=st.get("total"))
                break
            for si, (a, data) in enumerate(plan):
                obs = env.step(a, data=data)
                if tracer:
                    try:
                        cstate = None
                        try:
                            from arc_algebra import compact_state
                            cstate = compact_state(env)
                        except Exception:
                            cstate = None
                        sat, total, _ = director(env._game)
                        tracer.emit(step=si, action="ACTION6", coherence_R=st["R"],
                                    phi=(round(sat / total, 6) if total else None),
                                    goal_progress=sat,
                                    levels=getattr(obs, "levels_completed", 0) or 0,
                                    state=obs.state.name, won=(obs.state.name == "WIN"),
                                    cstate=cstate, force=True, edge="path", level=lvl, total=total)
                    except Exception:
                        pass
                if obs.state.name in ("WIN", "GAME_OVER"):
                    break
            levels = getattr(obs, "levels_completed", 0) or 0
            state = obs.state.name
            n_actions = len(plan)
            human = baseline[lvl] if lvl < len(baseline) else None
            rhae = round(human / n_actions, 3) if (human and n_actions) else None
            per_level.append({"level": lvl, "actions": n_actions, "human": human, "rhae": rhae,
                              "nodes": st["nodes"], "R": round(st["R"], 3)})
            if tracer:
                tracer.emit(step=lvl, action="per-tissue", coherence_R=st["R"],
                            goal_progress=st.get("best_sat"), levels=levels, state=state,
                            won=(state == "WIN"), force=True, actions=n_actions, rhae=rhae,
                            nodes=st["nodes"])
            if state in ("WIN", "GAME_OVER"):
                if state == "GAME_OVER":
                    stall = "game_over"
                break
        else:
            stall = stall or "max_levels"
    finally:
        if tracer:
            tracer.close()
    won = state == "WIN"
    eff = [p["rhae"] for p in per_level if p["rhae"] is not None]
    return {
        "game": short_id, "levels": levels, "won": won, "state": state,
        "stall": None if won else (stall or "stuck"),
        "n_levels_baseline": len(baseline), "per_level": per_level,
        "mean_rhae": round(sum(eff) / len(eff), 3) if eff else None,
        "win_phi": {"max": round(win_phi_max, 4), "total": win_phi_total},
        "coherence_R": round(R_max, 4),
        "wall_s": round(time.time() - t0, 1), "tags_note": "click ring-rotation; per-tissue director",
    }


def _alarm(signum, frame):
    raise TimeoutError("hard wall-clock budget hit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, default="lp85")
    ap.add_argument("--max-levels", type=int, default=8)
    ap.add_argument("--max-nodes", type=int, default=int(os.environ.get("ARC_MAX_NODES", "40000")))
    ap.add_argument("--max-depth", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("ARC_TIMEOUT", "90")))
    ap.add_argument("--trace", type=str, default=os.environ.get("ARC_TRACE_PATH",
                                                                "probes/logs/lp85_traces.jsonl"))
    ap.add_argument("--results", type=str, default="probes/logs/lp85_results.jsonl")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(a.timeout + 30)
    try:
        r = solve_game(a.game, max_levels=a.max_levels, max_nodes=a.max_nodes,
                       max_depth=a.max_depth, timeout=a.timeout, trace_path=a.trace)
    except TimeoutError:
        r = {"game": a.game, "levels": 0, "won": False, "state": "TIMEOUT", "stall": "hard_timeout"}
    except Exception as e:
        r = {"game": a.game, "levels": 0, "won": False, "state": "ERROR",
             "stall": f"{type(e).__name__}:{e}"[:160]}
    finally:
        signal.alarm(0)

    os.makedirs(os.path.dirname(a.results), exist_ok=True)
    with open(a.results, "a") as f:
        f.write(json.dumps(r) + "\n")
    try:
        import scorecard as _sc
        _sc.mint(r, method="per-tissue-lp85", source_coupled=True,
                 win_phi=r.get("win_phi"), coherence_R=r.get("coherence_R"))
    except Exception:
        pass
    if a.json:
        print(json.dumps(r))
    else:
        print(f"{a.game:8s} levels={r.get('levels',0)} won={r.get('won')} state={r.get('state')} "
              f"stall={r.get('stall')} rhae={r.get('mean_rhae')} win_phi={r.get('win_phi')} "
              f"({r.get('wall_s','?')}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
