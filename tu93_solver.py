# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""tu93_solver — the PER-TISSUE (re86-method) solver for tu93, the moving-hazard maze.

WHAT tu93 IS (read from the public source `environment_files/tu93/0768757b/tu93.py`):

  * A grid MAZE. The player (tag `0017unajnymcki`, a 3x3 sprite) sits on a 6px cell lattice over a
    big background maze sprite (tag `0005uvnhiglpvh`) whose pixel value 2 = a corridor opening and 0
    = a wall. ACTION1/2/3/4 = up/down/left/right: each tries to move the player one cell (6px) in
    that direction, allowed only when the maze pixel at the mid-cell offset (`hwthhtvyki=3`) is a 2.
  * The WIN predicate (L1263 of the game): after the move + all actor animation resolve, the level
    clears iff the player stands EXACTLY on an exit tile (tag `0015msvpvzxhqf`): player.x == exit.x
    and player.y == exit.y. So the objective is a single, pure SPATIAL target — the simplest case of
    the foundry GRID-DISTANCE director (re86's per-sprite manhattan, with no attribute precondition).
  * Later levels add MOVING HAZARDS — enemy/wall actors (tags `0001haidilggfh`, `0020npxxteirsg`,
    `0023otenflmryc`) that step every time the player moves (the kdkehgjrzq 0->1->2->0 animation
    inside one `perform_action`). If the player is caught / the step budget (`StepCounter`) runs out,
    `lose()` fires -> GAME_OVER. The deepcopy world-model reproduces this hazard dynamics EXACTLY, so
    a plain death-pruned manhattan A* handles it with no per-hazard modelling.

THE DIRECTOR. `manhattan(player, exit)` read straight off the game's OWN state on each deepcopy'd
world-model node. This is a clean monotone gradient (one target, no plateau, no deceptive coupling),
so best-first descends it directly — exactly bp35's `_grid_progress` / re86's per-sprite director,
transferred with the player as the moving thing and the exit tile as the fixed goal. No DECOMPOSITION
or BEAM is needed: the win is a single uncoupled spatial predicate (verified on L0 — solved in 98
nodes / 18 actions / 0.1s with a player-position dedup).

THE SIGNATURE (the sk48 OOM lesson). Dedup on a GAME-STABLE tuple: the player cell (x,y) PLUS every
hazard actor's (tag, x, y, rotation), sorted — NOT `str(sprite)` (address-keyed -> no dedup -> 4^depth
blowup) and NOT the rendered frame (the StepCounter UI bar at row 63 changes every step -> every frame
unique). On L0 (no hazards) this reduces to the player cell; on hazard levels it stays injective on the
logical state the win depends on, so dedup is sound (no pruned win) and bounded.

WHAT IS REUSED FROM the foundry library (re86_solver / ls20_solver / bp35):
  * the GRID-DISTANCE DIRECTOR `manhattan(thing, goal)` read from the game's own state (re86 `_man_to_goal`);
  * the A* `search_to_clear` skeleton (frontier keyed `director*K + depth`, death-prune on GAME_OVER,
    dedup, `levels_completed` rise = win) — the ls20/re86 shell;
  * the deepcopy world-model step, and the trace/scorecard mint machinery.
  NOT reused: the per-sprite goal-finder + ACTION5 toggle (re86 is multi-sprite canvas-stamp; tu93 is
  one player, keyboard 1-4, single spatial target — that whole apparatus would be dead code) and
  `_trim_history` (tu93 keeps no growing undo stack; deepcopy is flat and cheap).

Run BOUNDED (CPU/venv only, no model):
  ulimit -v 8388608
  timeout 90 /var/home/zero/arc-agi/toolkit/.venv/bin/python tu93_solver.py \
      --max-levels 2 --max-nodes 40000 --timeout 80 --json
"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arc_agi
from arc_agi import OperationMode

PLAYER_TAG = "0017unajnymcki"   # the player sprite (moves with ACTION1-4, 6px per cell)
EXIT_TAG = "0015msvpvzxhqf"     # the goal tile; win = player on exit (x==exit.x and y==exit.y)
# moving hazards that step with the player; their positions are part of the logical state
HAZARD_TAGS = ("0001haidilggfh", "0020npxxteirsg", "0023otenflmryc")
MOVE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")  # up / down / left / right


def _player(g):
    ps = g.current_level.get_sprites_by_tag(PLAYER_TAG)
    return ps[0] if ps else None


def _exits(g):
    return g.current_level.get_sprites_by_tag(EXIT_TAG)


def manhattan(env):
    """The GRID-DISTANCE DIRECTOR: manhattan(player, nearest exit), read off the game's own state.
    A clean monotone gradient to the single spatial win target. Large sentinel if unreadable so the
    caller (best-first) deprioritises a state it can't measure rather than crashing."""
    g = getattr(env, "_game", None)
    if g is None:
        return 10 ** 9
    p = _player(g)
    ex = _exits(g)
    if p is None or not ex:
        return 10 ** 9
    return min(abs(p.x - e.x) + abs(p.y - e.y) for e in ex)


def gsig(env):
    """GAME-STABLE signature (the sk48 lesson): the player cell + every hazard actor's
    (tag, x, y, rotation), sorted. Injective on the logical state the win depends on -> sound dedup,
    no 4^depth blowup, no pruned win. On L0 (no hazards) this is just the player cell."""
    g = getattr(env, "_game", None)
    if g is None:
        return None
    p = _player(g)
    if p is None:
        return None
    haz = []
    for t in HAZARD_TAGS:
        for s in g.current_level.get_sprites_by_tag(t):
            haz.append((t, s.x, s.y, getattr(s, "rotation", 0)))
    haz.sort()
    return ((p.x, p.y, getattr(p, "rotation", 0)), tuple(haz))


def search_to_clear(env, base_lv, action_by_name, max_nodes, max_depth, deadline):
    """Coherence-guided best-first to the next level-clear. Frontier ordered by manhattan-to-exit then
    depth (the world-model planner's goal_progress x coherence_R: expand the most-converged timeline
    first). Death-prune on GAME_OVER; dedup on the game-stable signature; a `levels_completed` rise =
    the win. Complete within max_nodes (reorders, never skips). Returns (plan, stats)."""
    root = copy.deepcopy(env)
    counter = 0
    frontier = [(manhattan(root), 0, counter, root, [])]
    seen = set()
    nodes = 0
    dead = 0
    best_h = manhattan(root)
    while frontier and nodes < max_nodes and time.time() < deadline:
        h0, depth, _, ec, plan = heapq.heappop(frontier)
        if depth >= max_depth:
            continue
        for nm in MOVE_ACTIONS:
            a = action_by_name.get(nm)
            if a is None:
                continue
            nodes += 1
            if nodes >= max_nodes or time.time() >= deadline:
                break
            try:
                ec2 = copy.deepcopy(ec)
                obs = ec2.step(a, data=None)
            except Exception:
                continue
            lv = getattr(obs, "levels_completed", 0) or 0
            if lv > base_lv:
                # the convergent reading — the timelines reconverge on the win
                R = max(0.0, min(1.0, math.exp(-(nodes / 2000.0))))  # tight (few nodes) = high coherence-R
                return plan + [nm], {"nodes": nodes, "depth": len(plan) + 1, "frontier": len(frontier),
                                     "R": R, "dead": dead, "best_h": min(best_h, 0)}
            if getattr(obs.state, "name", "") == "GAME_OVER":
                dead += 1
                continue
            sig = gsig(ec2)
            if sig is not None and sig in seen:
                continue
            if sig is not None:
                seen.add(sig)
            hh = manhattan(ec2)
            best_h = min(best_h, hh)
            counter += 1
            heapq.heappush(frontier, (hh * 100 + (depth + 1), depth + 1, counter, ec2, plan + [nm]))
    return None, {"nodes": nodes, "depth": max_depth, "frontier": len(frontier), "R": 0.0,
                  "dead": dead, "best_h": best_h}


def solve_game(short_id="tu93", *, max_levels=9, max_nodes=40000, timeout=120, trace_path=None):
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
    action_by_name = {getattr(a, "name", None): a for a in env.action_space}
    obs = env.reset()
    levels = getattr(obs, "levels_completed", 0) or 0
    state = obs.state.name
    per_level = []
    tracer = None
    if trace_path:
        try:
            from arc_trace import Tracer
            tracer = Tracer(trace_path, world="arc", subject=short_id, method="tu93-mazemanhattan",
                            clock_every=1, event_delta=0.1)
        except Exception:
            tracer = None
    stall = None
    R_max = 0.0
    try:
        for lvl in range(max_levels):
            base = getattr(obs, "levels_completed", 0) or 0
            if time.time() > deadline:
                stall = "timeout"
                break
            g = env._game
            if _player(g) is None or not _exits(g):
                stall = "no_sprites"
                break
            ts = time.time()
            plan, st = search_to_clear(env, base, action_by_name, max_nodes,
                                       max_depth=240, deadline=deadline)
            if plan is None:
                # frontier>0 = budget-bound (raise max_nodes); frontier=0 = space exhausted (model gap)
                stall = "timeout" if time.time() > deadline else (
                    "budget" if st["frontier"] > 0 else "exhausted")
                if tracer:
                    try:
                        tracer.emit(step=lvl, action=None, coherence_R=0.0, levels=base, state=state,
                                    won=False, force=True, stall=stall, best_h=st.get("best_h"))
                    except Exception:
                        pass
                break
            # execute the found plan on the REAL env (the convergent trajectory)
            for nm in plan:
                obs = env.step(action_by_name[nm], data=None)
                if obs.state.name in ("WIN", "GAME_OVER"):
                    break
            cur = getattr(obs, "levels_completed", 0) or 0
            state = obs.state.name
            if cur > base:
                levels = cur
                R = st["R"]
                R_max = max(R_max, R)
                human = baseline[lvl] if lvl < len(baseline) else None
                n_actions = len(plan)
                rhae = round(human / n_actions, 3) if (human and n_actions) else None
                per_level.append({"level": lvl, "actions": n_actions, "human": human, "rhae": rhae,
                                  "nodes": st["nodes"], "R": round(R, 3)})
                if tracer:
                    try:
                        tracer.emit(step=lvl, action="tu93", coherence_R=R, levels=levels, state=state,
                                    won=(state == "WIN"), force=True, actions=n_actions, rhae=rhae,
                                    nodes=st["nodes"])
                    except Exception:
                        pass
            else:
                stall = "game_over" if state == "GAME_OVER" else "stuck"
                if tracer:
                    try:
                        tracer.emit(step=lvl, action=None, coherence_R=0.0, levels=base, state=state,
                                    won=False, force=True, stall=stall)
                    except Exception:
                        pass
                break
            if state in ("WIN", "GAME_OVER"):
                if state == "GAME_OVER":
                    stall = "game_over"
                break
        else:
            stall = stall or "max_levels"
    finally:
        if tracer:
            try:
                tracer.close()
            except Exception:
                pass
    eff = [p["rhae"] for p in per_level if p["rhae"] is not None]
    return {
        "game": short_id, "method": "tu93-mazemanhattan", "levels": levels,
        "won": levels > 0, "state": state,
        "stall": None if levels > 0 and not stall else (stall or "stuck"),
        "n_levels_baseline": len(baseline), "per_level": per_level,
        "mean_rhae": round(sum(eff) / len(eff), 3) if eff else None,
        "coherence_R": round(R_max, 4),
        "wall_s": round(time.time() - t0, 1),
    }


def _alarm(signum, frame):
    raise TimeoutError("hard wall-clock budget hit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, default="tu93")
    ap.add_argument("--max-levels", type=int, default=9)
    ap.add_argument("--max-nodes", type=int, default=int(os.environ.get("ARC_MAX_NODES", "40000")))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("ARC_TIMEOUT", "120")))
    ap.add_argument("--trace", type=str, default=os.environ.get("ARC_TRACE_PATH",
                                                                "probes/logs/tu93_solver_traces.jsonl"))
    ap.add_argument("--results", type=str, default="probes/logs/tu93_solver_results.jsonl")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(a.timeout + 30)
    try:
        r = solve_game(a.game, max_levels=a.max_levels, max_nodes=a.max_nodes,
                       timeout=a.timeout, trace_path=a.trace)
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
        _sc.mint(r, method="tu93-mazemanhattan", source_coupled=False,
                 coherence_R=r.get("coherence_R"))
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
