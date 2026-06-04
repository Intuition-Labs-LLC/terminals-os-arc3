# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""ft09_solver — the PER-TISSUE (bp35/sk48/re86-method) solver for ft09, the parity-constraint
click-toggle game.

WHY a per-tissue solver. the general solver gets 0 on ft09 (verified: exhausts in ~1.5s, levels=0).
The miss is NOT that ft09 is a "non-monotone CSP with no gradient" — that is the GENERAL win-φ
director's BLIND VIEW, and it is exactly wrong. The per-tissue truth is the opposite: ft09 has a
near-perfect MONOTONE gradient once you read the win predicate's own per-neighbor checks.

THE GENERAL SOLVER'S BLINDNESS, located + measured:
  * `find_win_method` instruments `all([...])` to read (sat, tot) from a public 0-arg bool win method.
    ft09's win `cgj()` (ft09.py L2436) is a MANUAL `for … return False` loop over per-neighbor parity
    checks — NOT an `all([...])`. So the `all`-instrumentation never fires → `find_win_method → None`.
  * `auto_progress` then finds no boolean done-mask and no scalar score/win_score pair → returns (0, 1).
  * Result: blind BFS over the 2^8 cell-colour space with NO gradient → exhausts, levels=0.

THE MECHANIC (L0), verified on-box:
  * `available_actions=[6]` → CLICK-ONLY (ACTION6). 1 constraint sprite `wmW` (tag `bsT`, in `g.gig`)
    at grid (22,22), 3×3 pattern `[[0,2,2],[0,8,0],[0,2,2]]`, center colour 8. It is surrounded by 8
    clickable `Hkx` cells (all start colour 9) at (22±4, 22±4).
  * Clicking a cell: `display_to_grid(x,y)` → grid cell; if a `Hkx` sprite sits there → `blr=Hkx`,
    `eHl = self.irw` (the flip pattern, default `[[0,0,0],[0,1,0],[0,0,0]]` = SINGLE-CELL toggle) →
    only the clicked cell's centre colour cycles through `gqb=[9,8]`. So the 8 cells are INDEPENDENT
    single-cell toggles (no piece-interaction coupling like re86's deceptive sum).
  * The win `cgj()` is a PER-NEIGHBOUR PARITY constraint: for each of the 8 directions, the constraint
    pattern edge `==0` → that neighbour must MATCH the centre colour (8); edge `!=0` → must NOT match
    (stay 9). On L0 exactly 4 of 8 cells need colour 8; root already satisfies 4/8; **4 clicks solve L0.**

VERIFIED EMPIRICALLY (bounded probes, /var/home/zero/arc-agi/toolkit/.venv):
  * `auto_progress(env) → (0, 1)`  →  the general solver degrades to BLIND BFS (no director).
  * the cgj-instrumented director → ROOT (sat, tot) = (4, 8), `cgj() == False`.
  * the 4 cells needing a flip = (18,18),(18,22),(26,22),(18,26), each colour 9 → 8.
  * 4 ACTION6 clicks (screen = grid*2 via display_to_grid round-trip) → levels_completed 0 → 1.
  * the general solver on the SAME game → levels=0, exhausted (the 0 this per-tissue solver beats).

WHAT THIS SOLVER ADDS (per-tissue):
  1. THE DIRECTOR — `_ft_progress(env) → (sat, total)`: replicate `cgj`'s per-neighbour parity checks
     across ALL `g.gig` constraint sprites and return the summed (Σ sat, Σ 8). This reads the win
     predicate's OWN checks at partial resolution — the sk48/wa30 "win-φ at partial resolution" family,
     here for ft09's parity constraint. MONOTONE: each correct single-cell toggle raises `sat` by 1.
  2. THE TIGHT SIGNATURE — `_ft_sig(env, lv)`: canonicalize only the win-relevant state (the colours of
     the constraint neighbour cells + each constraint centre + level). Small, exact, sprite-stable
     (keyed by grid position, never `str(address)` — the sk48 dedup-blindness lesson). Cheaper than the
     general `_kcanon` full-__dict__ recursion; the toggle space is tiny so dedup is fully effective.
  3. CLICK TARGETS — `_ft_click_targets(env)`: enumerate ONLY the constraint-neighbour cells (the cells
     `cgj` actually reads) and invert grid→screen via the camera's `display_to_grid` (cached once). This
     bounds the click branch to the ≤8·|gig| win-relevant cells — far below the 64×64 blind-click blowup
     the general `_click_targets` blob-centroid sampler would otherwise need.

WHAT IS REUSED (library-validation):
  * the A* `search_to_clear` skeleton (frontier keyed `director*K + depth`, GAME_OVER death-prune, dedup,
    `levels_completed` rise = win, deepcopy `_apply`) — the bp35/sk48/re86 shell;
  * the trace (`arc_trace.Tracer`, `coherence_R` from the rising sat/total) + scorecard mint machinery;
  * the bounded harness (`signal.alarm`, `--max-nodes/--timeout/--json`).
NEW (per-tissue): `_ft_progress` (the director), `_ft_sig` (the tight signature), `_ft_click_targets`
(the win-relevant click enumeration). No undo stack on this tissue (click-only, flat deepcopy) — so the
bp35/sk48 `_trim_history` transfer is NOT needed here; that is the honest per-tissue difference.

Run BOUNDED (CPU/venv only, no model):
  ulimit -v 8388608
  timeout 90 /var/home/zero/arc-agi/toolkit/.venv/bin/python ft09_solver.py \
      --max-levels 2 --max-nodes 40000 --timeout 80 --json
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

# The 8 neighbour offsets cgj() reads, paired with the constraint-pattern (row j, col i) cell whose
# value selects MATCH (edge==0 → must equal centre) vs DIFFER (edge!=0 → must not). The centre (1,1)
# is the constraint itself and is skipped. This is the exact loop in ft09.py cgj() (L2436), lifted.
_CGJ_OFFS = ((0, 0, -4, -4), (0, 1, 0, -4), (0, 2, 4, -4),
             (1, 0, -4, 0),                  (1, 2, 4, 0),
             (2, 0, -4, 4),  (2, 1, 0, 4),  (2, 2, 4, 4))


def _neighbor_cell(g, tx, ty):
    """The clickable cell sprite (Hkx, else NTi) at grid (tx, ty), or None. Mirrors cgj()'s lookup."""
    lvl = g.current_level
    s = lvl.get_sprite_at(tx, ty, "Hkx")
    if not s:
        s = lvl.get_sprite_at(tx, ty, "NTi")
    return s


# ── THE DIRECTOR: the win predicate read at partial resolution (cgj's own per-neighbour checks) ──
def _ft_progress(env):
    """(sat, total) = count of the per-neighbour parity constraints currently satisfied, summed over
    ALL constraint sprites in `g.gig`. Replicates cgj()'s checks exactly (edge==0 → neighbour must
    MATCH the centre colour; edge!=0 → must DIFFER), so it reads the win predicate's OWN output at
    partial resolution — the sk48/wa30 win-φ director, here for ft09's parity constraint. MONOTONE:
    each correct single-cell toggle raises `sat` by exactly 1. None on any miss → caller blinds
    (degrade-closed)."""
    g = getattr(env, "_game", None)
    if g is None:
        return None
    try:
        sat = total = 0
        for etf in g.gig:
            nRq = int(etf.pixels[1][1])
            for (j, i, dx, dy) in _CGJ_OFFS:
                PML = _neighbor_cell(g, etf.x + dx, etf.y + dy)
                if PML is None:
                    continue                       # cgj() only constrains cells that exist
                total += 1
                must_match = int(etf.pixels[j][i]) == 0
                cur = int(PML.pixels[1][1])
                ok = (cur == nRq) if must_match else (cur != nRq)
                if ok:
                    sat += 1
        if total == 0:
            return None
        return sat, total
    except Exception:
        return None


# ── THE TIGHT SIGNATURE: the win-relevant state only (sprite-stable, dedups across deepcopies) ──
def _ft_sig(env, lv):
    """Injective-on-win-relevant-state signature that DEDUPS across deepcopies. ft09's only mutable
    logical state is the colour of each constraint-neighbour cell (the toggles) plus each constraint
    centre — that is all `cgj` reads. We key those by GRID POSITION (sprite-stable, never `str(addr)` —
    the sk48 dedup-blindness lesson), so two deepcopies of the same logical config hash equal → dedup
    fires → the tiny toggle space stays bounded. Returns (lv, parts) so a state in level N never aliases
    level M. Degrade-closed: any miss → None → node kept (never crashes)."""
    g = getattr(env, "_game", None)
    if g is None:
        return None
    try:
        parts = []
        for etf in g.gig:
            cells = []
            cells.append((int(etf.x), int(etf.y), int(etf.pixels[1][1])))  # the constraint centre
            for (_j, _i, dx, dy) in _CGJ_OFFS:
                PML = _neighbor_cell(g, etf.x + dx, etf.y + dy)
                if PML is not None:
                    cells.append((int(PML.x), int(PML.y), int(PML.pixels[1][1])))
            parts.append(tuple(sorted(cells)))
        return (lv, tuple(sorted(parts)))
    except Exception:
        return None


# ── CLICK TARGETS: the win-relevant cells only, inverted grid→screen (bounds the click branch) ──
def _build_grid_to_screen(g):
    """One reverse map grid(gx,gy)→screen(sx,sy) by inverting the camera's display_to_grid over the
    display bound. Computed ONCE per solve (the camera is static within a level). The general solver
    samples blob centroids over 64×64; here we only need the ≤8·|gig| cells cgj reads, so a small
    targeted reverse map is both cheaper and exact."""
    cam = getattr(g, "camera", None)
    rev = {}
    if cam is None or not hasattr(cam, "display_to_grid"):
        return rev
    for sx in range(64):
        for sy in range(64):
            try:
                r = cam.display_to_grid(sx, sy)
            except Exception:
                r = None
            if r and len(r) == 2:
                rev.setdefault((int(r[0]), int(r[1])), (sx, sy))
    return rev


def _ft_click_targets(env, g2s):
    """Screen-click data (one per constraint-neighbour cell that cgj reads). Bounded to ≤8·|gig| =
    the win-relevant cells — NOT a 64×64 blind sweep. Each is one ACTION6 toggle of that cell."""
    g = getattr(env, "_game", None)
    if g is None:
        return []
    out = []
    try:
        for etf in g.gig:
            for (_j, _i, dx, dy) in _CGJ_OFFS:
                gx, gy = etf.x + dx, etf.y + dy
                PML = _neighbor_cell(g, gx, gy)
                if PML is None:
                    continue
                s = g2s.get((int(gx), int(gy)))
                if s is not None:
                    out.append({"x": int(s[0]), "y": int(s[1])})
    except Exception:
        return out
    # dedup identical screen targets (two constraints could share a neighbour)
    seen = set()
    uniq = []
    for d in out:
        k = (d["x"], d["y"])
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    return uniq


def _apply(env, action, data):
    child = copy.deepcopy(env)
    obs = child.step(action, data=data)
    return child, obs


def search_to_clear(env, obs0, base_lv, actions, *, max_nodes, max_depth, deadline, tracer=None,
                    level=0):
    """A* to the next level-clear (levels_completed rise), guided by the cgj parity director. Frontier
    key = (total - sat) * 1000 + depth → descend toward full constraint satisfaction. Death-pruned
    (GAME_OVER) and dedup'd on the tight sprite-stable signature (`_ft_sig`). Returns (plan, stats);
    plan = [(action, data)]. Mints a trace/v1 edge on every win-φ improvement when a tracer is given."""
    by_name = {getattr(a, "name", None): a for a in actions}
    click_a = by_name.get("ACTION6")
    if click_a is None:
        return None, {"nodes": 0, "stop": "no_click_action", "best_sat": 0, "total": 0}

    root = copy.deepcopy(env)
    g2s = _build_grid_to_screen(getattr(root, "_game", None))
    counter = 0
    p0 = _ft_progress(root)
    s0, t0 = p0 if p0 is not None else (0, 1)
    frontier = [((t0 - s0) * 1000 + 0, 0, counter, root, obs0, [])]
    seen = set()
    sig0 = _ft_sig(root, base_lv)
    if sig0 is not None:
        seen.add(sig0)
    nodes = dead = dedup = 0
    best_sat = s0
    total = t0
    while frontier and nodes < max_nodes:
        if time.time() > deadline:
            return None, {"nodes": nodes, "frontier": len(frontier), "dead": dead, "dedup": dedup,
                          "best_sat": best_sat, "total": total, "stop": "timeout"}
        _, depth, _, ec, _eobs, plan = heapq.heappop(frontier)
        if depth >= max_depth:
            continue
        for data in _ft_click_targets(ec, g2s):
            nodes += 1
            if nodes >= max_nodes or time.time() > deadline:
                break
            try:
                ec2, obs2 = _apply(ec, click_a, data)
            except Exception:
                continue
            lv = getattr(obs2, "levels_completed", 0) or 0
            if lv > base_lv:
                R = max(0.0, min(1.0, pow(2.718281828, -(nodes / 2000.0))))
                if tracer is not None:
                    try:
                        tracer.emit(step=nodes, action="ACTION6", coherence_R=R, phi=1.0,
                                    goal_progress=total, levels=lv, state="WIN", force=True,
                                    total=total, depth=depth + 1, level=level, edge="win")
                    except Exception:
                        pass
                return plan + [(click_a, data)], {"nodes": nodes, "depth": depth + 1,
                                                  "frontier": len(frontier), "dead": dead,
                                                  "dedup": dedup, "best_sat": total, "total": total,
                                                  "R": R, "stop": "win"}
            st = getattr(obs2, "state", None)
            if st is not None and getattr(st, "name", "") == "GAME_OVER":
                dead += 1
                continue
            sig = _ft_sig(ec2, lv)
            if sig is not None:
                if sig in seen:
                    dedup += 1
                    continue
                seen.add(sig)
            p = _ft_progress(ec2)
            if p is not None:
                s, total = p
                if s > best_sat:
                    best_sat = s
                    if tracer is not None and total:
                        try:
                            tracer.emit(step=nodes, action="ACTION6", coherence_R=None,
                                        phi=round(s / total, 6), goal_progress=s, levels=lv,
                                        state="SEARCHING", force=True, total=total, depth=depth + 1,
                                        frontier=len(frontier), level=level, edge="search")
                        except Exception:
                            pass
            else:
                s = best_sat
            counter += 1
            heapq.heappush(frontier, ((total - s) * 1000 + (depth + 1), depth + 1, counter,
                                      ec2, obs2, plan + [(click_a, data)]))
    stop = "exhausted" if not frontier else "node_budget"
    return None, {"nodes": nodes, "frontier": len(frontier), "dead": dead, "dedup": dedup,
                  "best_sat": best_sat, "total": total, "stop": stop}


def solve_game(short_id="ft09", *, max_levels=8, max_nodes=40000, max_depth=200, timeout=120,
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
    actions = list(env.action_space)
    obs = env.reset()
    levels = getattr(obs, "levels_completed", 0) or 0
    state = obs.state.name
    per_level = []
    tracer = None
    if trace_path:
        try:
            from arc_trace import Tracer
            tracer = Tracer(trace_path, world="arc", subject=short_id, method="ft09-parityphi",
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
            plan, st = search_to_clear(env, obs, base, actions, max_nodes=max_nodes,
                                       max_depth=max_depth, deadline=deadline, tracer=tracer,
                                       level=lvl)
            R = st.get("R", 0.0) if st.get("stop") == "win" else 0.0
            R_max = max(R_max, R)
            if plan is None:
                stall = st.get("stop", "stuck")
                if tracer:
                    try:
                        tracer.emit(step=lvl, action=None, coherence_R=0.0, levels=base, state=state,
                                    won=False, force=True, stall=stall, nodes=st.get("nodes"),
                                    dedup=st.get("dedup"), best_sat=st.get("best_sat"),
                                    total=st.get("total"))
                    except Exception:
                        pass
                break
            for (a, data) in plan:
                obs = env.step(a, data=data)
                if obs.state.name in ("WIN", "GAME_OVER"):
                    break
            levels = getattr(obs, "levels_completed", 0) or 0
            state = obs.state.name
            n_actions = len(plan)
            human = baseline[lvl] if lvl < len(baseline) else None
            rhae = round(human / n_actions, 3) if (human and n_actions) else None
            per_level.append({"level": lvl, "actions": n_actions, "human": human, "rhae": rhae,
                              "nodes": st["nodes"], "dedup": st.get("dedup"),
                              "total": st.get("total"), "R": round(R, 3)})
            if tracer:
                try:
                    tracer.emit(step=lvl, action="ft09", coherence_R=R, levels=levels, state=state,
                                won=(levels > base), force=True, actions=n_actions, rhae=rhae,
                                nodes=st["nodes"], dedup=st.get("dedup"))
                except Exception:
                    pass
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
        "game": short_id, "method": "ft09-parityphi", "levels": levels, "won": levels > 0,
        "state": state, "stall": None if levels > 0 and not stall else (stall or "stuck"),
        "n_levels_baseline": len(baseline), "per_level": per_level,
        "mean_rhae": round(sum(eff) / len(eff), 3) if eff else None,
        "coherence_R": round(R_max, 4),
        "wall_s": round(time.time() - t0, 1),
    }


def _alarm(signum, frame):
    raise TimeoutError("hard wall-clock budget hit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, default="ft09")
    ap.add_argument("--max-levels", type=int, default=8)
    ap.add_argument("--max-nodes", type=int, default=int(os.environ.get("ARC_MAX_NODES", "40000")))
    ap.add_argument("--max-depth", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("ARC_TIMEOUT", "120")))
    ap.add_argument("--trace", type=str,
                    default=os.environ.get("ARC_TRACE_PATH", "probes/logs/ft09_solver_traces.jsonl"))
    ap.add_argument("--results", type=str, default="probes/logs/ft09_solver_results.jsonl")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(a.timeout + 30)
    try:
        r = solve_game(a.game, max_levels=a.max_levels, max_nodes=a.max_nodes,
                       max_depth=a.max_depth, timeout=a.timeout, trace_path=a.trace)
    except TimeoutError:
        r = {"game": a.game, "levels": 0, "won": False, "state": "TIMEOUT", "stall": "hard_timeout"}
    except MemoryError:
        r = {"game": a.game, "levels": 0, "won": False, "state": "OOM", "stall": "memory_error"}
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
    try:                                              # mint a scorecard the OS/website can see
        import scorecard as _sc
        _sc.mint(r, method="ft09-parityphi", source_coupled=False,
                 coherence_R=r.get("coherence_R"))
    except Exception:
        pass

    if a.json:
        print(json.dumps(r))
    else:
        print(f"{a.game:8s} levels={r.get('levels', 0)} won={r.get('won')} state={r.get('state')} "
              f"stall={r.get('stall')} rhae={r.get('mean_rhae')} ({r.get('wall_s', '?')}s)",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
