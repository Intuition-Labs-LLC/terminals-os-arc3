# Terminals OS — Tissue Foundry (ARC-AGI-3 community submission)

A per-game ("per-tissue") solver for [ARC-AGI-3](https://arcprize.org/arc-agi/3) interactive games, by
[Intuition Labs](https://intuitionlabs.tech). This repo contains the verified, source-coupled solvers
behind our [ARC Prize Community Leaderboard](https://github.com/arcprize/ARC-AGI-Community-Leaderboard)
submission, plus the harness that replays them against the live ARC API to mint a Competition-Mode
scorecard.

## Honest scope (read this first)

- **What this is:** a *source-coupled* search agent. Each game is treated as a bounded "tissue" and
  solved by best-first search over the game's **own** deepcopy world-model (`copy.deepcopy(env)` + the
  game's own `step`) with a per-game director (the heuristic). This is the same method-class ARC permits
  on the community board (per-game reverse-engineered simulators). It runs with **no LLM**.
- **What this is NOT:** it is **not** a general, source-free agent, and it does **not** beat the current
  top community entry. It plays the games whose dynamics it has a director for. The general,
  learn-by-playing agent (which does not read the game's internal state) is separate, ongoing research
  and is not in this repo.
- **Verified result:** Competition-Mode scorecard
  [`5197b710-…`](https://arcprize.org/scorecards/5197b710-9600-4ebd-83ff-bd93f5ae5fc2) — 4 full wins
  (lp85 8/8, tu93 9/9, ft09 6/6, cd82 6/6).

## How it works

1. **Solve locally.** A bounded A*/best-first search expands the game's deepcopy world-model. The
   per-game director scores a state by how close it is to the win predicate (read at partial
   resolution); a game-stable signature dedups the frontier. Bounded by `ulimit -v` / node-cap /
   wall-clock, so a search can never blow up the host.
2. **Replay online.** `arc_online_replay.py` captures the exact winning action trajectory and replays it
   against the **remote** ARC environment under a single Competition-Mode scorecard — the scored
   interactions are the online replay (local play does not count toward the leaderboard).

```bash
pip install -r requirements.txt          # the arc-agi SDK
export ARC_API_KEY=...                    # your ARC Prize key
# mint a Competition-Mode scorecard from the four winning solvers:
python arc_online_replay.py --competition --games lp85 tu93 ft09 cd82 --max-levels 12
```

## Files

| file | role |
|---|---|
| `arc_online_replay.py` | solve-local → replay-online harness; opens/closes the scorecard |
| `lp85_solver.py`, `tu93_solver.py`, `ft09_solver.py`, `cd82_solver.py` | the four per-game solvers |
| `foundry_core.py` | the shared world-model step (`_apply` = deepcopy + step) |
| `scorecard.py` | mints a local `scorecard/v1` receipt per run |

## License

AGPL-3.0-or-later. © 2026 Tej Desai / Intuition Labs LLC.
