# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""foundry_core — the one generic search primitive the per-tissue solvers share.

`_apply` is the world-model step: deepcopy the env and advance it by one action on the COPY, so the
search can roll candidates forward without committing to the live game. This is the only shared
mechanic the published solvers import; everything else (the per-game directors) lives in each solver.
"""
from __future__ import annotations

import copy


def _apply(env, action, data):
    """One world-model step on a deepcopy. Returns (child_env, obs). Sets click data when needed.
    The engine SWALLOWS a step error (e.g. a click whose game-step raises) and returns obs=None,
    leaving the child in a partial state. Treat that as a rejected move: raise so the search loop's
    except skips it (degrade-closed) instead of branching on a corrupted child."""
    child = copy.deepcopy(env)
    if data is not None:
        try:
            action.set_data(data)
        except Exception:
            pass
    obs = child.step(action, data=data)
    if obs is None:                                  # engine rejected the step -> not a real successor
        raise ValueError("step returned None (action rejected)")
    return child, obs
