# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OR problem classes: instance generators, benchmark loaders, verifiers and OR-Tools reference solvers.

Every module exposes the same functions:

- `generate(seed, size)`: a fresh instance (a JSON-serializable dict), reproducible from `(seed, size)`.
- `load(path)`: an instance from the standard benchmark file format of the problem.
- `evaluate(instance, solution)`: the objective value of a solution, raising `ValueError` if it is infeasible.
  Pure Python, so it can run in a verifier image without OR-Tools.
- `reference(instance, time_limit)`: `(solution, objective, lower_bound)` from OR-Tools (`lower_bound` may be None).

All objectives are minimized and non-negative.
"""

from . import cvrp, jobshop, rcpsp, rostering, tardiness

PROBLEMS = {
    module.NAME: module for module in (jobshop, rcpsp, cvrp, tardiness, rostering)
}


def score(objective: float, reference: float) -> float:
    """Reward for reaching `objective` when the reference reaches `reference`: 1.0 matches it, above 1.0 beats it."""
    if objective == 0:
        return 1.0
    return reference / objective
