# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OR-Tools environment: design solutions and algorithms for hard operations research problems.

The setup follows "LLMs Can Design Near-Optimal OR Algorithms" (https://huggingface.co/papers/2608.27296). Each
episode is one query: the agent gets a problem statement, a Python sandbox with a compute budget, and returns one
deliverable.

- Level 1: one instance; the agent submits a solution with `submit_solution`.
- Level 2: the problem class and an example instance; the agent submits code defining `solve(instance)` with
  `submit_algorithm`, which is run on unseen instances.

The reward is the reference objective divided by the agent's (minimization), averaged over the evaluation instances
at level 2, and 0 for an infeasible solution. References come from OR-Tools (CP-SAT, or the routing solver for CVRP)
and should be precomputed into the dataset with `build_dataset.py`; missing ones are computed at submission time.
"""

import asyncio
import json
import random
import shutil
import tempfile
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

try:
    from .. import sandbox
    from ..problems import PROBLEMS, score
    from ..prompts import task_prompt
except ImportError:
    import sandbox
    from problems import PROBLEMS, score
    from prompts import task_prompt


SUBMIT_HELP = {
    1: (
        "## Tools\n\n- `run_python(code)` runs a Python script with the instance bound to the global `instance` "
        "(also in `instance.json`); files you write persist between calls.\n- `submit_solution(solution)` submits "
        "your final solution as a JSON string. You can submit only once."
    ),
    2: (
        "## Tools\n\n- `run_python(code)` runs a Python script with the example instance bound to the global "
        "`instance` (also in `instance.json`); files you write persist between calls.\n- `submit_algorithm(code)` "
        "submits Python code defining `solve(instance)`. It runs in a fresh process for each evaluation instance. "
        "You can submit only once."
    ),
}


class ORToolsEnvironment(MCPEnvironment):
    """MCP environment for OR algorithm design, with tools `run_python`, `submit_solution` and `submit_algorithm`."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("ortools_env")

        # The tools run subprocesses and solvers, so they are offloaded to threads to keep the server loop (shared by
        # all sessions) responsive.
        @mcp.tool
        async def run_python(code: str) -> str:
            """
            Run a Python script in the sandbox and return its combined stdout and stderr.

            Args:
                code: The Python source to run. The instance is bound to the global `instance`.

            Returns:
                The script output, followed by the remaining compute budget.
            """
            return await asyncio.to_thread(self._run_python, code)

        @mcp.tool
        async def submit_solution(solution: str) -> str:
            """
            Submit the final solution for the instance (level 1). Ends the episode.

            Args:
                solution: The solution as a JSON string, in the format given in the problem statement.

            Returns:
                The objective, the reference objective and the resulting score.
            """
            return await asyncio.to_thread(self._submit_solution, solution)

        @mcp.tool
        async def submit_algorithm(code: str) -> str:
            """
            Submit the final algorithm (level 2): Python code defining `solve(instance) -> solution`. Ends the episode.

            Args:
                code: The Python source defining `solve`.

            Returns:
                The score on each evaluation instance and their mean.
            """
            return await asyncio.to_thread(self._submit_algorithm, code)

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0, submitted=False)
        self._workdir = None
        self._tool_timeout_s = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        problem: str = "jobshop",
        size: Optional[str] = None,
        level: int = 1,
        instance: Optional[dict] = None,
        instance_seed: Optional[int] = None,
        reference: Optional[float] = None,
        lower_bound: Optional[float] = None,
        eval_seeds: Optional[list[int]] = None,
        eval_references: Optional[list[Optional[float]]] = None,
        num_eval_instances: int = 4,
        allow_ortools: bool = False,
        compute_budget_s: float = 600.0,
        max_python_calls: int = 50,
        python_timeout_s: float = 120.0,
        solve_time_limit_s: float = 30.0,
        reference_time_limit_s: float = 10.0,
        **kwargs: Any,
    ) -> Observation:
        """
        Start an episode on one problem instance (level 1) or one problem class (level 2).

        Args:
            seed: Seeds the instance (and, at level 2, the evaluation instances) when `instance_seed` / `eval_seeds`
                are not given.
            episode_id: Optional episode ID.
            problem: One of `jobshop`, `rcpsp`, `cvrp`, `tardiness`.
            size: Instance size, a key of the problem's `SIZES` (defaults to its `DEFAULT_SIZE`).
            level: 1 to submit a solution, 2 to submit an algorithm.
            instance: An explicit instance (e.g. a benchmark loaded with `load`), instead of generating one.
            instance_seed: Seed of the generated instance.
            reference: Precomputed reference objective of the instance (level 1).
            lower_bound: Precomputed lower bound, reported in the state only.
            eval_seeds: Seeds of the hidden evaluation instances (level 2).
            eval_references: Precomputed reference objectives of the evaluation instances (level 2).
            num_eval_instances: Number of evaluation instances when `eval_seeds` is not given.
            allow_ortools: Whether the agent's code may import `ortools`.
            compute_budget_s: Total wall-clock seconds of `run_python` for the episode.
            max_python_calls: Maximum number of `run_python` calls.
            python_timeout_s: Time limit of a single `run_python` call.
            solve_time_limit_s: Time limit of `solve` on each evaluation instance (level 2).
            reference_time_limit_s: OR-Tools time limit when a reference must be computed at submission.
        """
        self._module = PROBLEMS[problem]
        size = size or self._module.DEFAULT_SIZE
        rng = random.Random(seed)
        if instance is None:
            instance_seed = rng.randrange(2**31) if instance_seed is None else instance_seed
            instance = self._module.generate(instance_seed, size)
        self._instance = instance
        self._eval_instances = []
        if level == 2:
            if eval_seeds is None:
                eval_seeds = [rng.randrange(2**31) for _ in range(num_eval_instances)]
            self._eval_instances = [self._module.generate(s, size) for s in eval_seeds]
            eval_references = list(eval_references or [None] * len(eval_seeds))
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = tempfile.mkdtemp(prefix="ortools_env_")

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            problem=problem,
            size=size,
            level=level,
            instance_seed=instance_seed,
            reference=reference,
            lower_bound=lower_bound,
            eval_seeds=eval_seeds,
            eval_references=eval_references,
            allow_ortools=allow_ortools,
            compute_budget_s=compute_budget_s,
            compute_used_s=0.0,
            max_python_calls=max_python_calls,
            python_calls=0,
            python_timeout_s=python_timeout_s,
            solve_time_limit_s=solve_time_limit_s,
            reference_time_limit_s=reference_time_limit_s,
            submitted=False,
            objective=None,
            reward=None,
        )
        # The MCP layer times tool calls out after 30s by default; give it room for the longest tool of this episode.
        num_evals = max(1, len(self._eval_instances))
        self._tool_timeout_s = max(python_timeout_s, num_evals * (solve_time_limit_s + reference_time_limit_s)) + 60
        budget = (
            f"## Compute budget\n\n{compute_budget_s:g} seconds of Python compute in total, over at most "
            f"{max_python_calls} `run_python` calls of at most {python_timeout_s:g} seconds each."
        )
        prompt = "\n\n".join(
            [
                task_prompt(self._module, level, size, instance, solve_time_limit_s, allow_ortools),
                budget,
                SUBMIT_HELP[level],
            ]
        )
        return Observation(done=False, reward=None, metadata={"prompt": prompt})

    def _run_python(self, code: str) -> str:
        if self._state.submitted:
            return "Error: the episode is over."
        if self._state.python_calls >= self._state.max_python_calls:
            return "Error: no Python calls left; submit now."
        remaining = self._state.compute_budget_s - self._state.compute_used_s
        if remaining <= 0:
            return "Error: compute budget exhausted; submit now."
        result = sandbox.run(
            code,
            self._instance,
            self._workdir,
            timeout_s=min(self._state.python_timeout_s, remaining),
            allow_ortools=self._state.allow_ortools,
        )
        self._state.python_calls += 1
        self._state.compute_used_s += result["elapsed_s"]
        status = "timed out" if result["timed_out"] else f"exit code {result['returncode']}"
        return (
            f"{result['output']}\n[{status}, {result['elapsed_s']:.1f}s; "
            f"{max(0.0, self._state.compute_budget_s - self._state.compute_used_s):.0f}s of compute and "
            f"{self._state.max_python_calls - self._state.python_calls} calls left]"
        )

    def _submit_solution(self, solution: str) -> str:
        if self._state.submitted:
            return "Error: the episode is over."
        if self._state.level != 1:
            return "Error: this is a level-2 task; use submit_algorithm."
        self._state.submitted = True
        try:
            objective = self._module.evaluate(self._instance, json.loads(solution))
        except (ValueError, TypeError, KeyError, IndexError) as e:  # JSONDecodeError is a ValueError
            self._state.reward = 0.0
            return f"Infeasible solution: {e}\nscore=0.0"
        self._state.reference = self._reference(self._instance, self._state.reference)
        self._state.objective = objective
        self._state.reward = score(objective, self._state.reference)
        return f"objective={objective} reference={self._state.reference} score={self._state.reward:.4f}"

    def _submit_algorithm(self, code: str) -> str:
        if self._state.submitted:
            return "Error: the episode is over."
        if self._state.level != 2:
            return "Error: this is a level-1 task; use submit_solution."
        self._state.submitted = True
        lines, scores = [], []
        for k, instance in enumerate(self._eval_instances):
            workdir = tempfile.mkdtemp(prefix="ortools_eval_")
            try:
                result = sandbox.run(
                    code,
                    instance,
                    workdir,
                    timeout_s=self._state.solve_time_limit_s,
                    allow_ortools=self._state.allow_ortools,
                    solve=True,
                )
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
            if result["timed_out"]:
                scores.append(0.0)
                lines.append(f"instance {k}: timed out")
                continue
            if result["solution"] is None:
                scores.append(0.0)
                lines.append(f"instance {k}: failed\n{result['output'][-1000:]}")
                continue
            try:
                objective = self._module.evaluate(instance, result["solution"])
            except (ValueError, TypeError, KeyError, IndexError) as e:
                scores.append(0.0)
                lines.append(f"instance {k}: infeasible ({e})")
                continue
            reference = self._reference(instance, self._state.eval_references[k])
            self._state.eval_references[k] = reference
            scores.append(score(objective, reference))
            lines.append(f"instance {k}: objective={objective} reference={reference} score={scores[-1]:.4f}")
        self._state.reward = sum(scores) / len(scores)
        return "\n".join(lines) + f"\nscore={self._state.reward:.4f}"

    def _reference(self, instance: dict, reference: Optional[float]) -> float:
        if reference is None:
            _, reference, _ = self._module.reference(instance, self._state.reference_time_limit_s)
        return reference

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s or self._tool_timeout_s, **kwargs)
        if self._state.submitted:
            obs.done = True
            obs.reward = self._state.reward
        return obs

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s or self._tool_timeout_s, **kwargs)
        if self._state.submitted:
            obs.done = True
            obs.reward = self._state.reward
        return obs

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None
        super().close()
