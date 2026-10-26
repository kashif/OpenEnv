# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the OR-Tools environment: generators, verifiers, references and the MCP episode flow."""

import pytest

pytest.importorskip("ortools")
pytest.importorskip("fastmcp")

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from ortools_env import sandbox
from ortools_env.problems import jobshop, PROBLEMS, score
from ortools_env.server.ortools_environment import ORToolsEnvironment

TINY_SIZES = {
    "jobshop": "6x6",
    "rcpsp": "j10",
    "cvrp": "n20",
    "tardiness": "n10",
    "rostering": "4x1",
}


def _text(obs) -> str:
    if obs.error is not None:
        return obs.error.message
    return obs.result.content[0].text


def _call(env, tool, **arguments):
    return env.step(CallToolAction(tool_name=tool, arguments=arguments))


def test_taillard_generator_reproduces_ta01():
    # ta01 (Taillard 1993): time seed 840612802, machine seed 398197754; first job of the published instance.
    instance = jobshop.taillard(15, 15, 840612802, 398197754)
    machines = [7, 13, 5, 8, 4, 3, 11, 12, 9, 15, 10, 14, 6, 1, 2]
    durations = [94, 66, 10, 53, 26, 15, 65, 82, 10, 27, 93, 92, 96, 70, 83]
    assert instance["jobs"][0] == [[m - 1, d] for m, d in zip(machines, durations)]


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_generate_is_deterministic(problem):
    module = PROBLEMS[problem]
    assert module.generate(3, TINY_SIZES[problem]) == module.generate(
        3, TINY_SIZES[problem]
    )


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_reference_solution_passes_the_verifier(problem):
    module = PROBLEMS[problem]
    instance = module.generate(0, TINY_SIZES[problem])
    solution, objective, lower_bound = module.reference(
        instance, time_limit=5, num_workers=1
    )
    assert module.evaluate(instance, solution) == objective
    assert lower_bound is None or lower_bound <= objective


def test_verifiers_reject_infeasible_solutions():
    instance = jobshop.generate(0, "6x6")
    with pytest.raises(ValueError):
        jobshop.evaluate(instance, {"starts": [[0] * 6 for _ in range(6)]})

    rcpsp = PROBLEMS["rcpsp"]
    instance = rcpsp.generate(0, "j10")
    with pytest.raises(ValueError):
        rcpsp.evaluate(instance, {"starts": [0] * 10})

    cvrp = PROBLEMS["cvrp"]
    instance = cvrp.generate(0, "n20")
    with pytest.raises(ValueError, match="not visited"):
        cvrp.evaluate(instance, {"routes": [[1]]})
    with pytest.raises(ValueError, match="more than once"):
        cvrp.evaluate(instance, {"routes": [[1], [1]]})

    tardiness = PROBLEMS["tardiness"]
    instance = tardiness.generate(0, "n10")
    with pytest.raises(ValueError):
        tardiness.evaluate(instance, {"sequence": [0] * 10})

    rostering = PROBLEMS["rostering"]
    instance = rostering.generate(0, "4x1")
    with pytest.raises(ValueError):
        rostering.evaluate(instance, {"schedule": [[0] * 7 for _ in range(4)]})


def test_score():
    # Normalized between the naive baseline (0.0) and the reference (1.0).
    assert score(100, reference=100, baseline=300) == 1.0
    assert score(300, reference=100, baseline=300) == 0.0
    assert score(200, reference=100, baseline=300) == 0.5
    assert score(50, reference=100, baseline=300) == 1.25
    assert score(500, reference=100, baseline=300) == -1.0
    assert score(None, reference=100, baseline=300) == -1.0
    # The naive solution already matches the reference.
    assert score(100, reference=100, baseline=100) == 1.0
    assert score(101, reference=100, baseline=100) == -1.0


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_baseline_is_no_better_than_the_reference(problem):
    module = PROBLEMS[problem]
    instance = module.generate(0, TINY_SIZES[problem])
    _, objective, _ = module.reference(instance, time_limit=5, num_workers=1)
    assert module.baseline(instance) >= objective


def test_naive_solution_scores_zero():
    env = ORToolsEnvironment()
    env.reset(problem="tardiness", size="n10", seed=0, reference_time_limit_s=5)
    obs = _call(env, "submit_solution", solution={"sequence": list(range(10))})
    assert obs.reward == 0.0


def test_sandbox_blocks_ortools_and_times_out(tmp_path):
    result = sandbox.run("import ortools", {}, str(tmp_path), timeout_s=30)
    assert "not allowed" in result["output"]
    result = sandbox.run(
        "import ortools", {}, str(tmp_path), timeout_s=30, allow_ortools=True
    )
    assert result["returncode"] == 0
    result = sandbox.run("while True: pass", {}, str(tmp_path), timeout_s=1)
    assert result["timed_out"]


def test_tools_are_listed():
    env = ORToolsEnvironment()
    names = {tool.name for tool in env.step(ListToolsAction()).tools}
    assert names == {"run_python", "submit_solution", "submit_algorithm"}


def test_level_1_episode():
    env = ORToolsEnvironment()
    obs = env.reset(problem="jobshop", size="6x6", seed=0, reference_time_limit_s=5)
    assert "Job shop scheduling" in obs.metadata["prompt"]

    obs = _call(env, "run_python", code="print(instance['num_jobs'])")
    assert _text(obs).startswith("6\n")
    assert not obs.done

    solution, objective, _ = jobshop.reference(env._instance, 5, 1)
    obs = _call(env, "submit_solution", solution=solution)
    assert obs.done
    assert obs.reward == 1.0
    assert env.state.objective == objective

    obs = _call(env, "submit_solution", solution=solution)
    assert "episode is over" in _text(obs)


def test_level_1_infeasible_solution_scores_minus_one():
    env = ORToolsEnvironment()
    env.reset(problem="tardiness", size="n10", seed=0, reference=1)
    obs = _call(env, "submit_solution", solution={"sequence": [0, 0]})
    assert obs.done
    assert obs.reward == -1.0


def test_level_2_episode():
    env = ORToolsEnvironment()
    env.reset(
        problem="tardiness",
        size="n10",
        level=2,
        seed=0,
        num_eval_instances=2,
        solve_time_limit_s=10,
        reference_time_limit_s=5,
    )
    code = (
        "def solve(instance):\n"
        "    n = len(instance['processing_times'])\n"
        "    return {'sequence': sorted(range(n), key=lambda j: instance['due_dates'][j])}\n"
    )
    obs = _call(env, "submit_algorithm", code=code)
    assert obs.done
    assert "naive=" in _text(obs)
    assert -1.0 <= obs.reward <= 1.0


def test_level_2_crashing_algorithm_scores_minus_one():
    env = ORToolsEnvironment()
    env.reset(
        problem="tardiness",
        size="n10",
        level=2,
        seed=0,
        eval_seeds=[1, 2],
        eval_references=[1, 1],
    )
    obs = _call(
        env, "submit_algorithm", code="def solve(instance):\n    raise RuntimeError"
    )
    assert obs.done
    assert obs.reward == -1.0
