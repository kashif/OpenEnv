<!-- openenv-source: ortools_env -->
# OR-Tools Environment

Train and evaluate LLMs on **designing solutions and algorithms for hard operations research problems**, with a
verifiable reward: the objective of the agent's solution against an OR-Tools reference.

The setup follows [LLMs Can Design Near-Optimal OR Algorithms](https://huggingface.co/papers/2608.27296). Each episode
is one query: the agent reads a problem statement, uses a Python sandbox under a compute budget, and returns one
deliverable.

| Level | The agent gets | It submits | Scored on |
|---|---|---|---|
| 1 | one instance | a solution (`submit_solution`) | that instance |
| 2 | the problem class and an example instance | code defining `solve(instance)` (`submit_algorithm`) | unseen instances of the same distribution |

## Problems

The problem classes were picked where exact methods still struggle at scale, so a strong agent has room to match or
beat the reference instead of saturating at the optimum. Each class has a generator that follows the standard benchmark
design, a loader for the standard file format, a pure-Python verifier, and an OR-Tools reference solver taken from the
[OR-Tools examples](https://github.com/google/or-tools/tree/stable/examples/python).

| `problem` | Problem | Generator | Sizes | Benchmark files | Reference |
|---|---|---|---|---|---|
| `jobshop` | Job shop, minimize makespan | Taillard (1993), reproduces `ta01`-`ta80` from their seeds | `6x6` … `100x20` | OR-Library / JSPLIB | CP-SAT (optimum and bound) |
| `rcpsp` | Resource-constrained project scheduling | PSPLIB design (resource factor, resource strength) | `j10` … `j300` | PSPLIB `.sm` | CP-SAT (optimum and bound) |
| `cvrp` | Capacitated vehicle routing | Uchoa et al. (2017) "X" design | `n20` … `n1000` | CVRPLIB `.vrp` | Routing solver, guided local search |
| `tardiness` | Single-machine total weighted tardiness | OR-Library (Potts & Van Wassenhove) | `n10` … `n500` | OR-Library `wtN.txt` | CP-SAT (optimum and bound) |
| `rostering` | Nurse rostering with hard and soft work rules | Rules of `shift_scheduling_sat.py`, random demands and requests | `4x1` … `80x8` (employees x weeks) | — | CP-SAT (optimum and bound) |

The smallest size of each problem is for smoke tests. Every generated instance is reproducible from `(seed, size)`,
and the generators randomize the structural parameters (resource strength, due-date tightness, depot and customer
placement, demand distribution) per seed.

## Reward

All objectives are minimized. The reward is `reference / objective` (1.0 if the objective is 0), averaged over the
evaluation instances at level 2, and 0 for an infeasible solution, a crash or a timeout. Matching the reference scores
1.0; beating it scores above 1.0. The episode ends at the first submission, whose observation carries `done=True` and
the reward.

References are best computed offline with a long time limit:

```bash
PYTHONPATH=src:envs python -m ortools_env.build_dataset --problem rcpsp --sizes j60 j120 --level 1 \
    --num-rows 256 --reference-time-limit 60 --out rcpsp_l1.jsonl
```

Each row holds the `reset` arguments of one episode (`problem`, `size`, `level`, `instance_seed`, `reference` or
`eval_seeds` and `eval_references`). A reference missing from `reset` is computed at submission time with
`reference_time_limit_s`.

## Quick start

```bash
PYTHONPATH=src:envs uvicorn ortools_env.server.app:app --port 8000
```

```python
from openenv.core.env_server.mcp_types import CallToolAction

from ortools_env import ORToolsEnv

with ORToolsEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(problem="jobshop", size="15x15", level=1, seed=0)
    print(result.observation.metadata["prompt"])

    print(env.call_tool("run_python", code="print(instance['num_jobs'])"))

    # `call_tool` returns only the tool text; use `step` to read the reward of the submission.
    result = env.step(
        CallToolAction(
            tool_name="submit_solution", arguments={"solution": '{"starts": [...]}'}
        )
    )
    print(result.reward, result.done)
    print(env.state())  # ORToolsState: objective, reference, compute used, ...
```

## Tools

| Tool | Description |
|---|---|
| `run_python(code)` | Runs a script in the sandbox with the instance bound to `instance` (and in `instance.json`); files persist between calls. Counts against the compute budget. |
| `submit_solution(solution)` | Level 1: submits a JSON solution. Ends the episode. |
| `submit_algorithm(code)` | Level 2: submits code defining `solve(instance)`, run in a fresh process per evaluation instance with `solve_time_limit_s`. Ends the episode. |

## Reset arguments

| Argument | Default | Description |
|---|---|---|
| `problem` | `"jobshop"` | `jobshop`, `rcpsp`, `cvrp`, `tardiness` or `rostering` |
| `size` | the problem's default | a key of the problem's `SIZES` |
| `level` | `1` | `1` (solution) or `2` (algorithm) |
| `instance` / `instance_seed` | generated from `seed` | an explicit instance, e.g. read with the problem's `load`, or the seed of a generated one |
| `reference`, `lower_bound` | computed at submission | precomputed reference objective and bound (level 1) |
| `eval_seeds`, `eval_references` | `num_eval_instances` seeds from `seed` | hidden evaluation instances and their references (level 2) |
| `allow_ortools` | `False` | whether the agent's code may import `ortools` |
| `compute_budget_s`, `max_python_calls`, `python_timeout_s` | `600`, `50`, `120` | sandbox budget |
| `solve_time_limit_s` | `30` | per-instance time limit of `solve` (level 2) |
| `reference_time_limit_s` | `10` | OR-Tools time limit for references computed at submission |

## Sandbox

Code runs in a subprocess with a wall-clock limit and a 4 GB address-space limit. Imports of `ortools` (unless
`allow_ortools=True`) and of the environment's own modules are blocked, so the agent can neither call the reference
solver nor read the verifier. This is resource control, not a security boundary: run the server in its container.

## Harbor export

The same episodes can be exported as [Harbor](https://harborframework.com) tasks to measure frontier agents
(claude-code, codex, ...) with their own harnesses, for example through the `harbor_env` environment:

```bash
PYTHONPATH=src:envs python -m ortools_env.export_harbor --dataset rcpsp_l1.jsonl --out tasks/
```

Each task gets the instance in `/app/instance.json` and asks for `/app/solution.json` (level 1) or `/app/solve.py`
(level 2). The verifier, the problem module and the hidden evaluation instances are in `tests/`, which Harbor only
uploads after the agent has finished.
