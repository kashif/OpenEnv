# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Verifier of an exported Harbor task, copied to `/tests/verify.py` next to `problem.py` (the problem module) and
`task.json` (level, references and, at level 2, the hidden evaluation instances). Writes the score to
`/logs/verifier/reward.txt`.
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, "/tests")
import problem  # noqa: E402

RUNNER = """\
import json, sys
sys.path.insert(0, "/app")
from solve import solve
with open(sys.argv[1]) as f:
    instance = json.load(f)
with open(sys.argv[2], "w") as f:
    json.dump(solve(instance), f)
"""


def score(objective, reference):
    return 1.0 if objective == 0 else reference / objective


def level_1(task):
    with open("/app/instance.json") as f:
        instance = json.load(f)
    with open("/app/solution.json") as f:
        solution = json.load(f)
    objective = problem.evaluate(instance, solution)
    print(f"objective={objective} reference={task['reference']}")
    return score(objective, task["reference"])


def level_2(task):
    scores = []
    for k, (instance, reference) in enumerate(
        zip(task["eval_instances"], task["eval_references"])
    ):
        workdir = tempfile.mkdtemp()
        instance_path, solution_path = (
            os.path.join(workdir, "instance.json"),
            os.path.join(workdir, "solution.json"),
        )
        with open(instance_path, "w") as f:
            json.dump(instance, f)
        try:
            subprocess.run(
                [sys.executable, "-c", RUNNER, instance_path, solution_path],
                cwd=workdir,
                timeout=task["solve_time_limit_s"],
                check=True,
            )
            with open(solution_path) as f:
                objective = problem.evaluate(instance, json.load(f))
        except Exception as e:  # noqa: BLE001 - any failure of the submitted code scores 0 on this instance
            print(f"instance {k}: {type(e).__name__}: {e}")
            scores.append(0.0)
            continue
        print(f"instance {k}: objective={objective} reference={reference}")
        scores.append(score(objective, reference))
    return sum(scores) / len(scores)


def main():
    with open("/tests/task.json") as f:
        task = json.load(f)
    try:
        reward = level_1(task) if task["level"] == 1 else level_2(task)
    except Exception as e:  # noqa: BLE001 - a missing or infeasible submission scores 0
        print(f"{type(e).__name__}: {e}")
        reward = 0.0
    print(f"reward={reward}")
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(str(reward))


if __name__ == "__main__":
    main()
