# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run model-written Python in a subprocess with a wall-clock limit, a memory limit and blocked imports.

This is resource control, not a security boundary: the environment server is expected to run inside its container.
Blocking `ortools` keeps the policy from calling the reference solver when the task asks it to design its own
algorithm, and blocking the environment's own packages keeps it from reading the verifier and generators.
"""

import json
import os
import resource
import subprocess
import sys
import tempfile
import time

BLOCKED_ALWAYS = ("ortools_env", "problems", "server", "sandbox")

_RUNNER = """\
import importlib.abc, json, sys

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"import of {name!r} is not allowed in this sandbox")

BLOCKED = set(json.loads(sys.argv[1]))
sys.meta_path.insert(0, _Blocker())
with open("instance.json") as f:
    instance = json.load(f)
namespace = {"__name__": "__main__", "instance": instance}
with open(sys.argv[2]) as f:
    exec(compile(f.read(), sys.argv[2], "exec"), namespace)
if sys.argv[3] == "solve":
    solution = namespace["solve"](instance)
    with open("solution.json", "w") as f:
        json.dump(solution, f)
"""


def _limit_memory(memory_mb: int):
    def apply():
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply


def run(
    code: str,
    instance: dict,
    workdir: str,
    timeout_s: float,
    allow_ortools: bool = False,
    solve: bool = False,
    memory_mb: int = 4096,
) -> dict:
    """
    Run `code` with `instance` bound as a global, from `workdir` (where files persist between calls).

    Args:
        code (`str`):
            Python source to run.
        instance (`dict`):
            Instance bound to the global `instance` and written to `instance.json`.
        workdir (`str`):
            Working directory of the subprocess.
        timeout_s (`float`):
            Wall-clock limit of the subprocess.
        allow_ortools (`bool`, *optional*, defaults to `False`):
            Whether `code` may import `ortools`.
        solve (`bool`, *optional*, defaults to `False`):
            Whether to call `solve(instance)`, which `code` must define, and return its result as the solution.
        memory_mb (`int`, *optional*, defaults to `4096`):
            Address-space limit of the subprocess.

    Returns:
        `dict` with keys `output` (combined stdout and stderr, truncated), `elapsed_s`, `timed_out`, `returncode` and,
        with `solve=True`, `solution` (`None` if the run failed).
    """
    blocked = list(BLOCKED_ALWAYS) + ([] if allow_ortools else ["ortools"])
    with open(os.path.join(workdir, "instance.json"), "w") as f:
        json.dump(instance, f)
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=workdir, delete=False) as f:
        f.write(code)
        code_path = f.name
    solution_path = os.path.join(workdir, "solution.json")
    if os.path.exists(solution_path):
        os.remove(solution_path)
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                _RUNNER,
                json.dumps(blocked),
                code_path,
                "solve" if solve else "run",
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_limit_memory(memory_mb),
        )
        output, returncode, timed_out = (
            proc.stdout + proc.stderr,
            proc.returncode,
            False,
        )
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(
            errors="replace"
        )
        returncode, timed_out = None, True
    finally:
        os.remove(code_path)
    result = {
        "output": output,
        "elapsed_s": time.monotonic() - start,
        "timed_out": timed_out,
        "returncode": returncode,
    }
    if len(output) > 8000:
        result["output"] = output[:4000] + "\n... [truncated] ...\n" + output[-4000:]
    if solve:
        result["solution"] = None
        if returncode == 0 and os.path.exists(solution_path):
            with open(solution_path) as f:
                result["solution"] = json.load(f)
    return result
