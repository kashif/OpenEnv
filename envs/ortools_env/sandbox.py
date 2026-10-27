# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run model-written Python in a subprocess with a wall-clock limit, a memory limit and blocked imports.

Keeping `ortools` and the environment's own packages out of reach is what keeps the reward honest: otherwise a policy
can call the reference solver. In the container (see `server/Dockerfile`), `SANDBOX_USER` is set and the code runs as
that unprivileged user with `SANDBOX_PYTHON`, an interpreter without `ortools`, while the environment source and the
server's packages are not readable by it. With `allow_ortools=True` it runs with `SANDBOX_ORTOOLS_PYTHON` instead,
which only the `ortools` group can read. Without `SANDBOX_USER` (local development), the code runs with the server's
interpreter and an import blocker that it can remove, so rewards are only trustworthy in the container.
"""

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

BLOCKED_ALWAYS = ("ortools_env", "problems", "server", "sandbox")
SANDBOX_USER = os.environ.get("SANDBOX_USER")
SANDBOX_PYTHON = os.environ.get("SANDBOX_PYTHON", sys.executable)
SANDBOX_ORTOOLS_PYTHON = os.environ.get("SANDBOX_ORTOOLS_PYTHON", sys.executable)

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
# A submitted `solve` is imported like a module, so its `if __name__ == "__main__":` tests do not run.
namespace = {"__name__": "__main__" if sys.argv[3] == "run" else "solution", "instance": instance}
with open(sys.argv[2]) as f:
    exec(compile(f.read(), sys.argv[2], "exec"), namespace)
if sys.argv[3] == "solve":
    if "solve" not in namespace:
        raise NameError("the code must define solve(instance)")
    solution = namespace["solve"](instance)
    if not isinstance(solution, dict):
        raise TypeError(f"solve(instance) must return a dict, not {type(solution).__name__}")
    with open("solution.json", "w") as f:
        json.dump(solution, f, default=lambda o: o.tolist())
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
    os.chmod(code_path, 0o644)
    solution_path = os.path.join(workdir, "solution.json")
    if os.path.exists(solution_path):
        os.remove(solution_path)
    user = {}
    if SANDBOX_USER is not None:
        shutil.chown(workdir, SANDBOX_USER, SANDBOX_USER)
        user = {
            "user": SANDBOX_USER,
            "group": SANDBOX_USER,
            "extra_groups": ["ortools"] if allow_ortools else [],
        }
    start = time.monotonic()
    # Output goes to a file, so processes the code leaves behind cannot hold the call open through a pipe.
    with tempfile.TemporaryFile("w+") as log:
        proc = subprocess.Popen(
            [
                SANDBOX_ORTOOLS_PYTHON if allow_ortools else SANDBOX_PYTHON,
                "-I",
                "-c",
                _RUNNER,
                json.dumps(blocked),
                code_path,
                "solve" if solve else "run",
            ],
            cwd=workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            # RLIMIT_AS cannot be lowered on macOS.
            preexec_fn=_limit_memory(memory_mb) if sys.platform == "linux" else None,
            **user,
        )
        try:
            returncode, timed_out = proc.wait(timeout=timeout_s), False
        except subprocess.TimeoutExpired:
            returncode, timed_out = None, True
        # Kill the whole process group, so nothing the code started keeps running after the call.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        os.remove(code_path)
        log.seek(0)
        output = log.read()
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
