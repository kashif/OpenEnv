# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sandbox regressions that run without optional OR-Tools/server dependencies."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ortools_sandbox", Path(__file__).parents[2] / "envs/ortools_env/sandbox.py"
)
sandbox = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sandbox)


def test_persistent_instance_symlink_cannot_redirect_server_write(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("untouched")
    code = (
        "import os\nos.unlink('instance.json')\n"
        f"os.symlink({str(target)!r}, 'instance.json')"
    )
    assert sandbox.run(code, {}, str(workdir), 10)["returncode"] == 0
    result = sandbox.run("print(instance['value'])", {"value": 42}, str(workdir), 10)
    assert target.read_text() == "untouched"
    assert result["output"] == "42\n"


def test_subprocess_does_not_inherit_parent_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENENV_TEST_SECRET", "synthetic-canary-1501")
    result = sandbox.run(
        "import os\nassert 'OPENENV_TEST_SECRET' not in os.environ\nprint(instance)",
        {"safe": True},
        str(tmp_path),
        10,
    )
    assert result["returncode"] == 0, result["output"]
    assert "synthetic-canary-1501" not in result["output"]
    assert "safe" in result["output"]


@pytest.mark.parametrize("contents", ["{broken", "\xff", "[]"])
def test_invalid_solution_is_a_failed_submission(tmp_path, contents):
    code = f"import os\nopen('solution.json', 'wb').write({contents.encode('latin1')!r})\nos._exit(0)"
    result = sandbox.run(code, {}, str(tmp_path), 10, solve=True)
    assert result["solution"] is None


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "oversized"])
def test_untrusted_solution_file_is_rejected(tmp_path, kind):
    target = tmp_path / "outside.json"
    target.write_text('{"private": true}')
    operations = {
        "symlink": f"os.symlink({str(target)!r}, 'solution.json')",
        "fifo": "os.mkfifo('solution.json')",
        "directory": "os.mkdir('solution.json')",
        "oversized": "open('solution.json', 'w').write(' ' * (1024 * 1024 + 1) + '{}')",
    }
    code = f"import os\n{operations[kind]}\nos._exit(0)"
    result = sandbox.run(code, {}, str(tmp_path), 10, solve=True)
    assert result["solution"] is None


def test_instance_global_does_not_trust_persistent_file(tmp_path, monkeypatch):
    # Mutate the file at launch to exercise the actual runner input boundary.
    popen = sandbox.subprocess.Popen

    def launch(*args, **kwargs):
        (tmp_path / "instance.json").write_text('{"value": 99}')
        return popen(*args, **kwargs)

    monkeypatch.setattr(sandbox.subprocess, "Popen", launch)
    result = sandbox.run(
        "def solve(instance):\n    return instance",
        {"value": 7},
        str(tmp_path),
        10,
        solve=True,
    )
    assert result["solution"] == {"value": 7}


def test_workdir_symlink_is_rejected_before_privileged_io(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    workdir = tmp_path / "work"
    workdir.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        sandbox.run("pass", {}, str(workdir), 10)
    assert list(target.iterdir()) == []


def test_solution_read_uses_pinned_directory(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    moved = tmp_path / "moved"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "solution.json").write_text('{"private": true}')
    code = f"""
import os
os.rename({str(workdir)!r}, {str(moved)!r})
os.symlink({str(outside)!r}, {str(workdir)!r})
open('solution.json', 'w').write('{{"submitted": true}}')
os._exit(0)
"""
    result = sandbox.run(code, {}, str(workdir), 10, solve=True)
    assert result["solution"] == {"submitted": True}
