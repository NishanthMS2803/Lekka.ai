"""Unit tests for the run_python sandbox execution tool.

Validates subprocess isolation, resource bounds, timeouts, restricted imports,
and filesystem write containment.
"""

from __future__ import annotations

import pytest

from lekka_ai.tools.base import default_registry
from lekka_ai.tools.python_sandbox import (
    RunPythonInput,
    RunPythonTool,
)


@pytest.fixture
def python_tool() -> RunPythonTool:
    """Fixture providing an instance of RunPythonTool."""
    return RunPythonTool()


@pytest.mark.asyncio
async def test_run_python_simple_math_works(python_tool: RunPythonTool) -> None:
    """Verify standard arithmetic, math library, and stdout capture."""
    code = """
import math
import statistics

val = math.sqrt(64) + 10
mean_val = statistics.mean([val, 20, 30])
print(f"Calculated value: {val}")
result = mean_val
"""
    output = await python_tool.run(RunPythonInput(code=code))

    assert output.exit_code == 0
    assert output.error is None
    assert "Calculated value: 18.0" in output.stdout
    assert output.result == pytest.approx(22.666666666666668)


@pytest.mark.asyncio
async def test_run_python_numpy_pandas_works(python_tool: RunPythonTool) -> None:
    """Verify numpy and pandas libraries are allowed and compute correctly."""
    code = """
import numpy as np
import pandas as pd

arr = np.array([10, 20, 30, 40])
df = pd.DataFrame({"values": arr})
df["doubled"] = df["values"] * 2
print("DataFrame created:")
print(df)
result = float(df["doubled"].mean())
"""
    output = await python_tool.run(RunPythonInput(code=code))

    assert output.exit_code == 0
    assert output.error is None
    assert "DataFrame created:" in output.stdout
    assert output.result == 50.0


@pytest.mark.asyncio
async def test_run_python_infinite_loop_times_out(python_tool: RunPythonTool) -> None:
    """Verify infinite loop triggers timeout and kills subprocess safely."""
    code = """
while True:
    pass
"""
    output = await python_tool.run(RunPythonInput(code=code, timeout=0.5))

    assert output.exit_code == -1
    assert output.result is None
    assert output.error is not None
    assert "timed out" in output.error.lower()
    assert "timed out" in output.stderr.lower()


@pytest.mark.asyncio
async def test_run_python_import_os_blocked(python_tool: RunPythonTool) -> None:
    """Verify attempt to import os is intercepted and blocked safely."""
    code = """
import os

result = os.getcwd()
"""
    output = await python_tool.run(RunPythonInput(code=code))

    assert output.exit_code != 0
    assert output.result is None
    assert output.error is not None
    assert "ImportError" in output.stderr
    assert "Import of module 'os' is disallowed" in output.stderr


@pytest.mark.asyncio
async def test_run_python_import_subprocess_blocked(
    python_tool: RunPythonTool,
) -> None:
    """Verify attempt to import subprocess is intercepted and blocked safely."""
    code = """
import subprocess

result = subprocess.run(["echo", "escape"])
"""
    output = await python_tool.run(RunPythonInput(code=code))

    assert output.exit_code != 0
    assert output.result is None
    assert output.error is not None
    assert "ImportError" in output.stderr
    assert "Import of module 'subprocess' is disallowed" in output.stderr


@pytest.mark.asyncio
async def test_run_python_import_socket_blocked(python_tool: RunPythonTool) -> None:
    """Verify attempt to import socket is intercepted and blocked safely."""
    code = """
import socket

s = socket.socket()
result = "connected"
"""
    output = await python_tool.run(RunPythonInput(code=code))

    assert output.exit_code != 0
    assert output.result is None
    assert output.error is not None
    assert "ImportError" in output.stderr
    assert "Import of module 'socket' is disallowed" in output.stderr


@pytest.mark.asyncio
async def test_run_python_file_writes_outside_temp_blocked(
    python_tool: RunPythonTool,
) -> None:
    """Verify writing files outside sandbox temp directory raises PermissionError."""
    code = """
with open("/tmp/sandbox_escape_leak.txt", "w") as f:
    f.write("unauthorized payload")
result = "written"
"""
    output = await python_tool.run(RunPythonInput(code=code))

    assert output.exit_code != 0
    assert output.result is None
    assert output.error is not None
    assert "PermissionError" in output.stderr
    assert "outside sandbox" in output.stderr.lower()


@pytest.mark.asyncio
async def test_run_python_file_writes_inside_temp_allowed(
    python_tool: RunPythonTool,
) -> None:
    """Verify writing files within the assigned temporary directory is permitted."""
    code = """
with open("scratch.txt", "w") as f:
    f.write("allowed temporary content")

with open("scratch.txt", "r") as f:
    read_back = f.read()

result = read_back
"""
    output = await python_tool.run(RunPythonInput(code=code))

    assert output.exit_code == 0
    assert output.error is None
    assert output.result == "allowed temporary content"


def test_run_python_in_default_registry() -> None:
    """Verify run_python is available in default_registry and documents Docker note."""
    tool = default_registry.get("run_python")
    assert tool.name == "run_python"
    assert "Docker isolation is a later upgrade" in tool.description
