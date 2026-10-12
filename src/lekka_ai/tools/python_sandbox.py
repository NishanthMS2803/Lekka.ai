"""Python code execution sandbox tool with restricted imports and subprocess isolation.

Note:
    Subprocess isolation, resource limits (RLIMIT_CPU, RLIMIT_AS), environment
    dropping, and audit hooks provide initial execution sandboxing.
    Full containerized Docker isolation is a later upgrade for untrusted multi-tenant
    deployments.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import tempfile
from typing import Any

from pydantic import BaseModel, Field

from lekka_ai.logging import get_logger
from lekka_ai.tools.base import Tool, default_registry

logger = get_logger(__name__)


class RunPythonInput(BaseModel):
    """Input payload for the run_python sandbox tool."""

    code: str = Field(..., description="Python source code to execute in the sandbox")
    timeout: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Execution timeout in seconds",
    )


class RunPythonOutput(BaseModel):
    """Execution output from the run_python sandbox tool."""

    stdout: str = Field(default="", description="Standard output from code execution")
    stderr: str = Field(default="", description="Standard error from code execution")
    result: Any = Field(
        default=None,
        description="Value of 'result' variable if defined in the script",
    )
    error: str | None = Field(
        default=None,
        description="Execution error message if failed or timed out",
    )
    exit_code: int = Field(default=0, description="Process exit code")


_RUNNER_TEMPLATE = """
import builtins
import json
import math
import os
import pathlib
import resource
import statistics
import sys

import numpy
import pandas

# Pre-warm pandas string formatters and console dependencies
str(pandas.DataFrame({{"_": [0]}}))

temp_dir = pathlib.Path({temp_dir_repr}).resolve()
code_file = pathlib.Path({code_file_repr}).resolve()
timeout_seconds = {timeout_val}

# 1. Apply CPU and Memory limits via resource.setrlimit
try:
    soft_cpu, hard_cpu = resource.getrlimit(resource.RLIMIT_CPU)
    cpu_limit = max(1, int(timeout_seconds) + 2)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, hard_cpu))
except Exception:
    pass

try:
    if hasattr(resource, "RLIMIT_AS"):
        soft_as, hard_as = resource.getrlimit(resource.RLIMIT_AS)
        mem_bytes = 512 * 1024 * 1024
        if hard_as == resource.RLIM_INFINITY or mem_bytes <= hard_as:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard_as))
except Exception:
    pass

# 2. Restricted import allowlist and explicit blocked blacklist
BLOCKED_MODULES = {{
    "os",
    "subprocess",
    "_posixsubprocess",
    "socket",
    "_socket",
    "sys",
    "shutil",
    "ctypes",
    "http",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "importlib",
}}

ALLOWED_MODULES = {{
    "__main__",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "numpy",
    "operator",
    "pandas",
    "re",
    "scipy",
    "statistics",
    "string",
    "traceback",
    "typing",
}}

_orig_import = builtins.__import__


def secure_import(name, globals=None, locals=None, fromlist=(), level=0):
    top_level = name.split(".")[0]
    if top_level in BLOCKED_MODULES or any(b in name for b in ("socket", "subprocess")):
        raise ImportError(f"Import of module '{{name}}' is disallowed in sandbox.")
    if top_level in ALLOWED_MODULES:
        return _orig_import(name, globals, locals, fromlist, level)
    is_blocked_sub = any(b in name for b in ("socket", "subproc", "posix"))
    if name.startswith("_") and not is_blocked_sub:
        return _orig_import(name, globals, locals, fromlist, level)
    raise ImportError(
        f"Import of module '{{name}}' is disallowed in sandbox. "
        f"Allowed modules: {{sorted(ALLOWED_MODULES)}}"
    )


builtins.__import__ = secure_import

# 3. Security audit hooks blocking sockets, subprocesses, and unauthorized writes
def audit_hook(event, args):
    if event.startswith("socket."):
        raise PermissionError(f"Network access disabled in sandbox: {{event}}")
    if event.startswith("subprocess."):
        raise PermissionError(f"Subprocess creation disabled in sandbox: {{event}}")
    if event in ("os.system", "os.posix_spawn", "os.spawn"):
        raise PermissionError(f"Process execution disabled in sandbox: {{event}}")
    if event == "open":
        file_path, mode, flags = args
        if any(m in mode for m in ("w", "a", "+", "x")):
            try:
                resolved = pathlib.Path(file_path).resolve()
                if not (resolved == temp_dir or temp_dir in resolved.parents):
                    raise PermissionError(
                        f"File write outside sandbox temp directory blocked: "
                        f"{{file_path}}"
                    )
            except PermissionError:
                raise
            except Exception as exc:
                raise PermissionError(f"Path resolution check failed: {{exc}}")
    if event in ("os.mkdir", "os.rmdir", "os.remove", "os.rename", "os.unlink"):
        try:
            target = pathlib.Path(args[0]).resolve()
            if not (target == temp_dir or temp_dir in target.parents):
                raise PermissionError(
                    f"File modification outside sandbox temp directory blocked: "
                    f"{{args[0]}}"
                )
        except PermissionError:
            raise
        except Exception:
            pass


sys.addaudithook(audit_hook)


def _serialize(val):
    if hasattr(val, "to_dict"):
        try:
            return val.to_dict()
        except Exception:
            pass
    if hasattr(val, "tolist"):
        try:
            return val.tolist()
        except Exception:
            pass
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    if isinstance(val, (int, float, str, bool, type(None), list, dict)):
        return val
    try:
        json.dumps(val)
        return val
    except Exception:
        return str(val)


# 4. Execute user code in isolated global scope
user_code = code_file.read_text(encoding="utf-8")
scope = {{
    "__name__": "__main__",
    "math": math,
    "statistics": statistics,
    "numpy": numpy,
    "pandas": pandas,
}}

try:
    exec(user_code, scope)
    result_val = scope.get("result")
    serialized_res = _serialize(result_val)
    res_path = temp_dir / "_sandbox_result.json"
    res_path.write_text(json.dumps({{"result": serialized_res}}), encoding="utf-8")
except BaseException:
    import traceback

    traceback.print_exc()
    sys.exit(1)
"""


class RunPythonTool(Tool[RunPythonInput, RunPythonOutput]):
    """Tool that executes Python code in an isolated subprocess sandbox.

    Features:
    - Subprocess isolation with separate memory space and dropped environment.
    - Timeouts with process termination.
    - Memory and CPU limits (resource.setrlimit).
    - No network access (dropped env and blocked sockets via audit hooks).
    - Dedicated temporary directory for file operations.
    - Restricted import allowlist (math, statistics, numpy, pandas).
    - Note: Full Docker container isolation is a later upgrade.
    """

    name: str = "run_python"
    description: str = (
        "Executes Python code in an isolated subprocess with restricted libraries "
        "(math, statistics, numpy, pandas). Note: Docker isolation is a later upgrade."
    )
    input_model: type[RunPythonInput] = RunPythonInput
    output_model: type[RunPythonOutput] = RunPythonOutput

    async def run(self, input_data: RunPythonInput) -> RunPythonOutput:
        """Execute Python code in isolated sandbox subprocess."""
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_path = pathlib.Path(temp_dir_obj.name).resolve()

        try:
            code_file = temp_path / "_user_code.py"
            code_file.write_text(input_data.code, encoding="utf-8")

            runner_file = temp_path / "_runner.py"
            runner_script = _RUNNER_TEMPLATE.format(
                temp_dir_repr=repr(str(temp_path)),
                code_file_repr=repr(str(code_file)),
                timeout_val=input_data.timeout,
            )
            runner_file.write_text(runner_script, encoding="utf-8")

            # Clean environment without credentials or parent environment vars
            isolated_env = {"PYTHONIOENCODING": "utf-8"}

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(runner_file),
                cwd=str(temp_path),
                env=isolated_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=input_data.timeout,
                )
            except TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception as exc:
                    logger.warning(
                        "Error killing timed out sandbox process", error=str(exc)
                    )

                timeout_msg = f"Execution timed out after {input_data.timeout} seconds"
                return RunPythonOutput(
                    stdout="",
                    stderr=timeout_msg,
                    result=None,
                    error=timeout_msg,
                    exit_code=-1,
                )

            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode if proc.returncode is not None else 0

            result_val: Any = None
            result_json_file = temp_path / "_sandbox_result.json"
            if result_json_file.exists():
                try:
                    payload = json.loads(result_json_file.read_text(encoding="utf-8"))
                    result_val = payload.get("result")
                except Exception as exc:
                    logger.warning("Failed parsing sandbox result JSON", error=str(exc))

            error_msg: str | None = None
            if exit_code != 0:
                error_msg = (
                    stderr_str.strip() or f"Process exited with code {exit_code}"
                )

            return RunPythonOutput(
                stdout=stdout_str,
                stderr=stderr_str,
                result=result_val,
                error=error_msg,
                exit_code=exit_code,
            )
        finally:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                pass


default_registry.register(RunPythonTool(), overwrite=True)
