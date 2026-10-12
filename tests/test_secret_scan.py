"""Secret scanning test to ensure no credentials or API keys leak into
git-tracked files.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# High-confidence regex patterns for real credentials and secret keys
SECRET_PATTERNS = [
    (
        "Private Key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "AWS Access Key ID",
        re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    ),
    (
        "Google API Key",
        re.compile(r"\b(AIza[0-9A-Za-z\-_]{35})\b"),
    ),
    (
        "Live Gemini API Key",
        re.compile(r"\b(AQ\.[0-9A-Za-z\-_]{40,})\b"),
    ),
    (
        "OpenAI API Key",
        re.compile(r"\b(sk-[a-zA-Z0-9]{32,})\b"),
    ),
    (
        "Slack Webhook / Token",
        re.compile(
            r"https://hooks\.slack\.com/services/T[0-9A-Z]+/B[0-9A-Z]+/[0-9A-Za-z]+"
        ),
    ),
]

# Files explicitly excluded from scanning (e.g. tests with mock patterns, lockfiles)
EXCLUDED_PATTERNS = [
    "tests/test_secret_scan.py",
    "tests/test_hardening.py",
    "tests/test_gemini_live.py",  # Reads from os.getenv
    "uv.lock",
    "frontend/package-lock.json",
]


def test_repo_has_no_tracked_secrets() -> None:
    """Ensure zero tracked files in git contain unredacted secrets or API keys."""
    # Retrieve all files tracked by git
    res = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]

    repo_root = Path(__file__).resolve().parent.parent
    findings: list[str] = []

    for rel_path in tracked_files:
        if any(rel_path.endswith(ex) or rel_path == ex for ex in EXCLUDED_PATTERNS):
            continue

        file_path = repo_root / rel_path
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for rule_name, pattern in SECRET_PATTERNS:
            matches = pattern.findall(content)
            for m in matches:
                # Ignore placeholders such as <GEMINI_API_KEY>
                if isinstance(m, str) and (m.startswith("<") or "mock" in m.lower()):
                    continue
                findings.append(f"{rel_path}: [{rule_name}] -> {m}")

    assert not findings, (
        f"Secret scan detected {len(findings)} potential leaked secrets:\n"
        + "\n".join(findings)
    )
