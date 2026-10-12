#!/usr/bin/env bash
set -euo pipefail

# Lekka AI Scripted Demonstration
# Runs two sample research questions (high-confidence equity valuation and safe abstention)
# and outputs the resulting synthesized report files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "================================================================="
echo "               LEKKA AI AUTOMATED SCRIPTED DEMO                  "
echo "================================================================="
echo "Working Directory: ${ROOT_DIR}"
echo ""

mkdir -p "${ROOT_DIR}/reports"

uv run python "${ROOT_DIR}/scripts/run_demo.py" "$@"

echo ""
echo "================================================================="
echo "Demo complete. View your generated equity reports at:"
echo "  Report 1 (AAPL Analysis):     ${ROOT_DIR}/reports/demo_run_1_aapl.md"
echo "  Report 2 (Abstention Report): ${ROOT_DIR}/reports/demo_run_2_abstention.md"
echo "================================================================="
