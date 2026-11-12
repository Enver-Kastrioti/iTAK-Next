#!/usr/bin/env bash
# Example: run iTAK Next in test mode using pre-generated results
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CMD=()
if [[ -x ".venv/bin/python" ]]; then
  CMD=(".venv/bin/python")
elif command -v python3 &>/dev/null; then
  CMD=("python3")
else
  CMD=("python")
fi

echo "=== iTAK Next Test Mode Example ==="
echo "This runs the classification pipeline using pre-computed"
echo "InterProScan and hmmscan results, without requiring"
echo "those external tools to be installed."
echo ""

# Run in test mode
# The -json file is copied to the output as the input basename + .json.
# For test_protein.fasta as input, the expected name is test_protein.fasta.json.
"${CMD[@]}" itaknext.py \
  -test \
  -i test_protein.fasta \
  -json test_data/test_ipr_result.json \
  -spechmm test_data/test_hmmscan_result.tbl \
  -o output/test_mode_demo

echo ""
echo "=== Test mode completed ==="
echo "Classification results: output/test_mode_demo/result/match_tbl.txt"
