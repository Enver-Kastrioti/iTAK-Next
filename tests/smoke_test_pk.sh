#!/usr/bin/env bash
# PK smoke test — runs the protein kinase pipeline and verifies outputs
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
INPUT_FASTA="$ROOT_DIR/test_protein_kinase.fasta"
OUTPUT_DIR="$ROOT_DIR/output/smoke_test_pk"
REQUIRE_PK=2

usage() {
  cat <<'EOF'
Usage: ./tests/smoke_test_pk.sh [options]

Options:
  --input PATH        Input FASTA file (default: test_protein_kinase.fasta)
  --output PATH       Output directory (default: output/smoke_test_pk)
  --require-pk N      Minimum PK classifications (default: 2)
  --help              Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_FASTA="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --require-pk) REQUIRE_PK="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -f "$INPUT_FASTA" ]]; then
  echo "Input not found: $INPUT_FASTA" >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

CMD=()
if [[ -x "$ROOT_DIR/run_itaknext_local.sh" ]]; then
  CMD=("$ROOT_DIR/run_itaknext_local.sh")
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  CMD=("$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/itaknext.py")
else
  CMD=("python3" "$ROOT_DIR/itaknext.py")
fi

# Run iTAK Next with --skip-pk disabled so PK analysis runs,
# but also skip-deps-check for environments without full InterProScan data
ARGS=(-i "$INPUT_FASTA" -o "$OUTPUT_DIR")

echo "=== PK Smoke Test ==="
echo "Input: $INPUT_FASTA"
echo "Output: $OUTPUT_DIR"
echo "Command: ${CMD[*]} ${ARGS[*]}"
echo
"${CMD[@]}" "${ARGS[@]}"

echo
echo "=== Verifying PK outputs ==="

pk_dir="$OUTPUT_DIR/protein_kinase"
check_file() {
  if [[ ! -f "$1" ]]; then
    echo "  [FAIL] Missing: $1"
    exit 1
  fi
  echo "  [OK] $1"
}

check_file "$pk_dir/match_tbl.txt"
check_file "$pk_dir/pk_classification.tsv"
check_file "$pk_dir/pk_sequence.fasta"
check_file "$pk_dir/shiu_classification.txt"
check_file "$pk_dir/PPC_classification.txt"

pk_count="$(tail -n +2 "$pk_dir/pk_classification.tsv" 2>/dev/null | awk 'NF{count++} END{print count+0}')"
echo "  PK classifications: $pk_count (required >= $REQUIRE_PK)"

if [[ "$pk_count" -lt "$REQUIRE_PK" ]]; then
  echo "  [FAIL] Expected >= $REQUIRE_PK PKs, got $pk_count"
  exit 1
fi

echo "  [OK] Sufficient PK classifications"

# Print summary
echo
echo "=== PK Classification Summary ==="
if [[ -f "$pk_dir/match_tbl.txt" ]]; then
  cat "$pk_dir/match_tbl.txt"
fi

echo
echo "PK smoke test PASSED!"
