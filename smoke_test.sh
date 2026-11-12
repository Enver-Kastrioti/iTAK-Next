#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT_DIR/test/test_protein.fasta" ]]; then
  INPUT_FASTA="$ROOT_DIR/test/test_protein.fasta"
else
  INPUT_FASTA="$ROOT_DIR/test_protein.fasta"
fi
OUTPUT_DIR="$ROOT_DIR/output/smoke_test"
APPL_LIST="PROSITEPROFILES"
RUN_PREDICT=0
REQUIRE_PK_COUNT=""

usage() {
  cat <<'EOF'
Usage: ./smoke_test.sh [options]

Options:
  --predict           Run the prediction workflow before InterProScan/classification
  --input PATH        Input FASTA file (default: test_protein.fasta)
  --output PATH       Output directory (default: output/smoke_test)
  --appl LIST         InterProScan applications (default: PROSITEPROFILES)
  --require-pk N      Require at least N protein kinase classifications
  --help              Show this help message

Examples:
  ./smoke_test.sh
  ./smoke_test.sh --predict
  ./smoke_test.sh --input test_protein_kinase.fasta --require-pk 2
  ./smoke_test.sh --predict --input test_protein_kinase.fasta --require-pk 2
  ./smoke_test.sh --appl CDD,Pfam,SMART
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predict)
      RUN_PREDICT=1
      shift
      ;;
    --input)
      INPUT_FASTA="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --appl)
      APPL_LIST="$2"
      shift 2
      ;;
    --require-pk)
      REQUIRE_PK_COUNT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$INPUT_FASTA" ]]; then
  echo "Input FASTA not found: $INPUT_FASTA" >&2
  exit 1
fi

if [[ "$INPUT_FASTA" != /* ]]; then
  INPUT_FASTA="$ROOT_DIR/$INPUT_FASTA"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi
INPUT_FASTA="$(cd "$(dirname "$INPUT_FASTA")" && pwd)/$(basename "$INPUT_FASTA")"

CMD=()
if [[ -x "$ROOT_DIR/run_itaknext_local.sh" ]]; then
  CMD=("$ROOT_DIR/run_itaknext_local.sh")
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  CMD=("$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/itaknext.py")
else
  if command -v python3 >/dev/null 2>&1; then
    CMD=("python3" "$ROOT_DIR/itaknext.py")
  elif command -v python >/dev/null 2>&1; then
    CMD=("python" "$ROOT_DIR/itaknext.py")
  else
    echo "No Python interpreter found (expected run_itaknext_local.sh, .venv/bin/python, python3, or python)" >&2
    exit 1
  fi
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

ARGS=(-i "$INPUT_FASTA" --appl "$APPL_LIST" -o "$OUTPUT_DIR")
if [[ "$RUN_PREDICT" -eq 1 ]]; then
  ARGS=(--predict "${ARGS[@]}")
fi

echo "Running smoke test..."
printf '+'
printf ' %q' "${CMD[@]}" "${ARGS[@]}"
printf '\n'
"${CMD[@]}" "${ARGS[@]}"

check_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Expected output was not created: $path" >&2
    exit 1
  fi
}

json_file="$(find "$OUTPUT_DIR/InterproScan" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null || true)"
check_file "$json_file"
check_file "$OUTPUT_DIR/hmmscan/result.tbl"
check_file "$OUTPUT_DIR/result/match_tbl.txt"
check_file "$OUTPUT_DIR/result/all_match_tbl.txt"
tf_classified_fasta="$(find "$OUTPUT_DIR/result" -maxdepth 1 -name '*_tf_classified.fasta' -print -quit 2>/dev/null || true)"
check_file "$tf_classified_fasta"

pk_dir="$OUTPUT_DIR/protein_kinase"
pk_count=0
pk_classified_fasta=""
if [[ -d "$pk_dir" ]]; then
  check_file "$pk_dir/match_tbl.txt"
  check_file "$pk_dir/pk_classification.tsv"
  check_file "$pk_dir/pk_sequence.fasta"
  pk_classified_fasta="$(find "$pk_dir" -maxdepth 1 -name '*_pk_classified.fasta' -print -quit 2>/dev/null || true)"
  check_file "$pk_classified_fasta"
  check_file "$pk_dir/shiu_classification.txt"
  check_file "$pk_dir/PPC_classification.txt"
  pk_count="$(tail -n +2 "$pk_dir/pk_classification.tsv" | awk 'NF{count++} END{print count+0}')"
fi
if [[ -n "$REQUIRE_PK_COUNT" ]]; then
  if [[ ! -d "$pk_dir" ]]; then
    echo "Expected protein kinase output directory, found none: $pk_dir" >&2
    exit 1
  fi
  if [[ "$pk_count" -lt "$REQUIRE_PK_COUNT" ]]; then
    echo "Expected at least $REQUIRE_PK_COUNT protein kinase classifications, found $pk_count" >&2
    exit 1
  fi
fi

if [[ "$RUN_PREDICT" -eq 1 ]]; then
  pred_csv="$(find "$OUTPUT_DIR/protein_model_preclassification" -maxdepth 1 -name '*_prediction.csv' -print -quit 2>/dev/null || true)"
  pred_tf_csv="$(find "$OUTPUT_DIR/protein_model_preclassification" -maxdepth 1 -name '*_prediction_tf_only.csv' -print -quit 2>/dev/null || true)"
  check_file "$pred_csv"
  check_file "$pred_tf_csv"
fi

echo "Smoke test passed."
echo "Output directory: $OUTPUT_DIR"
echo "InterProScan JSON: $json_file"
echo "Classification table: $OUTPUT_DIR/result/match_tbl.txt"
echo "TF classified FASTA: $tf_classified_fasta"
if [[ -n "$pk_classified_fasta" ]]; then
  echo "Protein kinase FASTA: $pk_classified_fasta"
fi
echo "Protein kinase classifications: $pk_count"
