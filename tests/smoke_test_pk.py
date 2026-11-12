#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PK smoke test — runs the full protein kinase pipeline on a small known-PK input
and verifies the output structure and minimum classification count.

Usage:
    python tests/smoke_test_pk.py [--input test_protein_kinase.fasta] [--require-pk N]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent


def main():
    parser = argparse.ArgumentParser(description="PK pipeline smoke test")
    parser.add_argument(
        "--input",
        default=str(PROJECT_DIR / "test_protein_kinase.fasta"),
        help="Input FASTA file with PK sequences",
    )
    parser.add_argument(
        "--require-pk",
        type=int,
        default=2,
        help="Minimum number of PK classifications required (default: 2)",
    )
    parser.add_argument(
        "--python-bin",
        default=shutil.which("python3") or sys.executable,
        help="Python interpreter to use",
    )
    args = parser.parse_args()

    input_fasta = Path(args.input).resolve()
    if not input_fasta.exists():
        print(f"Input FASTA not found: {input_fasta}")
        sys.exit(1)

    # Use a temp output dir
    output_dir = Path(tempfile.mkdtemp(prefix="pk_smoke_")).resolve()

    itak_script = PROJECT_DIR / "winnowtak.py"
    if not itak_script.exists():
        print(f"iTAK Next script not found: {itak_script}")
        sys.exit(1)

    cmd = [
        args.python_bin,
        str(itak_script),
        "-i", str(input_fasta),
        "-o", str(output_dir),
        "--skip-deps-check",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-1000:]}")
        print("PK smoke test FAILED (iTAK Next returned non-zero)")
        sys.exit(1)

    # Check PK output files
    pk_dir = output_dir / "protein_kinase"
    required_files = [
        pk_dir / "match_tbl.txt",
        pk_dir / "pk_classification.tsv",
        pk_dir / "pk_sequence.fasta",
        pk_dir / "shiu_classification.txt",
        pk_dir / "PPC_classification.txt",
    ]

    all_ok = True
    for f in required_files:
        if f.exists() and f.stat().st_size > 0:
            print(f"  [OK] {f.name}")
        else:
            print(f"  [FAIL] {f.name} missing or empty")
            all_ok = False

    # Check PK classification count
    tsv_path = pk_dir / "pk_classification.tsv"
    if tsv_path.exists():
        with open(tsv_path) as fh:
            lines = fh.readlines()
        pk_count = len([l for l in lines if l.strip() and not l.startswith("Sequence_ID")])
        print(f"  PK classifications found: {pk_count} (required >= {args.require_pk})")
        if pk_count < args.require_pk:
            print(f"  [FAIL] Expected >= {args.require_pk} PKs, got {pk_count}")
            all_ok = False
        else:
            print(f"  [OK] Sufficient PK classifications")

    # Also check combined result
    combined_tbl = output_dir / "result" / "all_match_tbl.txt"
    if combined_tbl.exists():
        print(f"  [OK] Combined TF + PK table: {combined_tbl}")

    # Cleanup
    shutil.rmtree(output_dir, ignore_errors=True)

    if all_ok:
        print("\nPK smoke test PASSED!")
    else:
        print("\nPK smoke test FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
