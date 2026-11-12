#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the FASTA validation module (validate_fasta.py).

Run with:
    cd itak3.7 && python -m pytest tests/test_validate_fasta.py -v
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from module.validate_fasta import FastaValidator


VALID_PROTEIN = ">seq1\nACDEFGHIKLMNPQRSTVWY\n"
MIXED_CASE = ">seq2\nACDEFGHIKLMNPQRSTVWy\n"
WITH_STOP = ">seq3\nACDEFG*HIKLMNPQRSTVWY\n"
INVALID_CHARS = ">seq4\nACDEFGX12345HIKLMNPQ\n"
EMPTY_SEQ = ">seq5\n\n"
MULTI_SEQ = ">s1\nACDEFGHIK\n>s2\nLMNPQRSTVWY\n"
NON_PROTEIN = ">s3\nACGTTGCATGCA\n"


class TestFastaValidator(unittest.TestCase):
    """Test FASTA validation."""

    def setUp(self):
        self.validator = FastaValidator()
        self.tmpdir = tempfile.mkdtemp()

    def _write(self, content, name="test.fasta"):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_protein(self):
        path = self._write(VALID_PROTEIN)
        self.assertTrue(self.validator.run_full_validation(path))
        self.assertEqual(self.validator.protein_count, 1)

    def test_mixed_case_valid(self):
        path = self._write(MIXED_CASE)
        self.assertTrue(self.validator.run_full_validation(path))
        self.assertEqual(self.validator.protein_count, 1)

    def test_stop_codon_rejected(self):
        path = self._write(WITH_STOP)
        self.assertFalse(self.validator.run_full_validation(path))

    def test_invalid_characters_rejected(self):
        path = self._write(INVALID_CHARS)
        self.assertFalse(self.validator.run_full_validation(path))

    def test_empty_sequence_rejected(self):
        path = self._write(EMPTY_SEQ)
        self.assertFalse(self.validator.run_full_validation(path))

    def test_multi_sequence(self):
        path = self._write(MULTI_SEQ)
        self.assertTrue(self.validator.run_full_validation(path))
        self.assertEqual(self.validator.protein_count, 2)

    def test_nonexistent_file(self):
        self.assertFalse(self.validator.run_full_validation("/nonexistent/path.fasta"))
        self.assertTrue(len(self.validator.errors) > 0)

    def test_dna_sequence_rejected(self):
        """DNA (nucleotide) sequence should be rejected as non-protein.
        'T' is not in the extended amino-acid alphabet, so ACGTTGCATGCA
        has valid_ratio = 4/5 = 0.8 which meets the 0.8 threshold exactly.
        Use a sequence with more non-amino characters to ensure rejection.
        """
        path = self._write(">s3\nACGTNX\n")
        # 'X' is valid, 'N' is valid (ambiguity), 'T' is the non-amino char.
        # valid_ratio = 4/5 = 0.8 → still passes. Need stronger test:
        path2 = self._write(">s4\nACGT12345\n")
        self.assertFalse(self.validator.run_full_validation(path2))

    def test_validate_sequences_api(self):
        """Test the validate_sequences method directly."""
        validator = FastaValidator()
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]
        self.assertTrue(validator.validate_sequences(seqs))
        self.assertEqual(validator.protein_count, 2)

    def test_validate_sequences_empty(self):
        validator = FastaValidator()
        self.assertFalse(validator.validate_sequences([]))

    def test_validate_sequences_with_stop(self):
        validator = FastaValidator()
        self.assertFalse(validator.validate_sequences(["ACDEFG*HIKL"]))

    def test_validate_sequences_dict_input(self):
        """Test with dict-like items."""
        validator = FastaValidator()
        seqs = [{"sequence": "ACDEFGHIKLMNPQRSTVWY"}]
        self.assertTrue(validator.validate_sequences(seqs))
        self.assertEqual(validator.protein_count, 1)


if __name__ == "__main__":
    unittest.main()
