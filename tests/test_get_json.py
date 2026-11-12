#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the InterProScan JSON processing module (get_json.py).

Run with:
    cd itak3.7 && python -m pytest tests/test_get_json.py -v
"""

import json
import os
import sys
import unittest
import tempfile
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from module.get_json import (
    load_input_data,
    is_valid_score,
    get_score_value,
    process_data,
    parse_specific_thresholds_from_rule,
)


# A minimal but valid InterProScan-like JSON
MINIMAL_IPR_JSON = {
    "results": [
        {
            "sequence": {"id": "seq1"},
            "xref": [{"id": "GENE001"}],
            "matches": [
                {
                    "signature": {
                        "accession": "PF00001",
                        "signatureLibraryRelease": {"library": "Pfam"},
                        "entry": {
                            "accession": "IPR001111",
                            "name": "TestDomain",
                            "description": "A test domain",
                        },
                    },
                    "score": "STRONG",
                    "locations": [
                        {
                            "start": 10,
                            "end": 50,
                            "score": 15.0,
                            "evalue": 1e-10,
                        }
                    ],
                }
            ],
        },
        {
            "sequence": {"id": "seq2"},
            "xref": [{"id": "GENE002"}],
            "matches": [
                {
                    "signature": {
                        "accession": "PF00002",
                        "signatureLibraryRelease": {"library": "Pfam"},
                        "entry": {
                            "accession": "IPR002222",
                            "name": "AnotherDomain",
                            "description": "Another test domain",
                        },
                    },
                    "score": 5.0,
                    "locations": [
                        {
                            "start": 5,
                            "end": 30,
                            "score": 5.0,
                            "evalue": 1e-5,
                        }
                    ],
                }
            ],
        },
    ]
}

EMPTY_JSON = {"results": []}

NO_MATCHES_JSON = {
    "results": [
        {
            "sequence": {"id": "seq3"},
            "xref": [{"id": "GENE003"}],
        }
    ]
}


class TestLoadInputData(unittest.TestCase):
    """Test loading InterProScan JSON files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_json(self, data, name="test.json"):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_valid_json(self):
        path = self._write_json(MINIMAL_IPR_JSON)
        result = load_input_data(path)
        self.assertIsNotNone(result)
        self.assertIn("results", result)

    def test_load_nonexistent_file(self):
        result = load_input_data("/nonexistent.json")
        self.assertIsNone(result)

    def test_load_empty_json(self):
        path = self._write_json(EMPTY_JSON)
        result = load_input_data(path)
        self.assertEqual(len(result["results"]), 0)

    def test_load_invalid_json(self):
        path = os.path.join(self.tmpdir, "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json")
        result = load_input_data(path)
        self.assertIsNone(result)


class TestIsValidScore(unittest.TestCase):
    """Test score filtering logic."""

    def test_score_above_threshold(self):
        self.assertTrue(is_valid_score(5.0, score_threshold=1.0))

    def test_score_below_threshold(self):
        self.assertFalse(is_valid_score(0.5, score_threshold=1.0))

    def test_strong_sentinel(self):
        """'STRONG' should pass threshold check."""
        self.assertTrue(is_valid_score("STRONG", score_threshold=1.0))

    def test_specific_threshold(self):
        """Specific accession threshold overrides global."""
        specific = {"PF00001": 20.0}
        self.assertTrue(is_valid_score(25.0, accession="PF00001",
                                        specific_thresholds=specific))
        self.assertFalse(is_valid_score(15.0, accession="PF00001",
                                         specific_thresholds=specific))

    def test_invalid_score_type(self):
        self.assertFalse(is_valid_score("INVALID", score_threshold=1.0))

    def test_none_score(self):
        self.assertFalse(is_valid_score(None, score_threshold=1.0))


class TestGetScoreValue(unittest.TestCase):
    """Test score value conversion."""

    def test_numeric(self):
        self.assertEqual(get_score_value(5.0), 5.0)

    def test_string_numeric(self):
        self.assertEqual(get_score_value("5.0"), 5.0)

    def test_strong(self):
        self.assertEqual(get_score_value("STRONG"), 10.0)

    def test_strong_custom_value(self):
        self.assertEqual(get_score_value("STRONG", strong_value=20.0), 20.0)

    def test_none(self):
        self.assertEqual(get_score_value(None), 0.0)


class TestProcessData(unittest.TestCase):
    """Test the full data processing pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_json(self, data, name="input.json"):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_process_minimal_data(self):
        path = self._write_json(MINIMAL_IPR_JSON)
        filtered, raw = process_data(path, output_dir=self.tmpdir)
        self.assertIsNotNone(filtered)
        self.assertIn("result", filtered)
        self.assertGreater(len(filtered["result"]["match"]), 0)

    def test_process_empty_data(self):
        path = self._write_json(EMPTY_JSON)
        filtered, raw = process_data(path, output_dir=self.tmpdir)
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered["result"]["match"]), 0)

    def test_process_no_matches(self):
        path = self._write_json(NO_MATCHES_JSON)
        filtered, raw = process_data(path, output_dir=self.tmpdir)
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered["result"]["match"]), 0)

    def test_processed_json_written(self):
        path = self._write_json(MINIMAL_IPR_JSON)
        process_data(path, output_dir=self.tmpdir)
        self.assertTrue(os.path.exists(
            os.path.join(self.tmpdir, "processed_ipr_domains.json")
        ))


class TestParseSpecificThresholds(unittest.TestCase):
    """Test parsing Score: lines from rule file."""

    def setUp(self):
        self.rule_file = str(PROJECT_DIR / "rule.txt")

    def test_parses_without_error(self):
        thresholds = parse_specific_thresholds_from_rule(self.rule_file)
        self.assertIsInstance(thresholds, dict)

    def test_returns_dict(self):
        thresholds = parse_specific_thresholds_from_rule(self.rule_file)
        # If any thresholds exist, they should be parseable as floats
        for acc, val in thresholds.items():
            self.assertIsInstance(val, (int, float))


if __name__ == "__main__":
    unittest.main()
