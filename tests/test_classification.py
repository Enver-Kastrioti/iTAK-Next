#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the classification module (classification.py).

Run with:
    cd itak3.7 && python -m pytest tests/test_classification.py -v
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from module.classification import (
    get_ipr_counts,
    get_ipr_max_scores,
    evaluate_logic,
    check_rule_match,
)


class TestGetIPRCounts(unittest.TestCase):
    """Test IPR copy number extraction."""

    def setUp(self):
        self.gene_data = [
            {"sequence": "ACDEFGHIK"},
            {
                "IPR001005": [
                    {"accession": "PF00001", "library": "Pfam",
                     "start": 1, "end": 50, "score": 10.0},
                ],
                "IPR001006": [
                    {"accession": "PF00002", "library": "Pfam",
                     "start": 10, "end": 30, "score": 5.0},
                    {"accession": "PF00002", "library": "Pfam",
                     "start": 40, "end": 60, "score": 8.0},
                ],
            },
        ]

    def test_counts_extracted(self):
        counts = get_ipr_counts(self.gene_data)
        self.assertIn("IPR001005", counts)
        self.assertIn("IPR001006", counts)
        self.assertEqual(counts["IPR001005"], 1)
        self.assertEqual(counts["IPR001006"], 2)

    def test_empty_data(self):
        counts = get_ipr_counts([{}, {}])
        self.assertEqual(counts, {})

    def test_data_with_count_suffix(self):
        data = [
            {"sequence": "A"},
            {
                "IPR001005&2": [
                    {"accession": "PF00001", "library": "Pfam",
                     "start": 1, "end": 50, "score": 10.0},
                ],
            },
        ]
        counts = get_ipr_counts(data)
        # &2 should be stripped; count is under IPR001005
        self.assertIn("IPR001005", counts)


class TestGetIPRMaxScores(unittest.TestCase):
    """Test IPR maximum score extraction."""

    def setUp(self):
        self.gene_data = [
            {"sequence": "A"},
            {
                "IPR001005": [
                    {"accession": "PF00001", "library": "Pfam",
                     "start": 1, "end": 50, "score": 10.0},
                ],
                "IPR001006": [
                    {"accession": "PF00002", "library": "Pfam",
                     "start": 10, "end": 30, "score": 5.0},
                    {"accession": "PF00002", "library": "Pfam",
                     "start": 40, "end": 60, "score": 20.0},
                ],
            },
        ]

    def test_max_score(self):
        scores = get_ipr_max_scores(self.gene_data)
        self.assertAlmostEqual(scores["IPR001005"], 10.0)
        self.assertAlmostEqual(scores["IPR001006"], 20.0)


class TestEvaluateLogic(unittest.TestCase):
    """Test recursive rule logic evaluation."""

    def setUp(self):
        self.ipr_counts = {"IPR001005": 1, "IPR001006": 2, "IPR001007": 0}

    def test_single_domain_present(self):
        self.assertTrue(evaluate_logic("IPR001005", self.ipr_counts))

    def test_single_domain_absent(self):
        self.assertFalse(evaluate_logic("IPR001007", self.ipr_counts))

    def test_and_all_present(self):
        node = {"op": "AND", "children": ["IPR001005", "IPR001006"]}
        self.assertTrue(evaluate_logic(node, self.ipr_counts))

    def test_and_one_absent(self):
        node = {"op": "AND", "children": ["IPR001005", "IPR001007"]}
        self.assertFalse(evaluate_logic(node, self.ipr_counts))

    def test_or_one_present(self):
        node = {"op": "OR", "children": ["IPR001005", "IPR001007"]}
        self.assertTrue(evaluate_logic(node, self.ipr_counts))

    def test_or_all_absent(self):
        node = {"op": "OR", "children": ["IPR001007", "IPR001008"]}
        self.assertFalse(evaluate_logic(node, self.ipr_counts))

    def test_count_requirement_met(self):
        self.assertTrue(evaluate_logic("IPR001006&2", self.ipr_counts))

    def test_count_requirement_not_met(self):
        self.assertFalse(evaluate_logic("IPR001006&3", self.ipr_counts))

    def test_none_node(self):
        self.assertTrue(evaluate_logic(None, self.ipr_counts))

    def test_deep_nesting(self):
        """AND(OR(A, B), C)"""
        or_node = {"op": "OR", "children": ["IPR001007", "IPR001006"]}
        and_node = {"op": "AND", "children": [or_node, "IPR001005"]}
        self.assertTrue(evaluate_logic(and_node, self.ipr_counts))


class TestCheckRuleMatch(unittest.TestCase):
    """Test full rule matching."""

    def setUp(self):
        self.ipr_counts = {"IPR001005": 1, "IPR001006": 2}
        self.ipr_scores = {"IPR001005": 10.0, "IPR001006": 20.0}

    def test_rule_with_logic_matches(self):
        rule = {
            "name": "TestFamily",
            "type": "TF",
            "forbidden": [],
            "logic": {"op": "AND", "children": ["IPR001005", "IPR001006"]},
        }
        self.assertTrue(check_rule_match(self.ipr_counts, rule, self.ipr_scores))

    def test_rule_with_forbidden(self):
        """Forbidden domain present with high score → no match."""
        ipr_counts = {"IPR001005": 1, "IPR_FORBIDDEN": 1}
        ipr_scores = {"IPR001005": 10.0, "IPR_FORBIDDEN": 15.0}
        rule = {
            "name": "TestFam",
            "type": "TF",
            "forbidden": ["IPR_FORBIDDEN"],
            "logic": "IPR001005",
        }
        self.assertFalse(check_rule_match(ipr_counts, rule, ipr_scores, forbidden_score_threshold=10.0))

    def test_rule_with_forbidden_low_score(self):
        """Forbidden domain present but low score → still match."""
        ipr_counts = {"IPR001005": 1, "IPR_FORBIDDEN": 1}
        ipr_scores = {"IPR001005": 10.0, "IPR_FORBIDDEN": 5.0}
        rule = {
            "name": "TestFam",
            "type": "TF",
            "forbidden": ["IPR_FORBIDDEN"],
            "logic": "IPR001005",
        }
        self.assertTrue(check_rule_match(ipr_counts, rule, self.ipr_scores, forbidden_score_threshold=10.0))

    def test_empty_rule(self):
        """Rule with no logic → match by default."""
        rule = {
            "name": "TestFam",
            "type": "TF",
            "forbidden": [],
            "logic": None,
        }
        self.assertTrue(check_rule_match(self.ipr_counts, rule, self.ipr_scores))


if __name__ == "__main__":
    unittest.main()
