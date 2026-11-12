#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the rule-parsing module (get_rule.py).

Run with:
    cd itak3.7 && python -m pytest tests/test_get_rule.py -v
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

from module.get_rule import (
    Parser,
    parse_required_logic,
    parse_rule_file,
    parse_score_thresholds,
)


class TestParser(unittest.TestCase):
    """Test the rule logic tokenizer/parser."""

    def test_single_domain(self):
        """A single domain name parses to a leaf token."""
        tree = parse_required_logic("IPR001005")
        self.assertEqual(tree, "IPR001005")

    def test_and_expression(self):
        """A#B parses as AND(A, B)."""
        tree = parse_required_logic("IPR001005#IPR001006")
        self.assertIsInstance(tree, dict)
        self.assertEqual(tree["op"], "AND")
        self.assertEqual(tree["children"], ["IPR001005", "IPR001006"])

    def test_or_expression(self):
        """A:B parses as OR(A, B)."""
        tree = parse_required_logic("IPR001005:IPR001006")
        self.assertIsInstance(tree, dict)
        self.assertEqual(tree["op"], "OR")
        self.assertEqual(tree["children"], ["IPR001005", "IPR001006"])

    def test_nested_expression(self):
        """(A:B)#C parses as AND(OR(A, B), C)."""
        tree = parse_required_logic("(IPR001005:IPR001006)#IPR001007")
        self.assertIsInstance(tree, dict)
        self.assertEqual(tree["op"], "AND")
        # First child should be OR(A, B)
        or_child = tree["children"][0]
        self.assertEqual(or_child["op"], "OR")
        self.assertEqual(tree["children"][1], "IPR001007")

    def test_empty_string(self):
        """Empty or 'NA' returns None."""
        self.assertIsNone(parse_required_logic(""))
        self.assertIsNone(parse_required_logic("NA"))

    def test_count_suffix(self):
        """Domain with &N suffix is kept as-is in the leaf token."""
        tree = parse_required_logic("IPR001471&2")
        self.assertEqual(tree, "IPR001471&2")

    def test_complex_expression(self):
        """A#(B:C)#D&2 parses without error."""
        tree = parse_required_logic("IPR001#(IPR002:IPR003)#IPR004&2")
        self.assertIsNotNone(tree)

    def test_multi_and_flattening(self):
        """A#B#C flattens to AND(A, B, C)."""
        tree = parse_required_logic("A#B#C")
        self.assertEqual(tree["op"], "AND")
        self.assertEqual(len(tree["children"]), 3)

    def test_multi_or_flattening(self):
        """A:B:C flattens to OR(A, B, C)."""
        tree = parse_required_logic("A:B:C")
        self.assertEqual(tree["op"], "OR")
        self.assertEqual(len(tree["children"]), 3)


class TestParseRuleFile(unittest.TestCase):
    """Test parsing of the full rule file."""

    def setUp(self):
        self.rule_file = str(PROJECT_DIR / "rule.txt")

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.rule_file))

    def test_parses_without_error(self):
        rules = parse_rule_file(self.rule_file)
        self.assertIsInstance(rules, dict)
        self.assertGreater(len(rules), 0)

    def test_required_fields_present(self):
        rules = parse_rule_file(self.rule_file)
        for rid, rdata in rules.items():
            self.assertIn("id", rdata, f"Rule {rid} missing id")
            self.assertIn("name", rdata, f"Rule {rid} missing name")
            self.assertIn("family", rdata, f"Rule {rid} missing family")
            self.assertIn("type", rdata, f"Rule {rid} missing type")
            self.assertIn("logic", rdata, f"Rule {rid} missing logic")

    def test_all_types_present(self):
        rules = parse_rule_file(self.rule_file)
        types = set(r["type"] for r in rules.values())
        self.assertIn("TF", types)
        self.assertIn("TR", types)
        self.assertIn("GTF", types)

    def test_forbidden_parsed(self):
        """Some rules should have forbidden domains."""
        rules = parse_rule_file(self.rule_file)
        has_forbidden = any(
            len(r.get("forbidden", [])) > 0 and r["forbidden"] != ["NA"]
            for r in rules.values()
        )
        self.assertTrue(has_forbidden, "No rules with forbidden domains found")


class TestParseScoreThresholds(unittest.TestCase):
    """Test parsing of Score: lines in the rule file."""

    def setUp(self):
        self.rule_file = str(PROJECT_DIR / "rule.txt")

    def test_parses_without_error(self):
        thresholds = parse_score_thresholds(self.rule_file)
        self.assertIsInstance(thresholds, dict)

    def test_thresholds_are_positive(self):
        thresholds = parse_score_thresholds(self.rule_file)
        for acc, val in thresholds.items():
            self.assertGreater(val, 0, f"Threshold for {acc} is not positive")


class TestParserEdgeCases(unittest.TestCase):
    """Test edge cases in the Parser class."""

    def test_parser_empty_tokens(self):
        p = Parser([])
        self.assertIsNone(p.parse())

    def test_parser_none_peek(self):
        p = Parser([])
        self.assertIsNone(p.peek())

    def test_parser_consume_empty(self):
        p = Parser([])
        self.assertIsNone(p.consume())

    def test_unmatched_paren(self):
        """Unmatched '(' should still produce some result, not crash."""
        tree = parse_required_logic("(IPR001005")
        # Should either return None or the partial expression
        self.assertIsNotNone(tree)

    def test_only_parens(self):
        tree = parse_required_logic("()")
        # Should not crash
        self.assertIsNone(tree)


if __name__ == "__main__":
    unittest.main()
