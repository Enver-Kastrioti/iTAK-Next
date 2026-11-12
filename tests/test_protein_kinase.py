#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the protein kinase (PK) analysis module.

These tests validate the core parsing, identification, and classification
functions used by the PK pipeline without requiring a full hmmscan run.

Run with:
    cd itak3.7 && python -m pytest tests/test_protein_kinase.py -v

Or directly:
    cd itak3.7 && python tests/test_protein_kinase.py
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure the project root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from module.protein_kinase import (
    HmmerTblHit,
    _load_ga_cutoffs,
    _load_pk_descriptions,
    _parse_tblout,
    _identify_protein_kinases,
    _classify_best_hit,
)


class TestHmmerTblHit(unittest.TestCase):
    """Test the HmmerTblHit dataclass."""

    def test_create_hit(self):
        hit = HmmerTblHit(
            query_name="seq1",
            hit_id="PF00069",
            score=150.5,
            evalue=1e-30,
            description="Protein kinase domain",
        )
        self.assertEqual(hit.query_name, "seq1")
        self.assertEqual(hit.hit_id, "PF00069")
        self.assertEqual(hit.score, 150.5)
        self.assertEqual(hit.evalue, 1e-30)


class TestParseTblout(unittest.TestCase):
    """Test parsing of hmmscan --tblout output."""

    def setUp(self):
        # Format matches hmmscan --tblout output (19 column format)
        self.tbl_content = (
            "#                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----\n"
            "# target name        accession  query name           accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target\n"
            "#------------------- ---------- -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------\n"
            "Pkinase              PF00069.29 AT1G01540.1          -            1.4e-47  155.2   0.0   1.4e-47  155.2   0.0   1.0   1   0   0   1   1   1   1 Protein kinase domain\n"
            "PK_Tyr_Ser-Thr       PF07714.21 AT1G01540.1          -            3.3e-46  150.5   0.0   3.3e-46  150.5   0.0   1.0   1   0   0   1   1   1   1 Protein tyrosine kinase\n"
            "Pkinase              PF00069.29 AT1G01450.1          -            6.8e-31  100.5   1.2   6.8e-31  100.5   1.2   1.0   1   0   0   1   1   1   1 Protein kinase domain\n"
            "PK_Tyr_Ser-Thr       PF07714.21 AT1G01450.1          -            6.0e-40  130.0   0.0   6.0e-40  130.0   0.0   1.0   1   0   0   1   1   1   1 Protein tyrosine kinase\n"
            "#\n"
            "# [ok]\n"
        )

    def test_parse_valid_tblout(self):
        tbl_path = TEST_DIR / "fixtures" / "pk_test.tbl"
        tbl_path.parent.mkdir(parents=True, exist_ok=True)
        tbl_path.write_text(self.tbl_content)

        hits = _parse_tblout(str(tbl_path))
        self.assertEqual(len(hits), 4)

        # Check first hit
        self.assertEqual(hits[0].query_name, "AT1G01540.1")
        # hit_id is the ACC field (PF00069.29) since it's not "-"
        self.assertEqual(hits[0].hit_id, "PF00069.29")
        self.assertAlmostEqual(hits[0].score, 155.2)
        self.assertAlmostEqual(hits[0].evalue, 1.4e-47)

        # Check PK_Tyr hit
        self.assertEqual(hits[1].hit_id, "PF07714.21")

        tbl_path.unlink(missing_ok=True)

    def test_parse_empty_tblout(self):
        """Empty tblout should return empty list."""
        tbl_path = TEST_DIR / "fixtures" / "pk_empty.tbl"
        tbl_path.parent.mkdir(parents=True, exist_ok=True)
        tbl_path.write_text("# hmmscan :: search\n# [ok]\n")

        hits = _parse_tblout(str(tbl_path))
        self.assertEqual(len(hits), 0)

        tbl_path.unlink(missing_ok=True)


class TestIdentifyPK(unittest.TestCase):
    """Test protein kinase identification logic."""

    def setUp(self):
        # GA cutoffs from Tfam_domain.hmm for PF00069 and PF07714
        self.ga_cutoff = {
            "PF00069": 20.7,
            "PF07714": 20.7,
        }
        # In _identify_protein_kinases, hit_id is split on "." so "PF00069.29" becomes "PF00069"
        # But the hit.hit_id field from _parse_tblout retains the version suffix.
        # _identify_protein_kinases calls hit.hit_id.split(".")[0], so these tests work correctly.
        self.hits = [
            HmmerTblHit("seq1", "PF00069.29", 155.2, 1e-47, "Pkinase"),
            HmmerTblHit("seq1", "PF07714.21", 150.5, 3e-46, "PK_Tyr"),
            HmmerTblHit("seq2", "PF00069.29", 100.5, 7e-31, "Pkinase"),
            HmmerTblHit("seq2", "PF07714.21", 130.0, 6e-40, "PK_Tyr"),
        ]

    def test_identify_pk_hits(self):
        pks = _identify_protein_kinases(self.hits, self.ga_cutoff)
        self.assertIn("seq1", pks)
        self.assertIn("seq2", pks)
        self.assertEqual(len(pks), 2)

    def test_below_ga_cutoff_not_identified(self):
        hits = [
            HmmerTblHit("seq3", "PF00069", 5.0, 1.0, "weak hit"),
        ]
        pks = _identify_protein_kinases(hits, self.ga_cutoff)
        self.assertNotIn("seq3", pks)

    def test_non_pk_domain_not_identified(self):
        hits = [
            HmmerTblHit("seq1", "PF00583", 200.0, 1e-50, "Acetyltransf"),
        ]
        pks = _identify_protein_kinases(hits, self.ga_cutoff)
        self.assertNotIn("seq1", pks)


class TestClassifyBestHit(unittest.TestCase):
    """Test the best-hit classification logic."""

    def setUp(self):
        self.candidates = {"seq1": 1, "seq2": 1, "seq3": 1}

    def test_best_hit_assignment(self):
        hits = [
            HmmerTblHit("seq1", "Kinase_A", 200.0, 0.0, ""),
            HmmerTblHit("seq1", "Kinase_B", 150.0, 0.0, ""),
            HmmerTblHit("seq2", "Kinase_C", 180.0, 0.0, ""),
        ]
        result = _classify_best_hit(hits, self.candidates, "Other")
        self.assertEqual(result["seq1"], "Kinase_A")
        self.assertEqual(result["seq2"], "Kinase_C")
        self.assertEqual(result["seq3"], "Other")  # no hits → other label

    def test_empty_hits(self):
        result = _classify_best_hit([], self.candidates, "Other")
        self.assertEqual(result["seq1"], "Other")
        self.assertEqual(len(result), 3)


class TestLoadGACutoffs(unittest.TestCase):
    """Test loading of GA cutoffs (requires actual files only if they exist)."""

    def test_ga_cutoff_file_exists(self):
        """The GA_table.txt and Tfam_domain.hmm should be present."""
        ga_table = PROJECT_DIR / "db" / "winnowtak_pk" / "GA_table.txt"
        tfam_hmm = PROJECT_DIR / "db" / "winnowtak_pk" / "Tfam_domain.hmm"
        self.assertTrue(ga_table.exists(), f"Missing: {ga_table}")
        self.assertTrue(tfam_hmm.exists(), f"Missing: {tfam_hmm}")


class TestLoadDescriptions(unittest.TestCase):
    """Test loading PK class descriptions."""

    def test_desc_file_exists(self):
        desc_file = PROJECT_DIR / "db" / "winnowtak_pk" / "PK_class_desc.txt"
        self.assertTrue(desc_file.exists(), f"Missing: {desc_file}")

    def test_desc_has_entries(self):
        desc = _load_pk_descriptions()
        self.assertGreater(len(desc), 0)
        # Should have well-known categories
        self.assertIn("PPC:4.1.5.1", desc)  # WNK
        self.assertIn("PPC:4.5.1.1", desc)  # MAK


if __name__ == "__main__":
    unittest.main()
