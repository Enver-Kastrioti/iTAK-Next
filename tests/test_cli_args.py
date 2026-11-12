#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the iTAK Next CLI definition and validation.

Run with:
    python -m pytest tests/test_cli_args.py -v
"""

import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import itaknext  # noqa: E402  (imported after sys.path setup)


def make_fasta(tmp_path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">seq1\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n", encoding="utf-8")
    return fasta


def make_json(tmp_path):
    json_file = tmp_path / "ipr.json"
    json_file.write_text("{}", encoding="utf-8")
    return json_file


def make_tbl(tmp_path):
    tbl = tmp_path / "result.tbl"
    tbl.write_text("# hmmscan tblout\n", encoding="utf-8")
    return tbl


def parse(argv):
    return itaknext.build_parser().parse_args(argv)


def parse_and_validate(argv):
    parser = itaknext.build_parser()
    args = parser.parse_args(argv)
    itaknext.validate_cli_args(parser, args)
    return args


# ---------------------------------------------------------------------------
# Parser definition / two-tier help
# ---------------------------------------------------------------------------

class TestParserDefinition:
    def test_defaults(self):
        args = parse(["-i", "input.fasta"])
        assert args.threshold == 0.1
        assert args.score == 1.0
        assert args.score_interpro is None
        assert args.score_hmm is None
        assert args.score_thresholds is None
        assert args.cpu == 4
        assert args.interproscan_cpu is None
        assert args.classification_mode == "score"
        assert args.predict_mode == "fast"
        assert args.appl == itaknext.DEFAULT_APPL
        assert args.grad_cam_mode == "none"
        assert args.help_all is False

    def test_default_help_is_compact(self):
        help_text = itaknext.build_parser().format_help()
        for common in ("-i INPUT", "--predict", "--appl", "--score",
                       "--cpu", "--grad-cam-mode", "--check-deps", "--help-all"):
            assert common in help_text
        for advanced in ("--test-mode", "--json-file", "--list-predict",
                         "--classification-mode", "--predict-mode",
                         "--score-interpro", "--score-hmm", "--score-thresholds",
                         "--interproscan-cpu", "--use-supplementary",
                         "--supplementary-only", "--supp-models",
                         "--debug", "--skip-deps-check",
                         "--run-pk", "--skip-pk"):
            assert f"\n  {advanced}" not in help_text

    def test_help_all_lists_advanced_options(self):
        help_text = itaknext.build_parser(show_advanced=True).format_help()
        assert "Advanced / developer options" in help_text
        for advanced in ("--test-mode", "--json-file", "--list-predict",
                         "--classification-mode", "--predict-mode",
                         "--score-interpro", "--score-hmm", "--score-thresholds",
                         "--interproscan-cpu", "--use-supplementary",
                         "--supplementary-only", "--supp-models",
                         "--debug", "--skip-deps-check"):
            assert f"\n  {advanced}" in help_text

    def test_advanced_options_still_parse(self):
        args = parse([
            "--test-mode", "--json-file", "x.json", "--spechmm-file", "x.tbl",
            "--classification-mode", "specific", "--predict-mode", "full",
            "--debug", "--skip-deps-check",
        ])
        assert args.test_mode is True
        assert args.json_file == "x.json"
        assert args.spechmm_file == "x.tbl"
        assert args.classification_mode == "specific"
        assert args.predict_mode == "full"
        assert args.debug is True
        assert args.skip_deps_check is True

    def test_argument_abbreviation_disabled(self):
        with pytest.raises(SystemExit):
            parse(["-i", "input.fasta", "--predic"])

    def test_legacy_alias_normalization(self):
        argv, used = itaknext.normalize_legacy_args(
            ["-test", "-i", "input.fasta", "-json", "ipr.json", "-spechmm", "x.tbl"]
        )
        assert argv == [
            "--test-mode", "-i", "input.fasta",
            "--json-file", "ipr.json", "--spechmm-file", "x.tbl",
        ]
        assert used == ["-test", "-json", "-spechmm"]

        args = parse(argv)
        assert args.test_mode is True
        assert args.json_file == "ipr.json"
        assert args.spechmm_file == "x.tbl"

    def test_normalize_leaves_long_options_untouched(self):
        argv, used = itaknext.normalize_legacy_args(["--test-mode", "-i", "x"])
        assert argv == ["--test-mode", "-i", "x"]
        assert used == []


# ---------------------------------------------------------------------------
# argparse type validation
# ---------------------------------------------------------------------------

class TestNumericValidation:
    @pytest.mark.parametrize("value", ["0", "-1", "abc"])
    def test_cpu_must_be_positive_int(self, value):
        with pytest.raises(SystemExit):
            parse(["-i", "input.fasta", "--cpu", value])

    @pytest.mark.parametrize("value", ["0", "-2"])
    def test_interproscan_cpu_must_be_positive_int(self, value):
        with pytest.raises(SystemExit):
            parse(["-i", "input.fasta", "--interproscan-cpu", value])

    @pytest.mark.parametrize("option", ["--score", "--score-interpro", "--score-hmm"])
    @pytest.mark.parametrize("value", ["-0.1", "-1"])
    def test_scores_must_be_non_negative(self, option, value):
        with pytest.raises(SystemExit):
            parse(["-i", "input.fasta", option, value])

    def test_scores_accept_zero(self):
        args = parse(["-i", "input.fasta", "--score", "0", "--score-hmm", "0"])
        assert args.score == 0.0
        assert args.score_hmm == 0.0


# ---------------------------------------------------------------------------
# Cross-argument validation (validate_cli_args)
# ---------------------------------------------------------------------------

class TestValidateCliArgs:
    def test_direct_mode_is_valid(self, tmp_path):
        fasta = make_fasta(tmp_path)
        args = parse_and_validate(["-i", str(fasta)])
        assert args.test_mode is False

    def test_test_mode_is_valid(self, tmp_path):
        fasta = make_fasta(tmp_path)
        json_file = make_json(tmp_path)
        tbl = make_tbl(tmp_path)
        args = parse_and_validate([
            "--test-mode", "-i", str(fasta),
            "--json-file", str(json_file), "--spechmm-file", str(tbl),
        ])
        assert args.test_mode is True

    def test_input_is_required(self):
        with pytest.raises(SystemExit):
            parse_and_validate([])

    def test_input_must_exist(self, tmp_path):
        with pytest.raises(SystemExit):
            parse_and_validate(["-i", str(tmp_path / "missing.fasta")])

    def test_predict_and_list_predict_conflict(self, tmp_path):
        fasta = make_fasta(tmp_path)
        with pytest.raises(SystemExit):
            parse_and_validate(["-i", str(fasta), "--predict", "--list-predict"])

    def test_test_mode_rejects_predict(self, tmp_path):
        fasta = make_fasta(tmp_path)
        json_file = make_json(tmp_path)
        with pytest.raises(SystemExit):
            parse_and_validate([
                "--predict", "--test-mode", "-i", str(fasta), "--json-file", str(json_file),
            ])

    def test_test_mode_requires_json_file(self, tmp_path):
        fasta = make_fasta(tmp_path)
        with pytest.raises(SystemExit):
            parse_and_validate(["--test-mode", "-i", str(fasta)])

    def test_test_mode_json_must_exist(self, tmp_path):
        fasta = make_fasta(tmp_path)
        with pytest.raises(SystemExit):
            parse_and_validate([
                "--test-mode", "-i", str(fasta), "--json-file", str(tmp_path / "missing.json"),
            ])

    def test_json_outside_test_mode_is_rejected(self, tmp_path):
        fasta = make_fasta(tmp_path)
        json_file = make_json(tmp_path)
        with pytest.raises(SystemExit):
            parse_and_validate(["-i", str(fasta), "--json-file", str(json_file)])

    def test_gradcam_all_requires_predict(self, tmp_path):
        fasta = make_fasta(tmp_path)
        with pytest.raises(SystemExit):
            parse_and_validate(["-i", str(fasta), "--grad-cam-mode", "all"])

    def test_list_predict_rejects_gradcam(self, tmp_path):
        fasta = make_fasta(tmp_path)
        with pytest.raises(SystemExit):
            parse_and_validate(["-i", str(fasta), "--list-predict", "--grad-cam-mode", "fast"])

    def test_score_thresholds_none_disables_per_model_thresholds(self, tmp_path):
        fasta = make_fasta(tmp_path)
        args = parse_and_validate(["-i", str(fasta), "--score-thresholds", "none"])
        assert args.score_thresholds is None

        args = parse_and_validate(["-i", str(fasta), "--score-thresholds", "NONE"])
        assert args.score_thresholds is None

    def test_score_thresholds_must_exist(self, tmp_path):
        fasta = make_fasta(tmp_path)
        with pytest.raises(SystemExit):
            parse_and_validate(["-i", str(fasta), "--score-thresholds", str(tmp_path / "missing.json")])

    def test_deprecated_pk_switches_warn(self, tmp_path, capsys):
        fasta = make_fasta(tmp_path)
        for flag in ("--run-pk", "--skip-pk"):
            capsys.readouterr()
            args = parse_and_validate(["-i", str(fasta), flag])
            captured = capsys.readouterr()
            assert "--run-pk/--skip-pk are deprecated" in captured.err
            assert getattr(args, flag.lstrip('-').replace('-', '_')) is True

    def test_supplementary_flags_warn_without_predict(self, tmp_path, capsys):
        fasta = make_fasta(tmp_path)
        args = parse_and_validate(["-i", str(fasta), "--use-supplementary"])
        assert args.use_supplementary is True
        captured = capsys.readouterr()
        assert "ignored without --predict" in captured.err

    def test_supplementary_flags_do_not_warn_with_predict(self, tmp_path, capsys):
        fasta = make_fasta(tmp_path)
        args = parse_and_validate(["-i", str(fasta), "--predict", "--use-supplementary"])
        assert args.predict is True
        captured = capsys.readouterr()
        assert "ignored without --predict" not in captured.err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
