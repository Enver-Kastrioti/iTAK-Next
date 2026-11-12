#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO

try:
    from module.runtime_tools import build_runtime_env, resolve_helper_executable
except ImportError:
    build_runtime_env = None
    resolve_helper_executable = None


SCRIPT_DIR = Path(__file__).resolve().parent.parent
PK_DB_DIR = SCRIPT_DIR / "db" / "itaknext_pk"
PK_IDENTIFY_DOMAINS = {"PF00069", "PF07714"}
PPC_SUBCLASS_RULES = (
    ("PPC:4.1.5", "Pkinase_sub_WNK1.hmm", 30.0, "PPC:4.1.5.1"),
    ("PPC:4.5.1", "Pkinase_sub_MAK.hmm", 460.15, "PPC:4.5.1.1"),
)


@dataclass(frozen=True)
class HmmerTblHit:
    query_name: str
    hit_id: str
    score: float
    evalue: float
    description: str


def _load_ga_cutoffs():
    tfam_hmm = PK_DB_DIR / "Tfam_domain.hmm"
    ga_table = PK_DB_DIR / "GA_table.txt"

    if not tfam_hmm.exists():
        raise FileNotFoundError(f"Protein kinase identification HMM not found: {tfam_hmm}")
    if not ga_table.exists():
        raise FileNotFoundError(f"GA cutoff file not found: {ga_table}")

    ga_cutoff = {}
    pfam_id = ""
    ga_score = None

    with open(tfam_hmm, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("ACC"):
                pfam_id = line.split()[1].split(".")[0]
            elif line.startswith("GA"):
                parts = line.split()
                if len(parts) >= 3:
                    ga_score = float(parts[2].rstrip(";"))
            elif line == "//":
                if pfam_id and ga_score is not None:
                    ga_cutoff[pfam_id] = ga_score
                pfam_id = ""
                ga_score = None

    with open(ga_table, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            pfam_id, score = parts
            ga_cutoff[pfam_id.split(".")[0]] = float(score.rstrip(";"))

    return ga_cutoff


def _load_pk_descriptions():
    pk_desc = PK_DB_DIR / "PK_class_desc.txt"
    if not pk_desc.exists():
        raise FileNotFoundError(f"Protein kinase description file not found: {pk_desc}")

    desc = {}
    with open(pk_desc, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                desc[parts[0]] = parts[1]
    return desc


def _resolve_hmmscan_executable():
    if resolve_helper_executable is not None:
        helper = resolve_helper_executable(SCRIPT_DIR, "hmmer3", "hmmscan")
        if helper is not None:
            return str(helper)

    hmmscan_path = shutil.which("hmmscan")
    if hmmscan_path:
        return hmmscan_path

    raise FileNotFoundError("hmmscan executable not found")


def _run_hmmscan_tbl(hmmscan_bin, hmm_db, fasta_file, output_tbl, cpu):
    cmd = [
        hmmscan_bin,
        "--acc",
        "--notextw",
        "--cpu",
        str(cpu),
        "--tblout",
        str(output_tbl),
        str(hmm_db),
        str(fasta_file),
    ]
    env = build_runtime_env(SCRIPT_DIR) if build_runtime_env else None
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


def _parse_tblout(tbl_path):
    hits = []
    with open(tbl_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            fields = raw_line.strip().split(maxsplit=18)
            if len(fields) < 18:
                continue
            target_name = fields[0]
            target_accession = fields[1]
            query_name = fields[2]
            hit_id = target_accession if target_accession != "-" else target_name
            description = fields[18] if len(fields) > 18 else target_name
            hits.append(
                HmmerTblHit(
                    query_name=query_name,
                    hit_id=hit_id,
                    score=float(fields[5]),
                    evalue=float(fields[4]),
                    description=description,
                )
            )
    return hits


def _identify_protein_kinases(summary_hits, ga_cutoff):
    pkinase_id = {}
    for hit in summary_hits:
        pfam_id = hit.hit_id.split(".")[0]
        if pfam_id in PK_IDENTIFY_DOMAINS and hit.score >= ga_cutoff.get(pfam_id, float("inf")):
            pkinase_id[hit.query_name] = 1
    return pkinase_id


def _classify_best_hit(hits, candidate_ids, other_label):
    candidate_ids = dict(candidate_ids)
    best_hit = {}
    best_score = {}

    for hit in hits:
        if hit.query_name not in candidate_ids:
            continue
        if hit.query_name not in best_hit or hit.score > best_score[hit.query_name]:
            best_hit[hit.query_name] = hit.hit_id
            best_score[hit.query_name] = hit.score

    for seq_id in sorted(candidate_ids.keys()):
        if seq_id not in best_hit:
            best_hit[seq_id] = other_label

    return best_hit


def _write_selected_fasta(source_fasta, selected_ids, output_fasta):
    selected_ids = set(selected_ids)
    count = 0
    with open(output_fasta, "w", encoding="utf-8") as out_handle:
        with open(source_fasta, "r", encoding="utf-8") as in_handle:
            for record in SeqIO.parse(in_handle, "fasta"):
                if record.id in selected_ids:
                    SeqIO.write(record, out_handle, "fasta")
                    count += 1
    return count


def _write_pk_outputs(output_dir, source_fasta, pkinase_id, plantsp_cat, shiu_cat, pk_desc, debug=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stem = Path(source_fasta).stem
    pk_fasta = output_dir / "pk_sequence.fasta"
    classified_pk_fasta = output_dir / f"{source_stem}_pk_classified.fasta"
    match_tbl = output_dir / "match_tbl.txt"
    combined_tsv = output_dir / "pk_classification.tsv"
    shiu_tsv = output_dir / "shiu_classification.txt"
    ppc_tsv = output_dir / "PPC_classification.txt"
    match_json = output_dir / "match.json"

    sequences = {}
    with open(source_fasta, "r", encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences[record.id] = str(record.seq)

    json_result = {}

    with open(pk_fasta, "w", encoding="utf-8") as fasta_handle, \
         open(classified_pk_fasta, "w", encoding="utf-8") as classified_fasta_handle, \
         open(match_tbl, "w", encoding="utf-8") as match_handle, \
         open(combined_tsv, "w", encoding="utf-8") as combined_handle, \
         open(shiu_tsv, "w", encoding="utf-8") as shiu_handle, \
         open(ppc_tsv, "w", encoding="utf-8") as ppc_handle:
        combined_handle.write("Sequence_ID\tShiu_Class\tPPC_Class\tPPC_Description\n")
        for seq_id in sorted(pkinase_id.keys()):
            shiu_class = shiu_cat.get(seq_id, "Group-other")
            ppc_class = plantsp_cat.get(seq_id, "PPC:5.2.1")
            ppc_desc = pk_desc.get(ppc_class, "NA")
            sequence = sequences.get(seq_id, "")
            header = f">{seq_id} | {ppc_class} | PK"
            fasta_handle.write(f"{header}\n{sequence}\n")
            classified_fasta_handle.write(f"{header}\n{sequence}\n")
            match_handle.write(f"{seq_id}\t{ppc_class}\t{ppc_class}\tPK\t{ppc_desc}\tShiu:{shiu_class}\n")
            combined_handle.write(f"{seq_id}\t{shiu_class}\t{ppc_class}\t{ppc_desc}\n")
            shiu_handle.write(f"{seq_id}\t{shiu_class}\n")
            ppc_handle.write(f"{seq_id}\t{ppc_class}\t{ppc_desc}\n")
            json_result[seq_id] = {
                "name": ppc_class,
                "family": ppc_class,
                "type": "PK",
                "desc": [ppc_desc] if ppc_desc != "NA" else [],
                "other_family": f"Shiu:{shiu_class}",
            }

    if debug:
        with open(match_json, "w", encoding="utf-8") as handle:
            json.dump(json_result, handle, indent=2, ensure_ascii=False)


def _ensure_pk_database():
    required = [
        "GA_table.txt",
        "PK_class_desc.txt",
        "Tfam_domain.hmm",
        "Plant_Pkinase_fam.hmm",
        "PlantsPHMM3_89.hmm",
        "Pkinase_sub_WNK1.hmm",
        "Pkinase_sub_MAK.hmm",
    ]
    missing = [name for name in required if not (PK_DB_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing protein kinase database files under db/itaknext_pk: " + ", ".join(missing)
        )


def run_protein_kinase_pipeline(fasta_file, project_output, cpu=None, debug=False):
    _ensure_pk_database()
    hmmscan_bin = _resolve_hmmscan_executable()
    cpu = cpu or min(4, os.cpu_count() or 1)

    project_output = Path(project_output)
    pk_output_dir = project_output / "protein_kinase"
    temp_dir = pk_output_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ga_cutoff = _load_ga_cutoffs()
    pk_desc = _load_pk_descriptions()

    primary_tbl = temp_dir / "tfam_domain.tbl"
    _run_hmmscan_tbl(hmmscan_bin, PK_DB_DIR / "Tfam_domain.hmm", fasta_file, primary_tbl, cpu)
    primary_hits = _parse_tblout(primary_tbl)
    pkinase_id = _identify_protein_kinases(primary_hits, ga_cutoff)

    if not pkinase_id:
        pk_output_dir.mkdir(parents=True, exist_ok=True)
        source_stem = Path(fasta_file).stem
        for file_name in (
            "pk_sequence.fasta",
            f"{source_stem}_pk_classified.fasta",
            "match_tbl.txt",
            "shiu_classification.txt",
            "PPC_classification.txt",
        ):
            (pk_output_dir / file_name).write_text("", encoding="utf-8")
        (pk_output_dir / "pk_classification.tsv").write_text(
            "Sequence_ID\tShiu_Class\tPPC_Class\tPPC_Description\n",
            encoding="utf-8",
        )
        if debug:
            (pk_output_dir / "match.json").write_text("{}\n", encoding="utf-8")
        return {
            "success": True,
            "count": 0,
            "output_dir": str(pk_output_dir),
        }

    pk_fasta = temp_dir / "pkinase_seq.fa"
    _write_selected_fasta(fasta_file, pkinase_id.keys(), pk_fasta)

    shiu_tbl = temp_dir / "plant_pkinase_fam.tbl"
    _run_hmmscan_tbl(hmmscan_bin, PK_DB_DIR / "Plant_Pkinase_fam.hmm", pk_fasta, shiu_tbl, cpu)
    shiu_hits = _parse_tblout(shiu_tbl)
    shiu_cat = _classify_best_hit(shiu_hits, pkinase_id, "Group-other")

    plantsp_tbl = temp_dir / "plantsphmm3.tbl"
    _run_hmmscan_tbl(hmmscan_bin, PK_DB_DIR / "PlantsPHMM3_89.hmm", pk_fasta, plantsp_tbl, cpu)
    plantsp_hits = _parse_tblout(plantsp_tbl)
    plantsp_cat = _classify_best_hit(plantsp_hits, pkinase_id, "PPC:5.2.1")

    for base_cat, hmm_name, cutoff, refined_cat in PPC_SUBCLASS_RULES:
        subset_ids = [seq_id for seq_id, cat in plantsp_cat.items() if cat == base_cat]
        if not subset_ids:
            continue
        subset_fasta = temp_dir / f"{base_cat.replace(':', '_')}.fa"
        _write_selected_fasta(fasta_file, subset_ids, subset_fasta)
        subset_tbl = temp_dir / f"{Path(hmm_name).stem}.tbl"
        _run_hmmscan_tbl(hmmscan_bin, PK_DB_DIR / hmm_name, subset_fasta, subset_tbl, cpu)
        for hit in _parse_tblout(subset_tbl):
            if hit.score >= cutoff:
                plantsp_cat[hit.query_name] = refined_cat

    _write_pk_outputs(pk_output_dir, fasta_file, pkinase_id, plantsp_cat, shiu_cat, pk_desc, debug=debug)

    return {
        "success": True,
        "count": len(pkinase_id),
        "output_dir": str(pk_output_dir),
    }
