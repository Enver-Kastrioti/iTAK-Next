---
# iTAK Next - Transcription Factor and Protein Kinase Analysis
---

Overview
--------
iTAK Next is a bioinformatics tool for transcription factor (TF) prediction and analysis. It integrates
deep-learning-based preclassification with conventional sequence/domain analyses to identify and
classify TFs from protein sequences. The current workflow also includes protein kinase (PK)
identification and classification with bundled HMM profiles.

Key Features
--------
1. TF prediction: use a deep learning model to predict whether protein sequences are potential TF/TR sequences
2. Domain analysis: run InterProScan and hmmscan for domain annotation
3. TF family classification: assign TF families based on rule-based logic
4. Protein kinase analysis: identify kinase domains and classify PKs with bundled Shiu/PPC models
5. Output reporting: generate detailed reports and classified sequence FASTA files

System Requirements
--------
- Python 3.7+
- Java 11+ (required by InterProScan)
- HMMER 3.0+ (hmmscan)

Dependencies
------
Required Python packages:
- Biopython (Bio)
- pandas
- numpy
- Standard library: json, csv, argparse, subprocess, pathlib, os, sys, time, datetime, shutil

Optional Python packages:
- PyTorch (required only for prediction)

External tools:
- hmmscan (HMMER)
- java (required by InterProScan)

Project Layout
--------
## Key Scripts
- `winnowtak.py`: main CLI entry point (analysis / prediction / Grad-CAM orchestration)
- `pre_model/predict.py`: deep-learning preclassification (+ Grad-CAM heatmaps)
- `module/`: pipeline modules (FASTA processing, IPR JSON parsing, hmmscan processing, classification, protein kinase, checks)
- `rule.txt`: TF/TR/GTF family classification rules (124 family entries, 187 IPR domains)

```text
itak3.7/
├── winnowtak.py
├── module/
│   ├── check_dependencies.py
│   ├── validate_fasta.py
│   ├── classification.py
│   ├── protein_kinase.py
│   ├── db_manager.py
│   ├── runtime_tools.py
│   ├── get_fasta.py
│   ├── get_json.py
│   ├── get_rule.py
│   └── selfbuild_hmm.py
├── pre_model/
│   ├── model.pth
│   ├── predict.py
│   └── supplementary_model/
├── db/
│   ├── interproscan/
│   └── winnowtak_pk/
├── hmm/
│   └── self_build.hmm
├── rule.txt
├── output/
├── temp/
├── test_protein.fasta
└── test_protein_kinase.fasta
```

Installation
--------
## Requirements
- Python 3.7+
- Java 11+ (InterProScan)
- HMMER 3.0+ (`hmmscan`)
- PyTorch (only if you use `--predict` / `--list-predict` / `--grad-cam-mode`)

## Recommended installer
```bash
./install_runtime.sh
```

Optional:
```bash
# inspect current local runtime status
./install_runtime.sh --status

# remove generated runtime files
./install_runtime.sh --clean-runtime

# also install PyTorch CPU packages
./install_runtime.sh --with-predict

# Apple Silicon / Metal
./install_runtime.sh --with-predict --predict-backend mps

# CUDA example
./install_runtime.sh --with-predict --predict-backend cuda --torch-index-url https://download.pytorch.org/whl/cu121

# run a post-install smoke test
./install_runtime.sh --smoke-test

# run the repository smoke test directly
./smoke_test.sh

# include the prediction workflow in the smoke test
./smoke_test.sh --predict

# run a positive PK smoke test
./smoke_test.sh --input test_protein_kinase.fasta --require-pk 2

# run the prediction workflow with a positive PK sample
./smoke_test.sh --predict --input test_protein_kinase.fasta --require-pk 2

# install into the current Python without creating .venv
./install_runtime.sh --no-venv
```

This installer will:
- create or reuse a Python runtime
- install required Python packages
- generate `db/interproscan/interproscan.local.properties`
- generate `run_interproscan_local.sh`
- generate `run_winnowtak_local.sh`
- configure the bundled InterProScan to use the platform helper binaries under `bin/`
- run `python winnowtak.py --check-deps`
- optionally run a post-install smoke test through both InterProScan and the main iTAK Next workflow

Generated runtime files can be removed with:
```bash
./install_runtime.sh --clean-runtime
./install_runtime.sh --clean-runtime --remove-venv
```

You can inspect the current local runtime state with:
```bash
./install_runtime.sh --status
```

## Python Packages
```bash
pip install biopython pandas numpy
```

Optional (prediction / Grad-CAM):
```bash
# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon / Metal
pip install torch torchvision torchaudio

# GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## InterProScan Database (db/)
- If you use the bundled InterProScan, make sure `db/` exists or `db.tar.gz` is available in the project root.
- If you use an external InterProScan via `--interproscan /path/to/interproscan.sh`, you can skip the bundled `db/`.

## Protein kinase database (`db/winnowtak_pk`)
- iTAK Next bundles a reduced protein kinase HMM database under `db/winnowtak_pk/`.
- This PK workflow uses `hmmscan` only; it does not depend on InterProScan.
- Required profiles include `Tfam_domain.hmm`, `Plant_Pkinase_fam.hmm`, `PlantsPHMM3_89.hmm`,
  `Pkinase_sub_WNK1.hmm`, `Pkinase_sub_MAK.hmm`, plus `GA_table.txt` and `PK_class_desc.txt`.

## Deploy / Install from GitLab
```bash
git clone git@git.moilab.net:moilab/projects/WinnowTAK.git "iTAK Next"
cd "iTAK Next/itak3.7"
```
> 项目已更名为 **iTAK Next**；克隆时显式指定目标目录名，使本地目录与项目名一致。
> （GitLab 仓库随后更名，届时可直接 `git clone .../iTAK-Next.git`。）

Usage
--------

## Syntax
```bash
python winnowtak.py [options] -i <input_file>
```

## Primary arguments
- `-i, --input`: Input FASTA file path (required except with --check-deps)
- `-o, --output`: Output directory path (optional; default: output/<input_basename>/)
- `-t, --threshold`: Prediction threshold in [0,1] (default: 0.1; affects prediction only)
- `--score`: Score threshold for InterProScan filtering (default: 1.0)
- `--classification-mode`: Classification mode: specific (specificity-first) or score (score-first, default)

Feature options:
- -test, --test-mode      Test mode: skip InterProScan/hmmscan and validate classification using existing result files
- -json, --json-file      InterProScan JSON result file for test mode
- -spechmm, --spechmm-file hmmscan result file (result.tbl) for test mode
- --predict               Enable prediction: run model inference, then analyze predicted TF sequences
- --predict-mode          Prediction splitting mode: fast (default) or full
- --use-supplementary     Enable supplementary models (optional)
- --supplementary-only    Use supplementary models only; skip the main model
- --supp-models           Optional: specify supplementary model files (multiple allowed); if omitted, use all models in the directory
- --grad-cam-mode         Grad-CAM mode: none (default)/fast/all/positive (all/positive require --predict; --list-predict currently unsupported)
- --appl                  InterProScan application list (comma-separated; defaults to common libraries)
- --interproscan          Path to an external interproscan.sh (hmmscan will follow that InterProScan)
- --skip-pk               Skip protein kinase identification/classification
- --debug                 Enable debug outputs (default: off)

Dependency checks:
--check-deps            Check dependencies and exit
--skip-deps-check       Skip dependency checks (not recommended)

Examples
--------

1. Basic analysis (analyze input directly):
   python winnowtak.py -i test_protein.fasta

2. Prediction workflow:
   python winnowtak.py --predict -i test_protein.fasta -t 0.1

3. Specify an output directory:
   python winnowtak.py --predict -i test_protein.fasta -o /path/to/output

4. Enable debug mode:
   python winnowtak.py --predict -i test_protein.fasta --debug

5. Restrict InterProScan applications:
   python winnowtak.py -i test_protein.fasta --appl CDD,Pfam,SMART

6. Test mode:
   python winnowtak.py -test -i input.fasta -json ipr_result.json -spechmm hmmscan_result.tbl

7. Check dependencies:
   python winnowtak.py --check-deps

8. Use supplementary models (requires --predict):
   python winnowtak.py --predict --use-supplementary -i input.fasta
   python winnowtak.py --predict --supplementary-only -i input.fasta
   python winnowtak.py --predict --use-supplementary --supp-models a.pth b.pth -i input.fasta

9. Grad-CAM heatmaps (requires --predict):
   - fast: runs after classification, using the classified TF FASTA under result/ (batch mode)
   - all: runs on the original input FASTA (all sequences)
   python winnowtak.py --predict --grad-cam-mode fast -i input.fasta
   python winnowtak.py --predict --grad-cam-mode all -i input.fasta

10. Skip protein kinase analysis:
   python winnowtak.py -i test_protein.fasta --skip-pk

11. Run a positive protein kinase example:
   python winnowtak.py -i test_protein_kinase.fasta --skip-deps-check

12. Run a positive protein kinase smoke test:
   ./smoke_test.sh --input test_protein_kinase.fasta --require-pk 2

13. Run prediction mode with a positive protein kinase sample:
   ./smoke_test.sh --predict --input test_protein_kinase.fasta --require-pk 2

Outputs
--------
The program creates the following files and directories under the output directory:

```text
<project_output>/                          # Project output directory (default: output/<input_basename>/)
├── protein_model_preclassification/       # Prediction outputs (only with --predict/--list-predict)
│   ├── <name>_prediction.csv              # Predictions for all sequences
│   ├── <name>_prediction_tf_only.csv      # Subset with TF_Probability >= threshold
│   └── <name>_tf_sequences.fasta          # Extracted TF sequences
├── InterproScan/                          # InterProScan output directory
│   └── <input>.json
├── hmmscan/                               # hmmscan output directory
│   └── result.tbl
├── protein_kinase/                        # Protein kinase outputs (default direct/predict modes)
│   ├── match_tbl.txt                      # PK classification table in the same 6-column style as TF/TR match_tbl.txt
│   ├── match.json                         # Debug output (--debug): PK classification JSON in the same schema as TF/TR match.json
│   ├── <name>_pk_classified.fasta         # Classified PK sequences in the same header style as TF classified FASTA
│   ├── pk_sequence.fasta
│   ├── pk_classification.tsv
│   ├── shiu_classification.txt
│   └── PPC_classification.txt
├── getrule.json                           # Debug output (--debug): parsed classification rules
└── result/                                # Classification results
    ├── match_tbl.txt                      # Final TF family classification results (table)
    ├── all_match_tbl.txt                  # Combined TF/TR + protein kinase summary table
    ├── <name>_tf_classified.fasta         # Classified TF sequences
    ├── match.json                         # Debug output (--debug)
    ├── processed_ipr_domains.json         # Domain architecture JSON (sequence ↔ domain hits; optional)
    └── pfamspec.json                      # PFAM spec JSON used for classification (optional; written with --debug)
```

Grad-CAM outputs:
- Written to `<project_output>/<name>_gradcam/` by default, with one PNG per sequence (only when --grad-cam-mode is enabled)

Workflow
--------
1. Input validation: validate FASTA structure and protein sequences
2. Dependency checks: verify required tools and libraries
3. Prediction stage (optional): run deep learning model inference for TF preclassification
4. Protein kinase analysis: run bundled PK HMMs on the processed protein FASTA
5. Domain analysis: run InterProScan and hmmscan
6. Result processing: parse and filter analysis outputs
7. TF family classification: apply rule-based family assignment
8. Output reporting: write classification reports and FASTA files

FAQ / Troubleshooting
--------

1) db download / preparation failed (timeout / network error)
   - Download `db.tar.gz` manually, then place it in the project root (same directory as `winnowtak.py`).
   - Re-run the program (it will try to extract / prepare db assets automatically), or extract manually:
     ```bash
     tar -xzf db.tar.gz
     ```
   - If you already have a working InterProScan, prefer using it directly:
     ```bash
     python winnowtak.py -i input.fasta --interproscan /path/to/interproscan.sh
     ```

2) "Dependency checks failed"
   - Ensure Python packages are installed and `java` / `hmmscan` are available on PATH.
   - Run:
     ```bash
     python winnowtak.py --check-deps
     ```

3) "FASTA validation failed"
   - Ensure input is valid FASTA and sequences are proteins.
   - Remove `*` (stop codon) and unexpected characters.

4) "InterProScan failed"
   - Check `java -version`, disk space, and InterProScan path (`--interproscan` if needed).

5) "Prediction is unavailable" / Grad-CAM not working
   - Install PyTorch and ensure `pre_model/model.pth` and `pre_model/predict.py` exist.

Citation
--------
If you use iTAK Next in your research, please cite:

> **iTAK Next: A deep-learning-enhanced tool for transcription factor and protein kinase annotation in plant genomes**
> [Authors], [Year]
> *[Journal/Preprint — to be updated]*

See also the manuscript and supplementary materials under `../paper/hapi-iTAK/` (manuscript body: `../paper/hapi-iTAK/docs/readme_paper_修改版.md`).
