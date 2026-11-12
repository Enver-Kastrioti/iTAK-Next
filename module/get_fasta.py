import os
import json
import argparse
from Bio import SeqIO
from Bio.Seq import Seq

def get_output_dir(base_dir=None):
    """Get the output directory path.
    
    Args:
        base_dir (str, optional): Base directory; if None, use the project root.
    
    Returns:
        str: Absolute path to the result directory.
    """
    if base_dir is None:
        # Use the parent directory of this module as the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
    
    # Build result directory path
    output_dir = os.path.join(base_dir, "result")
    
    # Create it if missing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def get_output_subdir(name, base_dir=None):
    base = get_output_dir(base_dir)
    sub = os.path.join(base, name)
    if not os.path.exists(sub):
        os.makedirs(sub)
    return sub

def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_project_output_dir(fasta_file):
    root = _repo_root()
    output_base = os.path.join(root, "output")
    base = os.path.splitext(os.path.basename(fasta_file))[0]
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    candidates = []
    for name in os.listdir(output_base):
        if name == base or (name.startswith(base + "_")):
            full = os.path.join(output_base, name)
            if os.path.isdir(full):
                candidates.append(full)
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    target = os.path.join(output_base, base)
    os.makedirs(target, exist_ok=True)
    return target

def read_fasta_to_dict(fasta_file):
    """
    Read a FASTA file and convert it to a dictionary.
    
    Args:
        fasta_file (str): Path to a FASTA file.
    
    Returns:
        dict: Mapping from record ID to sequence string.
    """
    fasta_dict = {}
    
    try:
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Use record.id as the key and the sequence as the value
                fasta_dict[record.id] = str(record.seq)
        
        print(f"Successfully read {len(fasta_dict)} sequences")
        return fasta_dict
        
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return {}

def format_tf_fasta_with_classification(fasta_file, classification_result, output_file):
    """
    Generate a standardized FASTA file with TF classification annotations.
    
    Args:
        fasta_file (str): Input FASTA file path.
        classification_result (dict): Classification results, keyed by gene_id.
        output_file (str): Output FASTA file path.
    
    Returns:
        bool: True on success, otherwise False.
    """
    try:
        fasta_dict = read_fasta_to_dict(fasta_file)
        if not fasta_dict:
            print("Unable to read the input FASTA file for classification")
            return False
        
        # Write only sequences that have classification results
        with open(output_file, 'w') as f:
            written_count = 0
            for gene_id, sequence in fasta_dict.items():
                # Only include entries present in the classification map
                if gene_id in classification_result:
                    tf_info = classification_result[gene_id]
                    
                    # FASTA header format: >gene_id | family | type
                    header = f">{gene_id} | {tf_info['family']} | {tf_info['type']}"
                    
                    # Write header and sequence
                    f.write(header + '\n')
                    f.write(sequence + '\n')
                    written_count += 1
            
            print(f"Successfully generated a FASTA file containing {written_count} classified sequences")
            print("All output sequences include TF classification annotations")
            return True
            
    except Exception as e:
        print(f"Error generating FASTA file: {e}")
        return False

def generate_classified_fasta(fasta_file, classification_result, output_file=None, output_dir=None):
    """
    Generate a FASTA file with TF classification annotations (for use by other modules).

    Only sequences with classification results are included.
    
    Args:
        fasta_file (str): Input FASTA file path.
        classification_result (dict): In-memory classification results.
        output_file (str, optional): Output FASTA file path. If None, a default is generated.
        output_dir (str, optional): Output directory path (e.g., project subdirectory).
    
    Returns:
        str: Output file path; returns None on failure.
    """
    try:
        # If output_file is not provided, generate a default path
        if not output_file:
            input_name = os.path.splitext(os.path.basename(fasta_file))[0]
            if output_dir:
                result_dir = output_dir
            else:
                result_dir = get_output_dir(output_dir)
            output_file = os.path.join(result_dir, f"{input_name}_tf_classified.fasta")
        
        # Run formatter
        success = format_tf_fasta_with_classification(fasta_file, classification_result, output_file)
        
        if success:
            return output_file
        else:
            return None
            
    except Exception as e:
        print(f"Error generating classified FASTA file: {e}")
        return None

def _best_orf_coverage(seq, table=1):
    """六个读框中最长无终止 ORF 占编码容量（len/3）的比例，范围 0–1。

    真 CDS 在其正确读框中几乎不含内部终止子，覆盖率≈1；而序列组成恰好
    只由核酸字母构成的真实蛋白，六帧翻译会散布终止子，覆盖率显著偏低。
    """
    s = Seq(seq.upper())
    if len(s) < 3:
        return 0.0
    capacity = len(s) / 3.0
    best = 0
    for strand in (s, s.reverse_complement()):
        for frame in (0, 1, 2):
            sub = strand[frame:]
            rem = len(sub) % 3
            if rem:
                sub = sub[:-rem]
            if len(sub) < 3:
                continue
            try:
                aa = str(sub.translate(table=table, to_stop=False))
            except Exception:
                continue
            longest = max((len(seg) for seg in aa.split('*')), default=0)
            if longest > best:
                best = longest
    return best / capacity


def _looks_like_cds(seq, min_orf_coverage=0.95):
    """判断输入序列是否为需六帧翻译的核酸 CDS。

    仅"字母全部落在 IUPAC 核酸集"不足以判定：蛋白质组中真实存在的蛋白
    若恰好只由 ACGTUNWSMKRYBDHV 这些字母组成（例如某读框的产物），也会
    满足该条件，从而被误送入六帧翻译、序列被翻译产物替换（污染下游）。
    因此追加"存在近乎完整的开放阅读框"作为必要条件。
    """
    s = seq.upper()
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    nuc = set("ACGTUNWSMKRYBDHV")
    if not set(letters).issubset(nuc):
        return False
    return _best_orf_coverage(s) >= min_orf_coverage

def _best_six_frame_translate(seq, table=1, min_orf_aa=30):
    s = Seq(seq)
    candidates = []
    for frame in (0, 1, 2):
        sub = s[frame:]
        rem = len(sub) % 3
        if rem:
            sub = sub[:-rem]
        aa = str(sub.translate(table=table, to_stop=False))
        seg = max(aa.split('*'), key=len) if aa else ""
        candidates.append((seg, f"+{frame+1}"))
    rc = s.reverse_complement()
    for frame in (0, 1, 2):
        sub = rc[frame:]
        rem = len(sub) % 3
        if rem:
            sub = sub[:-rem]
        aa = str(sub.translate(table=table, to_stop=False))
        seg = max(aa.split('*'), key=len) if aa else ""
        candidates.append((seg, f"-{frame+1}"))
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    best = candidates[0]
    if len(best[0]) < min_orf_aa:
        return "", best[1]
    return best

def _six_frame_translate_all(seq, table=1):
    s = Seq(seq)
    results = []
    for frame in (0, 1, 2):
        sub = s[frame:]
        rem = len(sub) % 3
        if rem:
            sub = sub[:-rem]
        aa = str(sub.translate(table=table, to_stop=False)).replace('*', '')
        results.append((aa, f"+{frame+1}"))
    rc = s.reverse_complement()
    for frame in (0, 1, 2):
        sub = rc[frame:]
        rem = len(sub) % 3
        if rem:
            sub = sub[:-rem]
        aa = str(sub.translate(table=table, to_stop=False)).replace('*', '')
        results.append((aa, f"-{frame+1}"))
    return results

def generate_protein_sequences_in_memory(fasta_file, genetic_code=1, min_orf_aa=30):
    try:
        sequences = []
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                if _looks_like_cds(seq):
                    translated = _six_frame_translate_all(seq, table=genetic_code)
                    for aa, frame in translated:
                        sequences.append({"header": f"{record.id}|frame={frame}", "sequence": aa})
                else:
                    sequences.append({"header": record.id, "sequence": seq})
        return sequences
    except Exception as e:
        print(f"Failed to generate in-memory sequences via six-frame translation: {e}")
        return []

def format_tf_fasta_with_classification_from_mem(seqs_dict, classification_result, output_file):
    try:
        with open(output_file, 'w') as f:
            written_count = 0
            for gene_id, sequence in seqs_dict.items():
                if gene_id in classification_result:
                    tf_info = classification_result[gene_id]
                    header = f">{gene_id} | {tf_info['family']} | {tf_info['type']}"
                    f.write(header + '\n')
                    f.write(sequence + '\n')
                    written_count += 1
        print(f"Successfully generated a FASTA file containing {written_count} classified sequences")
        print("All output sequences include TF classification annotations")
        return True
    except Exception as e:
        print(f"Error generating FASTA file (in-memory): {e}")
        return False
def get_processed_fasta_path(fasta_file, output_dir=None):
    input_name = os.path.splitext(os.path.basename(fasta_file))[0]
    project_dir = get_project_output_dir(fasta_file) if output_dir is None else output_dir
    # Previously: write into a "six_frame_translation" subdirectory.
    # Now: write directly under the project output directory.
    return os.path.join(project_dir, f"{input_name}_protein_replaced.fasta")

def generate_protein_fasta_with_translation(fasta_file, output_file=None, output_dir=None, genetic_code=1, min_orf_aa=30):
    try:
        if not output_file:
            output_file = get_processed_fasta_path(fasta_file, output_dir)
        total = 0
        kept = 0
        translated_frames = 0
        has_translation = False
        
        # Determine whether translation is needed. If all inputs are proteins, translation is skipped.
        
        with open(fasta_file, 'r') as handle, open(output_file, 'w') as out:
            for record in SeqIO.parse(handle, "fasta"):
                total += 1
                seq = str(record.seq)
                if _looks_like_cds(seq):
                    has_translation = True
                    try:
                        translated = _six_frame_translate_all(seq, table=genetic_code)
                        for aa, frame in translated:
                            out.write(f">{record.id}|frame={frame}\n")
                            out.write(aa + "\n")
                            translated_frames += 1
                    except Exception:
                        out.write(f">{record.id}\n")
                        out.write(seq + "\n")
                        kept += 1
                else:
                    out.write(f">{record.id}\n")
                    out.write(seq + "\n")
                    kept += 1
        
        print(f"Read {total} sequences; retained {kept} protein sequences; generated {translated_frames} translated-frame sequences via six-frame translation")
        print(f"Written to: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"Failed to write protein FASTA from six-frame translation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate a FASTA file with TF annotations or perform six-frame translation')
    parser.add_argument('-i', '--input', required=True, help='Path to input FASTA file')
    parser.add_argument('-o', '--output', help='Path to output FASTA file (optional)')
    parser.add_argument('--classification', help='Path to classification-result JSON file (for testing)')
    parser.add_argument('--translate-only', action='store_true', help='Run six-frame translation only and write protein FASTA output')
    parser.add_argument('--genetic-code', type=int, default=1, help='Genetic code table ID used for translation')
    parser.add_argument('--min-orf-aa', type=int, default=30, help='Minimum ORF length threshold (amino acids)')
    
    args = parser.parse_args()
    
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    if not args.output:
        if args.translate_only:
            args.output = get_processed_fasta_path(args.input)
        else:
            output_dir = get_output_dir()
            args.output = os.path.join(output_dir, f"{input_name}_tf_classified.fasta")

    if args.translate_only:
        out = generate_protein_fasta_with_translation(
            args.input,
            output_file=args.output,
            output_dir=None,
            genetic_code=args.genetic_code,
            min_orf_aa=args.min_orf_aa,
        )
        if out:
            print(f"Protein FASTA saved to: {out}")
        else:
            print("Six-frame translation failed")
    else:
        if args.classification:
            try:
                with open(args.classification, 'r', encoding='utf-8') as f:
                    classification_result = json.load(f)
                print(f"Loaded {len(classification_result)} classification results from file")
            except Exception as e:
                print(f"Error reading classification-result file: {e}")
                classification_result = {}
        else:
            classification_result = {}
            print("No classification results provided; generating a FASTA file without TF annotations")
        success = format_tf_fasta_with_classification(args.input, classification_result, args.output)
        if success:
            print(f"FASTA file saved to: {args.output}")
        else:
            print("FASTA generation failed")

if __name__ == "__main__":
    main()
