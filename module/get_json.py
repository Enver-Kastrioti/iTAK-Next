import json
import os
import argparse


def load_input_data(input_file):
    """Load an input JSON file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {input_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed - {e}")
        return None
    except Exception as e:
        print(f"Error: failed to read file - {e}")
        return None


def parse_specific_thresholds_from_rule(rule_file_path):
    """Parse all 'Score:' lines in rule.txt blocks into a global accession-to-threshold map."""
    thresholds = {}
    try:
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            blocks = ''.join(f.readlines()).strip().split('//')
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            for raw_line in block.split('\n'):
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if not line.startswith('Score:'):
                    continue
                content = line[len('Score:'):].strip()
                parts = [p for p in content.split(':') if p]
                for part in parts:
                    part = part.strip()
                    if '(' in part and part.endswith(')'):
                        try:
                            name, val_str = part.split('(', 1)
                            name = name.strip()
                            val = float(val_str[:-1])
                            if name:
                                thresholds[name] = max(val, thresholds.get(name, float('-inf')))
                        except Exception:
                            continue
    except Exception:
        # On parsing failure, return an empty map and continue the pipeline.
        return {}
    return thresholds


def load_specific_thresholds():
    """Locate and load accession-specific score thresholds from db/rule.txt."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    candidates = [
        os.path.join(project_root, 'db', 'rule.txt'),
        os.path.join(project_root, 'rule.txt'),
    ]
    for rule_file in candidates:
        if os.path.exists(rule_file):
            return parse_specific_thresholds_from_rule(rule_file)
    return {}


def is_valid_score(score, score_threshold=1.0, accession=None, specific_thresholds=None, strong_value=10.0):
    """
    Determine whether a score passes filtering:
    - If an accession-specific threshold is available, require score >= that threshold.
    - Otherwise, require score > the global threshold.
    - The sentinel value 'STRONG' is treated as a numeric value (default: 10.0).
    """
    if str(score).upper() == "STRONG":
        s = float(strong_value)
    else:
        try:
            s = float(score)
        except (ValueError, TypeError):
            return False

    if specific_thresholds and accession in specific_thresholds:
        return s >= float(specific_thresholds[accession])
    return s > float(score_threshold)


def get_score_value(score, strong_value=10.0):
    """Convert a score to a numeric value used for comparisons."""
    if str(score).upper() == "STRONG":
        return float(strong_value)
    return float(score or 0)


def process_ipr_groups(gene_data, gene_id, output_dir, debug=False, score_threshold=1.0, specific_thresholds=None):
    """Process IPR-grouped matches and resolve overlaps within each group."""
    new_gene_data = {}
    
    # Collect cases where overlap < 0
    overlap_records = []
    
    for ipr, matches in gene_data[1].items():
        # Filter matches by score, preferring accession-specific thresholds when available
        valid_matches = [
            m for m in matches
            if is_valid_score(m.get("score"), score_threshold, accession=m.get("accession"), specific_thresholds=specific_thresholds)
        ]
        
        if not valid_matches:
            continue
            
        # Group by (accession, library)
        groups = {}
        for match in valid_matches:
            key = (match["accession"], match["library"])
            if key not in groups:
                groups[key] = []
            groups[key].append(match)
        
        # Process each group and collect results
        final_matches = []
        max_group_size = 0
        
        for (acc, lib), group in groups.items():
            # Sort by start coordinate
            group.sort(key=lambda x: x["start"])
            
            # Resolve overlaps
            i = 0
            while i < len(group):
                j = i + 1
                while j < len(group):
                    # Compute overlap length
                    overlap = group[j]["start"] - group[i]["end"] - 1
                    
                    # Record all overlap < 0 cases
                    if overlap < 0:
                        overlap_record = {
                            "gene_id": gene_id,
                            "ipr": ipr,
                            "accession": acc,
                            "library": lib,
                            "match1": {
                                "start": group[i]["start"],
                                "end": group[i]["end"],
                                "score": group[i]["score"]
                            },
                            "match2": {
                                "start": group[j]["start"],
                                "end": group[j]["end"],
                                "score": group[j]["score"]
                            },
                            "overlap": overlap
                        }
                        overlap_records.append(overlap_record)
                        # Write to overlop.txt only in debug mode
                        if debug:
                            with open(os.path.join(output_dir, "overlop.txt"), "a") as f:
                                f.write(f"Gene ID: {gene_id}\n")
                                f.write(f"IPR: {ipr}\n")
                                f.write(f"Accession: {acc}\n")
                                f.write(f"Library: {lib}\n")
                                f.write(f"Match 1: start={group[i]['start']}, end={group[i]['end']}, score={group[i]['score']}\n")
                                f.write(f"Match 2: start={group[j]['start']}, end={group[j]['end']}, score={group[j]['score']}\n")
                                f.write(f"Overlap: {overlap}\n")
                                f.write("---\n")
                    
                    if overlap < -100:
                        # Compare scores and drop the lower-scoring hit
                        score_i = get_score_value(group[i]["score"])
                        score_j = get_score_value(group[j]["score"])
                        
                        if score_i < score_j:
                            group.pop(i)
                            j = i + 1  # Reset j after removing i
                            continue
                        else:
                            group.pop(j)
                            continue
                    j += 1
                i += 1
            
            # Track the maximum group size
            current_group_size = len(group)
            max_group_size = max(max_group_size, current_group_size)
            
            # Append processed matches
            final_matches.extend([
                {
                    "accession": m["accession"],
                    "library": m["library"],
                    "start": m["start"],
                    "end": m["end"],
                    "evalue": m["evalue"],
                    "score": m["score"]
                } for m in group
            ])
        
        # Add the primary match group plus descending empty placeholders
        if final_matches:
            if max_group_size > 2:
                # Primary group (using the maximum suffix)
                new_gene_data[f"{ipr}&{max_group_size}"] = final_matches
                # Descending empty placeholders
                for i in range(max_group_size-1, 1, -1):
                    new_gene_data[f"{ipr}&{i}"] = []
                # Final placeholder without numeric suffix
                new_gene_data[ipr] = []
            elif max_group_size == 2:
                # When the maximum group size is 2
                new_gene_data[f"{ipr}&2"] = final_matches
                new_gene_data[ipr] = []
            else:
                # When the maximum group size is 1, keep the original IPR key
                new_gene_data[ipr] = final_matches
    
    # Write overlap < 0 records only in debug mode
    if overlap_records and debug:
        with open(os.path.join(output_dir, "overlop.txt"), "a") as f:
            for record in overlap_records:
                f.write(f"Gene ID: {record['gene_id']}\n")
                f.write(f"IPR: {record['ipr']}\n")
                f.write(f"Accession: {record['accession']}, Library: {record['library']}\n")
                f.write(f"Match 1: Start={record['match1']['start']}, End={record['match1']['end']}, Score={record['match1']['score']}\n")
                f.write(f"Match 2: Start={record['match2']['start']}, End={record['match2']['end']}, Score={record['match2']['score']}\n")
                f.write(f"Overlap: {record['overlap']}\n")
                f.write("---\n")
    
    return new_gene_data
# Read data from InterProScan JSON output

def process_data(input_file, output_dir=None, debug=False, score_threshold=1.0):
    # Load input data
    old_data = load_input_data(input_file)
    if old_data is None:
        return None, None
    
    # Load global accession-specific thresholds (from 'Score' lines in db/rule.txt)
    specific_thresholds = load_specific_thresholds()

    # Build the output structure
    new_result = {
        "result": {
            "match": []
        }
    }

    # Filtered results in the new schema
    filtered_result = {
        "result": {
            "match": []
        }
    }

    # If an output directory is provided, ensure it exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Clear overlop.txt only in debug mode
        if debug:
            with open(os.path.join(output_dir, "overlop.txt"), "w") as f:
                f.write("")

    for item in old_data.get("results", []):
        gene_data_list = []
        filtered_gene_data_list = []
        
        for xref in item.get("xref", []):
            gene_id = xref["id"]
            
            if item.get("matches"):
                # Handle legacy-format InterProScan JSON
                gene_data = [
                    {"sequence": item["sequence"]},
                    {}
                ]

                for match in item.get("matches", []):
                    signature = match.get("signature", {})
                    signature_accession = signature.get("accession")
                    library = signature.get("signatureLibraryRelease", {}).get("library")
                    
                    entry = signature.get("entry")
                    if not isinstance(entry, dict):
                        continue
                    
                    ipr_accession = entry.get("accession")
                    if not ipr_accession:
                        continue

                    ipr_name = entry.get("name")
                    description = entry.get("description")
                    parent_score = match.get("score")
                    
                    for location in match["locations"]:
                        start = location.get("start")
                        end = location.get("end")
                        score = location.get("score", location.get("level", parent_score))
                        
                        if "evalue" in location:
                            evalue = location["evalue"]
                        else:
                            evalue = location.get("representative", None)
                        
                        match_item = {
                            "accession": signature_accession,
                            "library": library,
                            "ipr": ipr_accession or "null",
                            "ipr_name": ipr_name or "null",
                            "description": description or "null",
                            "start": start or "null",
                            "end": end or "null",
                            "evalue": evalue or "null",
                            "score": score or "null"
                        }

                        if ipr_accession not in gene_data[1]:
                            gene_data[1][ipr_accession] = []
                        gene_data[1][ipr_accession].append(match_item)
                
                # Append to the legacy-format list
                gene_data_list.append({gene_id: gene_data})
                
                # Process into the new schema (passing accession-specific thresholds)
                filtered_data = process_ipr_groups(gene_data, gene_id, output_dir or "", debug, score_threshold, specific_thresholds=specific_thresholds)
                if filtered_data:
                    # Build a data structure consistent with the test-mode consumer
                    filtered_gene_data = [
                        {"sequence": gene_data[0]["sequence"]},
                        filtered_data
                    ]
                    filtered_gene_data_list.append({gene_id: filtered_gene_data})
            else:
                pass  # No matches: skip

        # Merge per-gene data into the final output
        if gene_data_list:
            new_result["result"]["match"].extend(gene_data_list)
        if filtered_gene_data_list:
            filtered_result["result"]["match"].extend(filtered_gene_data_list)

    # If an output directory is provided, write results to disk
    if output_dir:
        with open(os.path.join(output_dir, "processed_ipr_domains.json"), "w") as f:
            json.dump(filtered_result, f, indent=2)

        # Save debug artifacts only in debug mode
        if debug:
            with open(os.path.join(output_dir, "raw_interproscan_data.json"), "w") as f:
                json.dump(new_result, f, indent=2)
            print("Conversion completed; results saved as processed_ipr_domains.json and raw_interproscan_data.json (debug mode)")
        else:
            print("Conversion completed; results saved as processed_ipr_domains.json")
    else:
        print("Processing completed; returning in-memory results")
    
    # Return processed data
    return filtered_result, new_result


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process an InterProScan JSON file and generate normalized outputs')
    parser.add_argument('-i', '--input', required=True, help='Path to the input JSON file (InterProScan output)')
    parser.add_argument('-o', '--output', required=True, help='Path to the output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode and write debug artifacts')
    parser.add_argument('--score', type=float, default=1.0, help='Score threshold; retain only results above this value (default: 1.0)')
    
    args = parser.parse_args()
    
    # Run the processing pipeline
    filtered_result, new_result = process_data(
        input_file=args.input,
        output_dir=args.output,
        debug=args.debug,
        score_threshold=args.score
    )
    
    if filtered_result is None:
        print("Data processing failed")
        return False
    
    return True


if __name__ == "__main__":
    main()
