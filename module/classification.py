import json
import os
import sys
import importlib.util
from pathlib import Path

# Dynamically import the get_rule and selfbuild_hmm modules
def import_module_dynamically(module_name):
    """Dynamically import a module by name."""
    try:
        # First, try a standard import
        return importlib.import_module(module_name)
    except ImportError:
        # If it fails, try importing from the current directory
        current_dir = Path(__file__).parent
        module_path = current_dir / f"{module_name}.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            raise ImportError(f"Cannot find module {module_name}")

# Import required modules
get_rule = import_module_dynamically('get_rule')
selfbuild_hmm = import_module_dynamically('selfbuild_hmm')

# Get function references
parse_rule_file = get_rule.parse_rule_file
parse_pfam_spec = selfbuild_hmm.parse_pfam_spec

# Global variables (set at runtime)
filtered_result = None
result_dir = None

def set_result_dir(dir_path):
    """Set the result directory path."""
    global result_dir
    result_dir = dir_path

def load_filtered_result(filtered_data=None):
    """Load filtered results, preferring in-memory data and falling back to file I/O.
    
    Args:
        filtered_data (dict, optional): Filtered data provided in memory.
    
    Returns:
        dict: Loaded data; returns a default empty structure on failure.
    """
    global result_dir
    
    # Use in-memory data when provided
    if filtered_data is not None:
        print("Using in-memory filtered data")
        return filtered_data
    
    # Otherwise, read from file (for backward compatibility)
    if not result_dir:
        print("[ERROR] Result directory is not set")
        return {"result": {"match": []}}
        
    processed_ipr_path = os.path.join(result_dir, "processed_ipr_domains.json")
    
    if os.path.exists(processed_ipr_path):
        print(f"Loading filtered data from file: {processed_ipr_path}")
        with open(processed_ipr_path, "r") as f:
            return json.load(f)
    else:
        print(f"[ERROR] processed_ipr_domains.json not found: {processed_ipr_path}")
        return {"result": {"match": []}}

def initialize_module(result_directory, filtered_data=None):
    """Initialize module state: set result directory and load filtered data.
    
    Args:
        result_directory (str): Result directory path.
        filtered_data (dict, optional): Filtered data provided in memory.
    """
    global filtered_result
    set_result_dir(result_directory)
    filtered_result = load_filtered_result(filtered_data)

# Absolute path to the current module directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables (set at runtime)
rules_dict = {}
result_spec = {}

def initialize_rules(rule_file_path):
    """Initialize rule definitions from a rule file."""
    global rules_dict
    try:
        if not os.path.exists(rule_file_path):
            raise FileNotFoundError(f"Rule file not found: {rule_file_path}")
            
        rules_dict = parse_rule_file(rule_file_path)
        print("[OK] Rule file loaded successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Exception occurred while loading the rule file - {str(e)}")
        rules_dict = {}
        return False

def merge_results(dict1, dict2):
    """Merge two result dictionaries."""
    print("[INFO] Starting result merge...")
    
    # Create a new match list
    merged_matches = []
    
    # Add all genes from the first dict
    for gene_match in dict1["result"]["match"]:
        merged_matches.append(gene_match)
    
    # Add or merge genes from the second dict
    merged_count = 0
    new_count = 0
    
    for gene_match in dict2["result"]["match"]:
        for gene_id, gene_data in gene_match.items():
            # Check whether this gene already exists
            found = False
            for existing_match in merged_matches:
                if gene_id in existing_match:
                    # Existing gene found: merge domain data
                    existing_gene_data = existing_match[gene_id]
                    
                    # Both formats are now [sequence_obj, {domain: [matches]}]
                    if isinstance(existing_gene_data, list) and len(existing_gene_data) >= 2:
                        if isinstance(existing_gene_data[1], dict) and isinstance(gene_data, list) and len(gene_data) >= 2:
                            # Merge the second element (domain dict)
                            for domain, hits in gene_data[1].items():
                                if domain in existing_gene_data[1]:
                                    # If domain exists, extend its hit list
                                    if isinstance(existing_gene_data[1][domain], list):
                                        existing_gene_data[1][domain].extend(hits)
                                    else:
                                        existing_gene_data[1][domain] = [existing_gene_data[1][domain]] + hits
                                else:
                                    # New domain: add directly
                                    existing_gene_data[1][domain] = hits
                    
                    merged_count += 1
                    found = True
                    break
            
            if not found:
                # New gene: add as-is
                merged_matches.append({gene_id: gene_data})
                new_count += 1
    
    print(f"[OK] Merge completed: {merged_count} genes merged, {new_count} new genes added")
    return {"result": {"match": merged_matches}}

# Downstream functions
def get_ipr_counts(gene_data):
    """Extract IPR copy counts from per-gene data.

    Note:
    - A single IPR key may include hits from multiple accessions (databases).
    - For copy counting, we use the maximum count among accessions, rather than summing
      all hits across accessions.
    """
    counts = {}
    if len(gene_data) > 1 and isinstance(gene_data[1], dict):
        for ipr_key, hits in gene_data[1].items():
            # Handle optional count suffixes in keys (e.g., IPR001005&3)
            real_ipr = ipr_key.split('&')[0]
            
            # Count hits per accession
            accession_counts = {}
            if isinstance(hits, list):
                for hit in hits:
                    acc = hit.get('accession')
                    if acc:
                        accession_counts[acc] = accession_counts.get(acc, 0) + 1
            
            # Use the maximum as the effective copy number; 0 if no hits
            num_hits = max(accession_counts.values()) if accession_counts else 0
            
            # Accumulate counts under real_ipr to be robust to accidental splitting across keys.
            # This assumes that hits for a real_ipr are not fragmented in a way that would
            # incorrectly merge different accessions; for maximum rigor, track counts as
            # {real_ipr: {accession: count}} and reduce with max at the end.
            
            if real_ipr not in counts:
                counts[real_ipr] = {}
            
            for acc, count in accession_counts.items():
                counts[real_ipr][acc] = counts[real_ipr].get(acc, 0) + count
                
    # Convert {ipr: {acc: count}} to {ipr: max_count}
    final_counts = {}
    for ipr, acc_dict in counts.items():
        final_counts[ipr] = max(acc_dict.values()) if acc_dict else 0
        
    return final_counts


def get_ipr_max_scores(gene_data):
    ipr_max_scores = {}
    if len(gene_data) > 1 and isinstance(gene_data[1], dict):
        for ipr_key, hits in gene_data[1].items():
            real_ipr = ipr_key.split('&')[0]
            max_score = 0.0
            if isinstance(hits, list):
                for hit in hits:
                    score_val = hit.get('score', 0)
                    if str(score_val).upper() == 'STRONG':
                        score_val = 10.0
                    else:
                        try:
                            score_val = float(score_val)
                        except Exception:
                            score_val = 0.0
                    if score_val > max_score:
                        max_score = score_val
            if max_score > ipr_max_scores.get(real_ipr, 0.0):
                ipr_max_scores[real_ipr] = max_score
    return ipr_max_scores

def evaluate_logic(node, ipr_counts):
    """Recursively evaluate a rule logic tree."""
    if node is None:
        return True # No requirement implies a match
        
    if isinstance(node, str):
        # Leaf node: check a single domain constraint
        domain = node
        required_count = 1
        # Handle count suffix, e.g. IPR001471&2
        if '&' in domain:
            parts = domain.split('&')
            domain = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                required_count = int(parts[1])
        
        actual_count = ipr_counts.get(domain, 0)
        return actual_count >= required_count

    if isinstance(node, dict):
        op = node.get('op')
        children = node.get('children', [])
        
        if op == 'AND':
            return all(evaluate_logic(child, ipr_counts) for child in children)
        elif op == 'OR':
            return any(evaluate_logic(child, ipr_counts) for child in children)
            
    return False

def check_rule_match(ipr_counts, rule_data, ipr_max_scores=None, forbidden_score_threshold=10.0):
    """Check whether IPR copy counts satisfy a rule."""
    # Forbidden constraints
    forbidden = rule_data.get("forbidden", [])
    for f in forbidden:
        if f == 'NA': continue
        if f in ipr_counts:
            if ipr_max_scores is None:
                return False
            try:
                max_score = float(ipr_max_scores.get(f, 0.0) or 0.0)
            except Exception:
                max_score = 0.0
            if max_score > float(forbidden_score_threshold):
                return False
    
    # Logic constraints
    logic_tree = rule_data.get("logic")
    if logic_tree is not None:
        return evaluate_logic(logic_tree, ipr_counts)
        
    # If no logic tree exists (legacy compatibility)
    mode = rule_data.get("mode", [])[0] if rule_data.get("mode") else None
    
    # If mode is missing or logic with an empty tree, treat as unconstrained (match)
    if not mode or mode == "logic":
        return True

    # Legacy fallback (should rarely execute)
    required = set(rule_data.get("required", []))
    ipr_set = set(ipr_counts.keys())
    
    if mode == "a":
        return required.issubset(ipr_set)
    elif mode == "b":
        return not required.isdisjoint(ipr_set)
    
    return False

def classify_genes(input_dict, mode='specific'):
    """Classify genes and produce TF family assignments.
    
    Args:
        input_dict (dict): Input gene-to-domain match data.
        mode (str): Classification mode: 'specific' (specificity-prioritized) or 'score' (score-prioritized).
    """
    result = {}
    
    print(f"[INFO] Classification mode: {mode}")
    
    # Iterate over all genes in the input dictionary
    for gene_match in input_dict["result"]["match"]:
        for gene_id, gene_data in gene_match.items():
            # Compute IPR copy counts for this gene
            ipr_counts = get_ipr_counts(gene_data)
            ipr_max_scores = get_ipr_max_scores(gene_data)
            
            # Collect all matching families
            matched_families = []
            final_rule = None
            
            # Evaluate all rules
            for rule_id, rule_data in rules_dict.items():
                if check_rule_match(ipr_counts, rule_data, ipr_max_scores):
                    matched_families.append(rule_data)
            
            # If any rules match, select the best candidate
            if matched_families:
                # Choose the highest-priority rule as the final assignment.
                # Rules appearing later in the rule file are often more specific, but not always.
                # Here we approximate specificity by the structure of the Required logic:
                # - rules with AND constraints and/or multiplicity (&N, N>1) are prioritized
                # - score is used as a secondary criterion among candidates of equal weight
                
                def calculate_rule_score_new(rule):
                    # 1) Rule categorization: class-a vs class-b
                    # - class-a: a single domain, or domains connected only by OR
                    # - class-b: contains AND, or multiplicity constraints (&N, N>1)
                    
                    if rule.get('name') == 'Others':
                        # Special handling for "Others": use a small weight/score as a last-resort fallback
                        return 0.1, 0.1 
                    
                    logic = rule.get('logic')
                    
                    is_b_class = False
                    hit_weight = 0
                    total_score = 0.0
                    
                    def analyze_logic(node):
                        nonlocal is_b_class, hit_weight, total_score
                        
                        if isinstance(node, str):
                            # Leaf node
                            domain = node
                            count = 1
                            if '&' in node:
                                try:
                                    count = int(node.split('&')[1])
                                except:
                                    pass
                            
                            real_ipr = domain.split('&')[0]
                            
                            if count > 1:
                                is_b_class = True
                                hit_weight += count
                                # Score aggregation: sum the top-N scores for multiplicity constraints (N=count).
                                # This requires consulting the raw hit list for the domain.
                                
                                # Collect and sort all scores for this domain
                                scores = []
                                if len(gene_data) > 1 and isinstance(gene_data[1], dict):
                                    for k, hits in gene_data[1].items():
                                        if k.split('&')[0] == real_ipr:
                                            if isinstance(hits, list):
                                                for hit in hits:
                                                    s = hit.get('score', 0)
                                                    if str(s).upper() == 'STRONG': s = 10.0
                                                    try: s = float(s)
                                                    except: s = 0.0
                                                    scores.append(s)
                                scores.sort(reverse=True)
                                # Take the top-N
                                total_score += sum(scores[:count])
                                
                            else:
                                # Single domain: class-a behavior unless wrapped by an AND upstream
                                hit_weight += 1
                                total_score += ipr_max_scores.get(real_ipr, 0)
                                
                        elif isinstance(node, dict):
                            op = node.get('op')
                            children = node.get('children', [])
                            
                            if op == 'AND':
                                is_b_class = True
                                for child in children:
                                    analyze_logic(child)
                            elif op == 'OR':
                                # OR: select the highest-scoring matched branch.
                                max_child_score = -1.0
                                best_child_weight = 0
                                child_is_b = False
                                
                                # OR may contain nested AND nodes. We therefore compute metrics per branch.
                                
                                # Save current accumulators
                                old_score = total_score
                                old_weight = hit_weight
                                old_b = is_b_class
                                
                                best_branch_score = 0
                                best_branch_weight = 0
                                best_branch_b = False
                                branch_matched = False
                                
                                for child in children:
                                    # Reset accumulators for this branch
                                    total_score = 0
                                    hit_weight = 0
                                    is_b_class = False
                                    
                                    # Evaluate whether this branch matches
                                    if evaluate_logic(child, ipr_counts):
                                        analyze_logic(child)
                                        branch_matched = True
                                        if total_score > best_branch_score:
                                            best_branch_score = total_score
                                            best_branch_weight = hit_weight
                                            best_branch_b = is_b_class
                                
                                # Restore accumulators and add the best branch contribution
                                total_score = old_score + best_branch_score
                                hit_weight = old_weight + best_branch_weight
                                if best_branch_b: is_b_class = True # If best branch behaves like class-b, propagate it
                                is_b_class = is_b_class or old_b

                    if logic:
                        analyze_logic(logic)
                    
                    # Return (weight, total_score). For class-a rules, normalize weight to 1.
                    final_weight = hit_weight if is_b_class else 1
                    return final_weight, total_score

                # Compute (weight, total_score) for all matched rules
                rule_metrics = []
                for rule in matched_families:
                    weight, score = calculate_rule_score_new(rule)
                    
                    # In score mode, force weight=1 for all non-"Others" rules so only scores are compared
                    if mode == 'score' and rule.get('name') != 'Others':
                        weight = 1.0
                        
                    rule_metrics.append({
                        'rule': rule,
                        'id': rule.get('id'), # Use ID as the unique identifier
                        'weight': weight,
                        'score': score,
                        'is_others': rule.get('name') == 'Others'
                    })
                
                # Prefer class-b rules (weight>1) over class-a rules (weight=1).
                # Note: "Others" uses weight=0.1 and is not treated as class-b.
                
                max_weight = max(m['weight'] for m in rule_metrics)
                
                # Keep rules with maximal weight
                candidates = [m for m in rule_metrics if m['weight'] == max_weight]
                
                # Break ties by score
                candidates.sort(key=lambda x: x['score'], reverse=True)
                
                # Pick the top-scoring candidate
                best_candidate = candidates[0]
                final_rule = best_candidate['rule']
                
                # Only retain the final classification (do not emit all matched families)
                # all_families = [rule["family"] for rule in matched_families]
                # other_families = ", ".join(all_families)
                
                # Build result dictionary.
                # We use Name instead of Family, while using ID internally to disambiguate rules.
                result[gene_id] = {
                    "name": final_rule["name"],
                    "family": final_rule["name"], # Populate Family using Name for consistency
                    "type": final_rule["type"],
                    "desc": final_rule.get("desc", []),
                    "other_family": "NA" # Do not display alternative families
                }
    
    # Persist results to disk (only for direct invocation, not for process_with_data)
    try:
        # Use legacy default paths only when result_dir is not set (backward compatibility)
        if result_dir is None:
            # Build JSON output path
            json_path = os.path.join(os.path.dirname(CURRENT_DIR), 'match.json')
            # Build TBL output path
            tbl_path = os.path.join(os.path.dirname(CURRENT_DIR), 'match_tbl.txt')
            
            # Write JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            # Write TBL
            with open(tbl_path, 'w', encoding='utf-8') as f:
                for gene_id, data in result.items():
                    desc_str = ';'.join(data['desc']) if data['desc'] else 'NA'
                    line = f"{gene_id}\t{data['name']}\t{data['family']}\t{data['type']}\t{desc_str}\t{data['other_family']}\n"
                    f.write(line)
                
        print("[OK] Results have been saved successfully")
    except Exception as e:
        print(f"[ERROR] Exception occurred while saving results - {str(e)}")
        
    return result

def process_with_data(result_directory, rule_file, filtered_data=None, spec_data=None, debug=False, mode='specific'):
    """
    Perform TF classification using in-memory inputs.
    
    Args:
        result_directory (str): Result directory path.
        rule_file (str): Rule file path.
        filtered_data (dict, optional): In-memory filtered data.
        spec_data (dict, optional): In-memory spec data.
        debug (bool): Whether to enable debug mode.
        mode (str): Classification mode: 'specific' (default) or 'score'.
    
    Returns:
        dict: Classification results.
    """
    # Initialize module state
    initialize_module(result_directory, filtered_data)
    
    # Initialize rules
    if not initialize_rules(rule_file):
        print("Failed to initialize rule file")
        return None
    
    # Obtain filtered_result
    if filtered_data is not None:
        current_filtered_result = filtered_data
    else:
        current_filtered_result = load_filtered_result()
    
    if current_filtered_result is None:
        print("Unable to obtain filtered data")
        return None
    
    # Handle spec data
    if spec_data is not None:
        print("Using in-memory spec data")
        current_spec_result = spec_data
    elif debug:
        # In debug mode, attempt to load spec data from disk
        pfamspec_file = os.path.join(result_directory, 'pfamspec.json')
        if os.path.exists(pfamspec_file):
            print(f"Loading spec data from file: {pfamspec_file}")
            with open(pfamspec_file, 'r') as f:
                current_spec_result = json.load(f)
        else:
            print("pfamspec.json not found in debug mode; using empty data")
            current_spec_result = {"result": {"match": []}}
    else:
        # In non-debug mode, use empty data
        current_spec_result = {"result": {"match": []}}
    
    # Merge results
    merged_dict = merge_results(current_filtered_result, current_spec_result)
    
    # Run classification
    classification_result = classify_genes(merged_dict, mode=mode)
    
    return classification_result

if __name__ == "__main__":
    # Merge filtered_result and result_spec
    merged_dict = merge_results(filtered_result, result_spec)
    
    # Classify using the merged dictionary
    classify_genes(merged_dict)
    print("[OK] Classification completed; results saved as match.json")
