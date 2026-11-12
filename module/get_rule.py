import os
import json
import argparse
import re


def _iter_clean_lines(block: str):
    """Yield non-empty, non-comment lines from a block."""
    for line in block.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        yield line


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        t = self.peek()
        if t:
            self.pos += 1
        return t

    def parse(self):
        if not self.tokens:
            return None
        try:
            return self.parse_expression()
        except Exception:
            return None

    def parse_expression(self):
        # Expression = Term { : Term }
        left = self.parse_term()
        while self.peek() == ':':
            self.consume() # eat ':'
            right = self.parse_term()
            if right is None:
                continue
            left_node = {'op': 'OR', 'children': [left, right]}
            if isinstance(left, dict) and left.get('op') == 'OR':
                left_node['children'] = left['children'] + [right]
            left = left_node
        return left

    def parse_term(self):
        # Term = Factor { # Factor }
        left = self.parse_factor()
        while self.peek() == '#':
            self.consume() # eat '#'
            right = self.parse_factor()
            if right is None:
                continue
            left_node = {'op': 'AND', 'children': [left, right]}
            if isinstance(left, dict) and left.get('op') == 'AND':
                left_node['children'] = left['children'] + [right]
            left = left_node
        return left

    def parse_factor(self):
        token = self.peek()
        if token == '(':
            self.consume()
            expr = self.parse_expression()
            if self.peek() == ')':
                self.consume()
            return expr
        elif token and token not in (')', ':', '#'):
            return self.consume()
        return None


def parse_required_logic(text):
    if not text or text == 'NA':
        return None
    # Tokenize: split by ( ) : # but keep them
    regex = r'([():#])'
    parts = re.split(regex, text)
    tokens = [p.strip() for p in parts if p.strip()]
    parser = Parser(tokens)
    return parser.parse()


def parse_rule_file(file_path):
    rules_dict = {}
    # Validate input path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The input file '{file_path}' was not found.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Concatenate lines and split into blocks using the '//' delimiter
        blocks = ''.join(file.readlines()).strip().split('//')
        
        for block in blocks:
            # Skip empty blocks
            if not block.strip():
                continue
            # Convert the block into non-empty, non-comment lines
            lines = list(_iter_clean_lines(block))
            if not lines:
                continue
            # Parse key/value pairs within the block
            rule = {}
            for line in lines:
                # Split only at the first ':' to preserve values containing ':'
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                rule[key.strip()] = value.strip()
            
            # Required fields
            id_value = rule.get('ID', None)
            if not id_value:
                # Not a rule block or missing ID
                continue
            
            # Build the normalized rule structure
            required_raw = rule.get('Required', '')
            
            # Parse the Required expression into a logic tree
            logic_tree = parse_required_logic(required_raw)
            
            # For consistency, always use the logic-mode representation.
            
            # Optional: extract all domains referenced in the Required expression for fast pre-filtering
            # flat_domains = [r for r in required_raw.replace(':', '#').replace('(', '').replace(')', '').split('#') if r]
            
            rule_data = {
                'id': id_value, # Add the ID field
                'name': rule.get('Name', 'NA'),
                'family': rule.get('Family', 'NA'),
                'type': rule.get('Type', 'NA'),
                'desc': [] if rule.get('Desc', 'NA') == 'NA' else [rule.get('Desc')],
                'mode': ['logic'], # Use logic mode uniformly
                'logic': logic_tree,
                'required': [], # Flat lists are no longer used
                'forbidden': rule.get('Forbidden', '').split(':') if 'Forbidden' in rule else []
            }
            
            rules_dict[id_value] = rule_data
    
    return rules_dict


def parse_score_thresholds(file_path):
    """
    Parse global accession-specific score thresholds defined in the rule file.

    Example line:
      Score:PS50863(10):cd10017(15)

    Notes:
    - Thresholds are global and independent of family assignment.
    - If an accession appears multiple times across blocks, the maximum threshold is retained
      (i.e., the most stringent criterion).

    Returns:
      dict, e.g. {"PS50863": 10.0, "cd10017": 15.0}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The input file '{file_path}' was not found.")
    
    thresholds = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        blocks = ''.join(file.readlines()).strip().split('//')
        for block in blocks:
            if not block.strip():
                continue
            # Scan for 'Score:' lines
            for line in _iter_clean_lines(block):
                if not line.startswith('Score:'):
                    continue
                # Remove the prefix and split each entry by ':'
                content = line[len('Score:'):].strip()
                parts = [p for p in content.split(':') if p]
                for part in parts:
                    part = part.strip()
                    # Format: PS50863(10) or cd10017(15)
                    if '(' in part and part.endswith(')'):
                        try:
                            name, val_str = part.split('(', 1)
                            name = name.strip()
                            val = float(val_str[:-1])  # Strip trailing ')'
                            if name:
                                # If duplicated, keep the maximum threshold (more stringent)
                                thresholds[name] = max(val, thresholds.get(name, float('-inf')))
                        except Exception:
                            # Ignore unparsable entries
                            continue
    return thresholds


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Parse a rule file and generate JSON output')
    parser.add_argument('-i', '--input', required=True, help='Path to the input rule file')
    parser.add_argument('-o', '--output', required=True, help='Path to the output directory')
    parser.add_argument('--with-score', action='store_true', help='Also output global accession-specific score thresholds')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_dir = args.output
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, "getrule.json")
    
    try:
        # Parse rule file
        rules = parse_rule_file(input_file)
        
        # Write results to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        
        # Optional: write accession-specific score thresholds as a separate JSON file
        if args.with_score:
            score_file = os.path.join(output_dir, "score_thresholds.json")
            score_thresholds = parse_score_thresholds(input_file)
            with open(score_file, 'w', encoding='utf-8') as f:
                json.dump(score_thresholds, f, indent=2, ensure_ascii=False)
        
        print(f"Rules were parsed successfully and saved to {output_file}")
        if args.with_score:
            print(f"Score thresholds were saved to {score_file}")
        
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()
