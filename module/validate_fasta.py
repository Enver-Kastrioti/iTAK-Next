#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTAK Next FASTA validation module.

Validates whether the input FASTA file is well-formed, contains protein sequences,
and does not include disallowed characters (e.g., '*').
"""

import os
import re
from pathlib import Path
from Bio import SeqIO

class FastaValidator:
    """FASTA validator."""
    
    def __init__(self):
        self.errors = []
        self.protein_count = 0
        
        # Canonical amino-acid alphabet
        self.valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        # Extended amino-acid alphabet (including ambiguity codes)
        self.extended_amino_acids = set('ACDEFGHIKLMNPQRSTVWYXBZJU')
        

    
    def validate_fasta_format(self, fasta_file):
        try:
            sequences = list(SeqIO.parse(fasta_file, "fasta"))
            if not sequences:
                self.errors.append("The file contains no FASTA records")
                return False
            
            sequence_count = len(sequences)
            print(f"  [#] Found {sequence_count} sequences")
            return True
            
        except Exception as e:
            self.errors.append(f"FASTA parsing error: {str(e)}")
            return False
    
    def validate_protein_sequences(self, fasta_file):

        try:
            sequences = list(SeqIO.parse(fasta_file, "fasta"))
            self.protein_count = 0
            
            for record in sequences:
                seq_str = str(record.seq).upper()
                
                # Check for disallowed characters
                if '*' in seq_str:
                    self.errors.append(f"Sequence {record.id} contains a disallowed character: *")
                    continue
                
                # Verify that the sequence is protein-like
                seq_chars = set(seq_str)
                
                # Estimate the fraction of valid amino-acid characters
                valid_chars = seq_chars & self.extended_amino_acids
                total_chars = len(seq_chars)
                
                if total_chars == 0:
                    self.errors.append(f"Sequence {record.id} is empty")
                    continue
                
                valid_ratio = len(valid_chars) / total_chars
                
                if valid_ratio < 0.8:  # Require at least 80% valid amino-acid characters
                    invalid_chars = seq_chars - self.extended_amino_acids
                    self.errors.append(f"Sequence {record.id} contains non-protein characters: {', '.join(invalid_chars)}")
                    continue
                
                # Protein sequence passes validation
                self.protein_count += 1
            
            if self.protein_count == 0:
                self.errors.append("No valid protein sequences were found")
                return False
            
            print(f"  [#] Found {self.protein_count} valid protein sequences")
            return True
            
        except Exception as e:
            self.errors.append(f"Error during sequence validation: {str(e)}")
            return False

    def validate_sequences(self, sequences):
        try:
            self.protein_count = 0
            for item in sequences:
                seq_str = str(item if isinstance(item, str) else item.get('sequence', '')).upper()
                if '*' in seq_str:
                    self.errors.append("Sequence contains a disallowed character: *")
                    continue
                seq_chars = set(seq_str)
                if len(seq_chars) == 0:
                    self.errors.append("Empty sequence")
                    continue
                valid_chars = seq_chars & self.extended_amino_acids
                valid_ratio = len(valid_chars) / len(seq_chars)
                if valid_ratio < 0.8:
                    invalid_chars = seq_chars - self.extended_amino_acids
                    self.errors.append(f"Sequence contains non-protein characters: {', '.join(invalid_chars)}")
                    continue
                self.protein_count += 1
            if self.protein_count == 0:
                self.errors.append("No valid protein sequences were found")
                return False
            return True
        except Exception as e:
            self.errors.append(f"Error during sequence validation: {str(e)}")
            return False
    

    
    def run_full_validation(self, fasta_file):

        print(f"[#] Starting FASTA validation: {fasta_file}")
        
        # Reset error state
        self.errors = []
        
        # 1) Basic FASTA format validation
        if not self.validate_fasta_format(fasta_file):
            print("[WARN] FASTA format validation failed")
            return False
        
        # 2) Protein-sequence validation (including '*' check and protein count)
        if not self.validate_protein_sequences(fasta_file):
            print("[WARN] Protein sequence validation failed")
            return False
        
        print("[#] FASTA validation passed")
        return True
    


def main():
    """Entry point."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python validate_fasta.py <fasta_file>")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    validator = FastaValidator()
    success = validator.run_full_validation(fasta_file)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
