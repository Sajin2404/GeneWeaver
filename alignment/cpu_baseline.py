import sys
import os

# Add the project root directory to sys.path so sibling packages can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
from parser.fasta_parser import FastaParser

class CPUBaselineAlignment:
    def __init__(self, max_mismatches: int = 4):
        self.max_mismatches = max_mismatches

    def align_sequence(self, target_dna: str, g_rna: str) -> list:
        """
        Scans a target DNA string against a CRISPR guide RNA (gRNA)
        using a sliding window and counts mismatches.
        """
        g_len = len(g_rna)
        t_len = len(target_dna)
        matches = []

        if t_len < g_len:
            return matches

        # Convert strings to ASCII byte/numpy arrays for efficient vectorized comparison
        target_arr = np.frombuffer(target_dna.encode('ascii'), dtype=np.uint8)
        g_arr = np.frombuffer(g_rna.encode('ascii'), dtype=np.uint8)

        # Sequential sliding window scan across the genome
        for i in range(t_len - g_len + 1):
            window = target_arr[i:i + g_len]
            mismatches = np.sum(window != g_arr)
            
            if mismatches <= self.max_mismatches:
                matches.append({
                    "position": i,
                    "mismatches": int(mismatches),
                    "sequence": target_dna[i:i + g_len]
                })

        return matches

    def benchmark_scan(self, genome_seq: str, g_rna: str) -> float:
        """Measures CPU execution time for a full scan and returns elapsed seconds."""
        print(f"[CPU Baseline] Starting scan on sequence of length {len(genome_seq):,} bp...")
        
        start_time = time.perf_counter()
        results = self.align_sequence(genome_seq, g_rna)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        print(f"[CPU Baseline] Completed in {elapsed:.4f} seconds. Found {len(results)} potential off-target matches.")
        return elapsed

if __name__ == "__main__":
    # Test stub using synthetic genomic data
    print("Running CPU Baseline test...")
    dummy_genome = "AGCTTAGCTAGCTCGATCGATCGATCGATCGATCGATC" * 10000  # ~380 KB test genome
    test_grna = "GATCGATCGA"
    
    engine = CPUBaselineAlignment(max_mismatches=2)
    engine.benchmark_scan(dummy_genome, test_grna)