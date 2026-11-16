import pytest
from parser.fasta_parser import FastaParser
from alignment.cpu_baseline import CPUBaselineAlignment

def test_chunking():
    parser = FastaParser("dummy.fasta", chunk_size=10, overlap=2)
    seq = "ATCGATCGATCGATCG"
    chunks = list(parser.chunk_sequence(seq))
    assert len(chunks) > 0

def test_cpu_alignment():
    engine = CPUBaselineAlignment(max_mismatches=1)
    results = engine.align_sequence("AAAC AAAA", "AAAA") # note spacing for clarity
    # Simple verification that alignment runs without errors
    assert isinstance(results, list)