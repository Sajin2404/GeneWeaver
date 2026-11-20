import os
import random
from Bio import SeqIO
import numpy as np

NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_sequence(sequence: str) -> np.ndarray:
    """Encodes a DNA sequence string into a uint8 NumPy array (A=0, C=1, G=2, T=3, other=4)."""
    return np.array(
        [NUCLEOTIDE_MAP.get(base.upper(), 4) for base in sequence],
        dtype=np.uint8,
    )


class FastaParser:
    def __init__(self, filepath: str, chunk_size: int = 10000, overlap: int = 20):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse_genomes(self):
        """Yields (record_id, sequence_str) from the FASTA file."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"FASTA file not found at: {self.filepath}")

        for record in SeqIO.parse(self.filepath, "fasta"):
            yield record.id, str(record.seq).upper()

    def chunk_sequence(self, sequence: str):
        """Splits a DNA sequence into overlapping chunks, yielding (start_idx, chunk_str)."""
        seq_len = len(sequence)
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, seq_len, step):
            end = min(start + self.chunk_size, seq_len)
            yield start, sequence[start:end]


def parse_fasta_chunks(filepath: str, chunk_size: int = 500_000, overlap: int = 20):
    """Yields (record_id, encoded_chunk_uint8_array) tuples for aligners."""
    parser = FastaParser(filepath=filepath, chunk_size=chunk_size, overlap=overlap)
    for record_id, sequence in parser.parse_genomes():
        for _, chunk_str in parser.chunk_sequence(sequence):
            yield record_id, encode_sequence(chunk_str)


def generate_mock_fasta(filepath: str, length_bp: int = 2_000_000, seed: int = 42) -> str:
    """Generates a synthetic mock FASTA genome file with length_bp base pairs."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rng = random.Random(seed)
    bases = ["A", "C", "G", "T"]
    seq = "".join(rng.choices(bases, k=length_bp))

    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(">chr_mock_1 GeneWeaver synthetic test genome\n")
        for i in range(0, len(seq), 80):
            handle.write(seq[i : i + 80] + "\n")

    return filepath


if __name__ == "__main__":
    print("FASTA Parser module initialized successfully.")