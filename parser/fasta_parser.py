from Bio import SeqIO
import os

class FastaParser:
    def __init__(self, filepath: str, chunk_size: int = 10000, overlap: int = 20):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse_genomes(self):
        """Yields sequence records from the FASTA file."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"FASTA file not found at: {self.filepath}")
        
        for record in SeqIO.parse(self.filepath, "fasta"):
            yield record.id, str(record.seq).upper()

    def chunk_sequence(self, sequence: str):
        """Splits a long DNA sequence into overlapping chunks."""
        seq_len = len(sequence)
        for start in range(0, seq_len, self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, seq_len)
            yield start, sequence[start:end]

if __name__ == "__main__":
    # Quick test stub
    print("FASTA Parser module initialized successfully.")