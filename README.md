GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

Overview

Searching large genomes for unintended off-target CRISPR DNA mutations requires evaluating high volumes of sequence windows against a guide RNA (gRNA) within a mismatch threshold. GeneWeaver takes a staged engineering approach: establishing a verified FASTA ingestion pipeline and CPU baseline aligner first, then offloading window mismatch checks to CUDA GPUs while persisting benchmark runs and alignment hits in SQLite.

Requirements

Python 3.10+ (developed locally in a Windows .venv and validated in Google Colab)

NumPy for array operations and sequence handling

BioPython for streaming and chunking FASTA sequences

Numba (numba.cuda) for CUDA GPU kernel execution and GPU availability checks

SQLite3 (built into Python's standard library) for persisting benchmark runs and off-target hits

Textual / Rich for terminal UI and dashboard telemetry

NVIDIA GPU with CUDA support (such as an NVIDIA Tesla T4 in Google Colab) for GPU acceleration, with safe CPU fallback when CUDA is unavailable

Project Structure

geneweaver/
├── data/
│   ├── mock_genome.fasta       # Synthetic FASTA sequence generated for testing
│   └── geneweaver.db           # SQLite database storing benchmark runs & alignment hits
├── parser/
│   └── fasta_parser.py         # BioPython chunked FASTA streaming & sequence encoding
├── alignment/
│   ├── cpu_baseline.py         # CPU sliding-window aligner & benchmark execution helper
│   └── gpu_aligner.py          # Numba CUDA kernel, GPU availability check & CPU fallback
├── storage/
│   └── db.py                   # SQLite schema initialization & run/hit logging helpers
├── ui/
│   └── dashboard.py            # Interactive TUI / 
├── .gitignore                  # Ignores virtualenvs, bytecode caches, and local artifacts
└── requirements.txt            # Project dependencies


