# GeneWeaver: Week 1 — CPU Baseline & BioPython Pipeline

> **Project 1: "GeneWeaver" — GPU-Accelerated CRISPR Alignment Engine**  
> *Domain: Bioinformatics & High-Performance Computing (HPC)*

---

## Overview

Searching the human genome (3.2 billion base pairs) for unintended "off-target" CRISPR DNA mutations requires massive compute. Standard Python string matching (regex or simple loops) takes days to process a single genome, forcing researchers to rely on expensive, proprietary C++ software clusters.

**Week 1** establishes our foundational bioinformatics ingestion pipeline and a pure Python/NumPy CPU baseline aligner. Before introducing GPU acceleration, this module measures baseline processing performance and validates our integer-encoded sequence chunking logic.

---

## Repository Structure

```text
geneweaver_week1/
├── data/
│   └── mock_genome.fasta       # Synthetic FASTA file generated for local testing
├── src/
│   ├── parser.py               # BioPython parser & 8-bit integer DNA encoder
│   └── cpu_aligner.py          # Pure Python/NumPy sliding-window alignment engine
├── run_week1.py                # Execution and benchmarking script
└── requirements.txt            # Project dependencies