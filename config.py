import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Alignment parameters
CHUNK_SIZE = 10000  # Number of base pairs per processing chunk
OVERLAP = 20        # Overlap between chunks to prevent boundary splits
MAX_MISMATCHES = 4  # Maximum allowed mismatches for off-target detection