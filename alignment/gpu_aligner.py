import time
import numpy as np
from typing import List, Tuple
from numba import cuda

@cuda.jit
def crispr_align_kernel(target, grna, max_mismatches, grna_len, target_len, out_mismatches, out_hits):
    """
    CUDA Kernel: Assigns 1 thread per sequence alignment position in the genome chunk.
    """
    idx = cuda.grid(1)
    
    if idx <= (target_len - grna_len):
        mismatches = 0
        for j in range(grna_len):
            if target[idx + j] != grna[j]:
                mismatches += 1
                if mismatches > max_mismatches:
                    break
                    
        if mismatches <= max_mismatches:
            out_mismatches[idx] = mismatches
            out_hits[idx] = 1
        else:
            out_mismatches[idx] = 255
            out_hits[idx] = 0

def is_cuda_available() -> bool:
    """Checks if an NVIDIA GPU and CUDA drivers are accessible."""
    try:
        return cuda.is_available()
    except Exception:
        return False

def run_gpu_alignment(target_seq: np.ndarray, grna_seq: np.ndarray, max_mismatches: int = 3) -> Tuple[List[Tuple[int, int]], float, str]:
    """
    Manages Host (CPU RAM) <-> Device (GPU VRAM) memory transfers and launches the CUDA kernel.
    Returns: (matches, elapsed_time, execution_mode)
    """
    if not is_cuda_available():
        raise RuntimeError(
            "No local NVIDIA GPU / CUDA driver detected. "
            "Run this script on Google Colab (T4 GPU runtime) to execute the @cuda.jit kernel."
        )

    start_time = time.perf_counter()
    
    target_len = len(target_seq)
    grna_len = len(grna_seq)
    num_positions = target_len - grna_len + 1
    
    if num_positions <= 0:
        return [], 0.0, "CUDA"

    # 1. Allocate output buffers on Host (CPU RAM)
    out_mismatches_host = np.full(num_positions, 255, dtype=np.uint8)
    out_hits_host = np.zeros(num_positions, dtype=np.uint8)

    # 2. Transfer arrays from Host (RAM) to Device (GPU VRAM)
    d_target = cuda.to_device(target_seq)
    d_grna = cuda.to_device(grna_seq)
    d_out_mismatches = cuda.to_device(out_mismatches_host)
    d_out_hits = cuda.to_device(out_hits_host)

    # 3. Configure 1D CUDA Thread Block & Grid Dimensions
    threads_per_block = 256
    blocks_per_grid = (num_positions + (threads_per_block - 1)) // threads_per_block

    # 4. Launch JIT-compiled CUDA Kernel
    crispr_align_kernel[blocks_per_grid, threads_per_block](
        d_target, d_grna, max_mismatches, grna_len, target_len, d_out_mismatches, d_out_hits
    )
    cuda.synchronize()  # Wait for all GPU threads to finish

    # 5. Transfer results back from Device (VRAM) to Host (RAM)
    out_mismatches_host = d_out_mismatches.copy_to_host()
    out_hits_host = d_out_hits.copy_to_host()

    # 6. Extract matching offsets
    hit_indices = np.where(out_hits_host == 1)[0]
    results = [(int(idx), int(out_mismatches_host[idx])) for idx in hit_indices]

    elapsed_time = time.perf_counter() - start_time
    return results, elapsed_time, "CUDA"