import numpy as np

from alignment.dask_gpu_aligner import build_balanced_chunks, run_distributed_gpu_alignment


def _mock_cpu_slice_kernel(
    seq_slice: np.ndarray, grna_encoded: np.ndarray, max_mismatches: int
) -> list[tuple[int, int]]:
    """Deterministic local stand-in for run_gpu_alignment when testing without CUDA."""
    n = len(seq_slice)
    m = len(grna_encoded)
    hits: list[tuple[int, int]] = []
    for i in range(n - m + 1):
        mm = int(np.count_nonzero(seq_slice[i : i + m] != grna_encoded))
        if mm <= max_mismatches:
            hits.append((i, mm))
    return hits


def test_two_gpu_even_balancing_and_boundary_hit():
    rng = np.random.default_rng(42)
    seq_len = 1_000_000
    grna_len = 20
    max_mismatches = 2

    seq = rng.integers(0, 4, size=seq_len, dtype=np.uint8)
    grna = rng.integers(0, 4, size=grna_len, dtype=np.uint8)

    # Plant an exact hit straddling the 500,000 boundary (starts at 499,990 -> ends at 500,010)
    boundary_pos = 499_990
    seq[boundary_pos : boundary_pos + grna_len] = grna

    # Plant a 2-mismatch hit near the end
    mm_pos = 850_000
    seq[mm_pos : mm_pos + grna_len] = grna
    seq[mm_pos] = (grna[0] + 1) % 4
    seq[mm_pos + 10] = (grna[10] + 1) % 4

    chunks = build_balanced_chunks(
        seq_len=seq_len, grna_len=grna_len, num_workers=2, partitions_per_worker=1
    )
    assert len(chunks) == 2
    assert (chunks[0].logical_end - chunks[0].logical_start) == 500_000
    assert (chunks[1].logical_end - chunks[1].logical_start) == 500_000
    assert chunks[0].padded_end >= boundary_pos + grna_len

    expected = _mock_cpu_slice_kernel(seq, grna, max_mismatches)
    out = run_distributed_gpu_alignment(
        seq_encoded=seq,
        grna_encoded=grna,
        max_mismatches=max_mismatches,
        gpu_kernel_runner=_mock_cpu_slice_kernel,
        n_gpus=2,
        partitions_per_worker=1,
    )

    actual = [(h["position"], h["mismatches"]) for h in out["hits"]]
    assert out["workers_used"] == 2
    assert actual == expected
    assert (boundary_pos, 0) in actual
    assert (mm_pos, 2) in actual