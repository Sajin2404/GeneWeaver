from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dask.distributed import Client, LocalCluster

try:
    from dask_cuda import LocalCUDACluster  # type: ignore
except ImportError:  # pragma: no cover
    LocalCUDACluster = None

try:
    from numba import cuda  # type: ignore
except ImportError:  # pragma: no cover
    cuda = None


@dataclass
class ChunkSpec:
    chunk_index: int
    logical_start: int
    logical_end: int
    padded_start: int
    padded_end: int


def detect_gpu_count() -> int:
    """Return the number of visible CUDA GPUs, or 0 if unavailable."""
    if cuda is None:
        return 0
    try:
        return len(cuda.gpus)
    except Exception:
        return 0


def build_balanced_chunks(
    seq_len: int,
    grna_len: int,
    num_workers: int,
    partitions_per_worker: int = 1,
) -> List[ChunkSpec]:
    """
    Partition [0, seq_len) evenly across num_workers * partitions_per_worker slices,
    adding (grna_len - 1) lookahead bases so boundary windows are never missed.
    """
    if seq_len < grna_len or grna_len <= 0:
        return []

    total_partitions = max(1, num_workers * max(1, partitions_per_worker))
    total_partitions = min(total_partitions, seq_len - grna_len + 1)

    base_size, remainder = divmod(seq_len, total_partitions)
    overlap = max(0, grna_len - 1)

    specs: List[ChunkSpec] = []
    cursor = 0
    for idx in range(total_partitions):
        span = base_size + (1 if idx < remainder else 0)
        logical_start = cursor
        logical_end = min(seq_len, logical_start + span)
        padded_start = logical_start
        padded_end = min(seq_len, logical_end + overlap)

        if padded_end - padded_start >= grna_len:
            specs.append(
                ChunkSpec(
                    chunk_index=idx,
                    logical_start=logical_start,
                    logical_end=logical_end,
                    padded_start=padded_start,
                    padded_end=padded_end,
                )
            )

        cursor = logical_end
        if cursor >= seq_len:
            break

    return specs


def _worker_align_slice(
    seq_slice: np.ndarray,
    grna_encoded: np.ndarray,
    max_mismatches: int,
    spec: ChunkSpec,
    gpu_kernel_runner: Callable[[np.ndarray, np.ndarray, int], List[Tuple[int, int]]],
) -> Dict[str, Any]:
    """
    Executed on a Dask worker. Runs the single-GPU aligner on seq_slice,
    filters to hits whose start position belongs to [logical_start, logical_end),
    and converts local slice offsets to global sequence indices.
    """
    worker_gpu_id = None
    if cuda is not None:
        try:
            worker_gpu_id = cuda.get_current_device().id
        except Exception:
            worker_gpu_id = None

    raw_hits = gpu_kernel_runner(seq_slice, grna_encoded, max_mismatches)

    adjusted_hits: List[Dict[str, Any]] = []
    for item in raw_hits:
        if isinstance(item, dict):
            local_pos = int(item["position"])
            mismatches = int(item["mismatches"])
        else:
            local_pos = int(item[0])
            mismatches = int(item[1])

        global_pos = spec.padded_start + local_pos
        if spec.logical_start <= global_pos < spec.logical_end:
            adjusted_hits.append(
                {
                    "position": global_pos,
                    "mismatches": mismatches,
                    "chunk_index": spec.chunk_index,
                    "gpu_id": worker_gpu_id,
                }
            )

    return {
        "chunk_index": spec.chunk_index,
        "logical_start": spec.logical_start,
        "logical_end": spec.logical_end,
        "slice_length": int(len(seq_slice)),
        "gpu_id": worker_gpu_id,
        "hits": adjusted_hits,
    }


def run_distributed_gpu_alignment(
    seq_encoded: np.ndarray,
    grna_encoded: np.ndarray,
    max_mismatches: int,
    gpu_kernel_runner: Callable[[np.ndarray, np.ndarray, int], List[Tuple[int, int]]],
    n_gpus: Optional[int] = None,
    partitions_per_worker: int = 1,
) -> Dict[str, Any]:
    """
    Coordinate sequence alignment across multiple GPUs using Dask.
    When n_gpus == 2, spawns 2 GPU workers and balances sequence slices evenly.
    """
    seq_arr = np.asarray(seq_encoded, dtype=np.uint8)
    grna_arr = np.asarray(grna_encoded, dtype=np.uint8)

    detected = detect_gpu_count()
    target_gpus = n_gpus if n_gpus is not None else max(1, detected)

    if len(seq_arr) < len(grna_arr) or len(grna_arr) == 0:
        return {
            "requested_gpus": target_gpus,
            "workers_used": 0,
            "chunks_processed": 0,
            "bases_per_gpu": {},
            "chunk_summaries": [],
            "hits": [],
        }

    specs = build_balanced_chunks(
        seq_len=len(seq_arr),
        grna_len=len(grna_arr),
        num_workers=target_gpus,
        partitions_per_worker=partitions_per_worker,
    )

    if LocalCUDACluster is not None and detected > 0:
        visible_devices = ",".join(str(i) for i in range(min(target_gpus, detected)))
        cluster = LocalCUDACluster(
            n_workers=min(target_gpus, detected),
            threads_per_worker=1,
            CUDA_VISIBLE_DEVICES=visible_devices,
        )
    else:
        cluster = LocalCluster(
            n_workers=target_gpus,
            threads_per_worker=1,
            processes=True,
        )

    try:
        with Client(cluster) as client:
            futures = [
                client.submit(
                    _worker_align_slice,
                    np.ascontiguousarray(seq_arr[spec.padded_start : spec.padded_end]),
                    grna_arr,
                    max_mismatches,
                    spec,
                    gpu_kernel_runner,
                    pure=False,
                )
                for spec in specs
            ]
            chunk_results = client.gather(futures)
    finally:
        cluster.close()

    all_hits: List[Dict[str, Any]] = []
    bases_per_gpu: Dict[Any, int] = {}
    for res in chunk_results:
        all_hits.extend(res["hits"])
        key = res["gpu_id"] if res["gpu_id"] is not None else f"chunk_{res['chunk_index']}"
        bases_per_gpu[key] = bases_per_gpu.get(key, 0) + (res["logical_end"] - res["logical_start"])

    all_hits.sort(key=lambda h: (h["position"], h["mismatches"]))
    active_workers = min(target_gpus, len(chunk_results))

    return {
        "requested_gpus": target_gpus,
        "workers_used": active_workers,
        "chunks_processed": len(chunk_results),
        "bases_per_gpu": bases_per_gpu,
        "chunk_summaries": [
            {
                "chunk_index": r["chunk_index"],
                "logical_start": r["logical_start"],
                "logical_end": r["logical_end"],
                "slice_length": r["slice_length"],
                "gpu_id": r["gpu_id"],
                "hit_count": len(r["hits"]),
            }
            for r in chunk_results
        ],
        "hits": all_hits,
    }