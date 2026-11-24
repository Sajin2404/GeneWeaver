import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from storage.db import DEFAULT_DB_PATH, export_dashboard_json, log_benchmark_run


def record_and_export(
    *,
    environment: str,
    device_name: str,
    execution_mode: str,
    sequence_length: int,
    grna_seq: str,
    max_mismatches: int,
    hits: Iterable[Any],
    cpu_time_sec: Optional[float] = None,
    gpu_time_sec: Optional[float] = None,
    chunk_size: Optional[int] = None,
    notes: str = "",
    db_path: Path = DEFAULT_DB_PATH,
    json_output_path: Path = Path("web/benchmark_data.json"),
    export_limit: int = 25,
) -> tuple[int, dict[str, Any]]:
    """Log a benchmark run to SQLite and export updated JSON for the web dashboard."""
    run_id = log_benchmark_run(
        environment=environment,
        device_name=device_name,
        execution_mode=execution_mode,
        sequence_length=sequence_length,
        grna_seq=grna_seq,
        max_mismatches=max_mismatches,
        hits=hits,
        cpu_time_sec=cpu_time_sec,
        gpu_time_sec=gpu_time_sec,
        chunk_size=chunk_size,
        notes=notes,
        db_path=db_path,
    )
    payload = export_dashboard_json(
        output_path=json_output_path,
        db_path=db_path,
        limit=export_limit,
    )
    return run_id, payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log a GeneWeaver benchmark run to SQLite and export web/benchmark_data.json."
    )
    parser.add_argument("--environment", default="Local VS Code")
    parser.add_argument("--device-name", default="CPU")
    parser.add_argument("--execution-mode", default="CPU Baseline")
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--grna-seq", required=True)
    parser.add_argument("--max-mismatches", type=int, default=3)
    parser.add_argument("--cpu-time", type=float, default=None)
    parser.add_argument("--gpu-time", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--hits-json", default="[]", help="JSON list of hits (dicts or [pos, mm] pairs)")
    parser.add_argument("--notes", default="")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--json-out", type=Path, default=Path("web/benchmark_data.json"))
    parser.add_argument("--limit", type=int, default=25)

    args = parser.parse_args()
    hits = json.loads(args.hits_json)

    run_id, payload = record_and_export(
        environment=args.environment,
        device_name=args.device_name,
        execution_mode=args.execution_mode,
        sequence_length=args.sequence_length,
        grna_seq=args.grna_seq,
        max_mismatches=args.max_mismatches,
        hits=hits,
        cpu_time_sec=args.cpu_time,
        gpu_time_sec=args.gpu_time,
        chunk_size=args.chunk_size,
        notes=args.notes,
        db_path=args.db_path,
        json_output_path=args.json_out,
        export_limit=args.limit,
    )
    print(f"Saved run_id={run_id} to {args.db_path} and exported {len(payload['runs'])} run(s) to {args.json_out}")


if __name__ == "__main__":
    main()