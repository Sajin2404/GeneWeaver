import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DB_PATH = Path("geneweaver.db")


def init_benchmark_tables(conn: sqlite3.Connection) -> None:
    """Create benchmark_runs and alignment_hits tables if they do not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            environment TEXT NOT NULL,
            device_name TEXT NOT NULL,
            execution_mode TEXT NOT NULL,
            sequence_length INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL,
            grna_seq TEXT NOT NULL,
            max_mismatches INTEGER NOT NULL,
            cpu_time_sec REAL,
            gpu_time_sec REAL,
            speedup REAL,
            total_hits INTEGER NOT NULL,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alignment_hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            mismatches INTEGER NOT NULL,
            aligned_seq TEXT,
            chunk_index INTEGER DEFAULT 0,
            FOREIGN KEY (run_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def log_benchmark_run(
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
) -> int:
    """Persist a benchmark run and its alignment hits into SQLite. Returns run_id."""
    normalized_hits: List[Tuple[int, int, Optional[str], int]] = []
    for item in hits:
        if isinstance(item, dict):
            pos = int(item["position"])
            mm = int(item["mismatches"])
            seq = item.get("sequence")
            chk = int(item.get("chunk_index", 0))
        else:
            # Supports (position, mismatches) tuples from gpu_aligner
            pos = int(item[0])
            mm = int(item[1])
            seq = item[2] if len(item) > 2 else None
            chk = int(item[3]) if len(item) > 3 else 0
        normalized_hits.append((pos, mm, seq, chk))

    speedup: Optional[float] = None
    if cpu_time_sec and gpu_time_sec and gpu_time_sec > 0:
        speedup = round(cpu_time_sec / gpu_time_sec, 4)

    effective_chunk_size = chunk_size if chunk_size is not None else sequence_length
    created_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        init_benchmark_tables(conn)

        cur = conn.execute(
            """
            INSERT INTO benchmark_runs (
                created_at, environment, device_name, execution_mode,
                sequence_length, chunk_size, grna_seq, max_mismatches,
                cpu_time_sec, gpu_time_sec, speedup, total_hits, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                environment,
                device_name,
                execution_mode,
                sequence_length,
                effective_chunk_size,
                grna_seq,
                max_mismatches,
                cpu_time_sec,
                gpu_time_sec,
                speedup,
                len(normalized_hits),
                notes,
            ),
        )
        run_id = int(cur.lastrowid)

        if normalized_hits:
            conn.executemany(
                """
                INSERT INTO alignment_hits (run_id, position, mismatches, aligned_seq, chunk_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(run_id, pos, mm, seq, chk) for pos, mm, seq, chk in normalized_hits],
            )

        conn.commit()
        return run_id
    finally:
        conn.close()


def export_dashboard_json(
    output_path: Path = Path("web/benchmark_data.json"),
    db_path: Path = DEFAULT_DB_PATH,
    limit: int = 25,
) -> Dict[str, Any]:
    """Export recent benchmark runs and their hits to a JSON file for web/index.html."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        init_benchmark_tables(conn)
        runs_rows = conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        runs_payload: List[Dict[str, Any]] = []
        for r in runs_rows:
            hit_rows = conn.execute(
                """
                SELECT position, mismatches, aligned_seq, chunk_index
                FROM alignment_hits
                WHERE run_id = ?
                ORDER BY position ASC
                """,
                (r["id"],),
            ).fetchall()

            runs_payload.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "environment": r["environment"],
                    "device_name": r["device_name"],
                    "execution_mode": r["execution_mode"],
                    "sequence_length": r["sequence_length"],
                    "chunk_size": r["chunk_size"],
                    "grna_seq": r["grna_seq"],
                    "max_mismatches": r["max_mismatches"],
                    "cpu_time_sec": r["cpu_time_sec"],
                    "gpu_time_sec": r["gpu_time_sec"],
                    "speedup": r["speedup"],
                    "total_hits": r["total_hits"],
                    "notes": r["notes"],
                    "hits": [dict(h) for h in hit_rows],
                }
            )

        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "runs": runs_payload,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    finally:
        conn.close()