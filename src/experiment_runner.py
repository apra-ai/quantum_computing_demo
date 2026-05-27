"""Batch runner for CPU and Grover runtime evaluation across multiple qubit sizes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from classical_bruteforce import search_hidden_key
from grover import GroverRunResult, run_grover_simulator
from ibm_backend import run_grover_ibm_or_fallback
from visualization import plot_runtime_scaling


DEFAULT_N_QUBITS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36]
DEFAULT_OUTPUT_DIR = Path("outputs") / "evaluation_runtimes"
CSV_COLUMNS = [
    "n_qubits",
    "search_space",
    "target",
    "cpu_checked_candidates",
    "cpu_time_seconds",
    "grover_iterations",
    "grover_time_seconds",
    "mode_used",
    "backend",
    "shots",
    "job_id",
    "success_probability",
    "most_frequent_state",
    "most_frequent_count",
    "target_was_top_result",
    "error",
]

INT_FIELDS = {
    "n_qubits",
    "search_space",
    "cpu_checked_candidates",
    "grover_iterations",
    "shots",
    "most_frequent_count",
}

FLOAT_FIELDS = {
    "cpu_time_seconds",
    "grover_time_seconds",
    "success_probability",
}

BOOL_FIELDS = {"target_was_top_result"}


@dataclass(frozen=True)
class ExperimentRow:
    """One experiment row that is written to CSV and JSON."""

    n_qubits: int
    search_space: int
    target: str
    cpu_checked_candidates: int | None
    cpu_time_seconds: float | None
    grover_iterations: int | None
    grover_time_seconds: float | None
    mode_used: str
    backend: str | None
    shots: int
    job_id: str | None
    success_probability: float | None
    most_frequent_state: str | None
    most_frequent_count: int | None
    target_was_top_result: bool | None
    error: str


def parse_bool(value: str) -> bool:
    """Parse a command-line boolean written as true or false."""

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""

    parser = argparse.ArgumentParser(
        description="Run repeated CPU and Grover experiments across multiple qubit sizes."
    )
    parser.add_argument("--mode", choices=["simulator", "ibm"], default="simulator")
    parser.add_argument(
        "--cpu-mode",
        choices=["run", "skip"],
        default="run",
        help="Choose whether the classical CPU brute-force part should run or be skipped.",
    )
    parser.add_argument("--backend", default=None, help="Optional IBM backend name such as ibm_kingston.")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--max-qubits", type=int, default=36)
    parser.add_argument(
        "--max-cpu-qubits",
        type=int,
        default=24,
        help="Run real CPU brute force only up to this qubit count to avoid extreme host CPU load.",
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit qubit sizes to run, for example --qubits 36 or --qubits 8 12 16.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-mode", choices=["last", "random"], default="last")
    parser.add_argument("--stop-on-error", type=parse_bool, default=False)
    parser.add_argument("--resume", type=parse_bool, default=False)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate the runner arguments."""

    if args.shots <= 0:
        raise ValueError("--shots must be a positive integer.")
    if args.max_qubits < 2:
        raise ValueError("--max-qubits must be at least 2.")
    if args.max_cpu_qubits < 2:
        raise ValueError("--max-cpu-qubits must be at least 2.")
    if args.mode == "simulator" and args.backend:
        raise ValueError("--backend is only valid together with --mode ibm.")
    if args.qubits is not None and any(n_qubits < 2 for n_qubits in args.qubits):
        raise ValueError("All values passed to --qubits must be at least 2.")


def qubit_schedule(max_qubits: int, explicit_qubits: list[int] | None = None) -> list[int]:
    """Return the configured qubit schedule capped by the requested maximum."""

    if explicit_qubits is not None:
        return sorted({n_qubits for n_qubits in explicit_qubits if n_qubits <= max_qubits})

    return [n_qubits for n_qubits in DEFAULT_N_QUBITS if n_qubits <= max_qubits]


def build_target(n_qubits: int, target_mode: str) -> str:
    """Build the target bitstring for one experiment size."""

    if target_mode == "last":
        return "1" * n_qubits
    random_value = random.getrandbits(n_qubits)
    return format(random_value, f"0{n_qubits}b")


def most_frequent_measurement(counts: dict[str, int]) -> tuple[str, int]:
    """Return the most frequent measured bitstring and its count."""

    return max(counts.items(), key=lambda item: (item[1], item[0]))


def run_quantum_experiment(
    mode: str,
    target: str,
    shots: int,
    backend_name: str | None,
) -> GroverRunResult:
    """Run the Grover experiment in simulator or IBM mode."""

    if mode == "ibm":
        return run_grover_ibm_or_fallback(target=target, shots=shots, backend_name=backend_name)
    return run_grover_simulator(target=target, shots=shots)


def output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    """Return the CSV, JSON, and PNG paths for one evaluation batch."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / "experiment_results.csv",
        output_dir / "experiment_results.json",
        output_dir / "runtime_scaling.png",
    )


def _parse_optional_value(raw_value: str, field_name: str):
    """Convert CSV text back into the typed ExperimentRow values."""

    if raw_value == "":
        return None
    if field_name in INT_FIELDS:
        return int(raw_value)
    if field_name in FLOAT_FIELDS:
        return float(raw_value)
    if field_name in BOOL_FIELDS:
        return raw_value.lower() == "true"
    return raw_value


def load_existing_rows(csv_path: Path) -> list[ExperimentRow]:
    """Load any existing experiment rows from CSV for resume or plot rebuilds."""

    if not csv_path.exists():
        return []

    rows: list[ExperimentRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw_row in reader:
            parsed_row = {
                field_name: _parse_optional_value(raw_row.get(field_name, ""), field_name)
                for field_name in CSV_COLUMNS
            }
            rows.append(ExperimentRow(**parsed_row))
    return rows


def merge_rows(existing_rows: list[ExperimentRow], new_rows: list[ExperimentRow]) -> list[ExperimentRow]:
    """Merge old and new rows by n_qubits, letting new rows replace old ones."""

    merged = {row.n_qubits: row for row in existing_rows}
    for row in new_rows:
        merged[row.n_qubits] = row
    return [merged[n_qubits] for n_qubits in sorted(merged)]


def completed_qubits(rows: list[ExperimentRow]) -> set[int]:
    """Return the qubit sizes that already completed without errors."""

    return {row.n_qubits for row in rows if not row.error}


def save_csv(rows: list[ExperimentRow], csv_path: Path) -> None:
    """Write all experiment rows as CSV."""

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_json(rows: list[ExperimentRow], json_path: Path) -> None:
    """Write all experiment rows as JSON."""

    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2),
        encoding="utf-8",
    )


def persist_progress(
    existing_rows: list[ExperimentRow],
    new_rows: list[ExperimentRow],
    csv_path: Path,
    json_path: Path,
) -> list[ExperimentRow]:
    """Persist the current experiment progress immediately after each finished run."""

    all_rows = merge_rows(existing_rows, new_rows)
    save_csv(all_rows, csv_path)
    save_json(all_rows, json_path)
    return all_rows


def build_runtime_plot(rows: list[ExperimentRow], plot_path: Path) -> Path | None:
    """Create the combined runtime scaling plot when timing data exists."""

    successful_rows = [
        row
        for row in rows
        if row.cpu_time_seconds is not None and row.grover_time_seconds is not None
    ]
    if not successful_rows:
        return None

    return plot_runtime_scaling(
        n_qubits_values=[row.n_qubits for row in successful_rows],
        cpu_times=[max(row.cpu_time_seconds or 0.0, 1e-9) for row in successful_rows],
        grover_times=[max(row.grover_time_seconds or 0.0, 1e-9) for row in successful_rows],
        output_path=plot_path,
    )


def execute_one_run(
    *,
    mode: str,
    cpu_mode: str,
    n_qubits: int,
    shots: int,
    target: str,
    backend_name: str | None,
    max_cpu_qubits: int,
) -> ExperimentRow:
    """Run one CPU and Grover experiment and return the summary row."""

    search_space = 2**n_qubits

    cpu_checked_candidates: int | None = None
    cpu_time_seconds: float | None = None
    cpu_note = ""
    if cpu_mode == "skip":
        cpu_note = "CPU brute force skipped because --cpu-mode skip was selected."
    elif n_qubits <= max_cpu_qubits:
        cpu_start = time.perf_counter()
        classical_result = search_hidden_key(n_qubits=n_qubits, target=target)
        cpu_time_seconds = time.perf_counter() - cpu_start
        cpu_checked_candidates = classical_result.checked_candidates
    else:
        cpu_note = (
            f"CPU brute force skipped above --max-cpu-qubits={max_cpu_qubits} "
            f"to avoid extreme host CPU load."
        )

    grover_result = run_quantum_experiment(
        mode=mode,
        target=target,
        shots=shots,
        backend_name=backend_name,
    )
    grover_time_seconds = grover_result.quantum_runtime_seconds

    most_frequent_state, most_frequent_count = most_frequent_measurement(grover_result.counts)
    return ExperimentRow(
        n_qubits=n_qubits,
        search_space=search_space,
        target=target,
        cpu_checked_candidates=cpu_checked_candidates,
        cpu_time_seconds=cpu_time_seconds,
        grover_iterations=grover_result.iterations,
        grover_time_seconds=grover_time_seconds,
        mode_used=grover_result.mode_used,
        backend=grover_result.backend_label,
        shots=shots,
        job_id=grover_result.job_id,
        success_probability=grover_result.success_probability,
        most_frequent_state=most_frequent_state,
        most_frequent_count=most_frequent_count,
        target_was_top_result=(most_frequent_state == target),
        error=grover_result.fallback_reason or cpu_note,
    )


def error_row(n_qubits: int, shots: int, target: str, error: Exception) -> ExperimentRow:
    """Create one row describing a failed experiment run."""

    return ExperimentRow(
        n_qubits=n_qubits,
        search_space=2**n_qubits,
        target=target,
        cpu_checked_candidates=None,
        cpu_time_seconds=None,
        grover_iterations=None,
        grover_time_seconds=None,
        mode_used="error",
        backend=None,
        shots=shots,
        job_id=None,
        success_probability=None,
        most_frequent_state=None,
        most_frequent_count=None,
        target_was_top_result=None,
        error=str(error),
    )


def print_progress(index: int, total: int, row: ExperimentRow) -> None:
    """Print progress after each experiment size."""

    print(
        f"[{index}/{total}] n={row.n_qubits} "
        f"cpu={row.cpu_time_seconds if row.cpu_time_seconds is not None else 'skipped'}s "
        f"grover_qpu={row.grover_time_seconds if row.grover_time_seconds is not None else 'error'}s "
        f"mode={row.mode_used} backend={row.backend or '-'} "
        f"top={row.most_frequent_state or '-'} error={row.error or '-'}"
    )


def main() -> None:
    """Run the full experiment batch and save CSV, JSON, and runtime plot outputs."""

    load_dotenv()
    args = parse_args()
    validate_args(args)

    requested_schedule = qubit_schedule(args.max_qubits, explicit_qubits=args.qubits)
    output_dir = Path(args.output_dir)
    csv_path, json_path, plot_path = output_paths(output_dir)
    existing_rows = load_existing_rows(csv_path)
    existing_done = completed_qubits(existing_rows)
    schedule = [
        n_qubits for n_qubits in requested_schedule if not (args.resume and n_qubits in existing_done)
    ]
    new_rows: list[ExperimentRow] = []

    print("Experiment runner")
    print("=" * 17)
    print(f"Mode: {args.mode}")
    print(f"CPU mode: {args.cpu_mode}")
    print(f"Backend override: {args.backend or 'auto'}")
    print(f"Shots: {args.shots}")
    print(f"Max CPU qubits: {args.max_cpu_qubits}")
    print(f"Resume: {args.resume}")
    print(f"Output directory: {output_dir}")
    print("Large Grover circuits may fail because of circuit depth, transpilation limits, or hardware constraints.")
    if existing_rows:
        print(f"Loaded existing rows: {len(existing_rows)}")
    if args.resume:
        print(f"Skipping completed qubits: {sorted(existing_done)}")

    for index, n_qubits in enumerate(schedule, start=1):
        target = build_target(n_qubits, args.target_mode)
        try:
            row = execute_one_run(
                mode=args.mode,
                cpu_mode=args.cpu_mode,
                n_qubits=n_qubits,
                shots=args.shots,
                target=target,
                backend_name=args.backend,
                max_cpu_qubits=args.max_cpu_qubits,
            )
        except Exception as exc:
            row = error_row(n_qubits=n_qubits, shots=args.shots, target=target, error=exc)
            new_rows.append(row)
            all_rows = persist_progress(existing_rows, new_rows, csv_path, json_path)
            print_progress(index, len(schedule), row)
            if args.stop_on_error:
                reloaded_rows = load_existing_rows(csv_path)
                build_runtime_plot(reloaded_rows, plot_path)
                raise
            continue

        new_rows.append(row)
        persist_progress(existing_rows, new_rows, csv_path, json_path)
        print_progress(index, len(schedule), row)

    all_rows = persist_progress(existing_rows, new_rows, csv_path, json_path)
    reloaded_rows = load_existing_rows(csv_path)
    plot_result = build_runtime_plot(reloaded_rows, plot_path)

    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    if plot_result is not None:
        print(f"Saved plot: {plot_result}")


if __name__ == "__main__":
    main()