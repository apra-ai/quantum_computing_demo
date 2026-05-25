"""Grover circuit construction and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import floor, pi, sqrt
from pathlib import Path
from time import perf_counter
import json

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


@dataclass(frozen=True)
class GroverRunResult:
    """Stores the result of one Grover search experiment."""

    n_qubits: int
    target: str
    iterations: int
    counts: dict[str, int]
    success_probability: float
    backend_label: str
    mode_used: str
    search_space_size: int
    shots: int
    quantum_runtime_seconds: float | None = None
    wall_clock_seconds: float | None = None
    queue_time_seconds: float | None = None
    job_id: str | None = None
    log_file_path: str | None = None
    fallback_reason: str | None = None


def calculate_grover_iterations(n_qubits: int) -> int:
    """Return the standard textbook Grover iteration count."""

    return floor(pi / 4 * sqrt(2**n_qubits))


def _apply_multi_controlled_phase_flip(circuit: QuantumCircuit, qubits: list[int]) -> None:
    """Apply a phase flip to the all-ones state of the given qubits."""

    target_qubit = qubits[-1]
    control_qubits = qubits[:-1]

    circuit.h(target_qubit)
    circuit.mcx(control_qubits, target_qubit)
    circuit.h(target_qubit)


def build_phase_oracle(target: str) -> QuantumCircuit:
    """Build a phase oracle that marks one target bitstring."""

    n_qubits = len(target)
    oracle = QuantumCircuit(n_qubits, name="Oracle")
    target_by_qubit = target[::-1]

    for qubit_index, bit in enumerate(target_by_qubit):
        if bit == "0":
            oracle.x(qubit_index)

    _apply_multi_controlled_phase_flip(oracle, list(range(n_qubits)))

    for qubit_index, bit in enumerate(target_by_qubit):
        if bit == "0":
            oracle.x(qubit_index)

    return oracle


def build_diffusion_operator(n_qubits: int) -> QuantumCircuit:
    """Build the standard Grover diffusion operator."""

    diffuser = QuantumCircuit(n_qubits, name="Diffuser")

    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    _apply_multi_controlled_phase_flip(diffuser, list(range(n_qubits)))
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    return diffuser


def build_grover_circuit(target: str) -> tuple[QuantumCircuit, int]:
    """Build the complete Grover search circuit for the chosen target."""

    n_qubits = len(target)
    iterations = calculate_grover_iterations(n_qubits)

    quantum_register = QuantumRegister(n_qubits, "q")
    classical_register = ClassicalRegister(n_qubits, "meas")
    circuit = QuantumCircuit(quantum_register, classical_register, name="GroverSearch")

    oracle = build_phase_oracle(target)
    diffuser = build_diffusion_operator(n_qubits)

    circuit.h(quantum_register)
    for _ in range(iterations):
        circuit.compose(oracle, qubits=quantum_register, inplace=True)
        circuit.compose(diffuser, qubits=quantum_register, inplace=True)

    circuit.measure(quantum_register, classical_register)
    return circuit, iterations


def _normalize_counts(counts: dict[str, int], n_qubits: int) -> dict[str, int]:
    """Fill in missing states so plots and summaries stay easy to read."""

    normalized_counts = {
        format(value, f"0{n_qubits}b"): 0 for value in range(2**n_qubits)
    }
    normalized_counts.update(counts)
    return normalized_counts


def _outputs_log_dir() -> Path:
    """Return the log directory used for execution traces."""

    log_dir = Path(__file__).resolve().parent.parent / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _serialize_circuit(circuit: QuantumCircuit) -> dict[str, object]:
    """Return a compact beginner-friendly summary of a circuit."""

    operations = {str(name): int(count) for name, count in circuit.count_ops().items()}
    return {
        "name": circuit.name,
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "operations": operations,
        "diagram": str(circuit.draw(output="text")),
    }


def write_execution_log(log_payload: dict[str, object]) -> str:
    """Persist one structured execution log to outputs/logs."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = _outputs_log_dir() / f"grover_run_{timestamp}.json"
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    return str(log_path)


def _parse_runtime_timestamp(value: str | None) -> datetime | None:
    """Parse IBM Runtime timestamps when present."""

    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _extract_queue_time_seconds(metrics: dict[str, object] | None) -> float | None:
    """Return queue wait time from IBM Runtime metrics when available."""

    if not metrics:
        return None

    timestamps = metrics.get("timestamps") if isinstance(metrics, dict) else None
    if not isinstance(timestamps, dict):
        return None

    created = _parse_runtime_timestamp(timestamps.get("created"))
    running = _parse_runtime_timestamp(timestamps.get("running"))
    if created is None or running is None:
        return None
    return max((running - created).total_seconds(), 0.0)


def _extract_quantum_runtime_seconds(job, metrics: dict[str, object] | None) -> float | None:
    """Return the pure IBM quantum runtime without queue time when available."""

    if isinstance(metrics, dict):
        usage = metrics.get("usage")
        if isinstance(usage, dict):
            quantum_seconds = usage.get("quantum_seconds")
            if isinstance(quantum_seconds, (int, float)):
                return float(quantum_seconds)

        bss = metrics.get("bss")
        if isinstance(bss, dict):
            bss_seconds = bss.get("seconds")
            if isinstance(bss_seconds, (int, float)):
                return float(bss_seconds)

    usage_estimation = getattr(job, "usage_estimation", None)
    if isinstance(usage_estimation, dict):
        quantum_seconds = usage_estimation.get("quantum_seconds")
        if isinstance(quantum_seconds, (int, float)):
            return float(quantum_seconds)

    usage_getter = getattr(job, "usage", None)
    if callable(usage_getter):
        usage_value = usage_getter()
        if isinstance(usage_value, (int, float)):
            return float(usage_value)

    return None


def _extract_runtime_metrics(job) -> dict[str, object] | None:
    """Read IBM Runtime job metrics if the job implementation exposes them."""

    metrics_getter = getattr(job, "metrics", None)
    if not callable(metrics_getter):
        return None

    try:
        metrics = metrics_getter()
    except Exception:
        return None

    return metrics if isinstance(metrics, dict) else None


def _build_run_result(
    *,
    target: str,
    shots: int,
    iterations: int,
    counts: dict[str, int],
    backend_label: str,
    mode_used: str,
    circuit: QuantumCircuit,
    transpiled_circuit: QuantumCircuit,
    result_metadata: object,
    quantum_runtime_seconds: float | None,
    wall_clock_seconds: float | None,
    queue_time_seconds: float | None,
    job_id: str | None,
    write_log: bool,
    fallback_reason: str | None,
) -> GroverRunResult:
    """Assemble the shared result object and optional JSON log."""

    success_probability = counts.get(target, 0) / shots
    log_file_path = None
    if write_log:
        log_file_path = write_execution_log(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "request": {
                    "target": target,
                    "shots": shots,
                    "requested_mode": mode_used,
                    "backend_label": backend_label,
                    "search_space_size": 2 ** len(target),
                    "iterations": iterations,
                    "circuit": _serialize_circuit(circuit),
                    "transpiled_circuit": _serialize_circuit(transpiled_circuit),
                },
                "response": {
                    "job_id": job_id,
                    "backend_label": backend_label,
                    "mode_used": mode_used,
                    "quantum_runtime_seconds": quantum_runtime_seconds,
                    "wall_clock_seconds": wall_clock_seconds,
                    "queue_time_seconds": queue_time_seconds,
                    "success_probability": success_probability,
                    "counts": counts,
                    "result_metadata": result_metadata,
                },
                "fallback_reason": fallback_reason,
            }
        )

    return GroverRunResult(
        n_qubits=len(target),
        target=target,
        iterations=iterations,
        counts=counts,
        success_probability=success_probability,
        backend_label=backend_label,
        mode_used=mode_used,
        search_space_size=2 ** len(target),
        shots=shots,
        quantum_runtime_seconds=quantum_runtime_seconds,
        wall_clock_seconds=wall_clock_seconds,
        queue_time_seconds=queue_time_seconds,
        job_id=job_id,
        log_file_path=log_file_path,
        fallback_reason=fallback_reason,
    )


def _extract_sampler_counts(primitive_result) -> tuple[dict[str, int], object]:
    """Extract counts and metadata from a SamplerV2 result."""

    pub_result = primitive_result[0]
    register_keys = list(pub_result.data.keys())
    if not register_keys:
        raise RuntimeError("Sampler result did not contain any classical register data.")

    register_name = register_keys[0]
    register_data = getattr(pub_result.data, register_name)
    return dict(register_data.get_counts()), pub_result.metadata


def run_grover_with_backend(
    target: str,
    shots: int,
    backend,
    backend_label: str,
    mode_used: str,
    write_log: bool = True,
    fallback_reason: str | None = None,
) -> GroverRunResult:
    """Execute the Grover circuit on any backend exposing a run method."""

    circuit, iterations = build_grover_circuit(target)
    transpiled_circuit = transpile(circuit, backend)
    start_time = perf_counter()
    job = backend.run(transpiled_circuit, shots=shots)
    job_id_getter = getattr(job, "job_id", None)
    job_id = str(job_id_getter()) if callable(job_id_getter) else None
    result = job.result()
    wall_clock_seconds = perf_counter() - start_time
    raw_counts = result.get_counts()
    counts = _normalize_counts(dict(raw_counts), len(target))
    return _build_run_result(
        target=target,
        shots=shots,
        iterations=iterations,
        counts=counts,
        backend_label=backend_label,
        mode_used=mode_used,
        circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        result_metadata=getattr(result, "to_dict", lambda: {})(),
        quantum_runtime_seconds=wall_clock_seconds,
        wall_clock_seconds=wall_clock_seconds,
        queue_time_seconds=0.0,
        job_id=job_id,
        write_log=write_log,
        fallback_reason=fallback_reason,
    )


def run_grover_with_sampler(
    target: str,
    shots: int,
    backend,
    sampler,
    backend_label: str,
    mode_used: str,
    write_log: bool = True,
    fallback_reason: str | None = None,
) -> GroverRunResult:
    """Execute the Grover circuit through a SamplerV2-compatible primitive."""

    circuit, iterations = build_grover_circuit(target)
    transpiled_circuit = transpile(circuit, backend)
    start_time = perf_counter()
    job = sampler.run([transpiled_circuit], shots=shots)
    job_id_getter = getattr(job, "job_id", None)
    job_id = str(job_id_getter()) if callable(job_id_getter) else None
    primitive_result = job.result()
    wall_clock_seconds = perf_counter() - start_time
    runtime_metrics = _extract_runtime_metrics(job)
    raw_counts, result_metadata = _extract_sampler_counts(primitive_result)
    counts = _normalize_counts(raw_counts, len(target))

    return _build_run_result(
        target=target,
        shots=shots,
        iterations=iterations,
        counts=counts,
        backend_label=backend_label,
        mode_used=mode_used,
        circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        result_metadata={
            "sampler_metadata": result_metadata,
            "runtime_metrics": runtime_metrics,
        },
        quantum_runtime_seconds=_extract_quantum_runtime_seconds(job, runtime_metrics),
        wall_clock_seconds=wall_clock_seconds,
        queue_time_seconds=_extract_queue_time_seconds(runtime_metrics),
        job_id=job_id,
        write_log=write_log,
        fallback_reason=fallback_reason,
    )


def run_grover_simulator(target: str, shots: int, write_log: bool = True) -> GroverRunResult:
    """Execute Grover locally with Qiskit Aer."""

    simulator = AerSimulator()
    return run_grover_with_backend(
        target=target,
        shots=shots,
        backend=simulator,
        backend_label=simulator.name,
        mode_used="simulator",
        write_log=write_log,
    )