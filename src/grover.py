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


# =============================================================================
# Grover building blocks
#
# This file is grouped by the 4 conceptual stages of Grover's algorithm:
#
#   1. SUPERPOSITION         -> see section "1. SUPERPOSITION / QUANTUM REGISTER"
#                               (Hadamards on all qubits in build_grover_circuit)
#   2. ORACLE                -> see section "2. ORACLE"
#                               (build_phase_oracle)
#   3. AMPLITUDE AMPLIFICATION -> see section "3. AMPLITUDE AMPLIFICATION"
#                               (build_diffusion_operator + iteration loop)
#   4. MEASUREMENT           -> see section "4. MEASUREMENT"
#                               (circuit.measure + backend execution further below)
#
# Supporting helpers are grouped into their own sections below the 4 stages:
#
#   - MEASUREMENT POST-PROCESSING  (count normalization, sampler decoding)
#   - LOGGING / CIRCUIT SERIALIZATION
#   - IBM RUNTIME METRICS          (queue + pure quantum time)
#   - RESULT ASSEMBLY              (_build_run_result)
#   - BACKEND EXECUTION ENTRY POINTS (run_grover_* functions)
# =============================================================================


# -----------------------------------------------------------------------------
# Iteration count helper
#
# Used by stage 3 (Amplitude Amplification) to decide how often Oracle and
# Diffuser are repeated. Formula: floor(pi/4 * sqrt(N)).
# -----------------------------------------------------------------------------
def calculate_grover_iterations(n_qubits: int) -> int:
    """Return the standard textbook Grover iteration count."""

    return floor(pi / 4 * sqrt(2**n_qubits))


# -----------------------------------------------------------------------------
# Shared low-level helper
#
# Multi-controlled phase flip on the all-ones state.
# Used inside both the Oracle (stage 2) and the Diffuser (stage 3).
# -----------------------------------------------------------------------------
def _apply_multi_controlled_phase_flip(circuit: QuantumCircuit, qubits: list[int]) -> None:
    """Apply a phase flip to the all-ones state of the given qubits."""

    target_qubit = qubits[-1]
    control_qubits = qubits[:-1]

    circuit.h(target_qubit)
    circuit.mcx(control_qubits, target_qubit)
    circuit.h(target_qubit)


# =============================================================================
# 2. ORACLE
#
# Marks the target bitstring omega by flipping its phase:
#     O_f |x> = -|x|  if x == omega
#     O_f |x> =  |x|  otherwise
# =============================================================================
def build_phase_oracle(target: str) -> QuantumCircuit:
    """Build a phase oracle that marks one target bitstring."""

    n_qubits = len(target)
    oracle = QuantumCircuit(n_qubits, name="Oracle")
    target_by_qubit = target[::-1]

    # Flip qubits that are 0 in the target so the marker fires on |target>.
    for qubit_index, bit in enumerate(target_by_qubit):
        if bit == "0":
            oracle.x(qubit_index)

    # Apply the phase flip on the (now aligned) all-ones state.
    _apply_multi_controlled_phase_flip(oracle, list(range(n_qubits)))

    # Undo the temporary X gates so only the phase of |target> is changed.
    for qubit_index, bit in enumerate(target_by_qubit):
        if bit == "0":
            oracle.x(qubit_index)

    return oracle


# =============================================================================
# 3. AMPLITUDE AMPLIFICATION (Diffuser part)
#
# The diffuser reflects all amplitudes around their mean. Combined with the
# Oracle this forms one Grover iteration G = (2|s><s| - I) O_f, which boosts
# the amplitude of the marked state.
# =============================================================================
def build_diffusion_operator(n_qubits: int) -> QuantumCircuit:
    """Build the standard Grover diffusion operator."""

    diffuser = QuantumCircuit(n_qubits, name="Diffuser")

    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    _apply_multi_controlled_phase_flip(diffuser, list(range(n_qubits)))
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    return diffuser


# =============================================================================
# Full Grover circuit
#
# This is where stages 1, 3 and 4 are wired together in order:
#   1. SUPERPOSITION       -> Hadamards on all qubits
#   3. AMPLITUDE AMPLIFY   -> repeat (Oracle + Diffuser) for `iterations` rounds
#   4. MEASUREMENT         -> measure all qubits into the classical register
# =============================================================================
def build_grover_circuit(target: str) -> tuple[QuantumCircuit, int]:
    """Build the complete Grover search circuit for the chosen target."""

    n_qubits = len(target)
    iterations = calculate_grover_iterations(n_qubits)

    # ---- 1. SUPERPOSITION / QUANTUM REGISTER --------------------------------
    # Allocate one quantum register for the search qubits and one classical
    # register that will receive the measurement results.
    quantum_register = QuantumRegister(n_qubits, "q")
    classical_register = ClassicalRegister(n_qubits, "meas")
    circuit = QuantumCircuit(quantum_register, classical_register, name="GroverSearch")

    # Build the reusable sub-circuits for stages 2 and 3.
    oracle = build_phase_oracle(target)
    diffuser = build_diffusion_operator(n_qubits)

    # Put all qubits into uniform superposition |s> = 1/sqrt(N) * sum_x |x>.
    circuit.h(quantum_register)

    # ---- 3. AMPLITUDE AMPLIFICATION ----------------------------------------
    # Repeat (Oracle + Diffuser) approximately pi/4 * sqrt(N) times.
    for _ in range(iterations):
        circuit.compose(oracle, qubits=quantum_register, inplace=True)
        circuit.compose(diffuser, qubits=quantum_register, inplace=True)

    # ---- 4. MEASUREMENT ----------------------------------------------------
    # Collapse the amplified state into a classical bitstring.
    circuit.measure(quantum_register, classical_register)
    return circuit, iterations


# =============================================================================
# MEASUREMENT POST-PROCESSING (helpers for stage 4)
#
# After the backend returns raw counts we still need to make them readable
# (fill in zero entries) and, for IBM SamplerV2, decode them from the
# primitive result structure.
# =============================================================================
def _normalize_counts(counts: dict[str, int], n_qubits: int) -> dict[str, int]:
    """Fill in missing states so plots and summaries stay easy to read."""

    normalized_counts = {
        format(value, f"0{n_qubits}b"): 0 for value in range(2**n_qubits)
    }
    normalized_counts.update(counts)
    return normalized_counts


# =============================================================================
# LOGGING / CIRCUIT SERIALIZATION
#
# Each Grover run is dumped as a JSON file under outputs/logs so you can
# inspect the circuit, transpiled circuit and the response from the backend
# after the fact.
# =============================================================================
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


# =============================================================================
# IBM RUNTIME METRICS (queue + pure quantum time)
#
# Helpers that pull timing information out of an IBM Runtime job so we can
# separate "time spent waiting in the queue" from "pure time on QPU".
# =============================================================================
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


# =============================================================================
# RESULT ASSEMBLY
#
# Wraps counts, timing info and metadata into a GroverRunResult and writes the
# JSON execution log.
# =============================================================================
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


# =============================================================================
# 4. MEASUREMENT / BACKEND EXECUTION (entry points)
#
# The functions below take the prepared Grover circuit, transpile it for the
# selected backend (simulator or IBM hardware), run it, and collect the
# measurement counts that the rest of the demo turns into probabilities.
# =============================================================================
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

    # Build stages 1-4 of the algorithm into a single circuit.
    circuit, iterations = build_grover_circuit(target)
    # Adapt the abstract circuit to the concrete backend gate set.
    transpiled_circuit = transpile(circuit, backend)
    start_time = perf_counter()
    # Submit the circuit to the backend (stage 4: measurement on hardware/sim).
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

    # Build stages 1-4 of the algorithm into a single circuit.
    circuit, iterations = build_grover_circuit(target)
    # Adapt the abstract circuit to the IBM backend's native gate set.
    transpiled_circuit = transpile(circuit, backend)
    start_time = perf_counter()
    # Stage 4: measurement happens on real IBM hardware through the sampler.
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