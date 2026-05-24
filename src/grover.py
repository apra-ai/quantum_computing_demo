"""Grover circuit construction and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, pi, sqrt

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


def run_grover_with_backend(
    target: str,
    shots: int,
    backend,
    backend_label: str,
    mode_used: str,
    fallback_reason: str | None = None,
) -> GroverRunResult:
    """Execute the Grover circuit on any backend exposing a run method."""

    circuit, iterations = build_grover_circuit(target)
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    raw_counts = job.result().get_counts()
    counts = _normalize_counts(dict(raw_counts), len(target))
    success_probability = counts.get(target, 0) / shots

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
        fallback_reason=fallback_reason,
    )


def run_grover_simulator(target: str, shots: int) -> GroverRunResult:
    """Execute Grover locally with Qiskit Aer."""

    simulator = AerSimulator()
    return run_grover_with_backend(
        target=target,
        shots=shots,
        backend=simulator,
        backend_label=simulator.name,
        mode_used="simulator",
    )