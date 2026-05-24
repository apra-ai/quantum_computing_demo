"""IBM Quantum backend helpers with simulator fallback."""

from __future__ import annotations

import os

from grover import GroverRunResult, run_grover_simulator, run_grover_with_backend


def _backend_name(backend) -> str:
    """Return a readable backend name across Qiskit backend variants."""

    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    if isinstance(name, str):
        return name
    return backend.__class__.__name__


def _backend_qubit_count(backend) -> int | None:
    """Extract the advertised qubit count from common backend APIs."""

    num_qubits = getattr(backend, "num_qubits", None)
    if isinstance(num_qubits, int):
        return num_qubits

    target = getattr(backend, "target", None)
    target_qubits = getattr(target, "num_qubits", None)
    if isinstance(target_qubits, int):
        return target_qubits

    configuration = getattr(backend, "configuration", None)
    if callable(configuration):
        config = configuration()
        config_qubits = getattr(config, "n_qubits", None)
        if isinstance(config_qubits, int):
            return config_qubits

    return None


def _select_ibm_backend(service, n_qubits: int):
    """Pick a hardware backend with enough qubits and low queue depth."""

    backends = service.backends()
    hardware_backends = [
        backend for backend in backends if not bool(getattr(backend, "simulator", False))
    ]
    eligible_backends = [
        backend
        for backend in hardware_backends
        if (_backend_qubit_count(backend) or 0) >= n_qubits
    ]

    if not eligible_backends:
        raise RuntimeError(f"No IBM hardware backend with at least {n_qubits} qubits was found.")

    return sorted(
        eligible_backends,
        key=lambda backend: (
            getattr(backend, "pending_jobs", 10**9),
            _backend_name(backend),
        ),
    )[0]


def run_grover_ibm_or_fallback(target: str, shots: int) -> GroverRunResult:
    """Run on IBM hardware when possible, otherwise explain the simulator fallback."""

    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        local_result = run_grover_simulator(target, shots=shots)
        return GroverRunResult(
            **{
                **local_result.__dict__,
                "fallback_reason": "IBM_QUANTUM_TOKEN is not set, so the local simulator was used.",
            }
        )

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:  # pragma: no cover - optional dependency
        local_result = run_grover_simulator(target, shots=shots)
        return GroverRunResult(
            **{
                **local_result.__dict__,
                "fallback_reason": f"qiskit-ibm-runtime could not be imported: {exc}",
            }
        )

    try:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backend = _select_ibm_backend(service, n_qubits=len(target))
        return run_grover_with_backend(
            target=target,
            shots=shots,
            backend=backend,
            backend_label=_backend_name(backend),
            mode_used="ibm",
        )
    except Exception as exc:  # pragma: no cover - hardware path is optional
        local_result = run_grover_simulator(target, shots=shots)
        return GroverRunResult(
            **{
                **local_result.__dict__,
                "fallback_reason": f"IBM execution failed, so the local simulator was used instead: {exc}",
            }
        )