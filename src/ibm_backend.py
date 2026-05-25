"""IBM Quantum backend helpers with simulator fallback."""

from __future__ import annotations

from datetime import datetime
import os

from dotenv import load_dotenv

from grover import GroverRunResult, run_grover_simulator, run_grover_with_sampler, write_execution_log


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


def _select_ibm_backend(service, n_qubits: int, backend_name: str | None = None):
    """Pick either a requested IBM backend or auto-select one with low queue depth."""

    if backend_name:
        backend = service.backend(backend_name)
        if bool(getattr(backend, "simulator", False)):
            raise RuntimeError(f"Requested backend {backend_name!r} is a simulator, not IBM hardware.")

        qubit_count = _backend_qubit_count(backend) or 0
        if qubit_count < n_qubits:
            raise RuntimeError(
                f"Requested backend {backend_name!r} only supports {qubit_count} qubits, "
                f"but {n_qubits} are required."
            )

        return backend

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


def _write_fallback_log(
    target: str,
    shots: int,
    requested_mode: str,
    requested_instance: str | None,
    requested_backend_name: str | None,
    fallback_reason: str,
    local_result: GroverRunResult,
) -> str:
    """Persist a log entry for an IBM request that fell back to the simulator."""

    return write_execution_log(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "request": {
                "target": target,
                "shots": shots,
                "requested_mode": requested_mode,
                "requested_backend_type": "ibm_quantum",
                "requested_instance": requested_instance,
                "requested_backend_name": requested_backend_name,
                "search_space_size": 2 ** len(target),
            },
            "response": {
                "mode_used": local_result.mode_used,
                "backend_label": local_result.backend_label,
                "job_id": local_result.job_id,
                "success_probability": local_result.success_probability,
                "counts": local_result.counts,
            },
            "fallback_reason": fallback_reason,
            "note": "This run requested IBM Quantum execution but used the simulator fallback.",
        }
    )


def run_grover_ibm_or_fallback(
    target: str,
    shots: int,
    backend_name: str | None = None,
) -> GroverRunResult:
    """Run on IBM hardware when possible, otherwise explain the simulator fallback."""

    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    requested_backend_name = backend_name or os.getenv("IBM_QUANTUM_BACKEND")
    if not token:
        print("Warning: IBM_QUANTUM_TOKEN is not set. Falling back to local simulator.")
        local_result = run_grover_simulator(target, shots=shots, write_log=False)
        fallback_reason = "IBM_QUANTUM_TOKEN is not set, so the local simulator was used."
        log_file_path = _write_fallback_log(
            target=target,
            shots=shots,
            requested_mode="ibm",
            requested_instance=instance,
            requested_backend_name=requested_backend_name,
            fallback_reason=fallback_reason,
            local_result=local_result,
        )
        return GroverRunResult(
            **{
                **local_result.__dict__,
                "fallback_reason": fallback_reason,
                "log_file_path": log_file_path,
            }
        )

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except Exception as exc:  # pragma: no cover - optional dependency
        local_result = run_grover_simulator(target, shots=shots, write_log=False)
        fallback_reason = f"qiskit-ibm-runtime could not be imported: {exc}"
        log_file_path = _write_fallback_log(
            target=target,
            shots=shots,
            requested_mode="ibm",
            requested_instance=instance,
            requested_backend_name=requested_backend_name,
            fallback_reason=fallback_reason,
            local_result=local_result,
        )
        return GroverRunResult(
            **{
                **local_result.__dict__,
                "fallback_reason": fallback_reason,
                "log_file_path": log_file_path,
            }
        )

    try:
        service_kwargs = {
            "channel": "ibm_quantum_platform",
            "token": token,
        }
        if instance:
            service_kwargs["instance"] = instance

        service = QiskitRuntimeService(**service_kwargs)
        backend = _select_ibm_backend(
            service,
            n_qubits=len(target),
            backend_name=requested_backend_name,
        )
        sampler = SamplerV2(mode=backend)
        return run_grover_with_sampler(
            target=target,
            shots=shots,
            backend=backend,
            sampler=sampler,
            backend_label=_backend_name(backend),
            mode_used="ibm",
        )
    except Exception as exc:  # pragma: no cover - hardware path is optional
        local_result = run_grover_simulator(target, shots=shots, write_log=False)
        fallback_reason = f"IBM execution failed, so the local simulator was used instead: {exc}"
        log_file_path = _write_fallback_log(
            target=target,
            shots=shots,
            requested_mode="ibm",
            requested_instance=instance,
            requested_backend_name=requested_backend_name,
            fallback_reason=fallback_reason,
            local_result=local_result,
        )
        return GroverRunResult(
            **{
                **local_result.__dict__,
                "fallback_reason": fallback_reason,
                "log_file_path": log_file_path,
            }
        )