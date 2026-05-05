"""Presentation demo targeting the current Qiskit stack in the local venv."""

from __future__ import annotations

import argparse
import os

import qiskit
from qiskit_aer import AerSimulator


CURRENT_QISKIT_VERSION = qiskit.__version__

def _resolve_backend_label(backend_name: str | None) -> str:
    """Resolve the selected execution target using the current Qiskit packages."""

    if backend_name:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "qiskit-ibm-runtime is required when --backend-name is used."
            ) from exc

        token = os.getenv("IBM_QUANTUM_TOKEN")
        service = QiskitRuntimeService(token=token) if token else QiskitRuntimeService()
        backend = service.backend(backend_name)
        return backend.name

    backend = AerSimulator()
    return backend.name


def print_quantum_result(backend_name: str | None) -> None:
    """Print the status of the quantum demo for the current Qiskit version."""

    print("Quantum (Shor) result")
    print("=" * 21)
    print(f"Installed Qiskit version: {CURRENT_QISKIT_VERSION}")

    try:
        backend_label = _resolve_backend_label(backend_name)
    except Exception as exc:
        print("Quantum backend setup could not be completed.")
        print(f"Reason: {exc}")
        return

    print(f"Backend: {backend_label}")
    print("Current Qiskit no longer ships a built-in high-level Shor factorization API.")
    print("So this environment cannot produce factors for N = 15 via qiskit.algorithms.Shor.")
    print("If you need an actual Shor run, the remaining options are:")
    print("  1. use an older Qiskit stack that still included the high-level Shor class")
    print("  2. implement the order-finding circuit manually for N = 15")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for backend selection."""

    parser = argparse.ArgumentParser(
        description="Demo: current Qiskit environment status for Shor factoring"
    )
    parser.add_argument(
        "--backend-name",
        default=None,
        help="Optional IBM Quantum backend name. If omitted, the local simulator is used.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the quantum presentation demo."""

    args = parse_args()
    print_quantum_result(backend_name=args.backend_name)


if __name__ == "__main__":
    main()