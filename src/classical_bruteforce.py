"""Classical brute-force search helpers for a hidden bitstring target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BruteForceResult:
    """Stores the outcome of one classical search run."""

    n_qubits: int
    target: str
    found_key: str
    checked_candidates: int
    search_space_size: int


def generate_bitstrings(n_qubits: int) -> list[str]:
    """Return all bitstrings from 0 to 2^n - 1 with leading zeros."""

    return [format(value, f"0{n_qubits}b") for value in range(2**n_qubits)]


def search_hidden_key(n_qubits: int, target: str) -> BruteForceResult:
    """Search linearly through the full key space until the target is found."""

    candidates = generate_bitstrings(n_qubits)

    for checked_candidates, candidate in enumerate(candidates, start=1):
        if candidate == target:
            return BruteForceResult(
                n_qubits=n_qubits,
                target=target,
                found_key=candidate,
                checked_candidates=checked_candidates,
                search_space_size=len(candidates),
            )

    raise ValueError(f"Target {target!r} is outside the {n_qubits}-qubit search space.")