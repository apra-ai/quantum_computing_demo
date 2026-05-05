"""Presentation demo: naive classical trial division for RSA-style factoring."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


@dataclass(frozen=True)
class ClassicalRun:
    """Stores the result of one trial-division benchmark run."""

    label: str
    modulus: int
    factors: tuple[int, int] | None
    elapsed_seconds: float


def trial_division_factor(n: int) -> tuple[int, int] | None:
    """Return a non-trivial factor pair using naive trial division."""

    if n < 2:
        return None

    limit = math.isqrt(n)
    for divisor in range(2, limit + 1):
        if n % divisor == 0:
            return divisor, n // divisor

    return None


def run_classical_case(label: str, modulus: int) -> ClassicalRun:
    """Factor one modulus and measure the execution time."""

    start = time.perf_counter()
    factors = trial_division_factor(modulus)
    elapsed_seconds = time.perf_counter() - start
    return ClassicalRun(label, modulus, factors, elapsed_seconds)


def benchmark_classical_examples() -> list[ClassicalRun]:
    """Run the requested classical benchmark examples."""

    examples = [
        ("Small example", 15),
        ("Medium example", 10007 * 10009),
        ("Larger example", 1000003 * 1000033),
    ]
    return [run_classical_case(label, modulus) for label, modulus in examples]


def estimate_runtime_years(
    reference_modulus: int,
    reference_seconds: float,
    target_log10_n: float,
) -> float:
    """Estimate runtime in years assuming trial division grows as O(sqrt(N))."""

    safe_reference_seconds = max(reference_seconds, 1e-9)
    reference_log10_n = math.log10(reference_modulus)
    estimated_log10_seconds = math.log10(safe_reference_seconds) + 0.5 * (
        target_log10_n - reference_log10_n
    )
    return 10 ** (estimated_log10_seconds - math.log10(SECONDS_PER_YEAR))


def format_scientific(value: float) -> str:
    """Format large values in scientific notation for readable console output."""

    return f"{value:.3e}"


def print_classical_results(runs: Iterable[ClassicalRun]) -> None:
    """Print the classical benchmark results and the RSA-2048 runtime estimate."""

    runs = list(runs)
    baseline = max(runs, key=lambda run: run.elapsed_seconds)

    print("Classical brute force results")
    print("=" * 32)
    for run in runs:
        print(f"{run.label}:")
        print(f"  N = {run.modulus}")
        if run.factors is None:
            print("  Factors: no non-trivial factors found")
        else:
            print(f"  Factors: {run.factors[0]} x {run.factors[1]}")
        print(f"  Time: {run.elapsed_seconds:.6f} seconds")

    rsa_2048_log10_n = 2048 * math.log10(2)
    estimated_years = estimate_runtime_years(
        reference_modulus=baseline.modulus,
        reference_seconds=baseline.elapsed_seconds,
        target_log10_n=rsa_2048_log10_n,
    )

    print()
    print("Estimated naive trial-division runtime for RSA-2048:")
    print(f"  Baseline modulus: {baseline.modulus}")
    print(f"  Estimated time: {format_scientific(estimated_years)} years")


def main() -> None:
    """Run the classical presentation demo."""

    print_classical_results(benchmark_classical_examples())


if __name__ == "__main__":
    main()