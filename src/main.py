"""Command-line entry point for the brute-force vs. Grover demo."""

from __future__ import annotations

import argparse

from classical_bruteforce import search_hidden_key
from grover import GroverRunResult, run_grover_simulator
from ibm_backend import run_grover_ibm_or_fallback
from visualization import plot_measurement_distribution, plot_search_comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""

    parser = argparse.ArgumentParser(
        description="Compare classical brute-force search with Grover's algorithm."
    )
    parser.add_argument(
        "--mode",
        choices=["simulator", "ibm"],
        default="simulator",
        help="Execution mode for the Grover run.",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        choices=[2, 3, 4, 5],
        required=True,
        help="Number of qubits and therefore search bits.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target bitstring to search for, for example 101.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of circuit shots used for quantum measurement.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Ensure the provided target matches the selected number of qubits."""

    if len(args.target) != args.n_qubits:
        raise ValueError("The target bitstring length must match --n-qubits exactly.")
    if set(args.target) - {"0", "1"}:
        raise ValueError("The target bitstring may only contain 0 and 1.")
    if args.shots <= 0:
        raise ValueError("--shots must be a positive integer.")


def run_quantum_demo(mode: str, target: str, shots: int) -> GroverRunResult:
    """Run Grover in the requested execution mode."""

    if mode == "ibm":
        return run_grover_ibm_or_fallback(target=target, shots=shots)
    return run_grover_simulator(target=target, shots=shots)


def format_distribution(counts: dict[str, int]) -> str:
    """Format the measurement counts for a beginner-friendly console summary."""

    lines = []
    for state, count in counts.items():
        lines.append(f"  {state}: {count}")
    return "\n".join(lines)


def print_summary(args: argparse.Namespace, grover_result: GroverRunResult) -> None:
    """Print the comparison in presentation-friendly form."""

    classical_result = search_hidden_key(args.n_qubits, args.target)
    counts_plot = plot_measurement_distribution(
        grover_result.counts,
        target=args.target,
        mode_label=grover_result.mode_used,
    )
    comparison_plot = plot_search_comparison(
        search_space_size=classical_result.search_space_size,
        classical_checks=classical_result.checked_candidates,
        grover_iterations=grover_result.iterations,
    )

    print("Brute-force vs. Grover demo")
    print("=" * 28)
    print(f"Search space size N = {classical_result.search_space_size}")
    print(f"Target bitstring: {args.target}")
    print()
    print("Classical CPU search")
    print(f"  Found key: {classical_result.found_key}")
    print(f"  Checked candidates: {classical_result.checked_candidates}")
    print(f"  Scaling reminder: O(N)")
    print()
    print("Grover search")
    print(f"  Requested mode: {args.mode}")
    print(f"  Mode used: {grover_result.mode_used}")
    print(f"  Backend: {grover_result.backend_label}")
    if grover_result.job_id:
        print(f"  Job ID: {grover_result.job_id}")
    print(f"  Grover iterations: {grover_result.iterations}")
    print(f"  Success probability: {grover_result.success_probability:.4f}")
    print(f"  Scaling reminder: O(sqrt(N))")
    if grover_result.fallback_reason:
        print(f"  Fallback note: {grover_result.fallback_reason}")
    print()
    print("Measured result distribution")
    print(format_distribution(grover_result.counts))
    print()
    print("Interpretation")
    print("  Wall-clock time is not the main comparison for this demo.")
    print("  IBM hardware can be slower here because of queueing, transpilation, measurement, and noise.")
    print("  The point is the algorithmic scaling difference: O(N) vs. O(sqrt(N)).")
    print()
    print(f"Saved plot: {counts_plot}")
    print(f"Saved plot: {comparison_plot}")
    if grover_result.log_file_path:
        print(f"Saved log: {grover_result.log_file_path}")


def main() -> None:
    """Validate input, run the selected demo mode, and print the results."""

    args = parse_args()
    validate_args(args)
    grover_result = run_quantum_demo(args.mode, args.target, args.shots)
    print_summary(args, grover_result)


if __name__ == "__main__":
    main()