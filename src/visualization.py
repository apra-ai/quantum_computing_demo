"""Plot helpers for the brute-force vs. Grover comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def ensure_output_dir() -> Path:
    """Create the output directory used for generated figures."""

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_measurement_distribution(counts: dict[str, int], target: str, mode_label: str) -> Path:
    """Save a bar chart showing the measured state distribution."""

    output_dir = ensure_output_dir()
    output_path = output_dir / f"grover_counts_{mode_label}_{target}.png"
    states = list(counts.keys())
    values = [counts[state] for state in states]
    colors = ["#d95f02" if state == target else "#1b9e77" for state in states]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(states, values, color=colors)
    axis.set_title(f"Grover measurement distribution for target {target}")
    axis.set_xlabel("Measured bitstring")
    axis.set_ylabel("Counts")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_search_comparison(
    search_space_size: int,
    classical_checks: int,
    grover_iterations: int,
) -> Path:
    """Save a small comparison chart for the two search strategies."""

    output_dir = ensure_output_dir()
    output_path = output_dir / f"search_comparison_N{search_space_size}.png"

    figure, axis = plt.subplots(figsize=(7, 5))
    labels = ["Classical checks", "Grover iterations"]
    values = [classical_checks, grover_iterations]
    colors = ["#7570b3", "#e7298a"]

    axis.bar(labels, values, color=colors)
    axis.set_title(f"Search effort comparison for N = {search_space_size}")
    axis.set_ylabel("Work units")
    axis.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        axis.text(index, value + 0.05, str(value), ha="center", va="bottom")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path