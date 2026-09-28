#!/usr/bin/env python3
"""Compare the five FCC Cu A75 runs using their raw simulation output."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SEEDS = (12345, 23456, 34567, 45678, 56789)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def load_response(path):
    # ExaDiS default columns: step, strain, stress [Pa], density [m^-2], ...
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[0] == 0 or data.shape[1] < 4:
        raise ValueError(f"{path}: expected nonempty data with at least four columns")
    values = data[:, 1:4]
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite strain, stress or density")
    if np.any(np.diff(values[:, 0]) < 0):
        raise ValueError(f"{path}: strain decreases; check overlapping restart output")
    return values[:, 0] * 100.0, values[:, 1] / 1e6, values[:, 2]


def main():
    responses = []
    for run, (seed, color) in enumerate(zip(SEEDS, COLORS), 1):
        path = (BASE_DIR / f"run_{run:02d}" / f"output_A75_seed{seed}"
                / "stress_strain_dens.dat")
        if not path.is_file():
            print(f"[missing] {path}")
            continue
        strain, stress, density = load_response(path)
        responses.append((f"Run {run:02d} (seed {seed})", color, strain, stress, density))
        print(f"[run_{run:02d}] {len(strain)} points, final strain={strain[-1]:.5g}%")
    if not responses:
        raise FileNotFoundError("No simulation output found for any of the five runs")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for label, color, strain, stress, density in responses:
        axes[0].plot(strain, stress, color=color, lw=1.2, label=label)
        axes[1].plot(strain, density, color=color, lw=1.2, label=label)
    axes[0].set(title="Stress-Strain", ylabel="Stress (MPa)")
    axes[1].set(title="Dislocation Density-Strain",
                ylabel=r"Dislocation density (m$^{-2}$)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    for axis in axes:
        axis.set_xlabel("Strain (%)")
        axis.set_xlim(left=0)
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.legend(fontsize=8.5)
    fig.suptitle(f"FCC Cu A75 | {len(responses)}/5 runs available")
    output = BASE_DIR / "plots"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output / f"five_runs_comparison.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"[saved] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
