#!/usr/bin/env python3
"""Compare 1 um and 3 um FR runs at seed 12345."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent




def load_response(path):
    # ExaDiS default columns: step, strain, stress [Pa], density [m^-2], ...
    import io
    # Read a snapshot of the growing file; discard only an unfinished final line.
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0] + "\n" if "\n" in text else ""
        print(f"[partial row ignored] {path}")
    data = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
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
    cases = (
        ("1 um | 128 sources | A75", "#0072B2",
         BASE_DIR.parent / "stress_overshoot_5runs/run_01/output_A75_seed12345/stress_strain_dens.dat"),
        ("3 um | 43 sources | high-Schmid 74.42%", "#D55E00",
         BASE_DIR / "output_A75_3um_seed12345/stress_strain_dens.dat"),
    )
    for label, color, path in cases:
        if not path.is_file():
            print(f"[missing] {path}")
            continue
        strain, stress, density = load_response(path)
        responses.append((label, color, strain, stress, density))
        print(f"[{label}] {len(strain)} points, latest strain={strain[-1]:.5g}%")
    if not responses:
        raise FileNotFoundError("No simulation output found for either FR length")

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
    fig.suptitle(f"FCC Cu | seed 12345 | {len(responses)}/2 runs available")
    output = BASE_DIR / "plots"
    output.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output / f"fr_1um_vs_3um_seed12345.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(f"[saved] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
