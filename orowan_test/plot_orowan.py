"""Plot the stress response and dislocation-density evolution of the Orowan case."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(filepath):
    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError("stress_strain_dens.dat must contain at least four columns")

    strain = data[:, 1]
    stress = data[:, 2] / 1.0e6
    density = data[:, 3]
    mask = np.concatenate(([True], np.diff(strain) > 0.0))
    return strain[mask], stress[mask], density[mask]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "post_orowan")
    os.makedirs(output_dir, exist_ok=True)

    strain, stress, density = load_data(
        os.path.join(base_dir, "output_orowan", "stress_strain_dens.dat"))
    strain_pct = 100.0 * strain
    peak = int(np.argmax(stress))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(strain_pct, stress, color="#1F77B4", lw=1.3)
    ax1.scatter(strain_pct[peak], stress[peak], color="#D62728", s=28)
    ax1.annotate(
        "%.1f MPa" % stress[peak],
        xy=(strain_pct[peak], stress[peak]),
        xytext=(8, 8),
        textcoords="offset points",
        color="#D62728",
    )
    ax1.set_xlabel("Total strain (%)")
    ax1.set_ylabel("Axial stress (MPa)")
    ax1.set_title("Orowan stress-strain response")

    ax2.plot(strain_pct, density, color="#2CA02C", lw=1.3)
    ax2.set_xlabel("Total strain (%)")
    ax2.set_ylabel(r"Dislocation density (m$^{-2}$)")
    ax2.set_title("Dislocation-density evolution")
    ax2.set_yscale("log")

    for axis in (ax1, ax2):
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.set_xlim(left=0.0)

    figure_path = os.path.join(output_dir, "orowan_response.png")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    summary_path = os.path.join(output_dir, "orowan_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as stream:
        stream.write("points = %d\n" % len(strain))
        stream.write("max_strain = %.6f %%\n" % strain_pct[-1])
        stream.write("peak_stress = %.6f MPa @ %.6f %%\n" %
                     (stress[peak], strain_pct[peak]))
        stream.write("final_stress = %.6f MPa\n" % stress[-1])
        stream.write("density = %.8e -> %.8e m^-2\n" %
                     (density[0], density[-1]))

    print("Figure written:", figure_path)
    print("Summary written:", summary_path)


if __name__ == "__main__":
    main()
