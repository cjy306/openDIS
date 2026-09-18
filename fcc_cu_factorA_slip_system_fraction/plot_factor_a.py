#!/usr/bin/env python3
"""比较A25、A50、A75、A100四组正式加载结果。"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 12345
CASES = (
    ("A25: 25% high-Schmid", "A25", "#1f77b4"),
    ("A50: 50% high-Schmid", "A50", "#2ca02c"),
    ("A75: 75% high-Schmid", "A75", "#ff7f0e"),
    ("A100: 100% high-Schmid", "A100", "#d62728"),
)


def load_response(path):
    """读取strain、stress和density，并去除restart导致的非递增应变点。"""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{path} 至少需要4列数据")
    strain = data[:, 1]
    stress_mpa = data[:, 2] / 1.0e6
    density = data[:, 3]
    mask = np.concatenate(([True], np.diff(strain) > 0.0))
    return strain[mask], stress_mpa[mask], density[mask]


def main():
    base_dir = Path(__file__).resolve().parent
    post_dir = base_dir / f"post_factor_a_seed{SEED}"
    post_dir.mkdir(parents=True, exist_ok=True)

    fig, (stress_axis, density_axis) = plt.subplots(1, 2, figsize=(14, 6))
    loaded = 0
    for label, case, color in CASES:
        data_file = (base_dir / f"output_{case}_seed{SEED}"
                     / "stress_strain_dens.dat")
        if not data_file.is_file():
            print(f"[跳过] 文件不存在: {data_file}")
            continue

        strain, stress_mpa, density = load_response(data_file)
        strain_percent = strain * 100.0
        stress_axis.plot(strain_percent, stress_mpa, color=color, lw=1.5, label=label)
        density_axis.plot(strain_percent, density, color=color, lw=1.5, label=label)
        peak_index = int(np.argmax(stress_mpa))
        print(f"[{case}] points={len(strain)}, "
              f"peak={stress_mpa[peak_index]:.2f} MPa "
              f"at {strain_percent[peak_index]:.4f}%, "
              f"rho: {density[0]:.3e} -> {density[-1]:.3e} m^-2")
        loaded += 1

    if loaded == 0:
        plt.close(fig)
        raise FileNotFoundError("没有找到任何因素A的stress_strain_dens.dat")

    stress_axis.set_xlabel("Strain (%)")
    stress_axis.set_ylabel("Stress (MPa)")
    stress_axis.set_title("Stress-Strain Response")
    stress_axis.set_xlim(left=0.0)
    stress_axis.set_ylim(bottom=0.0)
    stress_axis.grid(True, alpha=0.3, linestyle="--")
    stress_axis.legend()

    density_axis.set_xlabel("Strain (%)")
    density_axis.set_ylabel(r"Dislocation density (m$^{-2}$)")
    density_axis.set_title("Dislocation Density Evolution")
    density_axis.set_xlim(left=0.0)
    density_axis.grid(True, alpha=0.3, linestyle="--")
    density_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    density_axis.legend()

    fig.tight_layout(pad=2.0)
    output_file = post_dir / f"factor_a_compare_seed{SEED}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"图像已保存: {output_file}")


if __name__ == "__main__":
    main()
