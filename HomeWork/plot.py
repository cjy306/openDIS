#!/usr/bin/env python3
"""单次运行绘图：应力-应变曲线 + 应变-位错密度曲线
数据来源：模拟输出的 stress_strain_dens.dat
格式：step  strain  stress(Pa)  density(m^-2)  [plastic_strain]
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 路径配置（按需修改）==========
DATA_FILE  = "/data/home/dg000246b/openDIS/HomeWork/output_Cu_line/stress_strain_dens.dat"
OUTPUT_DIR = "/data/home/dg000246b/openDIS/HomeWork/Post-processing simulation"
LABEL      = "twin + precip 1e3 LINE"   # 图例标签，标注这是哪个工况
# =========================================


def load_data(filepath):
    """读取 stress_strain_dens.dat，返回 strain, stress(MPa), density(m^-2)"""
    data = np.loadtxt(filepath, comments='#')
    strain  = data[:, 1]          # 应变（无量纲）
    stress  = data[:, 2] / 1e6    # 应力 Pa → MPa
    density = data[:, 3]          # 位错密度 m^-2
    # 只保留应变单调递增的部分（防止 restart 导致应变倒退）
    mask = np.concatenate(([True], np.diff(strain) > 0))
    return strain[mask], stress[mask], density[mask]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    strain, stress, dens = load_data(DATA_FILE)
    strain_pct = strain * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    COLOR = '#1F77B4'

    # ── 左图：应力-应变 ──
    ax1.plot(strain_pct, stress, color=COLOR, lw=1.2, label=LABEL)
    ax1.set_xlabel('Strain (%)', fontsize=13)
    ax1.set_ylabel('Stress (MPa)', fontsize=13)
    ax1.set_title('Stress-Strain Curve', fontsize=14)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(labelsize=11)

    # 标注峰值应力
    idx_peak = np.argmax(stress)
    ax1.annotate(f"{stress[idx_peak]:.0f} MPa",
                 xy=(strain_pct[idx_peak], stress[idx_peak]),
                 xytext=(strain_pct[idx_peak] + 0.05, stress[idx_peak] + 3),
                 fontsize=10, color=COLOR,
                 arrowprops=dict(arrowstyle='->', color=COLOR, lw=1.2))

    # ── 右图：应变-位错密度 ──
    ax2.plot(strain_pct, dens, color=COLOR, lw=1.2, label=LABEL)
    ax2.set_xlabel('Strain (%)', fontsize=13)
    ax2.set_ylabel(r'Dislocation Density (m$^{-2}$)', fontsize=13)
    ax2.set_title('Strain vs Dislocation Density', fontsize=14)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xlim(left=0)
    ax2.tick_params(labelsize=11)

    plt.tight_layout(pad=2.0)
    output_path = os.path.join(OUTPUT_DIR, "results1.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # ── 终端摘要（判断崩溃/平台/过钉的关键数字）──
    print(f"图像已保存: {output_path}")
    print(f"数据点: {len(strain)}, 最大应变: {strain_pct[-1]:.3f}%")
    print(f"峰值应力: {stress[idx_peak]:.1f} MPa @ 应变 {strain_pct[idx_peak]:.3f}%")
    print(f"终态应力: {stress[-1]:.1f} MPa (终/峰 = {stress[-1]/stress[idx_peak]*100:.0f}%)")
    print(f"峰值密度: {dens.max():.3e} m^-2, 终态密度: {dens[-1]:.3e} m^-2")


if __name__ == '__main__':
    main()
