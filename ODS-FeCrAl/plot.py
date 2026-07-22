#!/usr/bin/env python3
"""多曲线对比绘图：应力-应变 + 应变-位错密度
每条曲线一个 stress_strain_dens.dat，标签/颜色自己填。
格式：step  strain  stress(Pa)  density(m^-2)  [plastic_strain]
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 在这里配置要对比的曲线（可加 2 条以上）==========
CURVES = [
    {
        "file":  "/public/home/cjy306/openDIS/ODS-FeCrAl/output_caseA_high/stress_strain_dens.dat",
        "label": "caseA_high",
        "color": "#030E15",   
    },
    {
        "file":  "/public/home/cjy306/openDIS/ODS-FeCrAl/output_caseA_low/stress_strain_dens.dat",
        "label": "caseA_low",
        "color": "#1F77B4",   
    }
]
OUTPUT_DIR  = "/public/home/cjy306/openDIS/ODS-FeCrAl/Post-processing simulation"
OUTPUT_NAME = "compare"
# =========================================================


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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    for i, c in enumerate(CURVES):
        if not os.path.exists(c["file"]):
            print(f"[警告] 文件不存在，跳过: {c['file']}")
            continue
        strain, stress, dens = load_data(c["file"])
        strain_pct = strain * 100
        color, label = c["color"], c["label"]

        # 左图：应力-应变
        ax1.plot(strain_pct, stress, color=color, lw=1.2, label=label)
        idx_peak = int(np.argmax(stress))
        ax1.annotate(f"{stress[idx_peak]:.0f} MPa",
                     xy=(strain_pct[idx_peak], stress[idx_peak]),
                     xytext=(strain_pct[idx_peak] + 0.05,
                             stress[idx_peak] + 3 + i * 8),  # 不同曲线标注错开
                     fontsize=10, color=color,
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

        # 右图：应变-位错密度
        ax2.plot(strain_pct, dens, color=color, lw=1.2, label=label)

        # 终端摘要
        print(f"\n=== {label} ===")
        print(f"  数据点: {len(strain)}, 最大应变: {strain_pct[-1]:.3f}%")
        print(f"  峰值应力: {stress[idx_peak]:.1f} MPa @ 应变 {strain_pct[idx_peak]:.3f}%")
        print(f"  终态应力: {stress[-1]:.1f} MPa (终/峰 = {stress[-1]/stress[idx_peak]*100:.0f}%)")
        print(f"  峰值密度: {dens.max():.3e} m^-2, 终态密度: {dens[-1]:.3e} m^-2")

    # 左图样式
    ax1.set_xlabel('Strain (%)', fontsize=13)
    ax1.set_ylabel('Stress (MPa)', fontsize=13)
    ax1.set_title('Stress-Strain Curve', fontsize=14)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(labelsize=11)

    # 右图样式
    ax2.set_xlabel('Strain (%)', fontsize=13)
    ax2.set_ylabel(r'Dislocation Density (m$^{-2}$)', fontsize=13)
    ax2.set_title('Strain vs Dislocation Density', fontsize=14)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xlim(left=0)
    ax2.tick_params(labelsize=11)

    plt.tight_layout(pad=2.0)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图像已保存: {output_path}")


if __name__ == '__main__':
    main()
