#!/usr/bin/env python3
"""
对比绘图脚本：有夹杂（Orowan）vs 无夹杂（纯Cu）
生成应力-应变曲线和应变-位错密度曲线
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 路径配置 ==========
FILE_WITH = "/data/home/dg000246b/openDIS/HomeWork/output_Cu_fcc/stress_strain_dens.dat"
FILE_PURE = "/data/home/dg000246b/openDIS/HomeWork/output_Cu_fcc_pure/stress_strain_dens.dat"
OUTPUT_DIR = "/data/home/dg000246b/openDIS/HomeWork/Post-processing simulation"
# ==============================

# 平滑窗口（单位：步数），调大可以更平滑
SMOOTH_WINDOW = 20


def load_data(filepath):
    """读取 stress_strain_dens.dat，返回 strain, stress(MPa), density"""
    data = np.loadtxt(filepath, comments='#')
    strain  = data[:, 1]           # 应变（无量纲）
    stress  = data[:, 2] / 1e6    # 应力 Pa → MPa
    density = data[:, 3]           # 位错密度 m^-2
    return strain, stress, density


def smooth(y, window):
    """滑动平均平滑"""
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    # 边缘用边界值填充，避免端点失真
    pad = window // 2
    y_pad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(y_pad, kernel, mode='valid')[:len(y)]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取数据
    strain_w, stress_w, dens_w = load_data(FILE_WITH)
    strain_p, stress_p, dens_p = load_data(FILE_PURE)

    # 平滑
    stress_w_sm = smooth(stress_w, SMOOTH_WINDOW)
    stress_p_sm = smooth(stress_p, SMOOTH_WINDOW)
    dens_w_sm   = smooth(dens_w,   SMOOTH_WINDOW)
    dens_p_sm   = smooth(dens_p,   SMOOTH_WINDOW)

    # 转换应变为百分比
    strain_w_pct = strain_w * 100
    strain_p_pct = strain_p * 100

    # ── 绘图 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    COLOR_WITH = '#D62728'   # 红色：有夹杂
    COLOR_PURE = '#1F77B4'   # 蓝色：无夹杂
    ALPHA_RAW  = 0.15        # 原始曲线透明度
    LW_RAW     = 0.8
    LW_SM      = 2.0

    # ── 左图：应力-应变曲线 ──
    # 原始曲线（淡色）
    ax1.plot(strain_w_pct, stress_w, color=COLOR_WITH, lw=LW_RAW, alpha=ALPHA_RAW)
    ax1.plot(strain_p_pct, stress_p, color=COLOR_PURE, lw=LW_RAW, alpha=ALPHA_RAW)
    # 平滑曲线
    ax1.plot(strain_w_pct, stress_w_sm, color=COLOR_WITH, lw=LW_SM,
             label='With precipitates (Orowan)')
    ax1.plot(strain_p_pct, stress_p_sm, color=COLOR_PURE, lw=LW_SM,
             label='Pure Cu')

    ax1.set_xlabel('Strain (%)', fontsize=13)
    ax1.set_ylabel('Stress (MPa)', fontsize=13)
    ax1.set_title('Stress-Strain Curve', fontsize=14)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(labelsize=11)

    # 标注峰值应力
    idx_peak_w = np.argmax(stress_w_sm)
    idx_peak_p = np.argmax(stress_p_sm)
    ax1.annotate(f"{stress_w_sm[idx_peak_w]:.0f} MPa",
                 xy=(strain_w_pct[idx_peak_w], stress_w_sm[idx_peak_w]),
                 xytext=(strain_w_pct[idx_peak_w]+0.05, stress_w_sm[idx_peak_w]+3),
                 fontsize=10, color=COLOR_WITH,
                 arrowprops=dict(arrowstyle='->', color=COLOR_WITH, lw=1.2))
    ax1.annotate(f"{stress_p_sm[idx_peak_p]:.0f} MPa",
                 xy=(strain_p_pct[idx_peak_p], stress_p_sm[idx_peak_p]),
                 xytext=(strain_p_pct[idx_peak_p]+0.05, stress_p_sm[idx_peak_p]+3),
                 fontsize=10, color=COLOR_PURE,
                 arrowprops=dict(arrowstyle='->', color=COLOR_PURE, lw=1.2))

    # ── 右图：应变-位错密度曲线 ──
    ax2.plot(strain_w_pct, dens_w, color=COLOR_WITH, lw=LW_RAW, alpha=ALPHA_RAW)
    ax2.plot(strain_p_pct, dens_p, color=COLOR_PURE, lw=LW_RAW, alpha=ALPHA_RAW)
    ax2.plot(strain_w_pct, dens_w_sm, color=COLOR_WITH, lw=LW_SM,
             label='With precipitates (Orowan)')
    ax2.plot(strain_p_pct, dens_p_sm, color=COLOR_PURE, lw=LW_SM,
             label='Pure Cu')

    ax2.set_xlabel('Strain (%)', fontsize=13)
    ax2.set_ylabel(r'Dislocation Density (m$^{-2}$)', fontsize=13)
    ax2.set_title('Strain vs Dislocation Density', fontsize=14)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xlim(left=0)
    ax2.tick_params(labelsize=11)

    plt.tight_layout(pad=2.0)

    output_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"图像已保存: {output_path}")
    print(f"有夹杂: {len(strain_w)} 个数据点, 最大应变 {strain_w_pct[-1]:.2f}%")
    print(f"无夹杂: {len(strain_p)} 个数据点, 最大应变 {strain_p_pct[-1]:.2f}%")
    print(f"有夹杂峰值应力: {stress_w_sm.max():.1f} MPa @ 应变 {strain_w_pct[idx_peak_w]:.3f}%")
    print(f"无夹杂峰值应力: {stress_p_sm.max():.1f} MPa @ 应变 {strain_p_pct[idx_peak_p]:.3f}%")


if __name__ == '__main__':
    main()