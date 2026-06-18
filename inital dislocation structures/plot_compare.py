#!/usr/bin/env python3
"""五工况最终对比:应力-应变 + 应变-位错密度。

读各工况"正式加载"段的 stress_strain_dens.dat,把 A/B/C1/C2/C3 五条画在一起。
只画正式加载段(predeform/relax2 是过程,不进对比)。

stress_strain_dens.dat 列(0 起):
  0 Step  1 Strain  2 Stress(Pa)  3 Density(m^-2)  4 Nnodes  5 Nsegs  6 dt
本脚本用 1/2/3 列(strain/stress/density),与旧 plot.py 一致。

设好 SEED 即自动定位五个输出;某个没跑的会跳过并提示。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 配置 ==========
SEED        = 12345
OUTPUT_PNG  = "compare_5cases.png"
# 力学指标参数(与 sim 的材料参数一致)
MU          = 82e9     # 剪切模量 [Pa] ⚠️与 caseA 等一致
NU          = 0.29     # 泊松比   ⚠️
FLOW_FRAC   = 0.40     # σ_flow = 尾部此比例数据的平均应力(稳定流动段)
OFFSET      = 0.002    # 0.2% offset 屈服
# 五工况:标签、相对脚本目录的 .dat 路径、颜色
#   A/B 直接在 output_caseX_seed/,C 在 output_caseC_pX_seed/load/
CASES = [
    ("A (无预处理)",      f"output_caseA_seed{SEED}/stress_strain_dens.dat",            "#000000"),
    ("B (预弛豫)",        f"output_caseB_seed{SEED}/stress_strain_dens.dat",            "#1F77B4"),
    ("C1 (预变形 0.1%)",  f"output_caseC_p0.1_seed{SEED}/load/stress_strain_dens.dat",  "#FFB000"),
    ("C2 (预变形 0.3%)",  f"output_caseC_p0.3_seed{SEED}/load/stress_strain_dens.dat",  "#FF6A00"),
    ("C3 (预变形 0.5%)",  f"output_caseC_p0.5_seed{SEED}/load/stress_strain_dens.dat",  "#D40010"),
]
# =========================


def load_data(filepath):
    """读 stress_strain_dens.dat,返回 strain, stress(MPa), density(m^-2)。
    只保留应变单调递增的部分(防 restart 导致应变倒退)。"""
    data = np.loadtxt(filepath, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    strain  = data[:, 1]
    stress  = data[:, 2] / 1e6     # Pa → MPa
    density = data[:, 3]
    mask = np.concatenate(([True], np.diff(strain) > 0))
    return strain[mask], stress[mask], density[mask]


def compute_metrics(strain, stress, E_MPa, flow_frac, offset):
    """返回力学指标:
       σ_peak, ε_peak(%), σ_flow, Δσ_overshoot, σ_0.2%, ε_0.2%(%)。
       (单位:应力 MPa,应变 %)"""
    ipk = int(np.argmax(stress))
    sigma_peak = stress[ipk]
    eps_peak   = strain[ipk] * 100.0

    # σ_flow:尾部 flow_frac 段平均应力(稳定流动)
    i0 = int(len(stress) * (1.0 - flow_frac))
    tail = stress[i0:]
    sigma_flow = float(np.mean(tail)) if len(tail) else np.nan
    d_over = sigma_peak - sigma_flow

    # σ_0.2%:offset 线 σ = E·(ε − offset) 与曲线首个交点(diff 由 + 变 -)
    line = E_MPa * (strain - offset)
    diff = stress - line
    sigma_02, eps_02 = np.nan, np.nan
    cross = np.where(np.diff(np.sign(diff)) < 0)[0]
    if len(cross):
        i = cross[0]
        d0, d1 = diff[i], diff[i + 1]
        t = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.0
        eps_c = strain[i] + t * (strain[i + 1] - strain[i])
        sigma_02 = E_MPa * (eps_c - offset)
        eps_02 = eps_c * 100.0
    return sigma_peak, eps_peak, sigma_flow, d_over, sigma_02, eps_02


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    E_MPa = 2.0 * MU * (1.0 + NU) / 1e6   # 杨氏模量 [MPa],用于 0.2% offset

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    for label, relpath, color in CASES:
        path = os.path.join(base_dir, relpath)
        if not os.path.exists(path):
            print(f"[跳过] 文件不存在: {relpath}")
            continue
        strain, stress, dens = load_data(path)
        strain_pct = strain * 100

        ax1.plot(strain_pct, stress, color=color, lw=1.4, label=label)
        ax2.plot(strain_pct, dens, color=color, lw=1.4, label=label)

        s_peak, e_peak, s_flow, d_over, s_02, e_02 = compute_metrics(
            strain, stress, E_MPa, FLOW_FRAC, OFFSET)

        ax1.plot(e_peak, s_peak, 'o', color=color, ms=4)              # 峰值点
        if not np.isnan(s_02):
            ax1.plot(e_02, s_02, 'x', color=color, ms=8, mew=2)      # 0.2% 屈服点

        print(f"=== {label} ===")
        print(f"  σ_peak       = {s_peak:.1f} MPa  @ ε_peak = {e_peak:.3f}%")
        print(f"  σ_flow       = {s_flow:.1f} MPa  (尾部 {int(FLOW_FRAC*100)}% 均值)")
        print(f"  Δσ_overshoot = {d_over:.1f} MPa  (= σ_peak − σ_flow)")
        if not np.isnan(s_02):
            print(f"  σ_0.2%       = {s_02:.1f} MPa  @ ε = {e_02:.3f}%")
        else:
            print(f"  σ_0.2%       = 未找到交点(曲线始终低于 offset 线)")
        print(f"  密度 初 {dens[0]:.3e} → 末 {dens[-1]:.3e} m^-2")

    # 左:应力-应变
    ax1.set_xlabel('Strain (%)', fontsize=13)
    ax1.set_ylabel('Stress (MPa)', fontsize=13)
    ax1.set_title('应力-应变(五工况对比)', fontsize=14)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, ls='--')
    ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)

    # 右:应变-位错密度
    ax2.set_xlabel('Strain (%)', fontsize=13)
    ax2.set_ylabel(r'Dislocation Density (m$^{-2}$)', fontsize=13)
    ax2.set_title('应变-位错密度(五工况对比)', fontsize=14)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, ls='--')
    ax2.set_yscale('log')
    ax2.set_xlim(left=0)

    plt.tight_layout(pad=2.0)
    out = os.path.join(base_dir, OUTPUT_PNG)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图像已保存: {out}")


if __name__ == '__main__':
    main()
