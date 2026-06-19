#!/usr/bin/env python3
"""弛豫收敛画图:读 stress_strain_dens.dat,画 Step 对 ρ / Nnodes&Nsegs / dt 三条曲线。

用于 Phase-1 诊断弛豫(diagnose_relax.py 的输出),判断"弛豫到位":
  - Density(ρ)走平
  - Nnodes / Nsegs 不再漂移
  - dt 爬升并稳定在 maxdt 附近
三者同时满足的步数(加余量)即 caseB/C 该用的 RELAX_STEPS。

新版 stress_strain_dens.dat 列(0 起):
  0 Step  1 Strain  2 Stress  3 Density  4 Nnodes  5 Nsegs  6 dt
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 配置 ==========
DAT_FILE   = "output_diag_relax_seed12345/stress_strain_dens.dat"  # 自己改
OUTPUT_PNG = "relax_convergence.png"
TAIL_FRAC  = 0.20    # 尾部窗口比例(判收敛用)
RHO_TOL    = 0.01    # ρ 相对变化阈值(尾部窗口内 < 此值视为走平)
# =========================


def main():
    if not os.path.exists(DAT_FILE):
        print(f"[错误] 文件不存在: {DAT_FILE}")
        return

    data = np.loadtxt(DAT_FILE, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 7:
        print(f"[错误] 列数={data.shape[1]} < 7,这不是新版含 Nnodes/Nsegs/dt 的输出。"
              f"\n  确认跑的是更新过的 pyexadis_base.py(7 列)。")
        return

    step   = data[:, 0]
    rho    = data[:, 3]
    nnodes = data[:, 4]
    nsegs  = data[:, 5]
    dt     = data[:, 6]

    # ---- 收敛判据:尾部窗口内 ρ 相对变化 ----
    n = len(step)
    i0 = int(n * (1.0 - TAIL_FRAC))
    rho_tail = rho[i0:]
    rho_rel_change = (rho_tail.max() - rho_tail.min()) / np.mean(rho_tail) if len(rho_tail) else np.nan
    converged = rho_rel_change < RHO_TOL

    print(f"=== 弛豫收敛诊断: {DAT_FILE} ===")
    print(f"  总步数: {int(step[-1])}, 数据点: {n}")
    print(f"  ρ: 初 {rho[0]:.3e} → 末 {rho[-1]:.3e} m^-2 (末/初 = {rho[-1]/rho[0]*100:.1f}%)")
    print(f"  尾部 {int(TAIL_FRAC*100)}% 窗口内 ρ 相对变化 = {rho_rel_change*100:.2f}%  "
          f"({'<' if converged else '>='} {RHO_TOL*100:.0f}% → {'已走平' if converged else '未走平'})")
    print(f"  Nnodes: 初 {int(nnodes[0])} → 末 {int(nnodes[-1])};  Nsegs: 初 {int(nsegs[0])} → 末 {int(nsegs[-1])}")
    print(f"  dt: 初 {dt[0]:.2e} → 末 {dt[-1]:.2e} s")
    if rho[-1] < 0.5 * rho[0]:
        print("  ⚠️ ρ 跌破初值一半 —— 可能是环坍缩/被 remesh 吃掉,不是正常弛豫,查环半径/maxseg")

    # ---- 画图 ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')

    # ρ - Step
    ax1.plot(step, rho, color="#1F77B4", lw=1.3)
    ax1.axvspan(step[i0], step[-1], color='gray', alpha=0.12, label=f'尾部 {int(TAIL_FRAC*100)}%')
    ax1.set_xlabel('Step'); ax1.set_ylabel(r'Dislocation Density (m$^{-2}$)')
    ax1.set_title('ρ vs Step'); ax1.grid(True, alpha=0.3, ls='--'); ax1.legend()

    # Nnodes / Nsegs - Step
    ax2.plot(step, nnodes, color="#D41010", lw=1.3, label='Nnodes')
    ax2.plot(step, nsegs,  color="#2CA02C", lw=1.3, label='Nsegs')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Count')
    ax2.set_title('拓扑 (Nnodes/Nsegs) vs Step'); ax2.grid(True, alpha=0.3, ls='--'); ax2.legend()

    # dt - Step (log y)
    ax3.plot(step, dt, color="#9467BD", lw=1.3)
    ax3.set_xlabel('Step'); ax3.set_ylabel('dt (s)')
    ax3.set_yscale('log')
    ax3.set_title('时间步 dt vs Step'); ax3.grid(True, alpha=0.3, ls='--', which='both')

    plt.tight_layout(pad=2.0)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图像已保存: {OUTPUT_PNG}")


if __name__ == '__main__':
    main()
