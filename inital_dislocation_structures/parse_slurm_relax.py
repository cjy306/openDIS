#!/usr/bin/env python3
"""从 slurm 作业日志 + stress_strain_dens.dat 合并出弛豫收敛曲线。

应急用:当超算上的 pyexadis_base.py 还是旧版(.dat 只有 4 列、无 Nnodes/dt)时,
nodes 和 dt 其实都在 slurm 作业日志的每步打印里:
    Step =   1: nodes = 3585, dt = 5.000000e-10, time = ..., elapsed = ...
本脚本从日志抓 (step, nodes, dt),从 .dat 抓 (step, density),按 step 合并画图。

(等 pyexadis_base.py 的 7 列版部署到超算后,直接用 plot_relax.py 即可,不再需要本脚本。)
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 配置 ==========
SLURM_LOG  = "slurm-c3.out"                              # 作业日志(含每步 nodes/dt)
DAT_FILE   = "/data/home/dg000246b/openDIS/inital_dislocation_structures/output_caseC_0.28%_seed12345/relax2/stress_strain_dens.dat"  # 含 density
OUTPUT_PNG = "relax_convergence_from_slurm_c3.png"
TAIL_FRAC  = 0.20
RHO_TOL    = 0.01
# =========================

# 匹配:Step =   123: nodes = 3585, dt = 5.000000e-10
STEP_RE = re.compile(
    r"[Ss]tep\s*=\s*(\d+):\s*nodes\s*=\s*(\d+),\s*dt\s*=\s*([0-9.eE+\-]+)")


def parse_slurm(path):
    steps, nodes, dts = [], [], []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                nodes.append(int(m.group(2)))
                dts.append(float(m.group(3)))
    return np.array(steps), np.array(nodes), np.array(dts)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    slurm_path = os.path.join(base, SLURM_LOG)
    dat_path   = os.path.join(base, DAT_FILE)

    if not os.path.exists(slurm_path):
        print(f"[错误] 日志不存在: {SLURM_LOG}"); return
    if not os.path.exists(dat_path):
        print(f"[错误] .dat 不存在: {DAT_FILE}"); return

    s_step, s_nodes, s_dt = parse_slurm(slurm_path)
    if len(s_step) == 0:
        print("[错误] 日志里没匹配到 'Step = N: nodes = ..., dt = ...' 行,检查日志格式"); return

    dat = np.loadtxt(dat_path, comments='#')
    if dat.ndim == 1:
        dat = dat.reshape(1, -1)
    d_step = dat[:, 0].astype(int)
    d_rho  = dat[:, 3]
    rho_of = {int(k): v for k, v in zip(d_step, d_rho)}

    # 按 step 合并(取日志与 .dat 都有的步)
    step, nodes, dt, rho = [], [], [], []
    for st, nd, dd in zip(s_step, s_nodes, s_dt):
        if st in rho_of:
            step.append(st); nodes.append(nd); dt.append(dd); rho.append(rho_of[st])
    step = np.array(step); nodes = np.array(nodes); dt = np.array(dt); rho = np.array(rho)
    if len(step) == 0:
        print("[错误] 日志步与 .dat 步无交集"); return

    # ---- 收敛判据 ----
    n = len(step); i0 = int(n * (1 - TAIL_FRAC))
    rt = rho[i0:]
    rel = (rt.max() - rt.min()) / np.mean(rt) if len(rt) else np.nan
    conv = rel < RHO_TOL
    print(f"=== 弛豫收敛(日志+dat 合并): {SLURM_LOG} ===")
    print(f"  合并点 {n}, 步范围 {step.min()}~{step.max()}")
    print(f"  ρ: 初 {rho[0]:.3e} → 末 {rho[-1]:.3e} (末/初 {rho[-1]/rho[0]*100:.1f}%)")
    print(f"  尾部 {int(TAIL_FRAC*100)}% ρ 相对变化 {rel*100:.2f}%  ({'已走平' if conv else '未走平'})")
    print(f"  nodes: 初 {nodes[0]} → 末 {nodes[-1]};  dt: 初 {dt[0]:.2e} → 末 {dt[-1]:.2e} s")
    if rho[-1] < 0.5 * rho[0]:
        print("  ⚠️ ρ 跌破初值一半 —— 可能环坍缩/被 remesh 吃掉,查环半径/maxseg")

    # ---- 画图 ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')
    ax1.plot(step, rho, color="#1F77B4", lw=1.3)
    ax1.axvspan(step[i0], step[-1], color='gray', alpha=0.12, label=f'tail {int(TAIL_FRAC*100)}%')
    ax1.set_xlabel('Step'); ax1.set_ylabel(r'Density (m$^{-2}$)'); ax1.set_title('ρ vs Step')
    ax1.grid(True, alpha=0.3, ls='--'); ax1.legend()

    ax2.plot(step, nodes, color="#D41010", lw=1.3, label='Nnodes')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Nodes'); ax2.set_title('Nnodes vs Step')
    ax2.grid(True, alpha=0.3, ls='--'); ax2.legend()

    ax3.plot(step, dt, color="#9467BD", lw=1.3)
    ax3.set_xlabel('Step'); ax3.set_ylabel('dt (s)'); ax3.set_yscale('log')
    ax3.set_title('dt vs Step'); ax3.grid(True, alpha=0.3, ls='--', which='both')

    plt.tight_layout(pad=2.0)
    out = os.path.join(base, OUTPUT_PNG)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图像已保存: {out}")


if __name__ == '__main__':
    main()
