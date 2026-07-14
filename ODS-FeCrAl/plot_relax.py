#!/usr/bin/env python3
"""弛豫判敛图:位错密度 vs 物理时间(不是步数、不是 dt)
数据源 = relax_config.py 输出的 stress_strain_dens.dat
列序(与 out_props 一致): step  time  dt  density  nnodes
曲线趋平 = 弛豫收敛;仍在下滑 = 从 relaxed_config.data 续跑。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 配置 ==========
DAT    = "output_relax_seed12345/stress_strain_dens.dat"
OUTPUT = "relax_density_time.png"
# =========================


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    dat = DAT if os.path.isabs(DAT) else os.path.join(base, DAT)

    data = np.loadtxt(dat, comments='#')
    time, dens = data[:, 1], data[:, 3]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.plot(time * 1e9, dens, color='#1F77B4', lw=1.5)
    ax.set_xlabel('Time (ns)', fontsize=13)
    ax.set_ylabel(r'Dislocation Density (m$^{-2}$)', fontsize=13)
    ax.set_title('Zero-stress Relaxation: Density vs Time', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

    out = os.path.join(base, OUTPUT) if not os.path.isabs(OUTPUT) else OUTPUT
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'数据点: {len(time)}, 总物理时间: {time[-1]*1e9:.2f} ns')
    # 起点跳过第一行:step=1 的记录可能异常偏低(2026-07 弛豫炉实测);
    # 权威保持率以 relax_config.py 打印的构型实算前/后密度为准
    d0 = dens[1] if len(dens) > 1 else dens[0]
    print(f'密度: {d0:.4e} -> {dens[-1]:.4e} /m^2 (保持率 {dens[-1]/d0*100:.1f}%)')
    tail = dens[int(0.8*len(dens)):]
    drift = (tail[-1] - tail[0]) / tail[0] * 100
    print(f'末段20%漂移: {drift:+.2f}%  (|漂移|<1% 可视为趋平)')
    print(f'图像: {out}')


if __name__ == '__main__':
    main()
