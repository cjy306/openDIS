"""
画弛豫收敛曲线: 位错密度 vs 时间 (relax_fcc.py 的输出)

判据: density vs Time 走平 = 弛豫收敛(Motz Fig.1a 横轴为时间)。

读 <dir>/stress_strain_dens.dat, 按表头名取列(稳健: 有无 Time 列都能处理):
  有 Time -> 直接用; 无 Time 有 dt -> cumsum(dt) 重建; 再无 -> 退回 Step。

顺带打印平台密度(尾部 20% 均值)= rho*, 拿去喂 FR/棱柱 gen_fcc_config.py --rho。

图内文字一律英文(超算 matplotlib 无 CJK 字形, 中文渲染成豆腐块)。

用法: python plot_relax_fcc.py --dir relax_glide_seed12345
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_props(datfile):
    """按表头名读列, 返回 dict{name: array}。表头形如 '# Step Strain ... Time'。"""
    with open(datfile) as f:
        header = f.readline().lstrip('#').split()
    data = np.loadtxt(datfile, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {name: data[:, i] for i, name in enumerate(header)}


def get_time(cols, n):
    """优先 Time 列; 无则 cumsum(dt); 再无则 Step。返回 (x, xlabel)。"""
    for key in ('Time', 'time'):
        if key in cols:
            return cols[key], 'Time [s]'
    for key in ('dt', 'DT'):
        if key in cols:
            return np.cumsum(cols[key]), 'Time [s] (rebuilt from cumsum dt)'
    for key in ('Step', 'step'):
        if key in cols:
            return cols[key], 'Step (no time/dt column!)'
    return np.arange(n), 'index'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='弛豫输出目录(含 stress_strain_dens.dat)')
    parser.add_argument('--out', default=None, help='输出图片; 默认 <dir>/relax_density_time.png')
    args = parser.parse_args()

    datfile = os.path.join(args.dir, 'stress_strain_dens.dat')
    cols = load_props(datfile)
    n = len(next(iter(cols.values())))
    x, xlabel = get_time(cols, n)
    dens = cols.get('Density', cols.get('density'))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, dens, '-', color='C0')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Dislocation density [1/m^2]')
    ax.set_title('Relaxation: %s' % os.path.basename(os.path.normpath(args.dir)))
    ax.grid(True, alpha=0.3)

    out = args.out or os.path.join(args.dir, 'relax_density_time.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print('saved:', out)

    tail = dens[int(0.8 * n):]
    print('plateau density (last 20%% mean) = %.4e /m^2  <- 用作 FR/棱柱 --rho' % tail.mean())


if __name__ == '__main__':
    main()
