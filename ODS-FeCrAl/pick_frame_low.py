#!/usr/bin/env python3
"""
pick_frame_low.py —— 从弛豫历史里挑"低档亚稳态"代表帧(分支敏感性检查用)

与选 config.9800(高档/主档)完全对称的判据,档位换成低档:
  1. 末段 40% 数据,10/90 分位中点 = 高低档分界线
  2. 低档水平 = 分界线以下点的中位数
  3. 候选帧 = %WRITE_FREQ==0 的存盘快照,且
     - 自身密度在低档
     - 前后 ±NEIGH 步邻域中位也在低档(驻留检查,排除跳变途中的过渡态)
     - 距低档水平最近
输出:低档推荐帧文件名 + 前 5 名候选表。

用法(昆山):
  python pick_frame_low.py
  python pick_frame_low.py --dir output_relax_seed12345 --neigh 100
"""
import os
import argparse
import numpy as np

WRITE_FREQ = 100   # 与 relax_config.py 的 write_freq 一致


def main():
    p = argparse.ArgumentParser(description='挑低档亚稳态代表帧')
    p.add_argument('--dir',   type=str, default='output_relax_seed12345')
    p.add_argument('--tail',  type=float, default=0.4, help='末段数据占比(默认 40%%)')
    p.add_argument('--neigh', type=int, default=100, help='驻留检查半窗 [步]')
    p.add_argument('--top',   type=int, default=5)
    args = p.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.dir if os.path.isabs(args.dir) else os.path.join(base, args.dir)
    dat = os.path.join(out_dir, 'stress_strain_dens.dat')

    d = np.loadtxt(dat, comments='#')
    if len(d) > 1:
        d = d[1:]                          # 跳过 step=1 异常行
    step, dens = d[:, 0].astype(int), d[:, 3]

    n = len(dens)
    lo = int((1.0 - args.tail) * n)
    S, R = step[lo:], dens[lo:]

    thr = (np.percentile(R, 10) + np.percentile(R, 90)) / 2
    low_pts = R[R < thr]
    if len(low_pts) == 0:
        print('末段没有低档数据点——曲线可能没有双稳态,直接用主档帧即可')
        return
    low_med = np.median(low_pts)
    print(f'历史: {dat}')
    print(f'末段 {args.tail*100:.0f}%: 高低档分界 {thr:.4e}, 低档水平(中位) {low_med:.4e}, '
          f'低档占时 {len(low_pts)/len(R)*100:.0f}%\n')

    cand = []
    for i in range(len(S)):
        if S[i] % WRITE_FREQ != 0:
            continue
        j = np.where(step == S[i])[0][0]
        nb = dens[max(0, j - args.neigh): j + args.neigh]
        if np.median(nb) >= thr:           # 邻域整体不在低档 → 跳变途中或高档,排除
            continue
        cand.append((abs(R[i] - low_med), S[i], R[i]))

    if not cand:
        print(f'无合格低档快照(驻留半窗 ±{args.neigh} 步内没有整段低档)。'
              f'试试 --neigh {args.neigh//2}')
        return
    cand.sort(key=lambda c: c[0])

    print(f'{"步号":>8} {"密度m^-2":>12} {"距低档水平%":>10}  文件')
    for dv, s_, r_ in cand[:args.top]:
        fname = f'config.{s_}.data'
        exists = '✓' if os.path.isfile(os.path.join(out_dir, fname)) else '✗缺失'
        print(f'{s_:>8d} {r_:>12.4e} {dv/low_med*100:>9.2f}   {fname} {exists}')

    best = cand[0]
    print(f'\n低档推荐帧: config.{best[1]}.data  (密度 {best[2]:.4e}, 距低档水平 {best[0]/low_med*100:.2f}%)')
    print('下一步: python test_caseA_baseline.py '
          f'--init {args.dir}/config.{best[1]}.data --out output_caseA_low')


if __name__ == '__main__':
    main()
