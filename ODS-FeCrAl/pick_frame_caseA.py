#!/usr/bin/env python3
"""
pick_frame_caseA.py —— 从 Case A 预变形历史里挑"流变平台代表帧"

判据(对应 PLAN_NOW 挑帧三条):
  1. 应变窗口 0.85%–1.0%(σ 平台 + ρ 趋稳同时满足的区段,默认可调)
  2. 只考虑磁盘上真实存在的帧:步号 % write_freq == 0 → config.<步号>.data
  3. 应力、密度都接近各自滑动均值(排掉尖峰帧=卡高应力构型、谷帧=刚雪崩完)
     分数 = max(|σ-σ̄|/σ̄, |ρ-ρ̄|/ρ̄),越小越好

用法(昆山,放在 ODS-FeCrAl 目录下):
  python pick_frame_caseA.py
  python pick_frame_caseA.py --dir output_caseA_seed12345 --wlo 0.85 --whi 1.0
输出:前 10 名候选帧,选定后 paraview 确认三维网络,再交给弛豫。
"""
import os
import argparse
import numpy as np

WRITE_FREQ = 100   # 与 test_caseA_baseline.py 的 write_freq 一致


def load_history(dat_file):
    """step strain stress(Pa) density(m^-2) [...];restart 单调保护同 plot.py。"""
    data = np.loadtxt(dat_file, comments='#')
    strain = data[:, 1]
    mask = np.concatenate(([True], np.diff(strain) > 0))
    return data[mask, 0].astype(int), strain[mask], data[mask, 2] / 1e6, data[mask, 3]


def running_mean(x_strain, y, half_win):
    """以应变为坐标的滑动均值(窗口 ±half_win,应变无量纲)。"""
    ybar = np.empty_like(y)
    for i, s in enumerate(x_strain):
        sel = np.abs(x_strain - s) <= half_win
        ybar[i] = y[sel].mean()
    return ybar


def main():
    p = argparse.ArgumentParser(description='挑 Case A 流变平台代表帧')
    p.add_argument('--dir',    type=str,   default='output_caseA_seed12345')
    p.add_argument('--wlo',    type=float, default=0.85, help='窗口下限 [%%]')
    p.add_argument('--whi',    type=float, default=1.00, help='窗口上限 [%%]')
    p.add_argument('--smooth', type=float, default=0.025, help='滑动均值半窗 [%%应变]')
    p.add_argument('--top',    type=int,   default=10)
    args = p.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.dir if os.path.isabs(args.dir) else os.path.join(base, args.dir)
    dat = os.path.join(out_dir, 'stress_strain_dens.dat')

    step, strain, stress, dens = load_history(dat)

    # 滑动均值在窗口外侧也要有数据支撑 → 先在"窗口±半窗"范围内算
    lo, hi, hw = args.wlo / 100, args.whi / 100, args.smooth / 100
    wide = (strain >= lo - hw) & (strain <= hi + hw)
    s_w, st_w, de_w, step_w = strain[wide], stress[wide], dens[wide], step[wide]
    st_bar = running_mean(s_w, st_w, hw)
    de_bar = running_mean(s_w, de_w, hw)

    # 窗口内 + 帧存在(步号是 write_freq 倍数)的候选
    cand = []
    for i in range(len(s_w)):
        if not (lo <= s_w[i] <= hi):
            continue
        if step_w[i] % WRITE_FREQ != 0:
            continue
        dev_s = abs(st_w[i] - st_bar[i]) / st_bar[i]
        dev_d = abs(de_w[i] - de_bar[i]) / de_bar[i]
        cand.append((max(dev_s, dev_d), step_w[i], s_w[i], st_w[i], de_w[i], dev_s, dev_d))

    if not cand:
        print(f'窗口 [{args.wlo}%, {args.whi}%] 内没有 %{WRITE_FREQ}==0 的帧,检查窗口或 write_freq')
        return
    cand.sort(key=lambda c: c[0])

    print(f'历史: {dat}')
    print(f'窗口 [{args.wlo}%, {args.whi}%], 滑动均值半窗 ±{args.smooth}%应变, '
          f'候选帧 {len(cand)} 个, 前 {min(args.top, len(cand))} 名:\n')
    print(f'{"步号":>8} {"应变%":>8} {"应力MPa":>9} {"密度m^-2":>11} '
          f'{"偏σ%":>7} {"偏ρ%":>7}  文件')
    for c in cand[:args.top]:
        score, stp, s, st, de, dev_s, dev_d = c
        fname = f'config.{stp}.data'
        exists = '✓' if os.path.isfile(os.path.join(out_dir, fname)) else '✗缺失'
        print(f'{stp:>8d} {s*100:>8.4f} {st:>9.1f} {de:>11.3e} '
              f'{dev_s*100:>6.1f} {dev_d*100:>6.1f}  {fname} {exists}')

    best = cand[0]
    print(f'\n推荐: config.{best[1]}.data  (应变 {best[2]*100:.4f}%, '
          f'应力 {best[3]:.1f} MPa, 密度 {best[4]:.3e} /m^2)')
    print('下一步: paraview 确认三维网络(非孤立弓、12 系均有) → 零外力弛豫')


if __name__ == '__main__':
    main()
