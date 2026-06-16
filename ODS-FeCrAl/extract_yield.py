"""
0.2% 偏移屈服强度提取 + 多随机实现平均

读取 Case A(及后续 B-F 工况)各随机种子目录下的 stress_strain_dens.dat,
对每个实现用 0.2% offset 法提取屈服强度,然后给出多实现的均值与误差棒。

stress_strain_dens.dat 列格式(由 SimulateNetworkPerf.write_results 写出):
    istep  strain  stress  density  elapsed
其中 strain = 加载方向投影应变(无量纲),stress = von Mises 等效应力(Pa)。

用法:
    # 自动汇总某工况所有种子目录(默认匹配 output_caseA_seed*)
    python extract_yield.py --glob "output_caseA_seed*"

    # 指定多个目录
    python extract_yield.py --dirs output_caseA_seed1234 output_caseA_seed1235

    # 调整弹性模量估计窗口与 offset
    python extract_yield.py --glob "output_caseA_seed*" --offset 0.002 --efit-max-strain 5e-4
"""
import os, sys, glob, argparse
import numpy as np


def read_ss(dat_file):
    """读取 stress_strain_dens.dat,返回 (strain, stress) 一维数组(单位:无量纲, Pa)。"""
    data = np.loadtxt(dat_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2:
        raise ValueError(f"{dat_file}: 数据点不足({data.shape[0]} 行),无法提取屈服")
    strain = np.abs(data[:, 1])
    stress = np.abs(data[:, 2])
    # 按应变排序(防止重启拼接乱序)
    order = np.argsort(strain)
    return strain[order], stress[order]


def estimate_youngs_modulus(strain, stress, efit_max_strain):
    """用初始弹性段(strain < efit_max_strain)线性拟合估计有效杨氏模量 E。
    若该窗口点数不足,退化为用前若干点。
    """
    mask = strain < efit_max_strain
    if np.count_nonzero(mask) >= 2:
        s, e = stress[mask], strain[mask]
    else:
        # 退化:取前 5 个点(至少 2 个)
        k = max(2, min(5, len(strain)))
        s, e = stress[:k], strain[:k]
    # 过原点最小二乘:E = (e·s) / (e·e)
    E = float(np.dot(e, s) / np.dot(e, e))
    return E


def offset_yield(strain, stress, offset, E):
    """0.2% offset 法:求 应力曲线 与 直线 stress = E*(strain - offset) 的首个交点。
    返回屈服应力(Pa)。找不到交点时返回曲线峰值应力作为兜底(并告警)。
    """
    offset_line = E * (strain - offset)
    diff = stress - offset_line
    # 寻找 diff 由正变负(曲线被 offset 线追上)的首个穿越点
    sign = np.sign(diff)
    cross = np.where(np.diff(sign) < 0)[0]
    if len(cross) == 0:
        return float(np.max(stress)), False
    i = cross[0]
    # 在 [i, i+1] 间线性插值求交点应力
    d0, d1 = diff[i], diff[i + 1]
    if d1 == d0:
        return float(stress[i]), True
    t = d0 / (d0 - d1)
    sy = stress[i] + t * (stress[i + 1] - stress[i])
    return float(sy), True


def main():
    ap = argparse.ArgumentParser(description="0.2% 偏移屈服提取 + 多实现平均")
    ap.add_argument('--dirs', nargs='+', help='显式指定种子输出目录列表')
    ap.add_argument('--glob', default='output_caseA_seed*', help='种子目录匹配模式(默认 output_caseA_seed*)')
    ap.add_argument('--offset', type=float, default=0.002, help='偏移量(默认 0.002 = 0.2%%)')
    ap.add_argument('--efit-max-strain', type=float, default=5e-4,
                    help='弹性模量拟合的应变上限(默认 5e-4)')
    ap.add_argument('--datname', default='stress_strain_dens.dat', help='数据文件名')
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dirs:
        dirs = [d if os.path.isabs(d) else os.path.join(base_dir, d) for d in args.dirs]
    else:
        dirs = sorted(glob.glob(os.path.join(base_dir, args.glob)))

    if not dirs:
        print(f"未找到匹配目录: {args.glob}")
        sys.exit(1)

    print("=" * 64)
    print(f"0.2% 偏移屈服提取  (offset={args.offset}, E 拟合窗口 strain<{args.efit_max_strain})")
    print("=" * 64)

    yields = []
    for d in dirs:
        dat = os.path.join(d, args.datname)
        if not os.path.isfile(dat):
            print(f"  [跳过] {os.path.basename(d)}: 无 {args.datname}")
            continue
        try:
            strain, stress = read_ss(dat)
            E = estimate_youngs_modulus(strain, stress, args.efit_max_strain)
            sy, ok = offset_yield(strain, stress, args.offset, E)
        except Exception as e:
            print(f"  [错误] {os.path.basename(d)}: {e}")
            continue
        flag = "" if ok else "  ⚠️未找到offset交点,取峰值兜底"
        print(f"  {os.path.basename(d):28s}  E≈{E/1e9:7.1f} GPa   σy = {sy/1e6:8.2f} MPa{flag}")
        yields.append(sy)

    if not yields:
        print("没有可用的屈服数据。")
        sys.exit(1)

    yields = np.array(yields)
    print("-" * 64)
    print(f"  实现数 N = {len(yields)}")
    print(f"  屈服强度均值  σy = {yields.mean()/1e6:.2f} MPa")
    if len(yields) > 1:
        print(f"  标准差        std = {yields.std(ddof=1)/1e6:.2f} MPa")
        print(f"  范围(误差棒)  [{yields.min()/1e6:.2f}, {yields.max()/1e6:.2f}] MPa")
    print("=" * 64)


if __name__ == "__main__":
    main()
