#!/usr/bin/env python3
"""
追踪触发 MAX_CONN 的节点从诞生到报错前的完整历史。

用法:
  python track_node.py --sim-dir output_Cu_twin
  python track_node.py --sim-dir output_Cu_twin --tag 0 1850
  python track_node.py --sim-dir output_Cu_twin --log debug_max_conn_0.log
"""
import os
import sys
import glob
import re
import argparse
import numpy as np


# ============================================================
# 解析 ParaDiS/ExaDiS .data 文件
# ============================================================
def parse_data_file(filepath):
    """
    解析 ExaDiS .data 文件，返回:
      nodes: list of dict { tag:(d,i), pos:np.array(3), arms:int }
      segs:  list of dict { n1:(d,i), n2:(d,i) }
    """
    nodes = {}   # tag -> {pos, arms, neighbors}
    segs  = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # ---- 找 fileParamList / Node data 区段 ----
    i = 0
    total_lines = len(lines)

    # 跳过文件头，定位到 "nodalData" 或节点列表开头
    nodal_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('nodalData') or s == 'nodalData':
            nodal_start = i + 1
            break
        # 有些格式用 "# nodes" 注释
        if re.match(r'#\s*nodes', s, re.IGNORECASE):
            nodal_start = i + 1
            break

    if nodal_start is None:
        # 尝试直接找节点行：格式 "domain index"
        nodal_start = 0

    # ---- 解析节点块 ----
    # ParaDiS 格式：
    #   <domain> <index>
    #   <x> <y> <z>  <constraint>
    #   <numArms>
    #   <arm_d> <arm_i>
    #   <bx> <by> <bz>
    #   <nx> <ny> <nz>
    #   ... (repeat numArms times)
    i = nodal_start
    while i < total_lines:
        line = lines[i].strip()
        # 跳过注释和空行
        if not line or line.startswith('#'):
            i += 1
            continue

        # 尝试匹配节点头行：两个整数（domain index）
        m = re.match(r'^(-?\d+)\s+(-?\d+)\s*$', line)
        if not m:
            i += 1
            continue

        domain = int(m.group(1))
        index  = int(m.group(2))
        tag    = (domain, index)

        # 下一行：位置 + 约束
        i += 1
        if i >= total_lines:
            break
        pos_line = lines[i].strip()
        if not pos_line or pos_line.startswith('#'):
            continue
        parts = pos_line.split()
        if len(parts) < 3:
            continue
        try:
            pos = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            i += 1
            continue

        # 下一行：numArms
        i += 1
        if i >= total_lines:
            break
        arms_line = lines[i].strip()
        try:
            num_arms = int(arms_line.split()[0])
        except (ValueError, IndexError):
            i += 1
            continue

        # 跳过每个 arm 的三行
        neighbors = []
        for _ in range(num_arms):
            i += 1
            if i >= total_lines:
                break
            arm_line = lines[i].strip()
            arm_parts = arm_line.split()
            if len(arm_parts) >= 2:
                try:
                    neighbors.append((int(arm_parts[0]), int(arm_parts[1])))
                except ValueError:
                    pass
            i += 1  # burg
            i += 1  # normal

        nodes[tag] = {
            'pos':       pos,
            'arms':      num_arms,
            'neighbors': neighbors,
        }
        i += 1

    # 从邻居信息重建线段（去重）
    seen = set()
    for tag, nd in nodes.items():
        for nb in nd['neighbors']:
            key = tuple(sorted([tag, nb]))
            if key not in seen:
                seen.add(key)
                segs.append({'n1': tag, 'n2': nb})

    return nodes, segs


# ============================================================
# 读取 log 文件，提取触发节点 tag
# ============================================================
def read_log_tags(log_file):
    """
    从 debug_max_conn_*.log 读取 n1, n2 的 tag。
    返回 list of (domain, index)
    """
    tags = []
    if not os.path.exists(log_file):
        return tags
    with open(log_file) as f:
        for line in f:
            # # n1: tag=(0,1850), ...
            m = re.search(r'tag=\((\d+),(\d+)\)', line)
            if m:
                tags.append((int(m.group(1)), int(m.group(2))))
    return tags


# ============================================================
# 提取步骤编号
# ============================================================
def step_of(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


# ============================================================
# 写 VTK PolyData（轨迹线 + 节点连接数随时间变化）
# ============================================================
def write_trajectory_vtk(outfile, history):
    """
    history: list of { step, pos, arms }
    生成折线 VTK，点颜色表示该步的 arms（连接数）。
    """
    if not history:
        print("没有历史数据，跳过 VTK 写入。")
        return

    pts    = np.array([h['pos']  for h in history])
    arms   = np.array([h['arms'] for h in history], dtype=int)
    steps  = np.array([h['step'] for h in history], dtype=int)
    N = len(pts)

    with open(outfile, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Node trajectory\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {N} float\n")
        for p in pts:
            f.write(f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")

        # Lines（把所有点连成一条折线）
        f.write(f"LINES 1 {N+1}\n")
        f.write(f"{N}")
        for k in range(N):
            f.write(f" {k}")
        f.write("\n")

        # Point data
        f.write(f"POINT_DATA {N}\n")

        f.write("SCALARS NodeDegree int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for a in arms:
            f.write(f"{a}\n")

        f.write("SCALARS Step int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for s in steps:
            f.write(f"{s}\n")

    print(f"轨迹 VTK 已写入: {outfile}  ({N} 个时间步)")


# ============================================================
# 写 VTK — 最终时刻节点 + 邻居（展示触发瞬间的局部网络）
# ============================================================
def write_local_network_vtk(outfile, nodes, target_tag, target_step):
    """
    输出触发瞬间的局部网络：目标节点 + 所有直接邻居。
    """
    if target_tag not in nodes:
        print(f"目标节点 {target_tag} 在 debug 文件中未找到。")
        return

    nd = nodes[target_tag]
    involved_tags = [target_tag] + nd['neighbors']
    # 只保留文件中存在的节点
    involved_tags = [t for t in involved_tags if t in nodes]

    # 建立 tag -> 点索引 映射
    tag_to_idx = {t: i for i, t in enumerate(involved_tags)}
    pts = np.array([nodes[t]['pos'] for t in involved_tags])
    arms_vals = np.array([nodes[t]['arms'] for t in involved_tags], dtype=int)
    is_target = np.array([1 if t == target_tag else 0 for t in involved_tags], dtype=int)

    # 收集线段（只取涉及目标节点的）
    local_segs = []
    for nb in nd['neighbors']:
        if nb in tag_to_idx:
            local_segs.append((tag_to_idx[target_tag], tag_to_idx[nb]))

    N = len(pts)
    M = len(local_segs)

    with open(outfile, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Local network at step {target_step} around node {target_tag}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {N} float\n")
        for p in pts:
            f.write(f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")

        if M > 0:
            f.write(f"LINES {M} {M*3}\n")
            for a, b in local_segs:
                f.write(f"2 {a} {b}\n")

        f.write(f"POINT_DATA {N}\n")
        f.write("SCALARS NodeDegree int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for a in arms_vals:
            f.write(f"{a}\n")
        f.write("SCALARS IsTarget int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in is_target:
            f.write(f"{v}\n")

    print(f"局部网络 VTK 已写入: {outfile}  ({N} 节点, {M} 线段)")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='追踪触发 MAX_CONN 节点的历史')
    parser.add_argument('--sim-dir',  type=str, default='output_Cu_twin',
                        help='模拟输出目录（含 config.*.data 和 debug_max_conn_*.log）')
    parser.add_argument('--log',      type=str, default=None,
                        help='指定 debug log 文件路径（默认自动在 sim-dir 中搜索）')
    parser.add_argument('--tag',      type=int, nargs=2, default=None,
                        metavar=('DOMAIN', 'INDEX'),
                        help='手动指定目标节点 tag（优先于 log 文件）')
    parser.add_argument('--out-dir',  type=str, default=None,
                        help='VTK 输出目录（默认与 sim-dir 相同）')
    args = parser.parse_args()

    sim_dir = args.sim_dir
    out_dir = args.out_dir or sim_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- 确定目标节点 tag ----
    target_tags = []

    if args.tag:
        target_tags = [tuple(args.tag)]
        print(f"手动指定目标节点: {target_tags}")
    else:
        # 自动寻找 debug log
        log_pattern = os.path.join(sim_dir, 'debug_max_conn_*.log')
        log_files   = sorted(glob.glob(log_pattern))

        # 也找当前目录
        if not log_files:
            log_files = sorted(glob.glob('debug_max_conn_*.log'))

        if args.log:
            log_files = [args.log] + log_files

        for lf in log_files[:1]:  # 只处理第一个
            tags = read_log_tags(lf)
            target_tags.extend(tags)
            print(f"从 {lf} 读取到触发节点: {tags}")

    if not target_tags:
        print("未找到目标节点，请通过 --tag DOMAIN INDEX 手动指定。")
        sys.exit(1)

    # ---- 收集所有 config.*.data 文件（按步骤排序）----
    config_pattern = os.path.join(sim_dir, 'config.*.data')
    config_files   = sorted(glob.glob(config_pattern), key=step_of)

    if not config_files:
        print(f"在 {sim_dir} 中未找到 config.*.data 文件。")
        sys.exit(1)

    print(f"\n共找到 {len(config_files)} 个配置文件，步骤范围: "
          f"{step_of(config_files[0])} ~ {step_of(config_files[-1])}")

    # ---- 同时加载 debug 文件（最终状态）----
    debug_files = sorted(
        glob.glob(os.path.join(sim_dir, 'debug_max_conn_*.data')) +
        glob.glob('debug_max_conn_*.data')
    )

    # ---- 对每个目标 tag 追踪历史 ----
    for target_tag in target_tags:
        print(f"\n{'='*60}")
        print(f"追踪节点 tag=({target_tag[0]},{target_tag[1]})")
        print(f"{'='*60}")

        history = []
        first_seen_step = None
        last_seen_step  = None

        for cf in config_files:
            step = step_of(cf)
            nodes, _ = parse_data_file(cf)
            if target_tag in nodes:
                nd = nodes[target_tag]
                history.append({
                    'step': step,
                    'pos':  nd['pos'],
                    'arms': nd['arms'],
                })
                if first_seen_step is None:
                    first_seen_step = step
                last_seen_step = step

        if not history:
            print(f"  在所有 config 文件中均未找到节点 {target_tag}。")
            print("  可能是该节点在最后一批步骤才新建，没有被 write_freq 捕获。")
            print("  建议减小 write_freq（如改为 10 或 1）后重新模拟。")
            continue

        print(f"  首次出现: 步骤 {first_seen_step}")
        print(f"  最后出现: 步骤 {last_seen_step}")
        print(f"  共记录  : {len(history)} 个快照")
        print(f"\n  连接数变化历史:")
        print(f"  {'步骤':>8}  {'连接数':>6}  {'位置 (b)':}")
        for h in history:
            p = h['pos']
            print(f"  {h['step']:>8}  {h['arms']:>6}  "
                  f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

        # ---- 写轨迹 VTK ----
        tag_str  = f"d{target_tag[0]}_i{target_tag[1]}"
        traj_vtk = os.path.join(out_dir, f"trajectory_{tag_str}.vtk")
        write_trajectory_vtk(traj_vtk, history)

        # ---- 从 debug 文件写局部网络 VTK ----
        for df in debug_files[:1]:
            debug_step = step_of(df)
            d_nodes, _ = parse_data_file(df)
            local_vtk  = os.path.join(out_dir, f"local_network_{tag_str}_step{debug_step}.vtk")
            write_local_network_vtk(local_vtk, d_nodes, target_tag, debug_step)

    print("\n完成。")
    print("在 ParaView 中:")
    print("  1. 打开 trajectory_*.vtk  → 用 NodeDegree 着色，观察连接数随时间增长")
    print("  2. 打开 local_network_*.vtk → 用 IsTarget 着色，红色节点为触发节点")
    print("  3. 同时加载对应步骤的 config.*.vtk，了解全局网络背景")


if __name__ == '__main__':
    main()
