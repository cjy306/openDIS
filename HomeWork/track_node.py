#!/usr/bin/env python3
"""
追踪触发 MAX_CONN 的节点从诞生到报错前的完整历史。

data 文件里写的是数组下标（0, 1, ..., N-1），不是 tag.index。
因此本脚本通过 **位置** 追踪节点：从 debug log 读取触发节点的坐标，
然后在每个 config 文件里找到离该位置最近的节点。

用法:
  # 自动从 debug log 读取节点位置来追踪
  python track_node.py --sim-dir output_Cu_twin

  # 手动指定位置（Burgers 单位）
  python track_node.py --sim-dir output_Cu_twin --pos 1234.5 5678.9 9012.3

  # 指定 log 文件
  python track_node.py --sim-dir output_Cu_twin --log debug_max_conn_0.log
"""
import os
import sys
import glob
import re
import argparse
import numpy as np


# ============================================================
# 解析 ExaDiS .data 文件
# ============================================================
def parse_data_file(filepath):
    """
    解析 ExaDiS write_data() 输出的 .data 文件。
    格式: domain, index x y z numarms constraint
          domain, neighbor bx by bz
                            nx ny nz
          ... (重复 numarms 次)

    返回:
      nodes: dict { (domain, index) -> {pos, arms, neighbors} }
    """
    nodes = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)

    # 跳过文件头，定位到 "nodalData"
    nodal_start = 0
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith('nodalData'):
            nodal_start = idx + 1
            break

    # 跳过注释行（nodalData 之后的 # 行）
    i = nodal_start
    while i < total_lines and lines[i].strip().startswith('#'):
        i += 1

    # 解析节点
    while i < total_lines:
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        # 节点行格式: "domain, index x y z numarms constraint"
        # 例: "0,    0   0.123456   0.234567   0.345678    2    0"
        line_clean = line.replace(',', ' ')
        parts = line_clean.split()

        if len(parts) < 7:
            i += 1
            continue

        try:
            domain     = int(parts[0])
            index      = int(parts[1])
            x          = float(parts[2])
            y          = float(parts[3])
            z          = float(parts[4])
            num_arms   = int(parts[5])
            constraint = int(parts[6])
        except (ValueError, IndexError):
            i += 1
            continue

        tag = (domain, index)
        pos = np.array([x, y, z])

        # 解析每个 arm（2 行: neighbor+burg, normal）
        neighbors = []
        for _ in range(num_arms):
            i += 1
            if i >= total_lines:
                break
            arm_line = lines[i].strip().replace(',', ' ')
            arm_parts = arm_line.split()
            if len(arm_parts) >= 2:
                try:
                    neighbors.append((int(arm_parts[0]), int(arm_parts[1])))
                except ValueError:
                    pass
            i += 1  # 跳过 normal 行

        nodes[tag] = {
            'pos':       pos,
            'arms':      num_arms,
            'neighbors': neighbors,
        }
        i += 1

    return nodes


# ============================================================
# 在节点集合中找到离目标位置最近的节点
# ============================================================
def find_nearest_node(nodes, target_pos, max_dist=200.0):
    """
    在 nodes dict 中找到离 target_pos 最近的节点。
    返回 (tag, node_dict, distance) 或 None。
    """
    best_tag  = None
    best_node = None
    best_dist = max_dist

    for tag, nd in nodes.items():
        d = np.linalg.norm(nd['pos'] - target_pos)
        if d < best_dist:
            best_dist = d
            best_tag  = tag
            best_node = nd

    if best_tag is None:
        return None
    return best_tag, best_node, best_dist


# ============================================================
# 读取 log 文件，提取触发节点的 tag 和位置
# ============================================================
def read_log_info(log_file):
    """
    从 debug_max_conn_*.log 读取 n1, n2 的 tag 和 pos。
    返回 list of { tag:(d,i), pos:np.array(3), conn:int }
    """
    results = []
    if not os.path.exists(log_file):
        return results
    with open(log_file) as f:
        for line in f:
            # # n1: tag=(0,4583), pos=(1.23e+03,4.56e+03,7.89e+03), conn=8
            m_tag  = re.search(r'tag=\((\d+),(\d+)\)', line)
            m_pos  = re.search(r'pos=\(([-+eE\d.]+),([-+eE\d.]+),([-+eE\d.]+)\)', line)
            m_conn = re.search(r'conn=(\d+)', line)
            if m_tag and m_pos:
                results.append({
                    'tag':  (int(m_tag.group(1)), int(m_tag.group(2))),
                    'pos':  np.array([float(m_pos.group(1)),
                                      float(m_pos.group(2)),
                                      float(m_pos.group(3))]),
                    'conn': int(m_conn.group(1)) if m_conn else -1,
                })
    return results


# ============================================================
# 提取步骤编号
# ============================================================
def step_of(path):
    # 匹配 config.123.data → 123, debug_max_conn_0.data → 0
    m = re.search(r'\.(\d+)\.', os.path.basename(path))
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


# ============================================================
# 写 VTK PolyData（轨迹线 + 节点连接数随时间变化）
# ============================================================
def write_trajectory_vtk(outfile, history):
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

        f.write(f"POINTS {N} float\n")
        for p in pts:
            f.write(f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")

        f.write(f"LINES 1 {N+1}\n")
        f.write(f"{N}")
        for k in range(N):
            f.write(f" {k}")
        f.write("\n")

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
    if target_tag not in nodes:
        print(f"目标节点 {target_tag} 在 debug 文件中未找到。")
        return

    nd = nodes[target_tag]
    involved_tags = [target_tag] + nd['neighbors']
    involved_tags = [t for t in involved_tags if t in nodes]

    tag_to_idx = {t: i for i, t in enumerate(involved_tags)}
    pts = np.array([nodes[t]['pos'] for t in involved_tags])
    arms_vals = np.array([nodes[t]['arms'] for t in involved_tags], dtype=int)
    is_target = np.array([1 if t == target_tag else 0 for t in involved_tags], dtype=int)

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
    parser.add_argument('--pos',      type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='手动指定目标节点位置（Burgers 单位），通过位置追踪')
    parser.add_argument('--radius',   type=float, default=200.0,
                        help='位置搜索半径（Burgers 单位，默认 200）')
    parser.add_argument('--out-dir',  type=str, default=None,
                        help='VTK 输出目录（默认与 sim-dir 相同）')
    args = parser.parse_args()

    sim_dir = args.sim_dir
    out_dir = args.out_dir or sim_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- 确定目标位置 ----
    targets = []  # list of { pos, tag, conn, label }

    if args.pos:
        targets.append({
            'pos':   np.array(args.pos),
            'tag':   None,
            'conn':  -1,
            'label': f"pos_{args.pos[0]:.0f}_{args.pos[1]:.0f}_{args.pos[2]:.0f}",
        })
        print(f"手动指定目标位置: ({args.pos[0]:.1f}, {args.pos[1]:.1f}, {args.pos[2]:.1f})")
    else:
        # 自动寻找 debug log
        log_pattern = os.path.join(sim_dir, 'debug_max_conn_*.log')
        log_files   = sorted(glob.glob(log_pattern))
        if not log_files:
            log_files = sorted(glob.glob('debug_max_conn_*.log'))
        if args.log:
            log_files = [args.log] + [f for f in log_files if f != args.log]

        for lf in log_files[:1]:
            infos = read_log_info(lf)
            for info in infos:
                targets.append({
                    'pos':   info['pos'],
                    'tag':   info['tag'],
                    'conn':  info['conn'],
                    'label': f"tag_{info['tag'][0]}_{info['tag'][1]}",
                })
            if infos:
                print(f"从 {lf} 读取到 {len(infos)} 个触发节点:")
                for info in infos:
                    print(f"  tag=({info['tag'][0]},{info['tag'][1]})  "
                          f"pos=({info['pos'][0]:.1f}, {info['pos'][1]:.1f}, {info['pos'][2]:.1f})  "
                          f"conn={info['conn']}")

    if not targets:
        print("未找到目标节点。请通过 --pos X Y Z 手动指定位置，")
        print("或确保目录下有 debug_max_conn_*.log 文件。")
        sys.exit(1)

    # ---- 收集所有 config.*.data 文件（按步骤排序）----
    config_pattern = os.path.join(sim_dir, 'config.*.data')
    config_files   = sorted(glob.glob(config_pattern), key=step_of)

    if not config_files:
        print(f"在 {sim_dir} 中未找到 config.*.data 文件。")
        sys.exit(1)

    print(f"\n共找到 {len(config_files)} 个配置文件，步骤范围: "
          f"{step_of(config_files[0])} ~ {step_of(config_files[-1])}")

    # ---- 同时加载 debug 文件 ----
    debug_files = sorted(
        glob.glob(os.path.join(sim_dir, 'debug_max_conn_*.data')) +
        glob.glob('debug_max_conn_*.data')
    )

    # ---- 对每个目标追踪历史（通过位置匹配）----
    search_radius = args.radius

    for tgt in targets:
        target_pos = tgt['pos']
        label      = tgt['label']

        print(f"\n{'='*60}")
        print(f"追踪节点 位置=({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
        if tgt['tag']:
            print(f"  原始 tag=({tgt['tag'][0]},{tgt['tag'][1]})  conn={tgt['conn']}")
        print(f"  搜索半径: {search_radius:.0f} b")
        print(f"{'='*60}")

        history = []
        # 用上一步找到的位置作为下一步的搜索中心（追踪移动的节点）
        current_pos = target_pos.copy()

        for cf in config_files:
            step  = step_of(cf)
            nodes = parse_data_file(cf)

            result = find_nearest_node(nodes, current_pos, max_dist=search_radius)
            if result is not None:
                found_tag, found_node, dist = result
                history.append({
                    'step':     step,
                    'pos':      found_node['pos'],
                    'arms':     found_node['arms'],
                    'file_tag': found_tag,
                    'dist':     dist,
                })
                # 更新搜索中心为本步找到的位置
                current_pos = found_node['pos'].copy()

        if not history:
            print(f"  在所有 config 文件中均未找到该位置附近的节点。")
            print(f"  可能是节点在最后几步才创建，没有被 write_freq 捕获。")
            continue

        print(f"  首次出现: 步骤 {history[0]['step']}")
        print(f"  最后出现: 步骤 {history[-1]['step']}")
        print(f"  共记录  : {len(history)} 个快照")
        print(f"\n  连接数变化历史:")
        print(f"  {'步骤':>8}  {'连接数':>6}  {'文件ID':>10}  {'偏移距离':>8}  {'位置 (b)':}")
        for h in history:
            p = h['pos']
            print(f"  {h['step']:>8}  {h['arms']:>6}  "
                  f"{h['file_tag'][1]:>10}  {h['dist']:>8.1f}  "
                  f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

        # ---- 写轨迹 VTK ----
        traj_vtk = os.path.join(out_dir, f"trajectory_{label}.vtk")
        write_trajectory_vtk(traj_vtk, history)

        # ---- 从 debug 文件写局部网络 VTK ----
        for df in debug_files[:1]:
            debug_step  = step_of(df)
            d_nodes     = parse_data_file(df)
            # 用位置在 debug 文件里找对应节点
            result = find_nearest_node(d_nodes, target_pos, max_dist=search_radius)
            if result:
                found_tag = result[0]
                local_vtk = os.path.join(out_dir, f"local_network_{label}_step{debug_step}.vtk")
                write_local_network_vtk(local_vtk, d_nodes, found_tag, debug_step)

    print("\n完成。")
    print("在 ParaView 中:")
    print("  1. 打开 trajectory_*.vtk  → 用 NodeDegree 着色，观察连接数随时间增长")
    print("  2. 打开 local_network_*.vtk → 用 IsTarget 着色，红色节点为触发节点")
    print("  3. 同时加载对应步骤的 config.*.vtk，了解全局网络背景")


if __name__ == '__main__':
    main()
