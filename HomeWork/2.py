"""
示意动画：演示多条位错线交汇 → merge_node → MAX_CONN 超限 → revert 的完整过程。
纯示意，不依赖任何 VTK 数据。

用法：
  pip install numpy matplotlib
  python demo_max_conn_process.py

输出：
  demo_max_conn_process.gif
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import os


# ============================================================
# 场景设置
# ============================================================
# 5条位错线从不同方向向中心汇聚
N_LINES = 5
CENTER = np.array([0.0, 0.0])
RADIUS_START = 4.0   # 初始距离
RADIUS_END = 0.15    # 汇聚时的距离

# 每条线有2个端点：一个外侧固定，一个内侧节点向中心移动
angles = np.linspace(0, 2 * np.pi, N_LINES, endpoint=False)
outer_pts = np.array([[RADIUS_START * 1.8 * np.cos(a), RADIUS_START * 1.8 * np.sin(a)] for a in angles])

# 各阶段帧数
PHASE1_FRAMES = 30   # 接近阶段
PHASE2_FRAMES = 8    # 碰撞检测
PHASE3_FRAMES = 10   # 尝试合并
PHASE4_FRAMES = 8    # MAX_CONN 报错
PHASE5_FRAMES = 15   # revert + 分离
PHASE6_FRAMES = 10   # SplitMultiNode 恢复
TOTAL = PHASE1_FRAMES + PHASE2_FRAMES + PHASE3_FRAMES + PHASE4_FRAMES + PHASE5_FRAMES + PHASE6_FRAMES


def ease_in_out(t):
    """缓入缓出插值"""
    return t * t * (3 - 2 * t)


def get_inner_positions(frame):
    """根据帧号计算每条线内侧节点的位置"""
    positions = []

    if frame < PHASE1_FRAMES:
        # Phase 1: 各节点从外围向中心移动
        t = ease_in_out(frame / PHASE1_FRAMES)
        r = RADIUS_START * (1 - t) + 0.5 * t
        for a in angles:
            positions.append([r * np.cos(a), r * np.sin(a)])

    elif frame < PHASE1_FRAMES + PHASE2_FRAMES:
        # Phase 2: 继续靠近到 rann 距离内
        t = ease_in_out((frame - PHASE1_FRAMES) / PHASE2_FRAMES)
        r = 0.5 * (1 - t) + RADIUS_END * t
        for a in angles:
            positions.append([r * np.cos(a), r * np.sin(a)])

    elif frame < PHASE1_FRAMES + PHASE2_FRAMES + PHASE3_FRAMES:
        # Phase 3: 尝试合并 → 所有节点移向中心点
        t = ease_in_out((frame - PHASE1_FRAMES - PHASE2_FRAMES) / PHASE3_FRAMES)
        r = RADIUS_END * (1 - t)
        for a in angles:
            positions.append([r * np.cos(a), r * np.sin(a)])

    elif frame < PHASE1_FRAMES + PHASE2_FRAMES + PHASE3_FRAMES + PHASE4_FRAMES:
        # Phase 4: MAX_CONN 报错，节点重叠在中心（闪烁）
        for a in angles:
            positions.append([0.0, 0.0])

    elif frame < PHASE1_FRAMES + PHASE2_FRAMES + PHASE3_FRAMES + PHASE4_FRAMES + PHASE5_FRAMES:
        # Phase 5: Revert → 节点弹回
        f5 = frame - PHASE1_FRAMES - PHASE2_FRAMES - PHASE3_FRAMES - PHASE4_FRAMES
        t = ease_in_out(f5 / PHASE5_FRAMES)
        r = RADIUS_END + t * (0.8 - RADIUS_END)
        for a in angles:
            positions.append([r * np.cos(a), r * np.sin(a)])

    else:
        # Phase 6: SplitMultiNode 重新整理
        f6 = frame - PHASE1_FRAMES - PHASE2_FRAMES - PHASE3_FRAMES - PHASE4_FRAMES - PHASE5_FRAMES
        t = ease_in_out(f6 / PHASE6_FRAMES)
        r = 0.8 + t * 0.5
        for i, a in enumerate(angles):
            # 稍微偏移角度，表示 split 后重新分布
            new_a = a + t * 0.15 * (1 if i % 2 == 0 else -1)
            positions.append([r * np.cos(new_a), r * np.sin(new_a)])

    return np.array(positions)


def get_phase_info(frame):
    """返回当前阶段的描述和颜色"""
    t1 = PHASE1_FRAMES
    t2 = t1 + PHASE2_FRAMES
    t3 = t2 + PHASE3_FRAMES
    t4 = t3 + PHASE4_FRAMES
    t5 = t4 + PHASE5_FRAMES

    if frame < t1:
        return "Phase 1: Dislocation lines moving toward junction", "#2196F3", "normal"
    elif frame < t2:
        return "Phase 2: Nodes within rann range, Collision detected", "#FF9800", "normal"
    elif frame < t3:
        return "Phase 3: Attempting merge_node()...", "#FF9800", "merging"
    elif frame < t4:
        return "Phase 4: ERROR! merged conn={} > MAX_CONN={} -> merge failed!".format(
            N_LINES * 2, 10), "#F44336", "error"
    elif frame < t5:
        return "Phase 5: Revert! Restoring original nodes", "#4CAF50", "revert"
    else:
        return "Phase 6: SplitMultiNode reorganizes topology -> normal", "#4CAF50", "normal"


def render_frame(frame):
    """渲染单帧"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    phase_text, phase_color, phase_mode = get_phase_info(frame)

    inner = get_inner_positions(frame)

    # 画 rann 范围圆（虚线）
    rann_circle = plt.Circle(CENTER, 0.5, fill=False, color='#666666',
                              linestyle='--', linewidth=1.5, label='rann radius')
    ax.add_patch(rann_circle)
    ax.text(0.55, 0.0, 'rann', color='#888888', fontsize=9, va='center')

    # 画位错线（外侧端点 → 内侧节点）
    line_colors = ['#E91E63', '#00BCD4', '#FFEB3B', '#76FF03', '#FF6D00']
    for i in range(N_LINES):
        p_out = outer_pts[i]
        p_in = inner[i]

        # 位错线
        ax.plot([p_out[0], p_in[0]], [p_out[1], p_in[1]],
                color=line_colors[i], linewidth=2.5, alpha=0.8)

        # 外侧固定节点（小圆）
        ax.plot(*p_out, 'o', color=line_colors[i], markersize=8,
                markeredgecolor='white', markeredgewidth=1)

    # 画内侧节点
    if phase_mode == 'error':
        # 闪烁效果
        sub_frame = frame - PHASE1_FRAMES - PHASE2_FRAMES - PHASE3_FRAMES
        flash = sub_frame % 2 == 0
        node_color = '#FF0000' if flash else '#FFFF00'
        node_size = 25 if flash else 20

        # 合并后的大节点
        ax.plot(0, 0, 'o', color=node_color, markersize=node_size,
                markeredgecolor='white', markeredgewidth=3, zorder=10)

        # 显示连接数
        ax.text(0, -0.8, f'conn = {N_LINES * 2}', color='#FF0000',
                fontsize=16, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#330000',
                          edgecolor='#FF0000', alpha=0.9))

        # MAX_CONN 标注
        ax.text(0, -1.5, f'MAX_CONN = 10', color='#FFFF00',
                fontsize=14, fontweight='bold', ha='center', va='top')

        # X 标记
        ax.plot(0, 0, 'x', color='white', markersize=35, markeredgewidth=4, zorder=11)

    elif phase_mode == 'merging':
        # 合并过程：节点变大，显示箭头
        for i in range(N_LINES):
            p_in = inner[i]
            ax.plot(*p_in, 'o', color=line_colors[i], markersize=12,
                    markeredgecolor='white', markeredgewidth=2, zorder=10)
            # 箭头指向中心
            dx = -p_in[0] * 0.5
            dy = -p_in[1] * 0.5
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                ax.annotate('', xy=(p_in[0] + dx, p_in[1] + dy),
                           xytext=(p_in[0], p_in[1]),
                           arrowprops=dict(arrowstyle='->', color='white',
                                          lw=2, ls='--'))

        # 中心虚线圆
        merge_circle = plt.Circle(CENTER, 0.3, fill=False, color='white',
                                   linestyle=':', linewidth=2)
        ax.add_patch(merge_circle)
        ax.text(0, -0.6, 'merge_node()', color='white',
                fontsize=12, ha='center', style='italic')

    elif phase_mode == 'revert':
        # Revert: 绿色节点弹回
        for i in range(N_LINES):
            p_in = inner[i]
            ax.plot(*p_in, 'o', color='#4CAF50', markersize=12,
                    markeredgecolor='white', markeredgewidth=2, zorder=10)
            # conn 标注
            ax.text(p_in[0] + 0.3, p_in[1] + 0.3, f'conn=2',
                    color='#81C784', fontsize=8, ha='left')

    else:
        # 正常状态
        for i in range(N_LINES):
            p_in = inner[i]
            node_size = 12
            ax.plot(*p_in, 'o', color=line_colors[i], markersize=node_size,
                    markeredgecolor='white', markeredgewidth=2, zorder=10)

            # Phase 2: 显示 conn 数
            t2_start = PHASE1_FRAMES
            t2_end = t2_start + PHASE2_FRAMES
            if frame >= t2_start and frame < t2_end:
                ax.text(p_in[0] + 0.3, p_in[1] + 0.3, f'conn=2',
                        color='#BBBBBB', fontsize=8, ha='left')

    # --- 标题和说明 ---
    ax.text(0, 5.0, phase_text, color=phase_color,
            fontsize=15, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d0d1a',
                      edgecolor=phase_color, alpha=0.95))

    # 帧号
    ax.text(-5.2, -5.2, f'Frame {frame}/{TOTAL-1}', color='#666666',
            fontsize=10, ha='left')

    # 图例
    legend_y = 4.0
    legend_items = [
        ('o  Dislocation node (conn=2)', '#E91E63'),
        ('-- Dislocation segment', '#00BCD4'),
        ('-- rann radius', '#666666'),
    ]
    for text, color in legend_items:
        ax.text(3.5, legend_y, text, color=color, fontsize=9, ha='left')
        legend_y -= 0.4

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'demo_frames')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Rendering {TOTAL} frames...')
    frame_paths = []
    for frame in range(TOTAL):
        fig = render_frame(frame)
        path = os.path.join(output_dir, f'frame_{frame:04d}.png')
        fig.savefig(path, dpi=100, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_paths.append(path)
        if (frame + 1) % 10 == 0 or frame == TOTAL - 1:
            print(f'  [{frame+1}/{TOTAL}]')

    # 合成 GIF
    gif_path = os.path.join(os.path.dirname(__file__), 'demo_max_conn_process.gif')
    try:
        from PIL import Image
        images = [Image.open(f) for f in frame_paths]
        # Phase 4 (error) 帧停留更久
        durations = []
        t3 = PHASE1_FRAMES + PHASE2_FRAMES + PHASE3_FRAMES
        t4 = t3 + PHASE4_FRAMES
        for i in range(TOTAL):
            if t3 <= i < t4:
                durations.append(500)   # 报错帧 500ms
            else:
                durations.append(150)   # 正常帧 150ms
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=durations, loop=0)
        print(f'\nGIF saved: {gif_path}')
    except ImportError:
        # 尝试 imageio
        try:
            import imageio
            images = [imageio.imread(f) for f in frame_paths]
            imageio.mimsave(gif_path, images, duration=0.15)
            print(f'\nGIF saved: {gif_path}')
        except ImportError:
            print(f'\nFrames saved in: {output_dir}')
            print('Install Pillow (pip install Pillow) or imageio to generate GIF')


if __name__ == '__main__':
    main()
