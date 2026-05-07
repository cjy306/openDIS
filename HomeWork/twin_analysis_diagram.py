"""
生成孪晶面阻挡机制分析图：几何投影 vs 力学方法
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 中文字体设置
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('white')

# ============================================================
# 上半部分：几何投影方法的问题
# ============================================================
ax1 = fig.add_axes([0.02, 0.52, 0.96, 0.46])
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 50)
ax1.axis('off')
ax1.set_title('几何投影方法的三重困境', fontsize=22, fontweight='bold', color='#C0392B', pad=15)

# --- 问题1: 速度断崖 ---
box1 = FancyBboxPatch((1, 28), 30, 19, boxstyle="round,pad=0.5",
                       facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2)
ax1.add_patch(box1)
ax1.text(16, 44, '问题1: 速度断崖', ha='center', fontsize=13, fontweight='bold', color='#C0392B')
ax1.text(16, 40.5, 'pre_integrate 把法向速度', ha='center', fontsize=10)
ax1.text(16, 38, '二值化归零 (blocked vs free)', ha='center', fontsize=10)
ax1.text(16, 35, '→ 相邻节点速度差巨大', ha='center', fontsize=10, color='#E74C3C')
ax1.text(16, 32, '→ 段被急剧拉长', ha='center', fontsize=10, color='#E74C3C')
ax1.text(16, 29.5, '→ Remesh 细化 → 节点爆炸', ha='center', fontsize=10, color='#E74C3C')

# --- 问题2: 投影-细化反馈循环 ---
box2 = FancyBboxPatch((35, 28), 30, 19, boxstyle="round,pad=0.5",
                       facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2)
ax1.add_patch(box2)
ax1.text(50, 44, '问题2: 投影-细化反馈循环', ha='center', fontsize=13, fontweight='bold', color='#C0392B')
ax1.text(50, 40.5, '投影节点到平面(d=0)', ha='center', fontsize=10)
ax1.text(50, 38, '→ 与远处邻居产生超长段', ha='center', fontsize=10, color='#E74C3C')
ax1.text(50, 35, '→ Remesh 细化插入新节点', ha='center', fontsize=10, color='#E74C3C')
ax1.text(50, 32, '→ 新节点再被投影', ha='center', fontsize=10, color='#E74C3C')
ax1.text(50, 29.5, '→ 无阻尼正反馈 → 指数增长', ha='center', fontsize=10, color='#E74C3C')

# --- 问题3: 共面碰撞堆积 ---
box3 = FancyBboxPatch((69, 28), 30, 19, boxstyle="round,pad=0.5",
                       facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2)
ax1.add_patch(box3)
ax1.text(84, 44, '问题3: 共面碰撞堆积', ha='center', fontsize=13, fontweight='bold', color='#C0392B')
ax1.text(84, 40.5, '多个节点投影到 d=0 平面', ha='center', fontsize=10)
ax1.text(84, 38, '→ 大量段共面', ha='center', fontsize=10, color='#E74C3C')
ax1.text(84, 35, '→ CollisionRetroactive 误判碰撞', ha='center', fontsize=10, color='#E74C3C')
ax1.text(84, 32, '→ merge_node 合并节点', ha='center', fontsize=10, color='#E74C3C')
ax1.text(84, 29.5, '→ MAX_CONN=15 溢出', ha='center', fontsize=10, color='#E74C3C')

# 根因箭头
ax1.annotate('', xy=(50, 27), xytext=(50, 22),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=3))

# 根因框
root_box = FancyBboxPatch((15, 4), 70, 17, boxstyle="round,pad=0.8",
                           facecolor='#F9EBEA', edgecolor='#922B21', linewidth=3)
ax1.add_patch(root_box)
ax1.text(50, 17, '根本原因: 几何投影打破了 DDD 框架的连续性假设',
         ha='center', fontsize=14, fontweight='bold', color='#922B21')
ax1.text(50, 13.5, 'DDD 框架假设: 力 → 速度 → 积分 → 位置变化 是连续的',
         ha='center', fontsize=11, color='#6C3483')
ax1.text(50, 10, '几何投影: 直接修改节点位置, 跳过力→速度→积分的物理链条',
         ha='center', fontsize=11, color='#922B21')
ax1.text(50, 6.5, 'Remesh / Collision / Integrator 模块看到的是不连续的位置跳变 → 产生各种副作用',
         ha='center', fontsize=11, color='#922B21')

# 连接线
for x in [16, 50, 84]:
    ax1.plot([x, 50], [28, 22], color='#C0392B', lw=1.5, ls='--')

# ============================================================
# 下半部分：力学方法
# ============================================================
ax2 = fig.add_axes([0.02, 0.02, 0.96, 0.48])
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 50)
ax2.axis('off')
ax2.set_title('力学方法: 指数排斥力', fontsize=22, fontweight='bold', color='#1A5276', pad=15)

# --- 左侧: 排斥力公式 ---
formula_box = FancyBboxPatch((1, 25), 32, 22, boxstyle="round,pad=0.8",
                              facecolor='#D4E6F1', edgecolor='#2471A3', linewidth=2)
ax2.add_patch(formula_box)
ax2.text(17, 44, '排斥力公式', ha='center', fontsize=14, fontweight='bold', color='#1A5276')
ax2.text(17, 40, r'$F_{repel}(d) = F_0 \cdot e^{-d/\lambda} \cdot \hat{n}_{active}$',
         ha='center', fontsize=16, color='#154360',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#AED6F1'))
ax2.text(17, 36, r'$F_0 = \mu b^2 / \lambda$  (峰值力)', ha='center', fontsize=11, color='#1A5276')
ax2.text(17, 33, r'$\lambda = minseg = 50b$  (衰减长度)', ha='center', fontsize=11, color='#1A5276')
ax2.text(17, 30, '只有法向分量, 切向为零', ha='center', fontsize=11, fontweight='bold', color='#117A65')
ax2.text(17, 27, '→ 节点可沿孪晶面自由滑动', ha='center', fontsize=11, color='#117A65')

# --- 中间: 驱动循环位置 ---
flow_box = FancyBboxPatch((35, 25), 30, 22, boxstyle="round,pad=0.8",
                           facecolor='#D5F5E3', edgecolor='#1E8449', linewidth=2)
ax2.add_patch(flow_box)
ax2.text(50, 44, '在 DDD 循环中的位置', ha='center', fontsize=14, fontweight='bold', color='#1E8449')

steps = [
    ('force->compute()', '#2C3E50', False),
    ('+ add_twin_repulsive_force()', '#E74C3C', True),
    ('mobility->compute()', '#2C3E50', False),
    ('integrator->integrate()', '#2C3E50', False),
    ('collision / topology / remesh', '#2C3E50', False),
]
y_start = 40
for i, (text, color, highlight) in enumerate(steps):
    y = y_start - i * 3
    if highlight:
        ax2.text(50, y, text, ha='center', fontsize=11, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FADBD8', edgecolor='#E74C3C'))
    else:
        ax2.text(50, y, text, ha='center', fontsize=11, color=color)
    if i < len(steps) - 1:
        ax2.annotate('', xy=(50, y - 1.8), xytext=(50, y - 0.5),
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5))

# --- 右侧: 优势 ---
adv_box = FancyBboxPatch((67, 25), 32, 22, boxstyle="round,pad=0.8",
                          facecolor='#D5F5E3', edgecolor='#1E8449', linewidth=2)
ax2.add_patch(adv_box)
ax2.text(83, 44, '力学方法的优势', ha='center', fontsize=14, fontweight='bold', color='#1E8449')
advantages = [
    '✓ 不修改位置 — Remesh 正常工作',
    '✓ 不修改速度 — Integrator 正常工作',
    '✓ 不产生共面段 — Collision 正常工作',
    '✓ 位错自然减速停住 — 物理正确',
    '✓ 产生背应力 — 抑制位错源发射',
]
for i, text in enumerate(advantages):
    ax2.text(83, 40 - i * 3, text, ha='center', fontsize=11, color='#145A32')

# --- 底部: 力随距离衰减图 ---
ax_force = fig.add_axes([0.06, 0.04, 0.25, 0.18])
d = np.linspace(0, 300, 500)
lam = 50
F = np.exp(-d / lam)
ax_force.plot(d, F, color='#E74C3C', lw=3)
ax_force.fill_between(d, F, alpha=0.15, color='#E74C3C')
ax_force.axvline(x=50, color='#3498DB', ls='--', lw=1.5, label=r'$\lambda$=50b')
ax_force.axvline(x=250, color='#95A5A6', ls=':', lw=1.5, label=r'$5\lambda$=250b (截断)')
ax_force.set_xlabel('距孪晶面距离 d (b)', fontsize=11)
ax_force.set_ylabel(r'$F/F_0$', fontsize=12)
ax_force.set_title('排斥力衰减曲线', fontsize=12, fontweight='bold')
ax_force.legend(fontsize=9, loc='upper right')
ax_force.set_xlim(0, 300)
ax_force.set_ylim(0, 1.05)
ax_force.grid(True, alpha=0.3)

# --- 底部: 物理行为示意 ---
ax_phys = fig.add_axes([0.38, 0.04, 0.25, 0.18])
ax_phys.set_xlim(0, 10)
ax_phys.set_ylim(0, 10)
ax_phys.axis('off')
ax_phys.set_title('位错接近孪晶面的行为', fontsize=12, fontweight='bold')

# 孪晶面
ax_phys.axvline(x=8, color='#8E44AD', lw=4, label='孪晶面')
ax_phys.fill_betweenx([0, 10], 8, 10, alpha=0.2, color='#8E44AD')
ax_phys.text(9, 9, '真空侧', ha='center', fontsize=10, color='#8E44AD')
ax_phys.text(4, 9, '活性区', ha='center', fontsize=10, color='#2C3E50')

# 位错节点
positions = [(2, 7), (4, 5.5), (6, 4), (7.2, 3), (7.6, 2)]
labels = ['v=100%', 'v=80%', 'v=40%', 'v=10%', 'v≈0']
colors_grad = ['#27AE60', '#58D68D', '#F4D03F', '#E67E22', '#E74C3C']
for (x, y), label, c in zip(positions, labels, colors_grad):
    ax_phys.plot(x, y, 'o', color=c, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
    ax_phys.text(x, y + 0.6, label, ha='center', fontsize=8, color=c, fontweight='bold')

# 箭头表示运动方向
for i in range(len(positions) - 1):
    ax_phys.annotate('', xy=positions[i+1], xytext=positions[i],
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

ax_phys.text(5, 0.5, '排斥力使位错自然减速\n最终在平面附近力平衡',
             ha='center', fontsize=10, color='#1A5276',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB', edgecolor='#AED6F1'))

# --- 底部: 对比表 ---
ax_table = fig.add_axes([0.68, 0.04, 0.30, 0.18])
ax_table.axis('off')
ax_table.set_title('几何投影 vs 力学方法', fontsize=12, fontweight='bold')

table_data = [
    ['', '几何投影', '力学方法'],
    ['原理', '强制改位置', '加排斥力'],
    ['速度', '断崖式归零', '渐进衰减'],
    ['Remesh', '触发爆炸', '正常工作'],
    ['Collision', 'MAX_CONN溢出', '正常工作'],
    ['切向运动', '需额外处理', '天然自由'],
    ['背应力', '无', '自动产生'],
]

for i, row in enumerate(table_data):
    for j, text in enumerate(row):
        y = 0.88 - i * 0.13
        x = 0.02 + j * 0.33
        if i == 0:
            ax_table.text(x, y, text, fontsize=10, fontweight='bold', color='#2C3E50',
                         transform=ax_table.transAxes)
        elif j == 1:
            ax_table.text(x, y, text, fontsize=9, color='#E74C3C',
                         transform=ax_table.transAxes)
        elif j == 2:
            ax_table.text(x, y, text, fontsize=9, color='#1E8449',
                         transform=ax_table.transAxes)
        else:
            ax_table.text(x, y, text, fontsize=9, color='#2C3E50',
                         transform=ax_table.transAxes)

# 保存
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'twin_analysis_diagram.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Diagram saved to: {output_path}")
plt.close()
