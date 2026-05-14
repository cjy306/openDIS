# 子循环积分（Subcycling Time-Integration）分组半径选取指南

> 基于文献：
> - Bertin et al., *Modelling Simul. Mater. Sci. Eng.* **27**, 075014 (2019)
> - Akhondzadeh et al., *Acta Materialia* **250**, 118851 (2023)

---

## 1. 基本原理

子循环积分的核心思想是：**按位移节点间相互作用距离将力分为若干组，不同组采用独立时间步长分别积分**。

- 长程相互作用（Group 0）：变化缓慢，可用较大时间步 Δt 积分；
- 短程相互作用（Groups 1–4）：变化剧烈，需在全局步 Δt 内多次子循环（subcycle）积分；
- 每组的划分边界由**分组半径** $r_{gi}$（grouping radius）决定。

节点对$(i,j)$的作用力被分配到满足以下条件的最小组 $k$：

$$
d_{ij} < r_{gk}
$$

其中 $d_{ij}$ 为两段之间的最近距离。

---

## 2. 标准分组结构

ParaDiS 中使用 **5组**（Group 0 ~ Group 4），典型参数如下表：

| 组别 | 分组半径 | 物理含义 |
|------|----------|----------|
| Group 0 | $r_{g0} = 0$ | 所有长程相互作用（FMM 以外的远场）；决定全局步长 Δt |
| Group 1 | $r_{g1} = 0\,b$ | 自相互作用（self-interaction）+ 外加应力 + FMM 远场贡献 |
| Group 2 | $r_{g2} = 10\,b$ | 近程短程相互作用 |
| Group 3 | $r_{g3} = 60\,b$ | 中等短程相互作用 |
| Group 4 | $r_{g4} = 200\,b$ | 最短程相互作用（最"刚硬"的节点对） |

> **说明**：$b$ 为 Burgers 矢量模长（对 FCC Cu：$b = 0.255\,\text{nm}$；对 FCC Al：$b = 0.286\,\text{nm}$）。上表数值直接来自 Akhondzadeh et al. (2023) Table A.1，是针对 FCC Cu 高应变率（$\dot{\varepsilon} \sim 10^4\,\text{s}^{-1}$）模拟的最终选定参数。

---

## 3. 各参数的物理意义与选取逻辑

### 3.1 Group 0（$r_{g0}$）

- 固定为 $0$，即所有节点对初始均分配至 Group 0。
- 之后在 GPU 上计算各对距离，距离满足 $d < r_{g1}$ 者上移至 Group 1，以此类推。
- Group 0 的积分步长即为**全局时间步** Δt，由误差控制方案自动确定。

### 3.2 Group 1（$r_{g1} = 0\,b$）

- $r_{g1}$ 设为 0 的含义：**只有自相互作用**才被放入 Group 1（节点与自身所在线段的相互作用距离为 0）。
- Group 1 还专门负责：核心能（core energy）、外加应力（applied stress）以及 FMM 的远场贡献（far-field FMM stress）。
- 这些慢变项在进入子循环前在 CPU 上预计算，在各子循环中只重新投影（不重新计算 FMM），从而节省开销。

### 3.3 Groups 2–4（$r_{g2}$, $r_{g3}$, $r_{g4}$）

这三组覆盖了真正的"短程刚硬"相互作用。选取原则：

**（a）间距比约为 6–7 倍递减**

$$
r_{g2} : r_{g3} : r_{g4} \approx 10 : 60 : 200\,(b)
$$

每级半径约为上一级的 3–6 倍，确保每组包含的节点对数量从 Group 2 到 Group 4 依次显著减少（典型情况：Group 0 含百万量级节点对，Group 4 仅含几千对）。

**（b）与网格参数联动**

- $r_{g4}$ 的上限应不超过**最小节点间距**的若干倍，通常取 $r_{g4} \lesssim l_{\min} \sim l_{\max}/5$。
  - Akhondzadeh et al. 中 $l_{\max} = 200\,b$，恰好与 $r_{g4}$ 相同，说明最短程组的半径与最大允许段长同量级。
- $r_{g2}$ 应足够小，使绝大多数节点对仍留在 Group 0（即大部分相互作用均属长程），保证子循环不频繁触发。

**（c）刚性节点（troublesome nodes）处理**

在 Group 0 积分中若某节点无法在误差限内收敛，其所有相互作用将被强制上移至 Group 4，以使全局步长 Δt 尽可能保持最大。因此 $r_{g4}$ 实际上也是"安全网"组，其半径不宜过小（否则无法捕获所有刚性相互作用）。

---

## 4. 容差参数与分组半径的配合

子循环分组半径需与以下容差参数共同调整（见 Table A.1）：

| 参数 | 符号 | 典型值 | 说明 |
|------|------|--------|------|
| 相对容差 | $f_{\text{tol}}$ | $0.1\,b$ | 控制 RKF 误差估计的相对阈值 |
| 绝对容差 | $r_{\text{tol}}$ | $0.5\,b$ | 节点位移的绝对容差 |
| 阈值容差 | $r_{\text{th}}$ | $1\,b$ | 前向进展检查（forward progress check）阈值，用于判断节点是否陷入振荡 |
| 碰撞半径 | $r_{\text{col}}$ | $2\,b$ | 触发碰撞处理的距离，应 $< r_{g2}$，否则碰撞可能被漏检 |

> **关键约束**：$r_{\text{col}} < r_{g2}$，保证即将发生碰撞的节点对始终处于被子循环频繁积分的短程组中。

---

## 5. 模拟尺寸与应变率的影响

Bertin et al. (2019) Table 1 给出了不同应变率下的计算时间参考：

| 应变率 $\dot{\varepsilon}$ | 达到 $\gamma = 1\%$ 所需时间（1 CPU + 1 GPU） |
|---------------------------|-----------------------------------------------|
| $10^4\,\text{s}^{-1}$ | ~3 h |
| $10^3\,\text{s}^{-1}$ | ~1 天 |
| $10^2\,\text{s}^{-1}$ | ~15 天 |

应变率越低，全局步长 Δt 越大（位移/时间步变化慢），子循环次数增多，计算量上升。此时可适当**放宽** $r_{g3}$、$r_{g4}$ 的值（扩大被子循环的短程区），或改用更粗的 FMM 格子以减少 Group 0 的计算量。

---

## 6. GPU 内存限制与可行的段数上限

分组半径的选取还受 GPU 显存约束。节点对数 $N_{\text{pair}}$ 的估算公式（来自 Bertin et al.）：

$$
N_{\text{pair}} \approx \frac{27}{N_{\text{FMM}}} \cdot \frac{N_{\text{seg}}^2}{2}
$$

每个节点对约占 150 bytes，对于 12 GB 显存的 GPU（如 GTX Titan X / Tesla K40）：

| FMM 格子数 $N_{\text{FMM}}$ | 最大可处理段数 $N_{\text{seg}}$ |
|-----------------------------|--------------------------------|
| $4^3 = 64$ | ~20,000 |
| $32^3 = 32768$ | ~450,000 |

因此，**较粗的 FMM 格子**（更多节点对被显式计算）要求 $r_{g2}$–$r_{g4}$ 覆盖范围不能过大，否则显存溢出。增大 FMM 格子精度（$N_{\text{FMM}}$ 增大）可允许更大系统，但 FMM 本身的 CPU 计算开销也随之增加。

---

## 7. 选取流程建议

```
1. 确定系统参数
   └─ b（Burgers矢量）、l_max（最大段长）、l_min（最小段长）、r_col（碰撞半径）

2. 固定 r_g0 = 0, r_g1 = 0

3. 设定 r_g2
   └─ 通常取 r_g2 = 5~15 b
   └─ 须满足 r_g2 > r_col（保证碰撞节点对在子循环中被高频积分）

4. 设定 r_g3 = 4~8 × r_g2
   └─ 典型：r_g3 = 60 b（当 r_g2 = 10 b）

5. 设定 r_g4 ≈ l_max 量级
   └─ 典型：r_g4 = 200 b（当 l_max = 200 b）
   └─ 须足以捕获所有"刚硬"节点对（troublesome nodes）

6. 验证显存约束
   └─ 估算 N_pair，确认 GPU 显存可容纳

7. 试运行 & 调整
   └─ 观察各组的节点对数量分布
   └─ 若 Group 4 过大（>总量的1%），适当增大 r_g4
   └─ 若全局步长 Δt 异常小，检查是否有大量节点被强制移入 Group 4
```

---

## 8. 与 ExaDiS / OpenDIS 的对应

在 ExaDiS（OpenDIS 的现代实现）中，上述参数在初始化时通过以下方式传入：

```python
# ExaDiS Python接口示例（参数名与ParaDiS Table A.1对应）
params = {
    'rg1': 0,      # b 单位，Group 1 半径
    'rg2': 10,     # b 单位
    'rg3': 60,     # b 单位
    'rg4': 200,    # b 单位
    'rtol': 0.5,   # b 单位，绝对容差
    'ftol': 0.1,   # b 单位，相对容差
    'rth':  1.0,   # b 单位，前向进展阈值
    'maxseg': 200, # b 单位，最大段长 l_max
}
```

> 注：ExaDiS 的具体参数名称可能与 ParaDiS 有出入，使用时请以源码中的 `Params` 结构体为准。

---

## 参考文献

1. N. Bertin, S. Aubry, A. Arsenlis, W. Cai, "GPU-accelerated dislocation dynamics using subcycling time-integration," *Modelling Simul. Mater. Sci. Eng.* **27**, 075014 (2019).
2. Sh. Akhondzadeh, M. Kang, R.B. Sills, K.T. Ramesh, W. Cai, "Direct comparison between experiments and dislocation dynamics simulations of high rate deformation of single crystal copper," *Acta Materialia* **250**, 118851 (2023).
3. R.B. Sills, A. Aghaei, W. Cai, "Advanced time integration algorithms for dislocation dynamics simulations of work hardening," *Modelling Simul. Mater. Sci. Eng.* **24**, 045019 (2016).
