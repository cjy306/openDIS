# DDD-FFT 方法论文详细总结

> **论文标题**: A FFT-based formulation for efficient mechanical fields computation in isotropic and anisotropic periodic discrete dislocation dynamics  
> **作者**: N. Bertin, M. V. Upadhyay, C. Pradalier, L. Capolungo  
> **期刊**: Modelling and Simulation in Materials Science and Engineering, **23** (2015) 065009  
> **机构**: Georgia Institute of Technology（佐治亚理工学院），UMI GT-CNRS 2958  

---

## 1. 研究背景与动机

### 1.1 离散位错动力学（DDD）简介

离散位错动力学（Discrete Dislocation Dynamics，DDD）是一种介观尺度模拟方法，用于研究大量位错集体行为对材料塑性和应变硬化的影响。DDD 在以下领域有重要应用：

- 薄膜和微柱中的小尺度塑性
- 不同晶体对称性材料中位错结的强度与形成
- 三重位错结的形成过程
- 立方和六方体系中的潜伏硬化效应
- 辐照损伤对强度的影响

### 1.2 现有方法的计算瓶颈

#### 常规 DDD 方法（叠加原理）

在传统节点型 DDD 中，每个位错线被离散为通过节点连接的线段。节点 $i$ 的速度与作用力的关系为：

$$\mathbf{B}\mathbf{v}_i = \mathbf{F}_i$$

其中 $\mathbf{B}$ 是过阻尼方程中的阻力系数矩阵。力 $\mathbf{F}_i$ 通过 Peach-Koehler 力沿线段积分得到：

$$\mathbf{f}_{ij} = \int_{x_i}^{x_j} N_i(\mathbf{x}) \mathbf{f}_{ij}^{pk}(\mathbf{x}) d\mathbf{x}, \quad \mathbf{f}_{ij}^{pk}(\mathbf{x}) = [\boldsymbol{\sigma}(\mathbf{x}) \cdot \mathbf{b}_{ij}] \times \mathbf{t}_{ij}$$

**核心问题**：计算所有节点力的复杂度为 $O(N_{seg}^2)$，当位错段数量 $N_{seg}$ 很大时，计算成本极高，构成传统 DDD 的主要瓶颈。

#### 各向同性 vs. 各向异性弹性

- **各向同性**：Cai 等人提出了非奇异解析公式，大幅降低计算成本
- **各向异性**：目前**无闭合解析解**，各向异性计算的相对成本高达各向同性的 200–500 倍，严重限制了实际应用

#### FEM 方法的局限

有限元法（FEM）虽然可以精确处理边界条件，但：
- 计算和内存需求大
- 限制了网格精度
- 进而影响可达到的应变水平

---

## 2. 核心思路：DCM + FFT 的耦合

### 2.1 离散-连续模型（DCM）

DCM 由 Lemarchand 等人提出，基于特征应变（eigenstrain）框架，将每条位错环视为板状夹杂（plate-like inclusion）。本构关系为：

$$\boldsymbol{\sigma}(\mathbf{x}) = \mathbb{C}(\boldsymbol{\varepsilon}(\mathbf{x}) - \boldsymbol{\varepsilon}^p(\mathbf{x}))$$

其中塑性应变 $\boldsymbol{\varepsilon}^p$ 直接来自位错运动。位错段 $ij$ 在网格点 $p$ 处产生的塑性剪切为：

$$d\gamma_{ij}^p = \frac{6b}{\pi h^3} dS_{ij}^p$$

$dS_{ij}^p$ 是位错滑移面积与以网格点 $p$ 为中心、半径为 $h/2$ 的球体的交集面积；$h$ 为正则化参数。

**DCM 的关键优势**：网格自动包含长程弹性相互作用，无需逐对计算远场位错-位错相互作用。

### 2.2 FFT 求解器

本文核心贡献是将 DCM 与 FFT 求解器结合，替代原有的 FEM 求解器。

#### 基本思路

在周期边界条件下，力学平衡方程为：

$$\mathbb{C}_{ijkl} u_{k,lj}(\mathbf{x}) - \varphi_{ij,j}(\mathbf{x}) = 0, \quad \forall \mathbf{x} \in V_s$$

其中极化张量 $\varphi_{ij}(\mathbf{x}) = \mathbb{C}_{ijkl} \varepsilon_{kl}^p(\mathbf{x})$。

将上式变换到 Fourier 空间（$\boldsymbol{\xi}$ 为频率坐标）：

$$\mathbb{C}_{ijkl} \xi_l \xi_j \hat{u}_k(\boldsymbol{\xi}) + \widehat{\varphi}_{ij,j}(\boldsymbol{\xi}) = 0$$

引入 Green 函数 $\hat{G}_{ik}(\boldsymbol{\xi}) = K_{ki}^{-1} = [\mathbb{C}_{kjil}\xi_l\xi_j]^{-1}$，应变场在 Fourier 空间中为：

$$\hat{\varepsilon}_{ij}(\boldsymbol{\xi}) = \hat{\Gamma}_{ijkl}(\boldsymbol{\xi})\hat{\varphi}_{kl}(\boldsymbol{\xi})$$

其中修正 Green 函数：

$$\hat{\Gamma}_{ijkl}(\boldsymbol{\xi}) = \frac{1}{2}(\xi_l \xi_j \hat{G}_{ik} + \xi_i \xi_j \hat{G}_{jk}), \quad \forall \boldsymbol{\xi} \neq \mathbf{0}; \quad \hat{\varepsilon}_{ij}(\mathbf{0}) = 0$$

实空间应变场：

$$\varepsilon_{ij}(\mathbf{x}) = \mathcal{FFT}^{-1}(\hat{\varepsilon}_{ij}(\boldsymbol{\xi})) + E_{ij}$$

其中 $E_{ij} = \langle \varepsilon \rangle$ 是施加的宏观平均应变。最终应力：

$$\boldsymbol{\sigma}(\mathbf{x}) = \mathbb{C}(\boldsymbol{\varepsilon}(\mathbf{x}) - \boldsymbol{\varepsilon}^p(\mathbf{x}))$$

该应力场**同时包含**位错内应力和外部加载应力，无冗余。

---

## 3. 数值实现细节

### 3.1 塑性剪切的正则化（DCM 数值实现）

#### 球形体积的选择

对于每个 Fourier 网格点，关联的基本体积（elementary volume）选为半径为 $h/2$ 的球体。为使所有球体的并集恰好覆盖整个模拟体积（不遗漏、不过多重叠），最小可接受的 $h$ 值为：

$$h_{\min} = \sqrt{3} L$$

其中 $L$ 为网格间距。

#### 位置依赖性修正

由于位错核心相对于网格的位置不同，塑性剪切分布会产生误差。为修正这一问题，令 $h = 2L$（即球体半径 $r = L$），位错距网格点距离为 $d$ 时，塑性剪切为：

$$d\gamma_{ij}^p(d) = \frac{b(L^2 - d^2)}{(4/3)L^3}$$

修正后的线性内插函数为：

$$d\gamma_{ij}^{p,\text{corr}}(d) = c(d) \cdot d\gamma_{ij}^p = \frac{L-d}{L} d\gamma_{ij}^{\text{ref}}$$

修正函数：

$$c(d) = \frac{4}{3} \frac{(L^2 - Ld)}{(L^2 - d^2)}$$

这一修正使得 DDD-FFT 方法对位错在 Fourier 网格上的任意位置和方向都能给出与解析解精确匹配的应力场。

### 3.2 交集面积的解析计算

计算 $dS_{ij}^p$（位错滑移面积四边形与球形基本体积的交集面积）使用 Green 定理进行解析计算：

$$dS_{ij}^p = \frac{1}{2} \oint_{C_{ij}^p} (-y\,dx + x\,dy)$$

闭合轮廓 $C_{ij}^p$ 由直线段和圆弧段组成，每段的线积分解析表达式分别为：

**直线段** $C_k$（端点 $(x_0, y_0)$ 到 $(x_1, y_1)$）：

$$I_{C_k}^{\text{seg}} = \frac{1}{2}(x_0 y_1 - y_0 x_1)$$

**圆弧段** $C_k$（半径 $r$，圆心 $(x_c, y_c)$，角度从 $\theta_0$ 到 $\theta_1$）：

$$I_{C_k}^{\text{arc}} = \frac{1}{2}[r^2(\theta_1 - \theta_0) + x_c(y_1 - y_0) - y_c(x_1 - x_0)]$$

### 3.3 Gibbs 现象的处理

由于位错引起的塑性应变场的不连续性，FFT 会产生 Gibbs 振荡（Gibbs phenomenon）。处理方法：将每个网格点计算得到的塑性应变，使用三角分布扩散到周围 $3 \times 3 \times 3$ 共 27 个体素上。这一"数值扩散"（numerical spreading）能有效消除振荡，代价是位错核心区域被适度弥散。

### 3.4 短程相互作用的补充计算

DCM 方法中，网格无法分辨距离小于 $h/2$ 的近场位错-位错相互作用，需要额外计算。具体做法：

1. 找出所有距离小于 $h/2 = L$ 的位错段对
2. 对这些近场段对，使用解析应力公式计算补充局域力
3. 由于 FFT 方法与解析方法之间存在常数平移张量 $\bar{\sigma}$（源于边界条件差异），需在每步确定 $\bar{\sigma}$ 并修正

---

## 4. 验证算例

### 4.1 各向同性弹性中的静态位错

**测试配置**：棱柱形方形位错环，由 4 段边缘位错组成，$l_0 = 1000b$，Burgers 矢量 $\mathbf{b} = [010]$，$b = 0.25$ nm，材料参数 $\mu = 51$ GPa，$\nu = 0.37$，模拟盒子 $a = 0.5\,\mu\text{m}$。

**结果**：
- 网格精度 $64^3$ 体素时，加入 Gibbs 扩散后，远场应力与 Cai 等人的非奇异解析解精确吻合
- 网格越细，核心区的精度越高（$16^3 \to 128^3$ 均测试）
- 位置修正对消除位错位置依赖性至关重要：三个不同位置（$d = 0, L/4, L/2$）的应力场在修正后完全重合

### 4.2 棱柱形位错偶极子

同时引入两个相反 Burgers 矢量的棱柱形位错环（分别位于 $d = L/4$ 和 $d = 3L/4$ 处），$128^3$ 网格下的 $\sigma_{23}$ 分量与解析解完全吻合。验证了 FFT 方法处理多位错叠加的能力。

### 4.3 各向异性弹性

测试了三个各向异性比：

| 情形 | 各向异性比 $A = 2C_{44}/(C_{11}-C_{12})$ | 对应材料 |
|------|------------------------------------------|----------|
| (a) | $A = 0.5$，$C_{44} = 25.5$ GPa | Nb 晶体 |
| (b) | $A = 1$，$C_{44} = 51$ GPa | 各向同性情形 |
| (c) | $A = 7.5$，$C_{44} = 382$ GPa | 高温 $\alpha$-Fe |

**关键结论**：在足够细的 Fourier 网格下，各向异性弹性计算的**计算成本与各向同性情形相同**，因为两者均通过 Green 函数在 Fourier 空间直接求解。这是相较于传统 DDD 方法的革命性优势（传统方法各向异性计算成本高出 200–500 倍）。

### 4.4 Frank-Read 源激活

**测试配置**：通过固定棱柱形位错环的三段边缘线，剩余一段可动段充当 Frank-Read 源，滑移面 $(001)$，各向同性 $\mu = 130$ GPa，$\nu = 0.309$，源长度从 $l_0 = 1000b$ 到 $l_0 = 10000b$。

**结果**：DDD-FFT 方法在不同网格分辨率（$32^3, 64^3, 128^3$）下得到的激活应力与常规 DDD 方法高度一致，验证了本方法处理动态情形的能力。

---

## 5. 计算效率分析

### 5.1 复杂度对比

| 方法 | 计算复杂度 |
|------|-----------|
| 常规 DDD（全计算） | $O(N_{seg}^2)$ |
| 常规 DDD + Box Method | 近似降低，但每 10 步才计算一次 |
| DCM 正则化过程 | $O(N_{seg})$ |
| FFT 求解器 | $O(N_{tot} \log N_{tot})$ |
| 标准 3D FEM | $O(N_{tot}^2)$ |

### 5.2 实测加速比

在单 CPU 条件下，以 $N_{seg}$ 为横轴对比运行时间：

- $N_{seg} = 20000$：DDD-FFT（$64^3$）比全计算快约 **230 倍**，比 FMM 方法快约 **10 倍**
- $N_{seg} = 100000$：$64^3$ 和 $128^3$ 网格的单步计算时间均在 **5s 以下**，比 Box Method 快约 **30 倍**
- $256^3$ 网格：单步约 9–11s，FFT 求解器本身占约 8s

### 5.3 最优网格选择

- 网格过粗（$32^3$）：随 $N_{seg}$ 增加，近场相互作用计算量急剧上升，效率反而下降
- $N_{seg} > 50000$ 时，$128^3$ 网格比 $64^3$ 更高效
- 建议根据预期最大 $N_{seg}$ 预先选择最优网格分辨率

### 5.4 DDD-FFT 理论适用上限

对于 $V_s = (5\,\mu\text{m})^3$、$64^3$ 网格，DDD-FFT 在位错密度 $\rho \lesssim 10^{14}$ m$^{-2}$ 时不需要大量近场计算，性能最优。

---

## 6. 方法局限性

1. **仅适用于全周期边界条件（PBC）**：当前公式无法模拟自由表面。（作者指出可借助谱方法的自由表面处理技术扩展，但超出本文范围）
2. **Gibbs 振荡**：必须引入数值扩散，导致核心区应力分布被弥散，不能精确反映近场核心处的应力
3. **短程相互作用**：对于近距离位错段（$< h/2 = L$），仍需解析计算补充力，各向异性情形下这一步更耗时
4. **计算时间依赖实现**：报告的数值依赖于具体代码优化、软硬件配置

---

## 7. 创新点与意义

### 7.1 主要创新点

1. **DDD-FFT 方法的提出**：将 FFT 求解器与 DCM 耦合，取代 FEM 求解器，大幅降低计算成本
2. **各向异性弹性的高效处理**：首次实现各向异性 DDD 模拟成本与各向同性相当
3. **解析正则化方案**：提出基于 Green 定理的解析交集面积计算方法（适用于节点型 DDD 中位错段的任意旋转运动）
4. **位置依赖性修正**：给出精确的塑性剪切修正函数 $c(d)$，消除位错位置对结果的影响

### 7.2 对 DDD 领域的意义

- **可在桌面计算机上进行真实微结构模拟**，无需大量 CPU
- 方法天然支持 GPU 加速和并行 FFT
- 为 DDD 尺度塑性与晶体塑性（crystal plasticity）之间的尺度过渡奠定基础
- 建立了 DDD 模拟与场位错力学（Field Dislocation Mechanics）谱方法之间的直接联系
- 为多晶 DDD 模拟（弹性非均质性）提供技术基础

---

## 8. 主要符号对照表

| 符号 | 含义 |
|------|------|
| $\mathbf{b}_{ij}$ | 位错段 $ij$ 的 Burgers 矢量 |
| $\mathbf{t}_{ij}$ | 位错段 $ij$ 的单位切向量 |
| $\boldsymbol{\sigma}$ | 总应力张量 |
| $\boldsymbol{\varepsilon}^p$ | 塑性应变张量 |
| $\mathbb{C}$ | 四阶弹性刚度张量 |
| $\hat{\Gamma}(\boldsymbol{\xi})$ | Fourier 空间中的修正 Green 函数 |
| $h$ | DCM 正则化参数（基本球体直径）|
| $L$ | Fourier 网格间距 |
| $N_{seg}$ | 位错段总数 |
| $N_{tot}$ | 网格点（体素）总数 |
| $E_{ij}$ | 施加的宏观平均应变 |
| $A$ | Zener 各向异性比 $= 2C_{44}/(C_{11}-C_{12})$ |
| $\bar{\sigma}$ | FFT 解与解析解之间的常数平移张量 |
| $d\gamma_{ij}^p$ | 位错段 $ij$ 在网格点 $p$ 处产生的塑性剪切增量 |
| $dS_{ij}^p$ | 位错滑移面积与球形基本体积的交集面积 |

---

## 9. 与 ExaDiS/OpenDIS 框架的关联

本论文的第一作者 **N. Bertin** 是 ExaDiS（即 OpenDIS 的核心计算库）的主要开发者之一。本文提出的 DDD-FFT 方法正是 ExaDiS 框架中弹性应力求解模块的重要理论基础之一：

- **DCM 正则化**：ExaDiS 中的 `force_ddd_fft` 模块直接实现了本文的 FFT 耦合方案
- **周期性边界条件**：本文的全周期 PBC 框架与 ExaDiS 默认的模拟边界条件一致
- **Peach-Koehler 力计算**：本文的力学框架（方程 B.2–B.4）是 ExaDiS 节点力计算流程的理论依据

---

*总结整理自：Bertin et al., Modelling Simul. Mater. Sci. Eng. 23 (2015) 065009*
