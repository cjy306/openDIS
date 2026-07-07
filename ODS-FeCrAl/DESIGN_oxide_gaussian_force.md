# 设计说明:氧化物高斯势力(Case D)—— 集成进 SUBCYCLING 力体系

> 动手写 C++ 之前的设计对齐文档。**最新版(2026-06,参数已解锁、集成方式已定)。**
> 状态速览:设计 ✅ / 参数 ✅(A、Rp、BKS 均已到手)/ 代码 ◻ 未开始。

---

## 0. 一句话

给每个氧化物颗粒造一个**排斥性高斯力场**,位错节点靠近时受到把它推开的力;
这个力**叠加进现有 SUBCYCLING 力体系**(加在 subcycling 内、用 oxides 空判断当开关),
不另起炉灶、不动几何投影那套。调强度参数 A 可从"软可剪切"连续过渡到"硬 Orowan 绕过"。

---

## 1. 物理:高斯势是什么

每个氧化物中心放各向同性排斥势,位错节点 r_node 到氧化物中心 r_ox 距离 r=|r_node−r_ox|:

    势能   U(r) = A · exp(−r² / Rp²)
    力     F(r) = −dU/dr = (2A r / Rp²) · exp(−r² / Rp²) · r̂    (r̂ 由氧化物指向节点,往外推)

- **A**  = 势强度 [Pa·m³] → 决定颗粒多"硬"。A 小=弱可剪切;A 大=强 Orowan。
- **Rp** = 势宽度 [长度] → 氧化物有效半径;r > ~3Rp 后力≈0(短程,可截断)。

**为什么用高斯**(Lehtinen 范式):① 空间连续光滑、对节点无力突变 → 不破坏 subcycling 积分;
② 单参数 A 覆盖软→硬全谱;③ 短程 → 只需查近邻氧化物,无长程求和负担。

⚠️ 这是**排斥不可剪切**建模:A 调强弱=改变绕过所需应力,位错始终绕过不穿切。
   氧化物本就不可剪切(CLAUDE §1.2;Takahashi 2011 实证:穿切需 21–128 GPa,不现实),故正确。
   # PHYS-APPROX: 各向同性高斯排斥势;ceiling=无界面/失配应力;upgrade=加 Eshelby 失配场按类型分强度。

---

## 2. 参数:全部已到手 ✅(2026-06 解锁,不再等 MD)

| 参数 | 值 | 来源 | 备注 |
|---|---|---|---|
| **A(高斯势强度)** | **1.56×10⁻¹⁸ Pa·m³** | Lehtinen 2018 (Sci.Rep. 8:6914) | BCC-Fe 强 Orowan 障碍标定值,MD 反推 |
| **Rp(势宽度)** | 5、10 nm(或用实验尺寸 6–27nm) | Lehtinen 2018 / Yan 2023 | 直接用氧化物有效半径 |
| **BKS 验证式** | 见 §5(完整) | Lehtinen 2018 Eq.2-3 | 含 3D 密度→间距换算 |
| **Y₂O₃ CRSS 交叉值** | ~400–900 MPa (D=5nm) | Takahashi 2011 | 用于 Y₂O₃ 专属反标微调 |
| **强度谱 α(r)** | 0.27–0.52 | Yan 2023 | 软→硬档位参照 |

**关键**:Lehtinen 2018 就是 §3.2 所指的"Lehtinen 范式"原文,它的势形式
U=A·e^(−r²/Rp²)、F=2Ar·e^(−r²/Rp²)/Rp² **与本设计完全一致**,A 值可直接采用。

⚠️ 三点诚实分寸:
- Lehtinen 的 A 是**通用强障碍**值(非专门 Y₂O₃);先直接用它起步,论文注明出处。
  要 Y₂O₃ 专属值,用 Takahashi 2011 的 CRSS(~900MPa)反标微调。
- Lehtinen 用 **750K 高温**参数(G=75GPa);本课题室温 Yan(μ=81GPa)。A 是能量量纲,
  换基体模量时可按 μ 比例微调(A ∝ μ 量级)。
- Rp 直接用实验尺寸,不必等任何标定。

---

## 3. 集成点:加进 SUBCYCLING 力体系(已定方案)

**现状**(已读源码确认):
- 用户固定用 `force_mode='SUBCYCLING_MODEL'` + Subcycling 积分,C++ 侧 = `ForceSubcycling`
  (`integrator_subcycling.h:189` typedef)。内部 = FLong(ForceFFT 远场)+ FSegSeg(近场),
  按相互作用距离分 group 0..4。

**已定集成方式(用户 2026-06 拍板)**:
> 氧化物力**直接加进 subcycling 力体系**,用 `oxides.empty()` 当自动开关——
> 有氧化物才算,没氧化物一行跳过、零开销、对现有纯 subcycling 仿真行为完全不变。
> (这与用户旧几何 Orowan 的 `if (obstacles.empty()) return;` 同款范式。)

**放哪个 group**:作为**外场力**(类比外加应力,非位错-位错相互作用),叠加到 **group 0**
(长程/慢变组)。理由:氧化物不动、节点在单个 subcycle 内位移小,力慢变,每全局步算一次即可,
不必在短程 subcycle 里反复重算 → 省算力、不扰动 subcycling 稳定性。

**开关逻辑(伪代码)**:
    // 在 ForceSubcycling 的 pre_compute / node_force 里
    if (system->oxides.empty()) return;      // 没氧化物 → 零开销跳过,行为不变
    for each node i:
        f_ox = 0
        for each oxide p:                     // 双循环(§4)
            r = pos[i] - oxide[p].center
            d = |r|
            if (d < 3*oxide[p].Rp):           // 短程截断
                f_ox += (2*A*d/Rp²)*exp(-d²/Rp²) * (r/d)
        add f_ox to node i force (group 0)

---

## 4. 数据结构 + 近邻查找

**氧化物几何**(复用现有 `SphericalObstacle` 思路,加力参数):

    struct OxideParticle {
        Vec3   center;   // 中心 (以 b 为单位)
        double Rp;       // 高斯势宽度 (以 b 为单位)
        double A;        // 高斯势强度
        int    id;
    };
    std::vector<OxideParticle> oxides;   // 存在 System 里

**注入接口**:复用现有 `load_obstacles` 的 Python→C++ 通路,扩展带 A/Rp → `load_oxides()`。
Python 侧生成氧化物中心/尺寸(复用撒球逻辑),A/Rp 作参数传入。

**近邻查找(已定:双循环)**:高斯短程(截断 3Rp),Case D 单排/几十个氧化物,
直接 O(Nnode×Nox) 双循环,简单不易错。F 工况氧化物多了再上格点近邻(ExaDiS neighbor_types)。

---

## 5. 验证:BKS 交叉检查(Case D 判据)

**做法**:DDD 里放**单排等间距氧化物**,测位错扫过临界切应力 τ_DDD,对照解析式。

**BKS 式(从 Lehtinen 2018 Eq.2-3 抄准)**:
    σ_r = C · Gb/(L−D) · [ln(L/rcore)]^(−1/2) · [ln(D̄/rcore) + 0.7]^(3/2)
      C = 1/(2π)
      L = (2·D·ρp)^(−0.5)      # 3D 数密度 ρp → 滑移面内颗粒间距
      D̄ = D·L/(D+L)            # L 与 D 的调和平均
      rcore = 位错核半径, b = 柏氏矢量, G = 剪切模量, D = 颗粒直径

**Yan 2023 Orowan 式(备用)**:σ = 0.176·M·μ·b·√f / d · ln(d/2b)。

**判据**:调 A/Rp 后,τ_DDD 与 BKS 吻合(量级 + 随间距 L 的标度 τ∝1/L)
→ 证明高斯势障碍物理可信,可进 E/F 耦合。
参考量级:Lehtinen 2018 报告 D=10–20nm、ρp=1e21–2e22 下 σy≈500–2000 MPa。

---

## 6. 代码待办(动 C++,需重编译)

1. **`system.h`**:加 `OxideParticle` 结构 + `std::vector<oxides>` + `load_oxides()` 接口。
2. **`integrator_subcycling.h`(ForceSubcycling)**:pre_compute/node_force 加氧化物高斯力,
   带 `oxides.empty()` 开关,叠加到 group 0。
3. **`exadis_pybind.cpp`**:暴露 `load_oxides()` 给 Python。
4. **Python 生成脚本**:`generate_caseD.py` 撒氧化物(中心/Rp/A),写进构型或单独注入。

⚠️ **编译约束**:改 C++ 核心要重编译。当前用户 GPU 在跑 Case A 预变形,
   建议 Case A 跑完、腾出编译环境后再动 C++,避免干扰。

---

## 7. 不做什么(边界)

- 不动几何投影那套(`collision_types/collision_orowan.h`)——弃用,不改造。
- 不模拟氧化物"被剪切/破坏"(展望,需第二层 MD)。
- 不模拟氧化物作为"缺陷阱"(属点缺陷动力学,本 DDD 不做)。
- 不加 Eshelby 失配应力场(Lehtinen 也未加,适合非共格颗粒;本阶段够用)。
