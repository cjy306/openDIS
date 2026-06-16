# ODS-FeCrAl 实施计划 (PLAN)

> 本文件 = "做什么 / 排程"。物理依据、机制、待验证项见同目录 `CLAUDE_FeCrAl.md`("为什么")。
> 最后更新:2026-06-16。每完成一项更新对应状态。

---

## 0. 已锁定前提(不再讨论)

| 项 | 结论 |
|---|---|
| 课题 | ODS-FeCrAl 三体 DDD:量化 环 + α′ + 氧化物 在**屈服阶段**的叠加规律 + 验证氧化物免疫相消 |
| 框架/平台 | ExaDiS / OpenDiS;跑在 **SCNet**(Slurm);工作区 `e:\openDIS\ODS-FeCrAl\` |
| 几何 | **全程 bulk 周期 RVE**(放弃自由表面纳米柱;ExaDiS 无自由表面/FEM-BVP 能力) |
| 承重墙 | §2.1 相消机制已对照 Pachaury 原文核实 ✅;氧化物免疫有 immobile-loop 对照实验支撑 ✅ |
| novelty | 三体在 FeCrAl 屈服阶段无人做;收紧到"**α′ + 相消免疫**"(见下 Phase 0) |

---

## 1. 状态总览

| 阶段 | 任务 | 状态 |
|---|---|---|
| Phase 0 | V1 novelty 核查 + 重表述 | ✅ 基本完成(残留:知网中文库、Robertson 正刊全文待补) |
| Phase 0 | V3 氧化物实验参数(尺寸/数密度) | ◑ 基准参数已提取(Yan 2023,见 CLAUDE §3.5);辐照后输入待 Zhang 2020 |
| Phase 0 | T2/V5 氧化物 BCC-MD 标定参数 | ◻ 待读(源非 arXiv,需下载 PDF) |
| Phase 1 | M1 Case A 基准 harness | ◑ 脚本已写+静态自查通过,待超算实跑 |
| Phase 1 | M0 SCNet 平台启动 | ◻ 阻塞:需用户上平台 |
| Phase 2 | Case D 氧化物高斯势(★创新核心) | ◻ 阻塞于 T2 |
| Phase 2 | Case B 位错环 | ◻ |
| Phase 2 | Case C α′ 相干应力(⚠️长杆) | ◻ |
| Phase 3 | Case E 环+α′(定性锚定 Pachaury) | ◻ |
| Phase 3 | Case F 三体(★论文落点) | ◻ |
| Phase 4 | 叠加律 / DBH 对比 / 免疫判决 | ◻ |

---

## 2. 分阶段计划(含验证判据与阻塞项)

### Phase 0 — 文献与参数地基(不依赖代码)
- **V1 novelty** ✅:三向检索(用户自查 + alphaXiv + Semantic Scholar)确认三体空白。
  最近邻已界定:Pachaury 2023 [环+α′]、Robertson/Gururaj 2011 [氧化物+环,无 α′、非 FeCrAl]。
  收紧后表述见 `CLAUDE_FeCrAl.md` §1.4。**残留**:SS 限速未覆盖知网;Robertson 正刊全文未读。
- **V3 氧化物参数** ◑:**基准参数已提取** — Yan et al. 2023 (arXiv:2309.03703),
  FeCrAlZrTi-ODS,未辐照:氧化物 5.9–27.2 nm、数密度 1.4–9.6×10²²/m³、α(r)=0.27–0.52、
  类型 YAG/Y-Ti-O/Y-Zr-O,详见 `CLAUDE_FeCrAl.md` §3.5。
  **残留(辐照后输入微结构)**:需 Zhang et al. 2020, J. Nucl. Mater. 533, 152094
  (DOI 10.1016/j.jnucmat.2020.152094;MA956 中子辐照,含环+氧化物+α′ 实测)——
  非 arXiv、ScienceDirect 反爬,**待用户下载 PDF**。
- **T2/V5 氧化物 BCC 标定**:候选源(均 bcc-Fe + Y₂O₃,**非 arXiv,需下载**):
  Yashiro 2012(MD 螺位错-Y₂O₃)、Takahashi 2011(原子-连续介质 位错-Y₂O₃)、
  Bakó 2007(DDD Y₂O₃/PM2000,方法学先例)。→ 目标:高斯势 A、Rp 或相互作用强度。

### Phase 1 — 把 harness 跑起来(无新物理)
- **工作流**:本地只写代码 + 静态自查(linux .so 本地不可跑);改完 git push,
  超算 git pull 后用已有 submit.sh 跑。SCNet 路径/env 待用户上平台后填。
- **M1 / Case A 基准** ◑:BCC FeCrAl、{110}⟨111⟩、bulk 周期 5×5×5 μm、0.2% 偏移屈服、3 随机实现。
  已写文件(均在 `ODS-FeCrAl/`):
  - `generate_glide_loop.py` — 可动滑移环生成器(b·n=0、闭合无钉扎、按密度生成);B/D 也复用
  - `test_caseA_baseline.py` — Case A 主脚本(无障碍物,参数 Yan 2023 一套)
  - `extract_yield.py` — 0.2% 偏移屈服提取 + 多实现均值/误差棒(读 stress_strain_dens.dat)
  已定决策:材料=Yan 2023 一套;构型=可动滑移环;ρ=1e12/m²;半径 0.10–0.20 μm;
  屈服=0.2% offset + 多 seed 平均。
  → 验证(待超算):出屈服曲线 + 均值/误差棒。这是 B–F 全复用的底座。
  ⚠️ 已知风险(见 §下方残留):滑移环线张力致初期密度回落;迁移率/盒子尺寸量级待实跑校。
- **M0 / SCNet 启动**:编译 OpenDiS/ExaDiS、跑通 stock example、确认 Kokkos OpenMP 后端。
  → 阻塞:用户上平台后才能做。

### Phase 2 — 单障碍物物理(按"价值/工作量"排序)
- **Case D / 氧化物(★创新核心,自洽、可独立验证)**:在 `core/exadis/src/force_types/`
  **新写高斯势力** U=A·e^(−r²/Rp²),F=2Ar·e(...)/Rp²;A、Rp 由 BCC-MD 标定(T2)。
  **不复用**现有几何投影 Orowan(`collision_types/collision_orowan.h`)。
  → 验证:单排氧化物强度与 **BKS 解析式**吻合(BKS 仅用于氧化物)。
- **Case B / 位错环**:BCC 配置生成 a/2⟨111⟩ 六边形(可动)+ a⟨100⟩ 方形(不可动)。
  → 验证:单环硬化随密度合理。
- **Case C / α′(⚠️ 长杆,最重)**:相干应力场。轻量路线 = **离线预计算应力场、采样进位错段当
  外部力**,不在 ExaDiS 里耦合活 FEM;迁移率先用成分平均 BCC。→ 验证:复现 α′ 弱摩擦行为。

### Phase 3 — 耦合与判据
- **Case E / 环+α′**:**定性**复现 Pachaury 相消(复现出=实现正确,不抠 MPa 数值)。需 B+C。
- **Case F / 三体(★论文落点)**:测 F 是否 = B+C+D 线性相加?氧化物贡献是否独立于 α′ 相消?
  → 回答免疫假设(及"是否偏构造性"的更强版本,见 `CLAUDE_FeCrAl.md` §2.2)。

### Phase 4 — 分析出论点
三体叠加律、与 DBH 模型对比、氧化物免疫判决、误差棒/分布。

---

## 3. 关键路径与风险
- **真正的长杆是 Case C(α′ 相干应力)**,而 F(论文落点)依赖它 → 它定时间线瓶颈。
- 对策:**前置 Case D(氧化物)**——既是创新核心,又自洽、BKS 可独立验证,不卡 α′,
  能早拿到一个可信的新结果;C 这条重活并行慢慢啃。
- **推荐总顺序:A → D → B → C → E → F**(D 提前;C 作为长杆并行推进)。

---

## 4. 待用户补的输入
- [ ] **SCNet 安装路径 + conda env 名**(上平台后填,用于 submit 脚本)。
- [ ] **Semantic Scholar API key**(已申请,到账后配进 MCP,解限速)。
- [ ] **Yashiro 2012 / Takahashi 2011 的 PDF**(非 arXiv,需下载放进仓库供 T2)。
- [ ] (可选)**Robertson/Gururaj J. Nucl. Mater. 正刊 PDF**(彻底闭合 V1 用)。

---

## 5. 眼下下一步(不依赖 key,可立即做)
1. **alphaxiv 读 arXiv:2309.03703** → 抠氧化物尺寸/数密度,给 V3 起头。
2. 之后:**本地起 Case A harness**(Phase 1,搭 B–F 底座)。
3. key 到账后:批量补扫 novelty + 拉 V3/T2 文献。
