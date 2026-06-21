# 交接摘要 — 预处理历史对棱柱环网络力学响应的影响 (DDD/ExaDiS)

> 给下一个对话/接手者看的。读完这份即可接上,不必回溯长对话。

## 1. 当前目标

研究 **预处理历史(弛豫、预变形)如何影响 BCC Fe 中棱柱环位错网络的力学响应**。
5 个工况共用**同一个**棱柱环初始网络 N0:

| 工况 | 流程 |
|------|------|
| A  | 原始 N0 → 直接加载(基线,无预处理) |
| B  | N0 → 预弛豫 → 加载 |
| C1/C2/C3 | N0 → 弛豫 → 预变形(0.1%/0.3%/0.5%) → 二次弛豫 → 加载 |

关键设计决定:
- **弛豫只做一次**,存成 `relaxed_config.data`,B 和 C 共用(不每个工况重跑)。
- **预变形量从 B 的加载曲线推导**(微塑性起始 / 过冲后 / 早期流动 三个点),不是拍脑袋定的固定值。等 A/B 跑完再定。
- 预变形用**总应变**(DDD 惯例,本低屈服体系下 ≈ 塑性应变)。
- 二次弛豫 = 卸载到零应力(模仿第一次弛豫,物理上自洽)。

## 2. 工作流 / 环境

- 本地 Windows 写代码(`e:\openDIS`) → git push → SCNet(Slurm)git pull → `submit.sh` 提交。
- 工作文件夹:**`inital_dislocation_structures/`**(注意拼写,中途从带空格改成下划线)。
- 用户跑的是 **ExaDiS v0.1.4**;`OpenDiS_old/` 是作者的 v0.2.0,**不要移植**(改动太散、构建方式也变了,当前课题不需要)。
- 材料参数(Fe BCC):b=0.248nm,μ=82GPa,ν=0.29。

## 3. 关键结论

- **rgroups 可留空 `[]` 让 ExaDiS 自动选子积分区间**(已追完 Python→pybind→C++ 链,见 `integrator_subcycling.h` 第220行 `if (rgroups.size()==0)`,rmax=min(maxseg,cutoff),rmin=max(0.3·minseg, 3·rann),MAXGROUPS=5)。
- **SUBCYCLING_MODEL 的 CalForce 必须传 `cell` 参数**,否则 `get_module_arg` 抛 KeyError。解决:先 `read` 网络拿到 `cell = net.get_disnet(ExaDisNet).cell`,再 `build_modules(cell)`。
- **dt 是自适应的(误差控制),不是固定的**;持续很低的 dt = 刚性模式信号,不是 bug。
- 弛豫收敛判据:ρ 走平 + Nnodes/Nsegs 不漂 + dt 爬升稳定。**1万步弛豫已结构收敛**(ρ≈9.42e11,降约2%,Nnodes 3300步后稳定),dt 仍低(~1e-12)但用户确认正常、决定不深究。`relaxed_config.data` 已验证有效(3677节点,5μm盒,健康 BCC 网络带 <100> junction)。
- 图内文字一律英文(matplotlib 无 CJK 字形,Linux 超算渲染成豆腐块)。

## 4. 文件路径(都在 `inital_dislocation_structures/`)

| 文件 | 作用 |
|------|------|
| `generate_loops.py` | 生成共用 N0。盒5μm,ρ=1e12,R=100nm,maxseg=200,b=0.248nm,~199环。`generate_prismatic_config('bcc', ...)` |
| `diagnose_relax.py` | 零应力长弛豫,存 `relaxed_config.data`。`--seed`/`--steps`(默认5000)。state: maxdt=1e-8, a=4.0 |
| `caseA_baseline.py` | 读原始 `init_config.data`,[001] strain_rate 加载。erate=1e4,max_strain=0.01 |
| `caseB_load.py` | 读 `relaxed_config.data`,正式加载。同上参数 |
| `caseC_prep.py` | 读 relaxed → 预变形 → 二次弛豫 → 存 `predeformed_relaxed_config.data`,**停**(不加载)。`--seed`/`--predeform`(%必填)/`--relax2-steps` |
| `caseC_load.py` | 读 `predeformed_relaxed_config.data`,正式加载。`--seed`/`--predeform`/`--restart` |
| `plot_relax.py` | 弛豫收敛 3 联图(ρ / Nnodes&Nsegs / dt vs Step),读 7列 .dat |
| `plot_compare.py` | 5工况最终对比(应力-应变 + 应变-密度)。compute_metrics: σ_peak/ε_peak/σ_flow(尾部40%均值)/Δσ_overshoot/σ_0.2%(E=2μ(1+ν)) |
| `parse_slurm_relax.py` | 应急:从 slurm 日志抓 (step,nodes,dt) 合并 .dat 画收敛图。等7列版部署到超算后即弃用 |
| `paraview.py` | data→vtk。顶部 `INPUT`/`OUTPUT`/`START`/`END` 配置式。INPUT 可为单文件或目录 |
| `check_relaxed.py` | 刚性模式检测器(查最短段/最近非键合节点对)。用户决定**不跑** |

**核心代码改动(已 commit+push 到 origin/main,但 SCNet 尚未 pull 生效):**
- `core/exadis/python/pyexadis_base.py`:给 `stress_strain_dens.dat` 加了 Nnodes/Nsegs/dt 三列。
  - `step_print_info`(~873行):加 `Nsegs = N.num_segments()`,append 改成 `[istep, strain, stress, density, Nnodes, Nsegs, dt]`
  - `write_results`(~744行):表头 `# Step Strain Stress Density Nnodes Nsegs dt`,`fmt='%d %e %e %e %d %d %e'`

## 5. 踩过的坑

- **.dat 列布局**:最初我假设有 Walltime 列,用户纠正"本身就没有walltime"。最终布局 `Step Strain Stress Density Nnodes Nsegs dt`(无 walltime)。
- **CalForce 缺 cell**:见第3节结论。caseB 最初没传 cell 报 KeyError。
- **图内中文乱码**:matplotlib 无 CJK 字形 → 全改英文(plot_relax/plot_compare/parse_slurm_relax)。已存记忆 `plot-text-english`。
- **版本不同步谜题**:SCNet 产出 4列 .dat,本地仓库是 7列改动。诊断为 SCNet 跑的是旧 `pyexadis_base.py`(没 pull)。改动确已 push 到 origin/main。
- **paraview.py 形式**:用户先要仿 HomeWork 版,后明确要 `input=/output=` 顶部配置式(已改)。同时删了本课题用不上的杂质球/孪晶面/wrap_vtk_pbc(死代码)/rorient。

## 6. 下一步任务

1. **(用户负责)** push 新脚本 + `pyexadis_base.py` 7列改动,在 SCNet `git pull` 让7列版生效;跑 `caseA_baseline.py` 和 `caseB_load.py`。
2. **(待 A/B 跑完)** 写 `extract_predeform.py`:从 B 的加载曲线推导 C1/C2/C3 三个预变形点(微塑性起始 / 过冲后 / 早期流动)。
3. 之后按推导出的值跑 caseC_prep(每个 predeform 一次)→ caseC_load,最后 `plot_compare.py` 出五条对比曲线。

## 7. 待确认/遗留

- `paraview.py` 里 `import numpy` 现在没用到(删杂质逻辑后),保留无害,可删。
- C 工况输出路径约定:`output_caseC_p0.1_seed{SEED}/load/stress_strain_dens.dat`(plot_compare.py 里已按此找)。
- 拆脚本策略:先只弄 C1,跑完用户改参数再跑 C2/C3。
