# FR 基体与辐照环对照

整个目录上传超算，OpenDiS核心不修改。默认安装路径为 `$HOME/openDIS`；不同路径请先 export OPENDIS_ROOT=/实际路径/openDIS。

## 文件
- 父目录generate.py：共用生成实现；每组generate.py：本组直接运行入口，支持--seed和--out。保持共用实现，防止两组FR生成逻辑分叉。
- no_loops/test.py、with_loops/test.py：分别整体弛豫，然后加载。
- plot.py：两组应力—总应变、总位错线密度—总应变对比，共两个子图。
- 每组vtk.py：弛豫及加载快照转换。
- 每组submit_test_kun.sh与submit_vtk_kun.sh：昆山ksagnormal01，1GPU/8CPU，120小时。
- relaxation.py：复用原3um FR算例的窗口判据。

## 提交
在各组目录提交，不要在父目录提交：

```bash
cd fecral_fr_loops_compare/no_loops
job=$(sbatch --parsable submit_test_kun.sh)
sbatch --dependency=afterok:$job submit_vtk_kun.sh
cd ../with_loops
job=$(sbatch --parsable submit_test_kun.sh)
sbatch --dependency=afterok:$job submit_vtk_kun.sh
```

test作业直接执行本组 `python generate.py` 自动生成自身初态，两个作业不写同一文件。相同seed控制随机数，两组FR源数量不同。也可以先在已配置ExaDiS的计算环境执行父目录的 `python generate.py` 生成两组。已有初态不覆盖。初态同时输出init_config_labeled.vtk及source_layout.csv，便于查看和核对FR位置、滑移系与面内角度。

两组完成后父目录执行 `python plot.py`，输出 `plots_pm2000_seg40_10/loops_compare.png`；其他输出目录可用 `python plot.py --input output_new`。对比图不是由独立组作业自动生成，以免另一组未完成。VTK位于各组vtk_output_pm2000_seg40_10和vtk_output_pm2000_seg40_10_relax目录。

test.py参照rho1e12算例组织：state在文件顶部、make_modules逐项定义模块、main读取构型并运行。支持--init、--out和--restart；保留本课题必须的整体弛豫。没有复制参考算例的FCC参数、5um盒子或无弛豫流程。两组修改seed时必须保持相同。

## 参数与物理近似
- 1 µm 周期立方盒，seed 12345，FR源长0.4 µm，12个滑移系均衡分配。未辐照250根（1e14/m²），辐照550根（2.2e14/m²）。
- FR使用现有两端固定开口线段，未补闭合返回段。这是人工源近似，不代表自然网络。
- PM2000，500 ℃、约21 dpa、仅中子辐照。按 Jung et al. 表3（DOI 10.1016/j.jnucmat.2016.11.022），加入800个 a<100> 方环和400个 a/2<111> 六边形环。
- 用各族平均直径17.0和19.6 nm作为对顶距离，环位置随机、轴向均衡。这是对实测尺寸分布和形貌的近似。

### 新旧算例目录

本版默认生成 init_pm2000_seg40_10，输出 output_pm2000_seg40_10_relax 和 output_pm2000_seg40_10。旧算例的 init、output 和 restart 请留作原工况。
- b=0.248nm、mu=81GPa、nu=0.3；a=2b、maxseg=40b、minseg=10b、rtol=0.5b、rann=1b；Ngrid128。
- BCC_NL tempK300，默认纯Fe拟合作为FeCrAl近似；不是合金专属迁移率标定。
- 两组弛豫和加载均设 num_bcc_plane_families=2，允许 {110} 和 {112} 面族。初始FR源分布在12个 {110}<111> 滑移系上。开启use_glide_planes及enforce_glide_planes；独立交滑移模块关闭，不加入扩散攀移机制。
- 无氧化物、无Cr起伏。两组整体零外载弛豫后重新开始[001]拉伸，erate1e3/s，总应变0.01（1%）。
- 自适应步长nextdt1e-12s、maxdt1e-9s。各组test.py参数直接写在使用位置；几何参数在generate.py。

## 弛豫与输出
复用真实dEp增量：完整非重叠>=10ns窗口内净塑性应变张量Frobenius范数<1e-7，一窗口满足就停止。整步累计可多出一个时间步。
window_activity是逐步范数和，仅诊断；净增量小不等于所有节点不动。结合活动量和构型观察。
默认最多1us/100万步；未收敛保存诊断并报错，不加载。
output_pm2000_seg40_10_relax保存relax.csv、summary.json和构型。加载重读弛豫构型并重置应变/时间起点。
两组弛豫后实际网络/密度允许不同；不能把环初始线长当成加载中增殖。
加载每100步输出曲线、保存构型快照，另保存loading_start.data及loading_final.data。
使用 --out output_new 避免覆盖；新目录VTK需手动 `python vtk.py --input output_new`。
墙钟中断后可 `sbatch submit_test_kun.sh --restart STEP` 恢复output_pm2000_seg40_10/restart.STEP.exadis。

## 验证
`python -B -m unittest check_scripts -v` 做离线检查，其中几何测试执行源仓库真实Python构造函数，但不等于原生DDD运行。首次超算试跑需要检查BCC_NL绑定、弛豫和小环存活。当前网格尚未做物理收敛测试。
