# FR 基体与辐照环对照

整个目录上传超算，OpenDiS核心不修改。默认安装路径为 `$HOME/openDIS`；不同路径请先 export OPENDIS_ROOT=/实际路径/openDIS。

## 文件
- 父目录generate.py：共用生成实现；每组generate.py：本组直接运行入口，支持--seed和--out。保持共用实现，防止两组FR生成逻辑分叉。
- no_loops/test.py、with_loops/test.py：分别整体弛豫，然后加载。
- plot.py：两组应力、总线密度及相对首条加载记录密度增量对比。
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

test作业直接执行本组 `python generate.py` 自动生成自身初态，两个作业不写同一文件。相同seed保证FR背景一致。也可以先在已配置ExaDiS的计算环境执行父目录的 `python generate.py` 生成两组。已有初态不覆盖。初态同时输出init_config_labeled.vtk及source_layout.csv，便于查看和核对FR位置、滑移系与面内角度。

两组完成后父目录执行 `python plot.py`，输出 `plots/loops_compare.png`；其他输出目录可用 `python plot.py --input output_new`。对比图不是由独立组作业自动生成，以免另一组未完成。VTK位于各组vtk_output和vtk_output_relax目录。

test.py参照rho1e12算例组织：state在文件顶部、make_modules逐项定义模块、main读取构型并运行。支持--init、--out和--restart；保留本课题必须的整体弛豫。没有复制参考算例的FCC参数、5um盒子或无弛豫流程。两组修改seed时必须保持相同。

## 参数与物理近似
- 1um立方PBC，seed12345，250根0.4um FR，基体密度1e14/m2。12个{110}<111>系各20或21根，面内方向随机。
- FR使用现有两端固定开口线段，未补闭合返回段。这是人工源近似，不代表自然网络。
- with_loops增加1000个六边形a/2<111>棱柱环，四种轴向各250个，随机位置、不固定节点。
- 20nm指六边形对顶点距离，周长60nm；环贡献6e13/m2，总初始密度1.6e14/m2。不是等面积圆直径，不是某一ODS材料实测标定。
- b=0.248nm、mu=81GPa、nu=0.3；a=2b、maxseg=20b、minseg=5b、rtol=0.5b、rann=1b；Ngrid128。
- BCC_NL tempK300，默认纯Fe拟合作为FeCrAl近似；不是合金专属迁移率标定。
- 开启use_glide_planes及enforce_glide_planes；独立交滑移模块关闭，不加入扩散攀移机制。
- 无氧化物、无Cr起伏。两组整体零外载弛豫后重新开始[001]拉伸，erate1e3/s，总应变0.005。
- 自适应步长nextdt1e-12s、maxdt1e-9s。各组test.py参数直接写在使用位置；几何参数在generate.py。

## 弛豫与输出
复用真实dEp增量：完整非重叠>=10ns窗口内净塑性应变张量Frobenius范数<1e-7，一窗口满足就停止。整步累计可多出一个时间步。
window_activity是逐步范数和，仅诊断；净增量小不等于所有节点不动。结合活动量和构型观察。
默认最多1us/100万步；未收敛保存诊断并报错，不加载。
output_relax保存relax.csv、summary.json和构型。加载重读弛豫构型并重置应变/时间起点。
两组弛豫后实际网络/密度允许不同；不能把环初始线长当成加载中增殖。
加载每100步输出曲线、1000步快照，另保存loading_start.data及loading_final.data。
使用 --out output_new 避免覆盖；新目录VTK需手动 `python vtk.py --input output_new`。
墙钟中断后可 `sbatch submit_test_kun.sh --restart STEP` 恢复output/restart.STEP.exadis。

## 验证
`python -B -m unittest check_scripts -v` 做离线检查，其中几何测试执行源仓库真实Python构造函数，但不等于原生DDD运行。首次超算试跑需要检查BCC_NL绑定、弛豫和小环存活。当前网格尚未做物理收敛测试。
