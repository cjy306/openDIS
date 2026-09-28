# FCC Cu A75 应力过冲：五个独立初态

每组目录含 `generate.py` 和 `test.py`，唯一参数差异是 seed：
run_01=12345、run_02=23456、run_03=34567、run_04=45678、run_05=56789。
生成器共用本目录的 `overshoot_common.py`。部署时将整个目录放在超算 openDIS 根目录下。

## 固定参数

- FCC Cu，5×5×5 μm，三方向周期边界；[001] 应变率控制，1000 /s，终止总应变 0.005（0.5%）。
- 128 条两端钉扎直线 FR 源，每条 1 μm，无额外闭合臂；实际初始总密度 1.024e12 /m²。
- A75：8 个高 Schmid 系各 12 条，4 个零 Schmid 系各 8 条。按两类分别均分，不是全部 12 系等分。
- 各源中心独立均匀分布，滑移系随机分配但计数固定；线方向角 theta 独立均匀取 [0°,180°)，0° 为螺型、90° 为刃型，其余为混合型。并非刃/螺两类各半。
- 无预变形、无弛豫，显式交滑移模块 `cross_slip=None`。
- b=2.55e-10 m，mu=54.6e9 Pa，nu=0.324；FCC_0，Medge=Mscrew=64103，vmax=4000。
- 沿用原 A75 数值设置：a=4b，maxseg=200b，minseg=40b，rtol=1b，rann=2b，nextdt=1e-10 s，maxdt=1e-9 s，Ngrid=64，Subcycling/rgroups=[]。
- print_freq=1，write_freq=1000。关闭显式交滑移不代表碰撞、拓扑或重网格永远不改变线段滑移面。

## 运行

在配置好 pyexadis 的超算环境，从 openDIS 根目录依次执行，例如：

```bash
python stress_overshoot_5runs/run_01/generate.py
python stress_overshoot_5runs/run_01/test.py
```

其余组将 run_01 换成 run_02 至 run_05。生成阶段输出各组自己的 init_A75_seed*/，加载阶段输出 output_A75_seed*/。
初态目录含 init_config.data、init_config_labeled.vtk 和 source_layout.csv；CSV 记录各源中心、滑移系和 theta_deg，便于追溯随机实现。
默认直接运行即可；若覆盖生成器 --seed 或 --out，加载时需要用 --init 指向实际生成目录。

## 昆山 GPU 提交

将整个 stress_overshoot_5runs 目录上传至 `/public/home/cjy306/openDIS/`，进入该目录后执行：

```bash
for i in 01 02 03 04 05; do sbatch submit_kun_run_${i}.sh; done
```

五个独立作业，各使用 ksagnormal01 队列、1 GPU、8 CPU、120 小时上限；环境沿用 ODS-FeCrAl/submit_kun.sh。每个作业仅运行本组 generate.py → test.py，生成失败即停止，不自动绘图。

## 五组曲线对比

在 openDIS 根目录运行 `python stress_overshoot_5runs/plot.py`。
脚本读取各组默认输出目录中的 `stress_strain_dens.dat`，绘制应力–应变和位错密度–应变双面板图，保存为 `plots/five_runs_comparison.png` 和同名 PDF。
应变单位为 %，应力为 MPa，密度为 m⁻²；不平滑、不平均。缺失组会提示并跳过，全部缺失则报错；非有限数值或应变回退会报错，避免绘制含重启拼接问题的曲线。

## 初态验证

`python -B stress_overshoot_5runs/verify_setup.py`：语法、五组仅 seed 不同、初态布局可复现、A75 计数、FCC Schmid 分类、FR 线段面内性、端点钉扎数量、段长和实算总密度通过。
离线几何验证调用仓库真实 insert_frank_read_src 函数，不需要加载 pyexadis；不等同完整求解器运行验证。尚未生成实际初态或提交模拟。
