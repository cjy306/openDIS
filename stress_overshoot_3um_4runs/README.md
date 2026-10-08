# 3 μm FR：四个新增随机初态

本目录独立生成 seed 23456、34567、45678、56789；不修改已有 stress_overshoot_3um_seed12345。
原 seed=12345 的数据只由绘图脚本读取。

| 目录 | seed |
|---|---:|
| run_02 | 23456 |
| run_03 | 34567 |
| run_04 | 45678 |
| run_05 | 56789 |

各组包含 generate.py、test.py、plot.py、vtk.py；根目录 common.py 和 vtk_common.py 提供共享实现。
部署时必须上传整个新目录到 /public/home/cjy306/openDIS/stress_overshoot_3um_4runs。

## 固定参数

FCC Cu，5×5×5 μm 三向周期盒；43条两端钉扎FR，每条3 μm；seed是四组间唯一参数差异。
总线长129 μm，实际密度1.032e12 m⁻²；高Schmid 8系各4条，零Schmid系2/5/8/11分别3/3/3/2条。
高Schmid比例32/43=74.4186%，不是严格75%。位置均匀随机，线方向角均匀取[0°,180°)，不是刃/螺各半。
[001]加载，1000 s⁻¹，终止应变0.005（0.5%），无预变形/弛豫，显式交滑移关闭。
b=0.255 nm，mu=54.6 GPa，nu=0.324；FCC_0，Medge=Mscrew=64103，vmax=4000 m/s。
a=4b、maxseg=200b、minseg=40b、rtol=1b、rann=2b、nextdt=1e-10 s、maxdt=1e-9 s。
SUBCYCLING_MODEL、Ngrid=64；Subcycling/rgroups=[]；Retroactive、TopologyParallel、LengthBased。
print_freq=1，write_freq=1000。每源60节点，总2580节点、2537段。

## 模拟提交：仅 generate → test

在新目录执行：
```bash
for i in 02 03 04 05; do sbatch submit_test_kun_run_${i}.sh; done
```

所有提交脚本采用昆山ksagnormal01、1 GPU、8 CPU、120小时。
模拟提交会重新生成初态并启动新模拟，请勿重复提交到正在运行的同组目录。
手动续跑可进入组目录使用 python test.py --restart STEP，不要重新执行生成步骤。

## VTK提交：仅转换已有快照

模拟结束后单独执行：
```bash
for i in 02 03 04 05; do sbatch submit_vtk_kun_run_${i}.sh; done
```

输出各组 vtk_A75_3um_seed*/config.0000001000.vtk 等序列，可在ParaView按文件组打开。
默认全部转换；手动可用 python run_02/vtk.py --start 0 --end 100000 --stride 2 筛选。
stride表示筛选后的快照每N帧选一帧。已有同名VTK将重写。

## 绘图

根目录运行 python plot.py：比较所有五个3 μm结果。
旧组读取 ../stress_overshoot_3um_seed12345/output_A75_3um_seed12345/stress_strain_dens.dat；
其余读取本目录 run_*/output_A75_3um_seed*/stress_strain_dens.dat。
新旧两个目录在超算需保持兄弟目录关系。
输出仅写入新目录plots/，含应力–应变、总密度–应变双面板PNG/PDF。
运行 python run_02/plot.py 等可画单组曲线。
未结束的运行只绘制完整记录，不补齐、不平滑；未换行末行提示跳过。
缺失组提示跳过，全部缺失则报错。

## 验证状态

本轮本地命令和Node运行环境均无法启动，尚未完成Python/Bash语法或求解器运行验证。
脚本依据本会话已核验的接口与3 μm参数编写；未提交作业，未修改原seed12345目录。
