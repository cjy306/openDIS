# 切过点障碍对照

no_particles：无颗粒；with_cutting：50个直径4.9 nm、临界脱钉角90度的点障碍。两组均无大颗粒、无孪晶，不预弛豫。点障碍是现有角度脱钉模型，不等同于完整析出相切过物理。

共用初始FR网络：300 nm三向周期盒、216个125 nm两端固定FR源，12滑移系各18个，全盒密度1e15 m^-2，源位于z=60~240 nm，seed=12345。颗粒分布在整个盒子内，无孪晶邻近排除带；保持颗粒不重叠且避开FR固定端。

沿晶体[112]以1e3/s直接拉伸到0.5%（0.005）；Ngrid=64，maxseg/minseg=20/5 b，交滑移关闭，构型每1000步输出，应力应变每步记录。其余材料、迁移率和数值参数沿用CuNiSi_content_scan。本组用于诊断，不自动报告屈服强度。

将整个目录上传到集群~/openDIS，在两个子目录分别执行sbatch submit_test_sd.sh（generate→test）。随后分别执行sbatch submit_vtk_sd.sh转换构型；python plot.py绘制单组。父目录python plot.py生成两组compare.png：Stress (MPa)—总应变、位错密度—总应变。提交脚本不自动绘图。

使用已有混合颗粒接口及CollisionOrowan中的type=1切过分支。无颗粒组仍用同一碰撞模块但障碍表为空。无需为本算例修改核心；已发现的捕获回退/应变记账风险尚未修复，此对照用于定位问题。

输入保留既有small/large/twin文件格式，large与twin数组为空；no_particles的small也为空。VTK球体显示应以Radius缩放，统一点大小不代表实际颗粒大小。

已验证几何、配对、数量、语法与合成数据绘图；未在本机运行原生DDD或提交集群。
