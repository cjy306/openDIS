# CuNiSi A/B

每组仅包含generate.py、test.py（直接加载）、vtk.py、plot.py及山东提交脚本；父目录plot.py用于两组对比。

参数：300 nm盒，216个125 nm FR源，全盒初始密度1e15；FR位于0.2L~0.8L。B组在60/240 nm放(111)阻挡面，A组无界面。小粒90度脱钉，大粒球面绕过。112加载、1e3/s、0.5%应变，交滑移关闭。

state和迁移率直接在各组test.py中；颗粒尺寸/体积分数在各组generate.py顶部。修改两组共用参数时保持一致。

山东每组目录执行sbatch submit_test_sd.sh；该任务只执行generate→test（直接加载），不自动绘图。之后用sbatch submit_vtk_sd.sh转换VTK，绘图由用户自行执行。

运行前，在现有OpenDiS GPU构建中应用core_patch/mixed_obstacles.patch并重新编译pyexadis。该补丁只增加两类颗粒同时加载的接口。本机未运行DDD，也未提交集群任务。

不进行预弛豫，直接从init/init_config.data加载；初始内部调整可能影响早期应力与屈服。旧output_relax保留但不再读取。普通data/restart不是完整捕获状态的无损恢复文件。

Cu迁移率Medge=Mscrew=62500为纯Cu拖曳近似（B=1.6e-5 Pa s），不是CuNiSi在293.15K的独立标定；20b/5b网格及碰撞距离为试跑值。文献：https://www.nature.com/articles/s41467-021-21939-1

PBC：Rorient将(111)转为xy面，中间区初始化用于降低边界歧义，未声称修复所有PBC问题。VTK内Radius以b计；slab_check统计原中间区域外节点，A组没有界面时该计数可正常增大。
