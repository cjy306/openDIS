# CuNiSi A/B

每组仅包含generate.py、test.py（含弛豫）、vtk.py、plot.py及昆山提交脚本；父目录plot.py用于两组对比。

参数：300 nm盒，216个125 nm FR源，全盒初始密度1e15；FR位于0.2L~0.8L。B组在60/240 nm放(111)阻挡面，A组无界面。小粒90度脱钉，大粒球面绕过。112加载、1e3/s、1%应变，交滑移关闭。

state和迁移率直接在各组test.py中；颗粒尺寸/体积分数在各组generate.py顶部。修改两组共用参数时保持一致。

昆山每组目录执行sbatch submit_test_kun.sh；该任务只执行generate→test（弛豫/加载），不自动绘图。之后用sbatch submit_vtk_kun.sh转换VTK，绘图由用户自行执行。

运行前，在现有OpenDiS GPU构建中应用core_patch/mixed_obstacles.patch并重新编译pyexadis。该补丁只增加两类颗粒同时加载的接口。本机未运行DDD，也未提交集群任务。

弛豫仍按完整>=10ns窗口的净塑性应变增量张量范数<1e-7判敛，未收敛不加载；保留同一内存网络进入加载，避免丢失颗粒捕获标记。普通data/restart不是完整捕获状态的无损恢复文件。

Cu迁移率Medge=Mscrew=62500为纯Cu拖曳近似（B=1.6e-5 Pa s），不是CuNiSi在293.15K的独立标定；20b/5b网格及碰撞距离为试跑值。文献：https://www.nature.com/articles/s41467-021-21939-1

PBC：Rorient将(111)转为xy面，中间区初始化用于降低边界歧义，未声称修复所有PBC问题。VTK内Radius以b计；slab_check统计原中间区域外节点，A组没有界面时该计数可正常增大。
