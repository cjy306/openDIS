# OpenDiS 0.1.0 开放位错动力学模拟器 — 完整中文文档

源文档：https://opendis.github.io/OpenDiS/
翻译整理：2026年4月

---

## 一、项目概述

OpenDiS（Open Dislocation Simulator，开放位错模拟器）是一个由社区驱动的开源项目，专注于为位错动力学（Dislocation Dynamics, DD）模拟提供一个高性能、可配置、可扩展的代码框架与开发平台。

该项目的核心使命是：通过开放协作，推动位错动力学模拟领域的科学创新，帮助研究者深入探索位错线的行为及其对材料宏观性能的影响。

项目快速链接：
- GitHub 仓库：https://github.com/OpenDiS/OpenDiS
- 官方文档：https://opendis.github.io/OpenDiS/
- 社区邮件：caiwei@stanford.edu（抄送 bertin1@llnl.gov，主题：OpenDiS signup）

---

## 二、项目宣言（Manifesto）

### 2.1 核心目标

**开放访问（Open Access）**：维护公开 Git 仓库，所有人均可免费下载和使用代码。

**社区开发（Community Development）**：优先考虑开发的便利性，让任何人都能创建扩展模块并贡献到代码库。

**拥抱创新（Embracing Innovation）**：不断适应 GPU 加速、大规模并行计算等新兴计算架构，始终保持计算效率的前沿地位。

### 2.2 项目动机

位错动力学领域已经积累了数十年的理论成果、模拟算法和计算范式方面的进展。开发者认为，现在正是启动社区共享式方法开发与代码编写工作的最佳时机。

通过民主化的 DD 开发模式，可以加速科学发现的步伐，使整个领域受益。同时，标准化的代码和数据管理实践将增强贡献代码与数据来源的可追溯性，提升研究的可重复性。

### 2.3 关键特性

**模块化设计与扩展性**：OpenDiS 采用模块化架构，便于开发和集成新的扩展模块，适用于广泛的研究应用场景。

**内置 Python 接口**：Python 接口大幅简化仿真配置，便于与其他程序及高性能计算模块对接；既可用作驱动层整合高性能模块，也是易学易用的原型开发工具。

**可替换 HPC 模块**：计算密集型部分被分配给可交换的独立代码模块，每个模块专门为最大化加速 DD 模拟而优化。

**严格测试与持续集成**：通过单元测试和 CI 实践，保证代码的稳定性和可靠性，让用户对模拟结果充满信心。

---

## 三、安装与系统要求

### 3.1 获取代码

OpenDiS 代码托管于 GitHub，可通过以下命令克隆：

```
git clone https://github.com/OpenDiS/OpenDiS.git
cd OpenDiS
git submodule update --init --recursive
```

### 3.2 系统要求

OpenDiS 的运行需要以下基础环境：

- Python 3.8 或更高版本
- CMake 3.16 或更高版本（用于 ExaDiS 编译）
- C++17 兼容的编译器（GCC 9+、Clang 10+、MSVC 2019+）
- Kokkos 框架（已作为子模块包含）
- pybind11（已作为子模块包含，用于 Python 绑定）
- 可选：CUDA 11+ 或 ROCm（用于 GPU 加速）
- 可选：OpenMP（用于多线程 CPU 并行）

### 3.3 支持的编译平台

文档提供了多种平台的编译指南：

- Mac（macOS）：支持 Apple Silicon 和 Intel 架构
- Linux（Ubuntu）：最常用的开发和生产环境
- Sherlock（Stanford HPC）：斯坦福大学高性能计算集群
- MC3 集群：Lawrence Livermore 国家实验室集群
- CMS3 集群：专用计算材料科学集群
- Raspberry Pi：嵌入式/ARM 平台
- Android（Termux）：移动设备实验性支持

### 3.4 基本编译流程（Linux/Mac 通用）

```
cd OpenDiS
mkdir build && cd build
cmake .. -DKokkos_ENABLE_OPENMP=ON
make -j$(nproc)
```

编译 GPU 版本（CUDA）：

```
cmake .. -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON
make -j$(nproc)
```

---

## 四、代码结构

### 4.1 整体架构概览

OpenDiS 的整体架构由两个层次构成：中央 OpenDiS 层和核心库层。

- OpenDiS 层（Python）：提供规范、接口、驱动器、工具和持续集成功能，作为整个框架的骨干
- 核心库 PyDiS：纯 Python 实现，用于学习、原型开发和小规模模拟
- 核心库 ExaDiS：C++ 实现，带 Python 接口，专为 HPC 和 GPU 大规模计算设计

### 4.2 OpenDiS 层详解

OpenDiS 层是整个框架的骨干，主要职责包括：

- 提供高层次规范（通过抽象 Python 类定义模块实现的规则和标准）
- 提供接口和驱动器，整合不同核心库和外部工具
- 提供工具函数（如网络数据转换、可视化辅助等）
- 支持持续集成（CI）测试，保障代码质量

整个 OpenDiS 层以 Python 编写，不仅简化了不同核心模块或外部库之间的交互，也极大降低了用户自定义增强功能的门槛。

### 4.3 核心库对比

PyDiS（Python）：
- 实现语言：纯 Python
- 适用场景：学习、原型、小规模仿真
- GPU 支持：否
- 并行模式：单线程
- 性能：较慢，但灵活易用
- 扩展难度：低（纯 Python）

ExaDiS（C++/GPU）：
- 实现语言：C++ + Kokkos，带 Python 绑定
- 适用场景：HPC 生产运行、大规模仿真
- GPU 支持：是（CUDA/ROCm）
- 并行模式：OpenMP / CUDA / HIP
- 性能：极高，针对硬件优化
- 扩展难度：中（需要 C++ 知识）

---

## 五、模块结构（Modules Structure）

### 5.1 模块规范

OpenDiS 基于模块化架构构建：每一项功能（如力的计算、迁移率定律、后处理分析等）都被实现为一个独立的模块，并以 Python 类的形式暴露给框架。

以下是一个典型的 OpenDiS 模块原型：

```python
class MyModule:
    """ OpenDiS 模块原型 """
    def __init__(self, state: dict, **kwargs) -> None:
        # 执行初始化操作
        pass

    def Compute(self, N: DisNetManager, state: dict) -> dict:
        # 以模块特定格式获取位错网络
        G = N.get_disnet(MyLibraryDisNet)

        # 对网络执行计算
        nodevalues = compute_values(G, state)

        # 对网络进行修改（如拓扑操作）
        G, mymoduleflags = modify_network(G, state)

        # 更新状态字典
        state["nodevalues"] = nodevalues
        state["mymoduleflags"] = mymoduleflags

        return state
```

### 5.2 模块规范要求

每个模块方法（如 Compute()）必须遵循以下规范：

- 接受 DisNetManager 对象和状态字典（state dict）作为输入参数
- 可以对位错网络执行计算
- 可以修改位错网络（如执行拓扑操作）
- 返回更新后的状态字典

特殊基础模块（如力计算模块和迁移率定律模块）有额外的规范要求：

- 力计算模块：参见 CalForce_Base 规范类（python/framework/calforce_base.py）
- 迁移率定律模块：参见 MobilityLaw_Base 规范类（python/framework/mobility_base.py）

### 5.3 模块间互操作性

不同模块之间的兼容性和互操作性通过以下两个核心对象来维护：

DisNetManager：容器类，允许各种位错网络数据结构共存并相互转换，使不同模块可以使用各自内部的数据结构来操作位错网络。

state 字典：包含模拟状态全局参数（如材料参数、仿真设置、节点力等）的字典，模块之间通过读写此字典进行通信。

### 5.4 构建完整 DDD 仿真流程

在 OpenDiS 中，一个完整的 DDD 仿真由一系列按顺序执行的模块构成。标准 DDD 仿真循环通常按以下顺序执行：

1. Force 模块 —— 计算位错节点上的力
2. Mobility 模块 —— 根据力计算节点速度
3. TimeIntegration 模块 —— 对系统进行时间积分
4. Collision 模块 —— 处理位错碰撞
5. Topology 模块 —— 实现拓扑操作（如节点分裂/合并）
6. CrossSlip 模块 —— 处理交滑移事件
7. Remesh 模块 —— 对位错线重新网格化
8. UpdateStress 模块 —— 更新应力状态
9. Output 模块 —— 输出数据（配置文件、统计信息等）

---

## 六、数据结构

### 6.1 DisNetManager 类

#### 描述

DisNetManager 是一个容器类，允许不同核心实现的位错网络数据结构在 OpenDiS 框架内共存并相互交互。它实现了来自不同核心库（如 PyDiS 和 ExaDiS）的模块之间的互操作性。

任何核心网络对象（如 DisNet 或 ExaDisNet）在被模块使用之前，都必须包装到 DisNetManager 对象中：

```python
from framework.disnet_manager import DisNetManager

G = ExaDisNet(...)
N = DisNetManager(G)
```

每个 OpenDiS 模块通过 get_disnet() 方法将 DisNetManager 转换为其内部所需的数据结构：

```python
class MyExaDisModule():
    def Foo(self, N: DisNetManager, state: dict):
        G = N.get_disnet(ExaDisNet)  # 转换为 ExaDisNet 对象
        # 在 ExaDisNet 对象上执行操作
```

#### 访问原始数据

DisNetManager 提供 export_data() 方法返回存储在字典中的原始网络数据：

```python
data = N.export_data()
```

返回的 data 字典包含以下条目：

- data["cell"]["h"]：仿真盒子矩阵（列向量为盒子向量）
- data["cell"]["origin"]：仿真盒子原点坐标
- data["cell"]["is_periodic"]：三个维度的周期性标志
- data["nodes"]["tags"]：节点标签数组（域、索引），尺寸为 (Nnodes, 2)
- data["nodes"]["positions"]：节点位置数组，尺寸为 (Nnodes, 3)
- data["nodes"]["constrains"]：节点约束数组，尺寸为 (Nnodes, 1)
- data["segs"]["nodeids"]：线段端节点的索引数组（node1, node2），尺寸为 (Nsegs, 2)
- data["segs"]["burgers"]：线段 Burgers 向量数组，尺寸为 (Nsegs, 3)
- data["segs"]["planes"]：线段滑移面法向量数组，尺寸为 (Nsegs, 3)

#### 属性与方法

- G（属性）：获取当前活跃的网络实例
- cell（属性）：获取仿真盒子对象
- __init__(disnet)：从已有位错网络对象构建 DisNetManager
- get_disnet(disnet_type)：将位错网络转换为指定类型的对象并返回
- get_active_type()：返回当前活跃的网络对象类型
- import_data(data)：从数据字典导入原始网络数据（需为 export_data() 的输出格式）
- export_data()：将原始网络数据导出为包含 'cell'、'nodes'、'segs' 键的字典
- write_json(filename)：将 DisNetManager 数据写入 JSON 文件
- read_json(filename)：从 JSON 文件读取 DisNetManager 数据
- num_nodes()：返回网络中节点的数量（int）
- num_segments()：返回网络中线段的数量（int）
- is_sane()：检查网络是否合理有效（bool）

### 6.2 DisNet 类（PyDiS 原生数据结构）

DisNet 是 PyDiS 核心库中位错网络的 Python 原生表示，以图（Graph）结构存储节点和线段信息。它基于 NetworkX 图库实现，适合小规模模拟和原型开发。

DisNet 继承自 DisNet_Base 规范类，并实现了 export_data() 和 import_data() 方法以支持与 DisNetManager 的数据转换。

### 6.3 ExaDisNet 类（ExaDiS 高性能数据结构）

ExaDisNet 是 ExaDiS 核心库在 Python 层面对位错网络的封装，其内部依托 SerialDisNet 和 DeviceDisNet 两个 C++ 类进行高性能计算，针对 GPU 内存布局进行了专门优化。

ExaDisNet 同样继承自 DisNet_Base，并实现了标准接口，支持与 DisNetManager 的无缝集成。

### 6.4 状态字典（State Dictionary）

#### 概述

全局 state 字典用于保存定义仿真状态的一组变量和数组。它在仿真过程中被传递通过各个模块，模块通过读写 state 字典中的条目来相互操作和通信。

state 字典有两个主要用途：

1. 定义对多个仿真模块通用的全局仿真参数（如材料参数）
2. 作为模块间的内存缓冲区，用于存储和检索信息（如力计算模块将节点力写入 state 字典，其他模块可以读取）

#### 全局仿真参数

在初始化模块和运行仿真之前，state 字典必须以全局仿真参数进行初始化，例如：

```python
state = {
    "crystal": "fcc",        # 晶体结构
    "burgmag": 2.55e-10,     # Burgers 向量大小（米）
    "mu": 54.6e9,            # 剪切模量（帕斯卡）
    "nu": 0.324,             # 泊松比
    "a": 6.0,                # 位错非奇异核半径（burgmag 单位）
    "maxseg": 2000.0,        # 最大网格线段长度（burgmag 单位）
    "minseg": 300.0,         # 最小网格线段长度（burgmag 单位）
    "rtol": 10.0,            # 积分容差（burgmag 单位）
    "rann": 10.0,            # 湮灭半径（burgmag 单位）
    "nextdt": 1e-10,         # 初始时间步长（秒）
    "maxdt": 1e-9,           # 最大时间步长（秒）
}
```

#### 主要参数说明

- crystal（可选）：晶体结构，支持 'fcc' 和 'bcc'，大多数迁移率定律需要此参数
- burgmag（必需）：Burgers 向量大小，单位为米，所有长度均以此值为基准缩放
- mu（必需）：剪切模量，单位为帕斯卡
- nu（必需）：泊松比
- a（必需）：位错非奇异核半径，以 burgmag 为单位
- maxseg（必需）：网格化最大线段长度，以 burgmag 为单位
- minseg（必需）：网格化最小线段长度，以 burgmag 为单位
- Rorient（可选）：晶体取向矩阵，默认为单位矩阵
- rtol（可选）：积分容差，以 burgmag 为单位，默认 0.25*a
- rann（可选）：湮灭半径，以 burgmag 为单位，默认 2*rtol
- maxdt（可选）：最大时间步长，默认 1e-7 秒
- nextdt（可选）：初始时间步长，默认 1e-12 秒

#### ExaDiS 专用参数

- split3node（可选）：是否允许分裂三节点（仅 BCC），默认为 1
- use_glide_planes（可选）：是否使用滑移面约束，FCC 默认 1，BCC 默认 0
- enforce_glide_planes（可选）：是否强制投影速度到滑移面，默认同 use_glide_planes
- num_bcc_plane_families（可选）：BCC 滑移面族数量，1={110}，2={110}+{112}，3={110}+{112}+{123}，默认 2

#### 模块状态变量示例

```python
calforce = CalForce(state=state, ...)
mobility = MobilityLaw(state=state, ...)

# 计算节点力并存入 state 字典
state = calforce.NodeForce(N, state)
print("节点力:", state["nodeforces"])

# 从 state 字典中读取节点力，计算节点速度
state = mobility.Mobility(N, state)
print("节点速度:", state["nodevels"])
```

确定哪些变量需要由模块保存的一个好原则是：将 state 字典视为从当前状态重新启动模拟所需的全部信息。

---

## 七、ExaDiS 代码架构（C++ 后端详解）

### 7.1 架构概述

ExaDiS 基于模块化设计实现，每种功能都可视为一个独立的模块。ExaDiS 的核心后端以现代 C++ 编写，并基于 Kokkos 框架构建。这使得同一份源代码可以面向多种硬件架构编译：

- 串行 CPU（Kokkos::Serial）
- 多线程 CPU（OpenMP）
- NVIDIA GPU（CUDA 架构）
- AMD GPU（HIP/ROCm 架构）

ExaDiS 类和函数在实现时充分考虑了高性能和高并行内核执行的需求。同时，代码设计尽可能地抽象复杂性，以降低开发新功能的门槛。例如，Topology 类中，TopologySerial 在主机（CPU）上以串行方式实现分裂多节点过程，而 TopologyParallel 则是同一过程的高度并行内核实现，可在设备（GPU）上执行。

### 7.2 项目目录结构

- cmake/：CMake 相关文件，包含预定义构建系统选项
- examples/：示例脚本和仿真案例，每个示例有独立编号子目录
- kokkos/：Kokkos 子模块目录
- python/pybind11/：pybind11 子模块目录（C++ 到 Python 绑定库）
- python/exadis_pybind.h 和 .cpp：C++ 类/函数绑定的定义与实现
- python/pyexadis_base.py：在 OpenDiS 内使用 pyexadis 的接口
- python/pyexadis_utils.py：pyexadis 工具函数
- src/：C++ 源代码和头文件，基类文件和通用函数位于根目录
- src/collision_types/：碰撞模块实现
- src/force_types/：力计算模块实现
- src/integrator_types/：时间积分模块实现
- src/mobility_types/：迁移率定律模块实现
- src/neighbor_types/：近邻搜索模块实现
- src/topology_types/：拓扑操作模块实现
- tests/benchmark/：性能基准测试
- tests/unit_tests/：单元测试

### 7.3 位错网络数据结构

#### SerialDisNet 类

SerialDisNet 是一个基于 STL 的类，用于在主机（CPU）上方便地创建、操作和修改位错网络。其执行空间为 Kokkos::Serial，内存空间为 Kokkos::HostSpace。

SerialDisNet 实现了所有底层拓扑操作，是创建初始位错网络和对已有网络进行拓扑操作的主要工具。

构造函数：
- SerialDisNet()：创建空位错网络
- SerialDisNet(double Lbox)：在边长为 Lbox 的立方体盒子内创建空网络
- SerialDisNet(const Cell& cell)：在指定盒子内创建空网络

主要属性：
- Cell cell：位错网络的仿真盒子
- std::vector<DisNode> nodes：节点数组
- std::vector<DisSeg> segs：线段数组
- std::vector<Conn> conn：节点连接关系数组
- int Nnodes_local：本地节点数量
- int Nsegs_local：本地线段数量

主要方法：
- number_of_nodes()：返回网络中节点总数
- number_of_segs()：返回网络中线段总数
- add_node(pos)：通过位置添加节点
- add_node(tag, pos, constraint)：通过标签、位置和约束添加节点
- add_seg(n1, n2, b)：添加连接节点 n1 和 n2 的线段，指定 Burgers 向量 b
- add_seg(n1, n2, b, p)：添加线段，同时指定滑移面法向量 p
- find_connection(n1, n2)：在 n1 的连接数组中查找到 n2 的连接（线段）
- generate_connectivity()：为所有节点生成连接数组
- seg_length(i)：返回线段 i 的长度
- constrained_node(i)：返回节点 i 是否为约束节点
- discretization_node(i)：返回节点 i 是否为离散化节点
- split_seg(i, pos)：在位置 pos 处分裂线段 i，返回新节点索引
- split_node(i, arms)：将节点 i 分裂，新节点包含指定的 arms 子集，返回新节点索引
- merge_nodes(n1, n2, dEp)：将节点 n2 合并到 n1，更新塑性应变张量 dEp，返回是否成功
- merge_nodes_position(n1, n2, pos, dEp)：将节点 n2 合并到 n1 的指定位置 pos
- remove_segs(seglist)：删除指定索引的线段列表
- remove_nodes(nodelist)：删除指定索引的节点列表
- purge_network()：清除孤立节点和零 Burgers 向量线段
- physical_links()：将网络分解为连接物理网络节点的位错链接集合
- dislocation_density(burgmag)：计算位错密度，单位 m^-2
- write_data(filename)：以 ParaDiS .data 格式写入位错网络
- save_node(i)：保存位错节点及其所有连接
- restore_node(saved_node)：从已保存的节点恢复位错节点及其所有连接

#### DeviceDisNet 类

DeviceDisNet 是使用 Kokkos Views 存储和访问位错节点、线段及连接信息的类，专门用于设备（GPU）上的执行。其执行空间为 Kokkos::DefaultExecutionSpace，在编译时自动映射到最高可用的执行/内存空间。

注意：DeviceDisNet 目前不实现拓扑操作，拓扑修改仍需通过 SerialDisNet 在主机上完成，然后同步到设备。

构造函数：
- DeviceDisNet(const Cell& cell)：在指定盒子内实例化空位错网络

主要属性：
- Cell cell：仿真盒子
- T_nodes nodes：节点的 Kokkos View
- T_segs segs：线段的 Kokkos View
- T_conn conn：连接关系的 Kokkos View
- int Nnodes_local：本地节点数量
- int Nsegs_local：本地线段数量

主要方法：
- update_ptr()：更新网络指针（对 DeviceDisNet 为空操作）
- get_nodes()：返回节点 View 数据的指针
- get_segs()：返回线段 View 数据的指针
- get_conn()：返回连接关系 View 数据的指针

#### ExaDiS 内的 DisNetManager 类

在 ExaDiS C++ 后端中，DisNetManager 作为容器类，用于在 SerialDisNet（主机）和 DeviceDisNet（设备）之间同步位错网络。

关键机制：只有当请求的网络类型实例未被标记为活跃时，才会触发内存拷贝，从而最小化主机与设备内存空间之间的数据传输开销。

构造函数：
- DisNetManager(SerialDisNet* n)：从 SerialDisNet 实例化 DisNetManager
- DisNetManager(DeviceDisNet* d)：从 DeviceDisNet 实例化 DisNetManager

方法：
- get_serial_network()：返回 SerialDisNet 实例的指针
- get_device_network()：返回 DeviceDisNet 实例的指针
- get_active()：返回当前活跃的网络类型（SERIAL_ACTIVE 或 DEVICE_ACTIVE）
- set_active(a)：设置活跃的网络类型
- Nnodes_local()：返回本地节点数量
- Nsegs_local()：返回本地线段数量

示例用法：

```cpp
// 获取主机（CPU）网络实例
SerialDisNet* net = disnet_manager->get_serial_network();

// 获取设备（GPU）网络实例
DeviceDisNet* dev_net = disnet_manager->get_device_network();
```

### 7.4 System 类

System 是 ExaDiS 中的基类，包含关于被模拟位错系统的所有信息，包括参数、晶体实例和位错网络对象。System 对象是在各模块之间传递的基本数据结构。

重要提示：System 对象必须使用 exadis_new() 或 make_system() 辅助函数来分配，以确保它被放置在所有执行空间都可访问的内存空间中。

构造函数：
- System()：实例化空 System 对象
- make_system(net, crystal, params)：从初始位错网络、晶体实例和参数对象创建 System 对象

主要属性：
- net_mngr：系统的位错网络管理器（DisNetManager*）
- neighbor_cutoff：系统的近邻截断距离
- params：系统的参数对象
- crystal：系统的晶体对象
- xold：存储旧节点位置的 Kokkos View
- extstress：外加应力张量（Mat33）
- realdt：当前全局时间步长
- dEp：当前时间步的塑性应变增量（Mat33）
- dWp：当前时间步的塑性自旋增量（Mat33）
- density：系统中的当前位错密度
- timer[]：系统计时器数组

主要方法：
- initialize(params, crystal, network)：用参数、晶体和初始网络初始化 System 对象
- register_neighbor_cutoff(cutoff)：注册系统中使用的最小截断距离
- get_serial_network()：返回 SerialDisNet 实例的指针
- get_device_network()：返回 DeviceDisNet 实例的指针
- Nnodes_local()：返回本地节点数量
- Nsegs_local()：返回本地线段数量
- Nnodes_total()：返回总节点数量
- Nsegs_total()：返回总线段数量
- plastic_strain()：计算所有位错线运动产生的塑性应变
- reset_glide_planes()：重置并为所有线段选择合适的滑移面
- write_config(filename)：将网络配置写入文件（ParaDiS 格式）
- print_timers(dev=false)：打印系统计时器信息

---

## 八、仿真设置完整流程

### 8.1 导入模块与核心库

```python
import os, sys

opendis_path = '/path/to/OpenDiS/python/'
if not opendis_path in sys.path: sys.path.append(opendis_path)
from framework.disnet_manager import DisNetManager
```

导入 PyDiS 模块：

```python
pydis_paths = ['/path/to/OpenDiS/core/pydis/python', '/path/to/OpenDiS/lib']
[sys.path.append(os.path.abspath(p)) for p in pydis_paths if not p in sys.path]
from pydis import DisNet, Cell, DisNode
from pydis import CalForce, MobilityLaw, TimeIntegration, Topology, Collision, Remesh
```

导入 PyExaDiS 模块：

```python
pyexadis_path = '/path/to/OpenDiS/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
import pyexadis
from pyexadis_base import ExaDisNet, SimulateNetwork, VisualizeNetwork
from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
```

注意：PyDiS 和 PyExaDiS 中的标准模块名称相同，同时导入时请使用别名：

```python
from pydis import CalForce as CalForce_pydis
from pyexadis_base import CalForce as CalForce_pyexadis
```

### 8.2 初始化 PyExaDiS

使用 pyexadis 模块时，必须在使用前显式初始化，并在完成后调用 finalize()：

```python
pyexadis.initialize()

# 在此处编写仿真设置和运行代码...

pyexadis.finalize()
```

initialize() 的可选参数：

- num_threads：指定 OpenMP 线程数，例如 pyexadis.initialize(num_threads=8)
- device_id：指定使用的 GPU 设备 ID，默认选择第一个可用设备

若未调用 initialize() 就执行 pyexadis 函数，会出现以下错误：

```
Constructing View and initializing data with uninitialized execution space
```

### 8.3 定义全局 State 字典

（参见第六章 6.4 节的详细说明）

```python
state = {
    "crystal": "fcc",
    "burgmag": 2.55e-10,
    "mu": 54.6e9,
    "nu": 0.324,
    "a": 6.0,
    "maxseg": 2000.0,
    "minseg": 300.0,
    "rtol": 10.0,
    "rann": 10.0,
    "nextdt": 1e-10,
    "maxdt": 1e-9,
}
```

### 8.4 创建初始位错构型

```python
G = ExaDisNet()
G.read_paradis('my_config.data')
N = DisNetManager(G)
```

### 8.5 定义仿真模块

定义各个仿真模块，例如迁移率模块：

```python
mobility = MobilityLaw(
    mobility_law='FCC_0',
    state=state,
    Medge=64103.0,
    Mscrew=64103.0,
    vmax=4000.0
)
```

定义仿真驱动器并运行：

```python
sim = SimulateNetwork(
    calforce=calforce,
    mobility=mobility,
    timeintegration=timeint,
    collision=collision,
    topology=topology,
    remesh=remesh,
    dt=1e-10,
    max_step=1000,
)
sim.run(N, state)
```

### 8.6 运行仿真

```
python my_script.py
```

交互模式（调试或事后分析）：

```
python -i my_script.py
```

交互模式下不要调用 pyexadis.finalize()，否则 ExaDiS 内存会被释放，数据将无法访问。建议：

```python
if not sys.flags.interactive:
    pyexadis.finalize()
```

### 8.7 性能考量

PyDiS 与 PyExaDiS 的性能对比：

- pydis 是纯 Python 代码，速度通常较慢，适合学习、原型开发或小规模仿真。
- ExaDiS 是专为利用现代计算架构（包括 GPU）高效性而设计的 HPC 代码，pyexadis 模块效率极高，适合生产运行和大规模仿真。

PyExaDiS 提供两种仿真驱动器：

SimulateNetwork：灵活的仿真驱动器，允许混合来自不同核心库的模块，支持用户自定义增强。数据需要流经 Python 管道，可能带来约 20% 的开销，特别是在 GPU 上运行时。

SimulateNetworkPerf：高性能仿真驱动器，消除了运行期间数据流经 Python 管道的开销，是对 ExaDiS C++ 后端驱动器的直接封装。缺点是不支持外部模块（仅限 pyexadis 模块），且不允许交互式数据可视化或用户自定义增强。推荐用于大型系统的生产运行（如应变硬化仿真）。

---

## 九、教程体系

### 9.1 Frank-Read 位错源

Frank-Read 源是位错动力学中的经典案例，展示位错在外加应力下如何增殖并形成位错环。OpenDiS 提供了多种实现方式：

- 纯 Python（PyDiS）实现：便于理解底层算法逻辑
- Python 调用 ExaDiS 实现：展示高性能模块的使用方式
- 图数据转换示例：展示 DisNet 与 ExaDisNet 之间的数据互转
- PyDiS 与 ExaDiS 混合使用：展示不同库模块的互操作性

### 9.2 Binary Junction（二元结）

展示两条不同 Burgers 向量的位错相遇时形成二元结（位错汇合）的过程。提供纯 Python 和调用 ExaDiS 两种实现。

### 9.3 Strain Hardening（应变硬化）

展示大量位错在外加应力下演化，导致材料流动应力随应变增加而升高的过程。这是最复杂的教程，提供 CPU 和 GPU 两个版本。

### 9.4 创建初始位错构型

介绍如何创建仿真所需的初始位错构型，包括：

- 从 ParaDiS .data 格式文件读取
- 手动构建（通过添加节点和线段）
- 从随机分布生成

### 9.5 可视化位错演化

OpenDiS 支持通过 VTK 格式输出和 ParaView 软件进行可视化，也支持实时 matplotlib 可视化（通过 VisualizeNetwork 模块）。

---

## 十、常见问题与注意事项

### 10.1 常见错误

Kokkos 未初始化错误：出现 "Constructing View and initializing data with uninitialized execution space" 时，说明未调用 pyexadis.initialize()。

模块名冲突：同时导入 pydis 和 pyexadis_base 中同名模块时，后者会覆盖前者，请使用别名区分。

内存访问错误：在 pyexadis.finalize() 后访问 ExaDiS 数据会导致内存访问错误，交互模式下应避免调用 finalize()。

### 10.2 使用建议

- 开发和调试阶段：优先使用 pydis 模块，代码简单易读，调试方便。
- 小至中等规模仿真：使用 SimulateNetwork 驱动器，灵活性高，支持混合模块。
- 大规模生产运行：使用 SimulateNetworkPerf 驱动器 + GPU，性能最优。
- 仿真参数一致性：通过统一的 state 字典传递参数，避免各模块参数不一致。
- 数据持久化：将重要状态量写入 state 字典，以支持断点续算（checkpoint/restart）。

### 10.3 社区与支持

- GitHub Issues：https://github.com/OpenDiS/OpenDiS/issues
- 社区邮件列表：caiwei@stanford.edu（加入后接收研讨会和教程通知）
- OpenDiS 研讨会：定期举办线上研讨会，内容涵盖框架使用和前沿算法

---

## 附录：参数速查表

state 字典常用参数汇总：

| 参数 | 类型 | 单位 | 是否必需 |
|------|------|------|----------|
| crystal | str，'fcc' 或 'bcc' | — | 可选 |
| burgmag | float | m（米） | 必需 |
| mu | float | Pa（帕斯卡） | 必需 |
| nu | float | 无量纲 | 必需 |
| a | float | burgmag 单位 | 必需 |
| maxseg | float | burgmag 单位 | 必需 |
| minseg | float | burgmag 单位 | 必需 |
| rtol | float | burgmag 单位 | 可选 |
| rann | float | burgmag 单位 | 可选 |
| nextdt | float | s（秒） | 可选 |
| maxdt | float | s（秒） | 可选 |
| Rorient | 3x3 矩阵 | — | 可选 |

---

文档结束
