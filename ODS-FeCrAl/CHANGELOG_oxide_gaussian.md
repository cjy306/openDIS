# 改动记录:氧化物高斯势力(Case D)C++ 实现

> 记录为实现 Case D 氧化物高斯势力,对 ExaDiS 核心 C++ 的每一处改动。
> 方案:预计算(方案1)—— 在 ForceSubcycling::pre_compute 里把每节点氧化物力预算成
> Kokkos::View,compute/node_force 里加上。氧化物不动、力慢变,每全局步算一次即可。
> 设计依据见 DESIGN_oxide_gaussian_force.md。开始:2026-06。

⚠️ **所有改动改完需重编译 ExaDiS 才能测**(linux .so,本地不可跑;push→超算/GPU 编译)。
⚠️ 所有改动带 `oxides.empty()` 保护 → 无氧化物时零开销、对现有仿真行为零影响。

---

## 改动清单

### [1] core/exadis/src/system.h — 数据结构 + 注入接口 ✅
- 新增 `struct OxideParticle { Vec3 center; double Rp; double A; int id; }`
  (放在 PlanarObstacle 之后)。
- System 类新增成员 `std::vector<OxideParticle> oxides;`
- 新增方法 `load_oxides(centers_b, Rp_b, A_vals)`(仿 load_obstacles 范式)。
- 位置:PlanarObstacle 结构后 / planar_obstacles 的 load 方法后。

### [2] core/exadis/src/integrator_types/integrator_subcycling.h — 高斯力预计算 ✅
ForceSubcycling 类内 5 处改动:

**(2a) 成员声明**(fgroup[] 之后):
- `Kokkos::View<Vec3*> foxide;` — 每节点氧化物力(device 端)
- `Kokkos::View<Vec3*>::HostMirror h_foxide;` — host 镜像(供 host 版 node_force,
  即 topology 分裂节点试算等 host 路径使用)
- `bool has_oxides = false;` — 开关标志

**(2b) pre_compute() 末尾加一行**:`compute_oxide_force(system);`
— 每全局步预计算一次(氧化物不动、力慢变,精度足够,不进 subcycle 循环)。

**(2c) 新增方法 compute_oxide_force(System*)**:
- `oxides.empty()` → has_oxides=false,直接 return(**零开销开关**)。
- oxides 拷 host→device:`Kokkos::View<OxideParticle*, T_memory_space>` +
  create_mirror_view + deep_copy(照抄 collision_orowan.h 的范式)。
- `Kokkos::resize(foxide, Nnodes)`(节点数变了自动重分配)。
- 并行 lambda:每节点对每氧化物算高斯力
  `F = 2*A*d/Rp² * exp(-d²/Rp²) * r̂`(方向氧化物→节点,排斥);
  **PBC**:用 `cell.pbc_position(center, p) - center` 取最近周期镜像位移
  (network.h:257,已确认是 KOKKOS_INLINE_FUNCTION,device 可调);
  **短程截断** 3*Rp;d²<1e-20 防除零。
- ⚠️ 踩坑:KOKKOS_LAMBDA 不能捕获 `this`(成员 View 隐式经 this 访问会挂),
  故 lambda 前先做局部拷贝 `Kokkos::View<Vec3*> fox = foxide;`(同 save_subforce 写法)。
- 末尾把 foxide deep_copy 到 h_foxide(host 镜像)。

**(2d) 新增方法 add_oxide_force(DeviceDisNet*)**:
- 并行把 foxide(i) 加到 nodes[i].f。
- **越界保护**:`if (i < foxide.extent(0))` — topology 操作在 pre_compute 之后新建的
  节点(i 超出预计算长度)本步拿不到氧化物力,下个全局步补上。可接受的近似。
- 在 **compute() 的 group==0 分支**调用(fsegseg 之后、drift 汇总之前)
  → 氧化物力归入 group 0(外场慢变力),与设计说明一致。

**(2e) 两个 node_force 都加**:
- host 版:`if (has_oxides && i < h_foxide.extent(0)) f += h_foxide(i);`
- device/team 版:`if (has_oxides && i < foxide.extent(0)) f += foxide(i);`
- 这两个函数被 topology(节点分裂试算)等调用,不加会导致试算力与实际力不一致。

### [3] Python 绑定 ✅(两个文件)
**(3a) core/exadis/python/exadis_pybind.h**:
- ExaDisNet 结构里加 `load_oxides(centers_b, Rp_b, A_vals)` 方法(转发到
  system->load_oxides;紧跟 load_twin_planes 之后,同款注释风格)。
- **SystemBind 构造函数**里加 `system->oxides = disnet.system->oxides;`
  — ⚠️ 关键坑:SystemBind 会 make_system 新建 System,若不拷贝,Python 侧
  load 进 ExaDisNet 的氧化物到了 driver 的 System 里就丢了(obstacles/planar
  原本就有同款拷贝,照加)。
**(3b) core/exadis/python/exadis_pybind.cpp**:
- `.def("load_oxides", ...)` 绑定(仿 load_obstacles 的 .def,含 docstring 与
  py::arg 签名),挂在 load_twin_planes 的 .def 之后。
**(3c) core/exadis/python/pyexadis_base.py【补丁,2026-07-12】**:
- Python 包装类 ExaDisNet 加 `load_oxides` 转发(load_twin_planes 之后三行)。
- ⚠️ 首跑踩坑:雄衡报 `AttributeError: 'ExaDisNet' object has no attribute 'load_oxides'`
  ——脚本里的 ExaDisNet 是 pyexadis_base 的 Python 包装类,不是 C++ 对象;
  只改 C++ 绑定不加 Python 转发层,方法到不了脚本。改完仍需重编译(.so 里才有 C++ 方法)。

### [4] ODS-FeCrAl/generate_caseD.py + test_caseD.py — Python 侧 (待做)
- 撒氧化物(中心/Rp/A),调 load_oxides 注入,跑单排 → BKS 验证。

---

## 关键实现细节 / 踩坑记录
1. **PBC 方法**:`Cell::pbc_position(r0, r)` 返回 r 相对 r0 的最近周期镜像**位置**
   (不是位移),所以位移 = `pbc_position(center, p) - center`。已读源码确认语义。
2. **KOKKOS_LAMBDA 捕获**:不能隐式捕获 this;所有成员 View 在 lambda 前做局部引用拷贝。
3. **新节点越界**:topology 分裂产生的新节点在本全局步 foxide 里没有条目,
   统一 bounds-guard 跳过,下步 pre_compute 自动覆盖。误差 = 一步内新节点缺氧化物力,
   慢变力下可忽略。
4. **单位约定**:center/Rp 以 b 为单位(与 obstacles 一致);A 的单位随 ExaDiS 内部
   力单位走,Python 侧注入时需把 Lehtinen 的 A=1.56e-18 Pa·m³ 换算成 ExaDiS 无量纲
   力单位——⚠️ **换算式还没推,写 generate_caseD.py 时必须先算清楚**(待办)。
5. **drift 模式兼容**:add_oxide_force 放在 drift 的 AddSubForces 之前,drift 汇总
   fgroup 时氧化物力已在 nodes[].f 里,不会重复计。
6. **与旧几何投影(CollisionOrowan)的共存规则**:两套机制完全独立——
   几何投影用 `obstacles` 列表 + 需 `collision_mode='Orowan'` 才实例化;
   高斯势用 `oxides` 列表(在 subcycling 力里,empty 即休眠)。
   现有脚本用 Retroactive 碰撞 + 两列表皆空 → **对 Case A/B 零影响**。
   ⚠️ 禁忌:同一批颗粒不得同时 load_obstacles + load_oxides 并开 Orowan 碰撞
   (会投影+推力双重作用)。Case D 正确用法 = 只 load_oxides + Retroactive。

---

### [5] ODS-FeCrAl/test_oxide_verify.py — 最小验证脚本 ✅(待编译后跑)
- 场景:一条穿盒无限刃位错(b=½[111], n=(0-11), 线向=n×b)+ 盒心一个氧化物,
  恒定纯剪应力 τ(b̂⊗n̂+n̂⊗b̂) 驱动位错沿 +b 滑向氧化物。500nm 小盒跑得快。
- 氧化物中心与位错同滑移面(起点=盒心沿 -b 退 1/4 盒,位移⊥n)→ 正面撞击。
- 支持 --A/--Rp/--tau/--steps;--A 0 为对照组(不注入,应自由通过)。
- 用 Retroactive 碰撞(不用 Orowan)→ 只有高斯势一种障碍机制在场。

**A 的单位换算(推导,写进脚本头,待 A 扫描验证)**:
  ExaDiS 内部长度以 b 计、应力 Pa;假设内部力单位=Pa·b² →
  **A_int = A_SI / b³** = 1.56e-18 / (0.248e-9)³ ≈ **1.02e11**(内部单位)。
  ⚠️ 该推导七成把握;第4级 A 扫描(0.1x/1x/10x)行为若与"强障碍"档吻合则证实,
  若 1.02e11 表现为隐形或无穷硬,按扫描结果修正换算式。

## 验证阶梯(编译后按序跑,每级有判据)
- [ ] **0 编译**:push → 超算/GPU 重编译,无错。
- [ ] **1 回归**:不加氧化物重跑一段已知 Case A → 曲线与改动前一致(开关有效,最重要)。
- [ ] **2 注入**:跑 test_oxide_verify.py → 日志出现 "[System] 1 oxide particles loaded"。
- [ ] **3 力存在/方向**:paraview 看演化 → 位错近颗粒减速弯曲、被挡、高应力下绕过留环;
      反例:径直穿过=A太小/没注入;远远弹飞=A单位错(大了几个量级)。
- [ ] **4 A 旋钮**:--A 取 0 / 0.1x / 1x / 10x → 临界 τ 单调上升,A=0 自由通过;
      顺带校验 A_int=1.02e11 换算是否落在"强障碍"档。
- [ ] **5 BKS 定量**(Case D 正式):单排氧化物,τ_DDD vs BKS 式,τ∝1/L 标度。

## 待办(接下来)
- [ ] generate_caseD.py:单排氧化物构型 + load_oxides 注入(等 3/4 级通过后写)
- [ ] test_caseD.py:仿 test_caseA,读构型跑扫掠 → BKS
