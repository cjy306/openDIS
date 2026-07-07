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

---

## 待办(接下来)
- [ ] **A 的单位换算**:Lehtinen A=1.56e-18 Pa·m³ → ExaDiS 内部单位(关键,勿跳过)
- [ ] generate_caseD.py:单排氧化物构型 + load_oxides 注入
- [ ] test_caseD.py:仿 test_caseA,读构型跑扫掠
- [ ] push → 超算/GPU 重编译 → 单排 BKS 验证(τ_DDD vs BKS 式,τ∝1/L 标度)
