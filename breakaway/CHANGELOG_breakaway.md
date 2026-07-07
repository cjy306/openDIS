# 改动记录:切过(breakaway)机制 C++ 实现

> 记录为实现"小夹杂切过"(Foreman-Makin 点障碍)对 ExaDiS 核心 C++ 的每一处改动,
> 以及设计演化历程。机制:位错段扫过障碍→插钉扎节点(冻结);两臂弓到破断角→释放。
> 与已有 Orowan 绕过互补(小→切过,大→Orowan)。开始:2026-07。

⚠️ **所有 C++ 改动改完需重编译 ExaDiS 才能测**(linux .so,本地不可跑;push→超算编译)。
⚠️ 所有改动带 `obstacles.empty()` 保护 → 无障碍时零开销、对现有仿真零影响。
⚠️ 触发条件:测试脚本 `collision_mode='Orowan'` 才进含 handle_breakaway 的 CollisionOrowan;'Retroactive' 不进。

---

## 改动清单

### [1] core/exadis/src/system.h — 障碍数据结构 + 强度字段 ✅
- `struct SphericalObstacle` 增 `double phi_crit;`(破断临界角,弧度)。
- `load_obstacles(centers_b, radii_b, phi_crit=M_PI/2)` — 默认 90°;逐障碍初始化 phi_crit。
- 位置:仿已有 load_obstacles 范式。

### [2] core/exadis/src/collision_types/collision_orowan.h — 机制主体 ✅
- 新增 `static bool point_in_tri(a,b,c,p,n)` — 面内点在三角形内判断(叉积同号)。
- 新增 `void handle_breakaway(System*)`(串行网络,split_seg 只能串行):
  - **Phase A 捕获**:反推步初位 o=pos−dt·v,拼扫掠四边形 [o1,o2,p2,p1];
    垂直门 h²<R²(面切 R 球);投影 Cproj=C−h·ng(切圆圆心);
    point_in_tri 判 Cproj 在扫掠面积内 → 命中。命中后解接触时刻 τ、端点回退、
    split_seg(s,Cproj) 插 PINNED 节点、记 sphere_id=j。仅端点护栏防同段相邻臂重钉。
  - **Phase B 释放**:R=|Σ 单位臂切向| > 2cos(φc/2) → 解钉(constraint→0, sphere_id→−1)。
- 在 `handle()` 里接线:CollisionRetroactive::handle() → **handle_breakaway** → handle_orowan → handle_twin_wall。

### [3] breakaway/ Python 侧 ✅
- `generate_breakaway_line.py` — 单根刃型直线 + 锚定障碍(沿 ±b̂ 偏);含 OBS_OFFPLANE_B(验证投影用)、盒内筛选。
- `generate_breakaway_field.py` — 单线 + 滑移面内随机撒一片点障碍(h=0、盒内、等强度);打印平均间距 L。
- `test_breakaway_prismatic.py` — 读构型+障碍、load_obstacles、SimulateNetworkPerf 跑;collision_mode='Orowan';--restart、max_step。
- `paraview.py` — *.data→VTK + obstacles.data→obstacles.vtk(带 VERTICES 才渲染得出点)。

---

## 设计演化历程 / 踩坑记录

### 判据选择:角判据(几何)vs 力判据
- 力判据要全节点力,但 Subcycling 只留**部分**力(pre_compute 后各组力会清零),要额外强制全力重算(贵);且 F_max 实验测不出、力判据不带新可测信息。→ **选几何角判据**。取向:以几何修改为主,避开力模块。

### 五个初版短板 → 处置
1. **抖动(脱钉后立即重捕)** → 采用**小 R**(不搞按步冷却,冷却不物理)。
2. **钉点离面** → 钉在 **Cproj**(球心投影到滑移面,=切圆圆心),保证在面内、不破坏滑移。
3. **快线漏检**(接近判据抓不住一步跨过)→ 换**扫掠面积检测**(o=pos−dt·v 拼四边形)。
4. 线张力各向异性 → 暂等张力近似(deferred)。
5. 角判据网格依赖 → deferred(后在障碍林里坐实为真问题,见下)。

### 点障碍 vs 小球(定案)
纯切过 = **点障碍**(Foreman-Makin):位错从粒子穿过、剪断,不碰表面,阻力点状,球面几何多余(球留给 Orowan)。故"接触点 vs 圆心投影"对点模型不成立;有限尺寸办法(入口接触/随切前移/力场软接触)超出点模型、搁置。Cproj=切圆圆心,r_p 的物理只经 φc(r_p)+面上点密度进,不进几何。

### τ 回退(办法1,消过冲尖)
旧:命中后就地在 Cproj 插钉,端点留在前进位置 → 过冲尖(回退相)。
新:解段随步线性运动碰到 Cproj 的接触时刻 τ(f(τ)=dot(cross(b−a,Cproj−a),ng)=0,τ 二次式取 [0,1] 根),端点 `pos−=(1−τ)·dt·v` 退回接触构型再插钉。检测、钉点(Cproj)不变,只多端点回退;解不出则 τ=1(退化为旧行为,不会更差)。纯几何、不碰力。

### 验证历程
- **单线**:钉→弓→断→恢复(φc=90°)、无抖动 ✓。
- **投影**:用离面障碍(OBS_OFFPLANE_B=20b<R=40b)边视角确认钉点落面内、障碍在面外 ✓。共面障碍(h=0)证不了投影(Cproj≡C)。
- **障碍林(field)**:集体钉→弓→逐个突破跑通;障碍严格 h=0 是接触钉、无投影(曾误读为投影)✓。
- **不脱钉疑点排除**:1e3/5000 步不脱钉,是步数不够、应力没到;1e3/10000 步 ~9000 步脱钉。机制没问题。
- **网格依赖坐实(硬约束)**:障碍林释放时线弓成"细针"(远过 φc=90° 才放),慢速 1e3 也出针 → maxseg=200b 太粗,角判据用远邻点算夹角把真尖平滑、释放偏晚 → **量 τ_c 会偏高**。看现象无妨(自愈),但**进入 Δτ/τ_c 定量前必须先细网格(如 maxseg=50)让释放回 ~90°**。
- **#5 决定:暂不改(2026-07)**。走"**标定 φc 到实验 Δτ**"这条路时,网格偏差会被**吸进 φc**、模型自洽,不必修。前提:(1) **全程网格固定**(所有 run 同一 maxseg),否则标定失效;(2) 偏差最好随 φc/r_p 大致恒定(将来花一次 maxseg=50 spot-check 验一下)。**但若改走"第一性算 F_max → 预测 τ_c → 对实验"那条路,#5 必须回来收**(细化/收敛)。同理 **#4 线张力各向异性**也暂缓(二阶精度,且加各臂 T 权重会破坏角判据"无量纲、不依赖 T"的优点)。
  - 修法备忘(将来要修时):**"定长 L_ref 测切向"没用**(离散线段内直,钉点→段内任一点方向不变)。真·病根是钉点第一段太长,唯一解是**钉点附近加分辨率**:全局细 maxseg(快、零代码,单线/小林够用)或钉点局部细化(省算力,bulk 阶段才值得写)。
- **max-conn 警告**:erate=1e4 下出现;疑因高速一步大量捕获+碰撞+拓扑 churn,或针状释放猛弹触发自交(已排除同点堆钉:红针经点节点确认只有一个 constraint=7)。1e3 无此问题;先用 1e3,想上高速先调小 maxdt。

---

## 尚未做(deferred)
φc(r_p) 尺寸-强度 + 面上点密度;Orowan/breakaway 尺寸分流(需球);角判据网格依赖量化;线张力各向异性;bulk 3D(与滑移面无关的扫掠版、段跨面处理);多线共障碍合并(C1,疑与 max-conn 相关)。
