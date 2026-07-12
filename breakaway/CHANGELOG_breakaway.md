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

### [5] 扫掠重建升级:步初位置改用 xold 记录(替代 pos−dt·v 反推)✅ 待编译验证
- 动机:旧反推隐含"步内匀速直线"假设;凡在积分后、检测前挪过节点的工序都会污染重建
  (retroactive 碰撞挪点、交滑移挪点;另 subcycling 步末 v 是否=步平均未查证,若否则反推每步轻微失真)。
- 改法(collision_orowan.h handle_breakaway):照抄 retroactive_collision 的 xold 访问范式
  (UNIFIED_MEMORY 宏分支 + HostMirror);o = pbc_position(p, xold(n));τ 回退改
  `pos −= (1−τ)·(p−o)`,v 从公式消失。当步新生节点(索引 ≥ xold 长度)退回速度反推(bounds 守卫)。
- 覆盖:位移类污染全部;**不覆盖交滑移的 plane 字段错位**(方案 B=换面段当步打标豁免,开交滑移时另合;
  时序调整方案 C 经审查不推荐:动全局流水线+交滑移挪点可能逃检)。
- 2D 投影与垂直门逻辑不变——本升级只改"步初位置从哪来"。
- 预期:现有单线 case 若 v 恰为步平均则行为逐位不变;有差异=旧反推本有误差的证据。
- 验证脚本:`test_xold_sweep.py`(run + verify 两模式,判据 V1-V4)。

### [4] 障碍分型:切过/Orowan 互斥(正确性修复)✅ 待编译验证
- 起因:释放节点被 handle_orowan 硬球以任意方向瞬移 R(见"尖角一步愈合追查")。
- `system.h`:`enum ObstacleType {OBSTACLE_OROWAN=0, OBSTACLE_BREAKAWAY=1}`;
  `SphericalObstacle` 增 `int type;`;`load_obstacles(..., type=OBSTACLE_OROWAN)` 默认 0 保旧行为。
- `collision_orowan.h`:handle_orowan 跳过 type==BREAKAWAY(点障碍无球面几何);
  handle_breakaway Phase A 只捕获 type==BREAKAWAY(Orowan 球不插钉)。
- pybind 两层穿透 `type` 参数(默认 0);`test_breakaway_prismatic.py` 改为 `type=1` 加载。
- **踩坑:调用链实为三层**(脚本 → pyexadis_base.ExaDisNet(纯Python)→ pybind → System),首改漏了第一层 → 超算报 `unexpected keyword argument 'type'`(纯 Python 错误格式,pybind 层报错是 "incompatible function arguments"——可用报错格式区分卡在哪层)。已补 pyexadis_base.py:load_obstacles 加 type=0 透传。.py 文件改动不需重编译,pull 即生效。
- 未动:handle_twin_wall 的球邻近检查仍看全体障碍(超出本修复范围,记录待议);φc 数组、#4 加权本轮不带。
- 预期重跑差异:释放帧不再有 R=40b 瞬移;钉扎期两臂不再被 1.025R 壳外推;**抖动可能回归**(瞬移此前可能客串防重捕垫片,见追查条目)。

### [3] breakaway/ Python 侧 ✅
- `generate_breakaway_line.py` — 单根刃型直线 + 锚定障碍(沿 ±b̂ 偏);含 OBS_OFFPLANE_B(验证投影用)、盒内筛选。
- `generate_breakaway_field.py` — 单线 + 滑移面内随机撒一片点障碍(h=0、盒内、等强度);打印平均间距 L。
- `test_breakaway_prismatic.py` — 读构型+障碍、load_obstacles、SimulateNetworkPerf 跑;collision_mode='Orowan';--restart、max_step。
- `paraview.py` — *.data→VTK + obstacles.data→obstacles.vtk(带 VERTICES 才渲染得出点)。

---

## 设计演化历程 / 踩坑记录

### 判据选择:角判据(几何)vs 力判据
- 力判据要全节点力,但 Subcycling 只留**部分**力(pre_compute 后各组力会清零),要额外强制全力重算(贵);且 F_max 实验测不出、力判据不带新可测信息。→ **选几何角判据**。取向:以几何修改为主,避开力模块。
- **复审(2026-07,已设计、暂缓未实施)**:"贵"的否决理由失效——`Force::node_force(system,i)` 单节点全力接口现成(ForceSubcycling 逐组求和,拓扑 split 试探同款,钉扎节点逐个调廉价);且角判据加权撞上"T 一词多义"难题(见 #4 重审警告),力判据天然无此问题(E/E'/扭矩/两臂作用全在代码力里)。两阶段迁移方案已评审:阶段1=接线 Force* 进 CollisionOrowan + Phase B 诊断打印 |F_glide|(零行为改变,采 phi<->F 词典);阶段2=判据切 |F_glide|>F_max_j,#4 作废、φc(r)改F_max(r)。已知妥协:flong 一步陈旧(与拓扑同级)、当步新钉节点 fsegseg 配对缺失、PK 份额网格依赖(力判据版#5,更轻)。**用户决定先放(2026-07-10),实施前需再拍板。**

### 五个初版短板 → 处置
1. **抖动(脱钉后立即重捕)** → 采用**小 R**(不搞按步冷却,冷却不物理)。
2. **钉点离面** → 钉在 **Cproj**(球心投影到滑移面,=切圆圆心),保证在面内、不破坏滑移。
3. **快线漏检**(接近判据抓不住一步跨过)→ 换**扫掠面积检测**(o=pos−dt·v 拼四边形)。
4. 线张力各向异性 → 暂等张力近似(deferred);加权方案已定见下节"#4 线张力各向异性加权",未合入。
5. 角判据网格依赖 → deferred(后在障碍林里坐实为真问题,见下)。

### 点障碍 vs 小球(定案)
纯切过 = **点障碍**(Foreman-Makin):位错从粒子穿过、剪断,不碰表面,阻力点状,球面几何多余(球留给 Orowan)。故"接触点 vs 圆心投影"对点模型不成立;有限尺寸办法(入口接触/随切前移/力场软接触)超出点模型、搁置。Cproj=切圆圆心,r_p 的物理只经 φc(r_p)+面上点密度进,不进几何。

### τ 回退(办法1,消过冲尖)
旧:命中后就地在 Cproj 插钉,端点留在前进位置 → 过冲尖(回退相)。
新:解段随步线性运动碰到 Cproj 的接触时刻 τ(f(τ)=dot(cross(b−a,Cproj−a),ng)=0,τ 二次式取 [0,1] 根),端点 `pos−=(1−τ)·dt·v` 退回接触构型再插钉。检测、钉点(Cproj)不变,只多端点回退;解不出则 τ=1(退化为旧行为,不会更差)。纯几何、不碰力。

### #4 线张力各向异性加权(方案,**未合入**)
问题:Phase B 的 R=|Σ t̂| 假设两臂等线张力 T;真实 T 随位错性质变(螺型拉得狠)。
方案:每臂按 **g=T(θ)/T_edge** 加权,判据变 **R=|Σ g·t̂| > 2cos(φc/2)**(阈值不变)。
- g = (1 − 2ν + 3ν·cos²θ)/(1 − 2ν),cos²θ = (t̂·b̂)²;螺型 g=(1+ν)/(1−2ν)、刃型 g=1。ν=0.29 → 螺/刃比 **3.07**(非 3.25,那是 ν=0.3)。
- **关键:比值形式,T_ref 与 log 截断都约掉 → 仍无量纲、不依赖绝对 T**(不破坏角判据优点。早先"会破坏无量纲"的说法是误判,已撤)。
- 接口:ν 从 `system->params.NU` 读(=state["nu"]);b 从 `segs[conn[i].seg[k]].burg` **逐臂**读。归一到刃型 → φc 定义为"刃型参考破断角"。
- 状态:**已规划、曾试改被否、未合入**;当前 Phase B 仍无加权(=g≡1 等张力)。
- 暂缓真理由:**二阶影响(仅非对称弓有差)+ 难验**(单根对称线 g 相等、看不出;要重编译 + 造含螺型段的非对称构型 + printf 打印每臂 g 才验得了)。**不是**"破坏无量纲"。
- 局限:cosθ 仍从第一段方向算 → 同样吃 #5 网格粗细;与 #5 独立、可分开加。
- ⚠️ **重审警告(2026-07,"T 一词多义"排查)**:本节 g 用的是**刚度** T=E+E''(螺/刃 3.07),但脱钉判据是**节点力平衡**,严格权重是能量+扭矩形式 F=E(θ)t̂+E'(θ)p̂(Herring/节点平衡;E 比值螺/刃=1−ν≈0.71,**与 3.07 方向相反**)。E+E'' 回答弓弯形状/刚度问题(DDD 动力学自己解,不需判据代劳)。core_force 的 fL(∝E)+ft(∝E' 扭矩)结构佐证节点力用 E,E'。代数优点(无量纲、log 约掉、逐臂可算)两版共享。**合入前必须重推导定案,3.07 版不可直接合。**

### 2D 扫掠适用范围(定案 2026-07):精确性条件是逐段的,bulk 3D 改触发式
2D 扫掠的精确条件**不是**"全局单滑移面",而是**逐段**:检测用 `segs[s].plane`(各段各自的 ng),
只要该段运动在自己面内,扫掠四边形严格共面、检测精确。→ 多滑移系 bulk 构型只要仍是纯滑移
运动(mobility 投影 + enforce_glide_planes),2D 版照样精确,"多面"本身不是破绽。
真正打破"本质 2D"的只有三件事(任一进入构型才需要 bulk 3D 扫掠版):
1. **交滑移开启**:段换面那一步,位移按旧面挪、扫掠按新面算,单步错判;
2. **非滑移位移分量**:爬移迁移率,或碰撞/重网格把节点挪出面(enforce_glide_planes 兜底则是小量,被 h<R 门宽容);
3. **junction 段**:ng≈0 被直接跳过(continue),junction 臂扫过障碍不可见;单线阶段无此事。
备忘(将来要做时):3D 版核心数学 = 点对动段的时空最短距离,ExaDiS 已有现成原语
`MinDistPtSegInTime`(collision_retroactive.cpp,KOKKOS 可 device);设计点:捕获语义
(2D 是"跨越判据",3D 距离<R 是"接近判据",会多抓擦边,阈值语义要重定)、钉点落点、
bulk 规模 Nseg×Nobs 串行全配对的 cell 剪枝。

### 验证历程
- **单线**:钉→弓→断→恢复(φc=90°)、无抖动 ✓。
- **投影**:用离面障碍(OBS_OFFPLANE_B=20b<R=40b)边视角确认钉点落面内、障碍在面外 ✓。共面障碍(h=0)证不了投影(Cproj≡C)。
- **障碍林(field)**:集体钉→弓→逐个突破跑通;障碍严格 h=0 是接触钉、无投影(曾误读为投影)✓。
- **不脱钉疑点排除**:1e3/5000 步不脱钉,是步数不够、应力没到;1e3/10000 步 ~9000 步脱钉。机制没问题。
- **脱钉后尖角"一步愈合"追查(2026-07,导师质疑坐实)**:逐帧追同一节点(NodeTag 不变)查坐标:释放帧位移 Δ=(0,+28.284,+28.284)、|Δ|=R=40b,与 handle_orowan"节点在球心投影 → 任意面内方向"分支(d_norm<1e-10 → cross(n,x̂),n=(0,-1,1)/√2 → d_dir=(0,1,1)/√2)**逐位吻合**。真相:释放(Phase B,4b)后**同一步**的 handle_orowan(4c)对刚释放、坐在球心(h=0)的节点失去 PINNED 豁免,按硬球投影瞬移 R 到交圆——**该帧尖顶位置是非物理瞬移,不是弛豫**;真弛豫在下一步(~155b 回线)。此前"纯物理弛豫、无伪影"的记录**作废**,本条替代。
  - **根因:切过点障碍与 Orowan 硬球共用同一障碍表,语义冲突**(定案明说"球留给 Orowan")。派生影响:①每次 h=0 脱钉都有 R 量级任意向瞬移且不记塑性应变;②钉扎期两臂非钉节点被 1.025R 壳每步外推,近钉弓弯被不该存在的硬盘扭曲(污染角判据定量)。
  - **处置:尺寸分流升级为正确性修复**——SphericalObstacle 加类型字段(切过/Orowan),handle_orowan 跳过切过型、handle_breakaway 跳过 Orowan 型。**已实现,见改动 [4],待编译验证**(逐障碍 φc 扩展按用户决定本轮未带)。
  - 判别法备忘:分辨节点"弹回去/被删/被投影"只能追 NodeTag+坐标,渲染图分不出;位移长度恰等于某特征尺度(R、maxseg…)= 几何操作的指纹。
- **网格依赖坐实(硬约束)**:障碍林释放时线弓成"细针"(远过 φc=90° 才放),慢速 1e3 也出针 → maxseg=200b 太粗,角判据用远邻点算夹角把真尖平滑、释放偏晚 → **量 τ_c 会偏高**。看现象无妨(自愈),但**进入 Δτ/τ_c 定量前必须先细网格(如 maxseg=50)让释放回 ~90°**。
  - **首个定量点(2026-07,分型修复后 run,step 3807)**:depin 打印 R=1.601、φ=73.7°(crit=90°)→ 单事件滞后 ~16°;该节点 NodeDegree=2(排除多臂稀释)。R 一步从 ≤1.414 跳 1.601 → 滞后=时间离散+空间离散混合(当步还有 captured=1 的 split 可能改了邻居)。**测量法备忘:grep "depin" slurm日志 → 全 run 释放角分布,零成本量化 #5 偏差**;maxseg=50 重跑同 case 应收回 ~90°(=欠的 spot-check)。
  - **同 run"直角弯折"疑点结案**:度=2、无障碍点、无碰撞/MAX_CONN 输出,5868→5877 约 10 帧自愈 → 释放/捕获后的弛豫瞬态,非结构性。浅长波愈合(~10帧)慢于脱钉尖角(1步),符合回复力∝曲率,互为旁证。分型修复 [4] 经 5866→5868 释放事件验证:无 R 瞬移。
- **#5 决定:暂不改(2026-07)**。走"**标定 φc 到实验 Δτ**"这条路时,网格偏差会被**吸进 φc**、模型自洽,不必修。前提:(1) **全程网格固定**(所有 run 同一 maxseg),否则标定失效;(2) 偏差最好随 φc/r_p 大致恒定(将来花一次 maxseg=50 spot-check 验一下)。**但若改走"第一性算 F_max → 预测 τ_c → 对实验"那条路,#5 必须回来收**(细化/收敛)。同理 **#4 线张力各向异性**也暂缓,但理由是**二阶影响 + 难验**(仅非对称弓有差),**不是**"破坏无量纲"(比值权重仍无量纲,方案见上节"#4 线张力各向异性加权")。
  - 修法备忘(将来要修时):**"定长 L_ref 测切向"没用**(离散线段内直,钉点→段内任一点方向不变)。真·病根是钉点第一段太长,唯一解是**钉点附近加分辨率**:全局细 maxseg(快、零代码,单线/小林够用)或钉点局部细化(省算力,bulk 阶段才值得写)。
- **max-conn 警告**:erate=1e4 下出现;疑因高速一步大量捕获+碰撞+拓扑 churn,或针状释放猛弹触发自交(已排除同点堆钉:红针经点节点确认只有一个 constraint=7)。1e3 无此问题;先用 1e3,想上高速先调小 maxdt。

---

## 尚未做(deferred)
φc(r_p) 尺寸-强度 + 面上点密度;Orowan/breakaway 尺寸分流(需球);角判据网格依赖量化;线张力各向异性;bulk 3D 扫掠版(**触发式**:交滑移/非滑移位移/junction 段三条件任一出现才做,见上节"2D 扫掠适用范围");多线共障碍合并(C1,疑与 max-conn 相关)。
