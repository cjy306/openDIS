import os, sys
import numpy as np
import traceback

# Import pyexadis
pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
for path in pyexadis_paths:
    abspath = os.path.abspath(path)
    if abspath not in sys.path:
        sys.path.append(abspath)
np.set_printoptions(threshold=20, edgeitems=5)

try:
    import pyexadis
    # import core API from pyexadis_base and the DisNetManager implementation from framework
    from pyexadis_base import ExaDisNet, SimulateNetworkPerf, read_restart, NodeConstraints
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
    from framework.disnet_manager import DisNetManager
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e

pyexadis.initialize()

def insert_prismatic_loop(crystal, cell, nodes, segs, burg, radius, center, maxseg=-1, Rorient=None):
    """Insert a prismatic dislocation loop into the list of nodes and segments
    Input Burgers vector must be of the 1/2<111> type for bcc and 1/2<110> type for fcc.
    Arguments:
    cell: network cell
    nodes: list of nodes
    segs: list of segments
    burg: Burgers vector of the loop
    radius: radius of the loop
    center: center position of the loop
    maxseg: maximum discretization length
    Rorient: crystal orientation matrix
    """ 
    b = -1.0*burg
    
    if crystal in ['BCC', 'bcc']:
        b0 = 1.0/np.sqrt(3.0)*np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.]])#标准化的1/2<111>类型柏氏矢量
        bcol = np.abs(np.abs(np.dot(b0, b))-1.0)#计算输入柏氏矢量与标准柏氏矢量的夹角余弦值的差值
        ib = bcol.argmin()#找到差值最小的标准柏氏矢量索引
        if bcol[ib] > 1e-5:#如果最小差值仍然较大，说明输入柏氏矢量不符合要求
            raise ValueError('BCC Burgers vector must be of the 1/2<111> type in insert_prismatic_loop()')
        Nsides = 6#棱柱形环的边数
        if 1:
            # Loop with arms on {110} planes (default)#在{110}平面上生成位错环
            e = np.array([[-2.0*b[0],b[1],b[2]],[-b[0],-b[1],2.0*b[2]],
                          [b[0],-2.0*b[1],b[2]],[2.0*b[0],-b[1],-b[2]],
                          [b[0],b[1],-2.0*b[2]],[-b[0],2.0*b[1],-b[2]]])
        else:
            # Loop with arms on {112} planes
            e = np.array([[-b[0],0.0,b[2]],[0.0,-b[1],b[2]],
                          [b[0],-b[1],0.0],[b[0],0.0,-b[2]],
                          [0.0,b[1],-b[2]],[-b[0],b[1],0.0]])
        
        n = np.cross(b, e[(np.arange(6)+1)%6]-e[np.arange(6)])#计算每个边的法向量
        e = e / np.linalg.norm(e, axis=1)[:,None]
        
    elif crystal in ['FCC', 'fcc']:
        Nsides = 4
        b0 = 1.0/np.sqrt(2.0)*np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]])#标准化的1/2<110>类型柏氏矢量
        n01 = np.array([[-1,-1,1],[-1,1,1],[-1,1,1],[1,1,1],[-1,1,1],[1,1,1]])#对应的习惯面法向量
        n02 = np.array([[1,-1,1],[1,1,1],[-1,-1,1],[1,-1,1],[-1,1,-1],[1,1,-1]])#对应的第二习惯面法向量
        bcol = np.abs(np.abs(np.dot(b0, b))-1.0)#计算输入柏氏矢量与标准柏氏矢量的夹角余弦值的差值
        ib = bcol.argmin()#找到差值最小的标准柏氏矢量索引
        if bcol[ib] > 1e-5:
            raise ValueError('FCC Burgers vector must be of the 1/2<110> type in insert_prismatic_loop()')
        p1 = n01[ib] / np.linalg.norm(n01[ib])#标准化第一个习惯面法向量
        p2 = n02[ib] / np.linalg.norm(n02[ib])#标准化第二个习惯面法向量
        l1 = np.cross(p1, b)#计算第一个边的方向向量
        l1 = l1 / np.linalg.norm(l1)#标准化第一个边的方向向量
        l2 = np.cross(p2, b)#计算第二个边的方向向量
        l2 = l2 / np.linalg.norm(l2)#标准化第二个边的方向向量
        e = np.array([-0.5*l1-0.5*l2, +0.5*l1-0.5*l2, +0.5*l1+0.5*l2, -0.5*l1+0.5*l2])#计算四个边的方向向量
        n = np.array([p1, p2, p1, p2])#计算四个边的法向量
        
    else:
        raise ValueError('Error: unsupported crystal type = %s in insert_prismatic_loop()' % crystal)
    
    n = n / np.linalg.norm(n, axis=1)[:,None]
    if Rorient is not None:#如果提供了晶体取向矩阵，则对b、e、n进行旋转
        Rorient = np.array(Rorient)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:,None]
        b = np.matmul(b, Rorient.T)
        e = np.matmul(e, Rorient.T)
        n = np.matmul(n, Rorient.T)
    
    istart = len(nodes)#记录当前节点数作为新插入节点的起始索引
    Nnodes = 0#新插入节点的计数
    for i in range(Nsides):
        l = radius*(e[(i+1)%Nsides]-e[i])#计算当前边的长度向量
        Nseg = int(np.ceil(np.linalg.norm(l)/maxseg)) if maxseg > 0 else 1#计算当前边的离散化段数
        for j in range(Nseg):
            p = radius*e[i]+1.0*j/Nseg*l+center#计算当前节点位置
            nodes.append(np.concatenate((p, [NodeConstraints.UNCONSTRAINED])))#添加节点
            n1 = istart+Nnodes#计算当前节点索引
            n2 = istart if (i == Nsides-1 and j == Nseg-1) else n1+1#计算下一个节点索引，最后一个节点连接回起始节点
            segs.append(np.concatenate(([n1, n2], b, n[i])))#添加段
            Nnodes += 1#增加节点计数
            
    return nodes, segs

def generate_prismatic_config_by_density(crystal, Lbox_m, burgmag, target_density, 
                                       radius_min_m=0.2e-6, radius_max_m=1e-6, 
                                       seed=12345):
    """
    根据目标密度生成棱柱位错环配置
    """
    # 转换为无量纲单位
    Lbox = int(round(Lbox_m / burgmag))
    R_min = radius_min_m / burgmag
    R_max = radius_max_m / burgmag
    
    # 计算需要的总长度
    volume_m3 = Lbox_m**3
    total_line_length_m = target_density * volume_m3
    
    # 创建随机数生成器
    rng = np.random.RandomState(seed)
    
    # 逐步生成环直到满足长度要求
    cell = pyexadis.Cell(Lbox)
    nodes, segs = [], []
    accumulated_length_m = 0.0
    radii_m = []
    centers = []
    
    # Burgers 矢量集合
    if crystal.lower() == 'bcc':
        bset = np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.]])
    else:
        raise ValueError(f"Unsupported crystal: {crystal}")
    
    bset = bset / np.linalg.norm(bset, axis=1)[:, None]
    nburg = bset.shape[0]
    
    max_attempts = 10000
    attempt = 0
    loop_count = 0
    
    while accumulated_length_m < total_line_length_m * 0.99 and attempt < max_attempts:
        # 生成随机半径和位置
        radius_m = rng.uniform(radius_min_m, radius_max_m)
        radius = radius_m / burgmag
        
        # 确保环在盒子内（留出边界）
        margin = radius_m * 1.2
        low = margin
        high = Lbox_m - margin
        if high <= low:
            attempt += 1
            continue
            
        center_m = np.array([rng.uniform(low, high) for _ in range(3)])
        center = center_m / burgmag
        
        # 检查是否与已有环重叠
        overlap = False
        for i, existing_center in enumerate(centers):
            existing_radius = radii_m[i]
            distance = np.linalg.norm(center_m - existing_center)
            if distance < (radius_m + existing_radius) * 1.2:
                overlap = True
                break
                
        if overlap:
            attempt += 1
            continue
            
        # 选择合适的 Burgers 矢量
        burg = bset[loop_count % nburg]
        
        # 计算分段数
        circumference_m = 2 * np.pi * radius_m
        segs_per_loop = max(8, min(200, int(np.ceil(circumference_m / (burgmag * 15)))))
        
        try:
            # 插入环
            new_nodes, new_segs = insert_prismatic_loop(
                crystal.lower(), cell, nodes.copy(), segs.copy(), 
                burg, radius, center, segs_per_loop
            )
            
            # 计算新增长度
            added_length = circumference_m
            nodes, segs = new_nodes, new_segs
            radii_m.append(radius_m)
            centers.append(center_m)
            accumulated_length_m += added_length
            loop_count += 1
            
            if loop_count % 10 == 0:
                print(f"已插入 {loop_count} 个环，当前密度: {accumulated_length_m/volume_m3:.2e} /m²")
                
        except Exception as e:
            print(f"插入环失败: {e}")
            attempt += 1
            continue
            
        attempt = 0  # 重置尝试计数
    
    print(f"生成完成: {loop_count} 个环，总长度: {accumulated_length_m:.2e} m，密度: {accumulated_length_m/volume_m3:.2e} /m²")
    
    # 创建网络
    G = ExaDisNet(cell, nodes, segs)
    
    # 添加元数据
    G.material = {'name': 'W', 'b_m': burgmag, 'mu': 161e9, 'nu': 0.28}
    G.box = Lbox
    G.pbc = True
    G.generation_info = {
        'target_density': target_density,
        'achieved_density': accumulated_length_m/volume_m3,
        'num_loops': loop_count,
        'total_length_m': accumulated_length_m,
        'radius_range_m': [radius_min_m, radius_max_m]
    }
    
    return G

def run_simulation(net, output_dir='output_W_bcc1', rho_target=1e12, restart_id=None):
    state = {
        "crystal": 'bcc',
        "burgmag": 0.274e-9,
        "mu": 161e9,
        "nu": 0.28,
        "a": 3,
        "maxseg": 400, 
        "minseg": 80,  
        "rtol": 0.75,  
        "rann": 1.5,
        "nextdt": 1e-8,
        "maxdt": 1e-6,
        "write_freq": int(os.environ.get('WRITE_FREQ', '300')),  # 减少输出
        "max_collisions": int(os.environ.get('MAX_COLLISIONS', '100000')),
        "max_steps": int(os.environ.get('MAX_STEPS', '10000000')),
        "use_glide_planes": 1, 
        "num_bcc_plane_families": 2,
    }
    
    if restart_id is None:
        net_manager = DisNetManager(net)
        restart = None
    else:
        restart_filename = f'restart.{restart_id}.exadis'
        print(f"从 {restart_filename} 重启")
        net_manager, restart = read_restart(
            state=state, 
            restart_file=os.path.join(output_dir, restart_filename)
        )
    
    calforce = CalForce(
        force_mode='SUBCYCLING_MODEL', 
        state=state, 
        Ngrid=64,  
        cell=net_manager.cell
    )
    mobility = MobilityLaw(
        mobility_law='BCC_0B', 
        state=state,
        Medge=15000.0,  
        Mscrew=3000.0,    
        Mclimb=100.0,      
        vmax=30000.0     
    )

    timeint = TimeIntegration(
        integrator='Subcycling', 
        rgroups=[0.0, 50.0, 300.0, 800.0],  
        state=state, 
        force=calforce, 
        mobility=mobility
    )
    
    collision = Collision(collision_mode='Retroactive', state=state)
    topology = Topology(
        topology_mode='TopologyParallel', 
        state=state, 
        force=calforce, 
        mobility=mobility
    )
    remesh = Remesh(remesh_rule='LengthBased', state=state)
    
    vis = None
    
    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh, vis=vis,
        loading_mode='strain_rate', 
        erate=2e3, 
        edir=np.array([0., 0., 1.]),
        max_strain=0.01, 
        max_steps=state["max_steps"],
        burgmag=state["burgmag"], 
        state=state,
        print_freq=1, 
        plot_freq=1000,
        plot_pause_seconds=0.0,
        write_freq=100, 
        write_dir=output_dir, 
        restart=restart
    )
    
    sim.run(net_manager, state)
    pyexadis.finalize()

def main():
    """主程序 - 超算版本"""
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='运行位错网络模拟')
    parser.add_argument('--restart', type=int, help='从指定的重启文件继续运行')
    args = parser.parse_args()
    
    # 参数设置
    crystal = 'BCC'
    burgmag = 0.274e-9
    Lbox_m = 5e-6  # 5微米
    target_density = float(os.environ.get('RHO_TARGET', '1e12'))
    
    # 输出配置信息
    print("="*60)
    print("生成位错网络配置")
    print("="*60)
    # ...其他输出...
    
    try:
        output_dir = 'output_W_bcc'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if args.restart is not None:
            # 如果提供了restart参数，直接从重启文件继续
            print(f"\n从重启点 {args.restart} 继续模拟...")
            # 创建一个空的网络对象(不会使用)
            G = ExaDisNet()
            run_simulation(G, output_dir=output_dir, rho_target=target_density, restart_id=args.restart)
        else:
            # 否则从头开始
            print("\n正在生成位错网络...")
            G = generate_prismatic_config_by_density(
                crystal=crystal,
                Lbox_m=Lbox_m,
                burgmag=burgmag,
                target_density=target_density,
                radius_min_m=0.2e-6,  
                seed=12345
            )
            
            # 显示生成信息...
            
            # 保存网络...
            
            # 运行模拟
            print("\n超算环境检测到,自动开始模拟运行...")
            run_simulation(G, output_dir=output_dir, rho_target=target_density)
            
    except Exception as e:
        print(f"程序执行失败: {e}")
        traceback.print_exc()
        try:
            pyexadis.finalize()
        except:
            pass


if __name__ == "__main__":
    main()