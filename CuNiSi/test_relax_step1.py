"""验证实验：纯弛豫(零外力)能否消除 FR 源的屈服骤降
   —— 第 1 段：零外力弛豫

机制：loading_mode='stress' 且 applied_stress=0 → 不加载(strain恒为0)，
位错网络只在【内应力 + 线张力】下自发演化(弛豫)。

预期(我的判断，待验证)：
  FR 源在零外力下不会放出可动位错(开动需 τ>μb/L，零外力达不到) →
  弛豫后可动密度仍≈0 → 骤降不会被消除。
  本段先把弛豫后的构型存下来(config)，交给第 2 段加载验证。

用法：读原始 FR 初始构型 init_data_twin_rorient/，弛豫 N 步，
输出到 output_relax/，取末态 config 当第 2 段的 init。
"""
import os, sys
import numpy as np

pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e


RELAX_STEPS = 2000   # 弛豫步数（够让网络达到力平衡；可按需调）

Rorient = [
    [ 1/np.sqrt(2), -1/np.sqrt(2),  0           ],
    [ 1/np.sqrt(6),  1/np.sqrt(6), -2/np.sqrt(6)],
    [ 1/np.sqrt(3),  1/np.sqrt(3),  1/np.sqrt(3)],
]

state = {
    "crystal": 'fcc',
    "Rorient": Rorient,
    "burgmag": 0.2556e-9,
    "mu":      48e9,
    "nu":      0.324,
    "a":       4.0,
    "maxseg":  200,
    "minseg":  50,
    "rtol":    1.0,
    "rann":    2.0,
    "nextdt":  1e-9,
    "maxdt":   1e-8,
}


def main():
    pyexadis.initialize()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    init_dir   = os.path.join(base_dir, 'init_data_twin_rorient')   # 原始 FR 构型
    output_dir = os.path.join(base_dir, 'output_relax')
    os.makedirs(output_dir, exist_ok=True)

    obs_data = np.loadtxt(os.path.join(init_dir, 'obstacles.data'))
    if obs_data.ndim == 1:
        obs_data = obs_data.reshape(1, -1)
    centers_b, radii_b = obs_data[:, :3], obs_data[:, 3]

    tp_data = np.loadtxt(os.path.join(init_dir, 'twin_planes.data'))
    if tp_data.ndim == 1:
        tp_data = tp_data.reshape(1, -1)
    twin_points_b  = tp_data[:, :3].tolist()
    twin_normals_b = tp_data[:, 3:].tolist()

    G = ExaDisNet()
    G.read_paradis(os.path.join(init_dir, 'init_config.data'))
    net = DisNetManager(G)
    exadis_net = net.get_disnet(ExaDisNet)
    if len(centers_b) > 0:
        exadis_net.load_obstacles([list(c) for c in centers_b], list(radii_b))
    if len(twin_points_b) > 0:
        exadis_net.load_twin_planes(twin_points_b, twin_normals_b)

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=exadis_net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=50.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 10.0, 60.0, 200.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Orowan', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    # ★ 关键：零外力弛豫 = loading_mode='stress' + applied_stress=0
    #   网络只在内应力/线张力下演化，strain 恒为 0，按步数停止
    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        loading_mode='stress',
        applied_stress=np.zeros(6),
        num_steps=RELAX_STEPS,
        burgmag=state["burgmag"],
        state=state,
        print_freq=1,
        write_freq=100,
        write_dir=output_dir,
    )
    sim.run(net, state)
    pyexadis.finalize()
    print(f"\n弛豫完成。取 {output_dir}/ 里最后一个 config.*.data 作为第2段加载的 init。")


if __name__ == "__main__":
    main()
