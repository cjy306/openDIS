"""@package docstring
Sim_DisNet: class for simulating dislocation network

Provide simulation functions based on other utlitity classes
"""

import numpy as np
import os, pickle
from ..disnet import DisNet
from ..calforce.calforce_disnet import CalForce
from ..mobility.mobility_disnet import MobilityLaw
from ..timeint.timeint_disnet import TimeIntegration
from ..visualize.vis_disnet import VisualizeNetwork
from framework.disnet_manager import DisNetManager

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except ImportError:
    print('-----------------------------------------')
    print(' cannot import matplotlib or mpl_toolkits')
    print('-----------------------------------------')

#class SimulateNetwork(SimulateNetwork_Base):
class SimulateNetwork:
    """SimulateNetwork: class for simulating dislocation network

    """
    def __init__(self, state: dict, calforce=None,
                 mobility=None, timeint=None, topology=None,
                 collision=None, remesh=None, cross_slip=None, vis=None,
                 dt0: float=1.0e-8,
                 max_step: int=10,
                 loading_mode: str=None,
                 applied_stress: np.ndarray=np.zeros(6),
                 # 新增应变速率相关参数
                 erate: float=0.0,
                 strain_direction: np.ndarray=np.array([0.0, 0.0, 1.0]),
                 max_strain: float=0.01,
                 print_freq: int=None,
                 plot_freq: int=None,
                 plot_pause_seconds: float=None,
                 write_freq: int=None,
                 write_dir: str=".",
                 save_state: bool=False,
                 **kwargs) -> None:
        self.calforce = calforce
        self.mobility = mobility
        self.timeint = timeint
        self.topology = topology
        self.collision = collision
        self.remesh = remesh
        self.cross_slip = cross_slip
        self.vis = vis
        self.dt0 = dt0
        self.max_step = max_step
        self.loading_mode = loading_mode
        self.applied_stress = np.array(applied_stress)
        
        # 新增应变速率相关属性
        self.erate = erate  # 应变速率 (s⁻¹)
        self.strain_direction = np.array(strain_direction)  # 应变方向
        self.max_strain = max_strain  # 最大应变
        self.current_strain = 0.0  # 当前应变
        self.initial_cell = None  # 初始单元尺寸
        
        self.print_freq = print_freq
        self.plot_freq = plot_freq
        self.plot_pause_seconds = plot_pause_seconds
        self.write_freq = write_freq
        self.write_dir = write_dir
        self.save_state = save_state

        # 根据加载模式初始化状态
        if self.loading_mode == 'stress':
            state["applied_stress"] = np.array(applied_stress)
        elif self.loading_mode == 'strain_rate':
            state["erate"] = self.erate
            state["strain"] = self.current_strain
            state["applied_stress"] = np.zeros(6)  # 初始应力为零
        elif self.loading_mode is None:
            state["applied_stress"] = np.zeros(6)
        else:
            raise ValueError(f"Invalid loading_mode: {loading_mode}. "
                           f"Supported modes: 'stress', 'strain_rate', or None")

    def step_begin(self, DM: DisNetManager, state: dict):
        """step_begin: invoked at the begining of each time step
        """
        pass

    def step_integrate(self, DM: DisNetManager, state: dict):
        """step_integrate: invoked for time-integration at each time step
        """
        #self.save_old_nodes(DM, state)
        state = self.calforce.NodeForce(DM, state)
        state = self.mobility.Mobility(DM, state)
        state = self.timeint.Update(DM, state)
        #self.plastic_strain(DM, state)

    def step_post_integrate(self, DM: DisNetManager, state: dict):
        """step_post_integrate: invoked after time-integration of each time step
        """
        pass

    def step_topological_operations(self, DM: DisNetManager, state: dict):
        """step_topological_operations: invoked for handling topological events at each time step
        """
        if self.cross_slip is not None:
            self.cross_slip.Handle(DM, state)

        # The order of topology vs collision is opposite to ExaDiS
        if self.topology is not None:
            self.topology.Handle(DM, state)

        if self.collision is not None:
            self.collision.HandleCol(DM, state)

        if self.remesh is not None:
            self.remesh.Remesh(DM, state)

    def step_update_response(self, DM: DisNetManager, state: dict):
        """step_update_response: update applied stress and rotation if needed
        """
        if self.loading_mode == 'stress':
            # 现有的应力加载逻辑
            # 这里可以根据需要添加应力更新逻辑
            pass
            
        elif self.loading_mode == 'strain_rate':
            # 新增的应变速率加载逻辑
            istep = state.get('istep', 0)
            
            # 第一步：保存初始单元尺寸
            if istep == 0:
                self.initial_cell = DM.cell.h.copy()
                if self.print_freq is not None:
                    print(f"Strain rate loading initialized:")
                    print(f"  Strain rate: {self.erate:.2e} s⁻¹")
                    print(f"  Maximum strain: {self.max_strain}")
            
            # 更新应变
            if self.erate > 0 and self.current_strain < self.max_strain:
                # 计算应变增量
                strain_increment = self.erate * self.timeint.dt
                self.current_strain += strain_increment
                
                if self.current_strain > self.max_strain:
                    self.current_strain = self.max_strain
                    strain_increment = self.max_strain - (self.current_strain - strain_increment)
                
                # 更新单元尺寸 (仿射变形)
                if self.initial_cell is not None:
                    # 计算变形梯度 (假设小变形)
                    deformation_gradient = np.eye(3)
                    for i in range(3):
                        deformation_gradient[i, i] = 1.0 + strain_increment * self.strain_direction[i]
                    
                    # 更新单元矩阵
                    DM.cell.h = np.dot(deformation_gradient, self.initial_cell)
                    
                    # 更新位错节点位置 (仿射变换)
                    G = DM.get_disnet()
                    if hasattr(G, 'rn') and G.rn is not None:
                        # 对每个节点应用仿射变换
                        for i in range(G.rn.shape[0]):
                            old_pos = G.rn[i, :3].copy()
                            new_pos = np.dot(deformation_gradient, old_pos)
                            G.rn[i, :3] = new_pos
                
                # 更新状态
                state["strain"] = self.current_strain
                state["erate"] = self.erate
                
                # 可选：根据应变计算应力 (使用胡克定律的简化版本)
                # 这对于某些材料模型可能是有用的
                mu = state.get('mu', 161e9)  # 剪切模量
                nu = state.get('nu', 0.28)   # 泊松比
                
                # 计算体积应变
                volumetric_strain = strain_increment * np.sum(self.strain_direction)
                
                # 简化的应力更新 (各向同性线性弹性)
                # 注意：这是一个简化模型，实际应用可能需要更复杂的本构关系
                youngs_modulus = 2 * mu * (1 + nu)
                
                # 更新应力张量 (Voigt记号: [11, 22, 33, 23, 13, 12])
                for i in range(3):
                    state["applied_stress"][i] += youngs_modulus * strain_increment * self.strain_direction[i]
                
                if self.print_freq is not None and istep % self.print_freq == 0:
                    print(f"Step {istep}: Strain = {self.current_strain:.4e}, "
                          f"Stress = {state['applied_stress'][:3]}")
            
            elif self.current_strain >= self.max_strain:
                # 达到最大应变，停止加载
                if self.print_freq is not None and istep % self.print_freq == 0:
                    print(f"Reached maximum strain: {self.max_strain:.4f}")
        
        elif self.loading_mode is None:
            # 无加载模式
            pass
        
        else:
            raise ValueError(f"invalid loading_mode in PyDiS SimulateNetwork. "
                           f"Supported modes: 'stress', 'strain_rate', or None. "
                           f"Got: '{self.loading_mode}'")
        
        return state

    def step_write_files(self, DM: DisNetManager, state: dict):
        if self.write_freq != None:
            istep = state['istep']
            if istep % self.write_freq == 0:
                DM.write_json(os.path.join(self.write_dir, f'disnet_{istep}.json'))
                if self.save_state:
                    with open(os.path.join(self.write_dir, f'state_{istep}.pickle'), 'wb') as file:
                        pickle.dump(state, file)

    def step_print_info(self, DM: DisNetManager, state: dict):
        if self.print_freq != None:
            istep = state['istep']
            if istep % self.print_freq == 0:
                if self.loading_mode == 'strain_rate':
                    print("step = %d dt = %e strain = %e" % 
                          (istep, self.timeint.dt, self.current_strain))
                else:
                    print("step = %d dt = %e" % (istep, self.timeint.dt))

    def step_visualize(self, DM: DisNetManager, state: dict):
        if self.vis != None and self.plot_freq != None:
            istep = state['istep']
            if istep % self.plot_freq == 0:
                self.vis.plot_disnet(DM, fig=self.fig, ax=self.ax, 
                                     trim=True, block=False, 
                                     pause_seconds=self.plot_pause_seconds)

    def step_end(self, DM: DisNetManager, state: dict):
        """step_end: invoked at the end of each time step
        """
        pass

    def step(self, DM: DisNetManager, state: dict):
        """step: take a time step of DD simulation on DisNetManager DM
        """
        # Step begin
        self.step_begin(DM, state)

        # Step time-integrate
        self.step_integrate(DM, state)

        # Step post-integrate
        self.step_post_integrate(DM, state)

        # Step topological operations
        self.step_topological_operations(DM, state)

        # Step update response
        self.step_update_response(DM, state)

        self.step_write_files(DM, state)
        self.step_print_info(DM, state)
        self.step_visualize(DM, state)

        # Step end
        self.step_end(DM, state)

        return state

    def run(self, DM: DisNetManager, state: dict):
        if self.write_freq != None:
            os.makedirs(self.write_dir, exist_ok=True)

        if self.vis != None and self.plot_freq != None:
            try: 
                self.fig = plt.figure(figsize=(8,8))
                self.ax = plt.axes(projection='3d')
            except NameError: 
                print('plt not defined')
                return
            
            # plot initial configuration
            self.vis.plot_disnet(DM, fig=self.fig, ax=self.ax, trim=True, block=False)

        for istep in range(self.max_step):
            state['istep'] = istep
            self.step(DM, state)

        # plot final configuration
        if self.vis != None and self.plot_freq != None:
            self.vis.plot_disnet(DM, fig=self.fig, ax=self.ax, trim=True, block=False)

        return state