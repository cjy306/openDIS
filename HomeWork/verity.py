# 在服务器上单独测试
import os
import sys
import numpy as np
pyexadis_paths = ['../python', '../lib', '../core/pydis/python', '../core/exadis/python/']
for path in pyexadis_paths:
    abspath = os.path.abspath(path)
    if abspath not in sys.path:
        sys.path.append(abspath)

np.set_printoptions(threshold=20, edgeitems=5)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, SimulateNetwork, read_restart, NodeConstraints
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh
    from framework.disnet_manager import DisNetManager
    from pyexadis_utils import read_paradis
except ImportError as e:
    raise ImportError('Cannot import pyexadis') from e

pyexadis.initialize()
print(hasattr(pyexadis.ExaDisNet, 'load_obstacles'))
pyexadis.finalize()