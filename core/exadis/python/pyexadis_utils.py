"""@package docstring

ExaDiS python utilities

Implements utility functions for the ExaDiS python binding

* insert_frank_read_src()
* insert_infinite_line()
* insert_prismatic_loop()
* generate_line_config()
* generate_prismatic_config()

* get_segments_end_points()
* get_segments_length()
* dislocation_density()
* dislocation_charge()

* replicate_network()
* combine_networks()
* extract_segments()
* delete_segments()

* read_paradis()
* write_data()
* write_vtk()

Nicolas Bertin
bertin1@llnl.gov
"""

import numpy as np
import pyexadis
from pyexadis_base import NodeConstraints, ExaDisNet
try:
    # Try importing DisNetManager from OpenDiS
    from framework.disnet_manager import DisNetManager
except ImportError:
    # Use dummy DisNetManager if OpenDiS is not available
    from pyexadis_base import DisNetManager

from typing import Tuple
import time


def insert_frank_read_src(cell, nodes, segs, burg, plane, length, center, theta=0.0, linedir=None, numnodes=10):
    """Insert a Frank-Read source into the list of nodes and segments
    """
    plane = plane / np.linalg.norm(plane)
    if np.abs(np.dot(burg, plane)) >= 1e-5:
        print('Warning: Burgers vector and plane normal are not orthogonal')
    
    if not linedir is None:
        ldir = np.array(linedir)
        ldir = ldir / np.linalg.norm(ldir)
    else:
        b = burg / np.linalg.norm(burg)
        y = np.cross(plane, b)
        y = y / np.linalg.norm(y)
        ldir = np.cos(theta*np.pi/180.0)*b+np.sin(theta*np.pi/180.0)*y
    
    istart = len(nodes)
    for i in range(numnodes):
        p = center -0.5*length*ldir + i*length/(numnodes-1)*ldir
        constraint = NodeConstraints.PINNED_NODE if (i == 0 or i == numnodes-1) else NodeConstraints.UNCONSTRAINED
        nodes.append(np.concatenate((p, [constraint])))
    
    for i in range(numnodes-1):
        segs.append(np.concatenate(([istart+i, istart+i+1], burg, plane)))
    
    return nodes, segs


def insert_infinite_line(cell, nodes, segs, burg, plane, origin, theta=0.0, linedir=None, maxseg=-1, trial=False):
    """Insert an infinite line into the list of nodes and segments
    """
    plane = plane / np.linalg.norm(plane)
    if np.abs(np.dot(burg, plane)) >= 1e-5:
        print('Warning: Burgers vector and plane normal are not orthogonal')
    
    if not linedir is None:
        ldir = np.array(linedir)
        ldir = ldir / np.linalg.norm(ldir)
    else:
        b = burg / np.linalg.norm(burg)
        y = np.cross(plane, b)
        y = y / np.linalg.norm(y)
        ldir = np.cos(theta*np.pi/180.0)*b+np.sin(theta*np.pi/180.0)*y

    h = np.array(cell.h)
    Lmin = np.min(np.linalg.norm(h, axis=0))
    seglength = 0.15*Lmin
    
    if maxseg > 0:
        seglength = np.min([seglength, maxseg])

    length = 0.0
    meet = 0
    maxnodes = 1000
    numnodes = 0
    origin = np.array(origin)
    p = 1.0*origin
    originpbc = 1.0*origin
    while ((~meet) & (numnodes < maxnodes)):
        p += seglength*ldir
        pp = np.asarray(cell.closest_image(Rref=origin, R=p))
        dist = np.linalg.norm(pp-origin)
        if ((numnodes > 0) & (dist < seglength)):
            originpbc = np.asarray(cell.closest_image(Rref=p, R=origin))
            meet = 1
        numnodes += 1

    if numnodes == maxnodes:
        if trial:
            return -1.0
        else:
            print('Warning: infinite line is too long, aborting')
            return nodes, segs

    if trial:
        return np.linalg.norm(originpbc-origin)
    else:
        istart = len(nodes)
        for i in range(numnodes):
            p = origin + 1.0*i/numnodes*(originpbc-origin)
            constraint = NodeConstraints.UNCONSTRAINED
            nodes.append(np.concatenate((p, [constraint])))
        for i in range(numnodes):
            segs.append(np.concatenate(([istart+i, istart+(i+1)%numnodes], burg, plane)))
        return nodes, segs


def insert_prismatic_loop(crystal, cell, nodes, segs, burg, radius, center, maxseg=-1, Rorient=None):
    """Insert a prismatic dislocation loop into the list of nodes and segments
    """
    b = -1.0*burg
    
    if crystal in ['BCC', 'bcc']:
        b0 = 1.0/np.sqrt(3.0)*np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.]])
        bcol = np.abs(np.abs(np.dot(b0, b))-1.0)
        ib = bcol.argmin()
        if bcol[ib] > 1e-5:
            raise ValueError('BCC Burgers vector must be of the 1/2<111> type in insert_prismatic_loop()')
        Nsides = 6
        if 1:
            # Loop with arms on {110} planes (default)
            e = np.array([[-2.0*b[0],b[1],b[2]],[-b[0],-b[1],2.0*b[2]],
                          [b[0],-2.0*b[1],b[2]],[2.0*b[0],-b[1],-b[2]],
                          [b[0],b[1],-2.0*b[2]],[-b[0],2.0*b[1],-b[2]]])
        else:
            # Loop with arms on {112} planes
            e = np.array([[-b[0],0.0,b[2]],[0.0,-b[1],b[2]],
                          [b[0],-b[1],0.0],[b[0],0.0,-b[2]],
                          [0.0,b[1],-b[2]],[-b[0],b[1],0.0]])
        
        n = np.cross(b, e[(np.arange(6)+1)%6]-e[np.arange(6)])
        e = e / np.linalg.norm(e, axis=1)[:,None]
        
    elif crystal in ['FCC', 'fcc']:
        Nsides = 4
        b0 = 1.0/np.sqrt(2.0)*np.array([[0,1,1],[0,-1,1],[1,0,1],[-1,0,1],[1,1,0],[-1,1,0]])
        n01 = np.array([[-1,-1,1],[-1,1,1],[-1,1,1],[1,1,1],[-1,1,1],[1,1,1]])
        n02 = np.array([[1,-1,1],[1,1,1],[-1,-1,1],[1,-1,1],[-1,1,-1],[1,1,-1]])
        bcol = np.abs(np.abs(np.dot(b0, b))-1.0)
        ib = bcol.argmin()
        if bcol[ib] > 1e-5:
            raise ValueError('FCC Burgers vector must be of the 1/2<110> type in insert_prismatic_loop()')
        p1 = n01[ib] / np.linalg.norm(n01[ib])
        p2 = n02[ib] / np.linalg.norm(n02[ib])
        l1 = np.cross(p1, b)
        l1 = l1 / np.linalg.norm(l1)
        l2 = np.cross(p2, b)
        l2 = l2 / np.linalg.norm(l2)
        e = np.array([-0.5*l1-0.5*l2, +0.5*l1-0.5*l2, +0.5*l1+0.5*l2, -0.5*l1+0.5*l2])
        n = np.array([p1, p2, p1, p2])
        
    else:
        raise ValueError('Error: unsupported crystal type = %s in insert_prismatic_loop()' % crystal)
    
    n = n / np.linalg.norm(n, axis=1)[:,None]
    if Rorient is not None:
        Rorient = np.array(Rorient)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:,None]
        b = np.matmul(b, Rorient.T)
        e = np.matmul(e, Rorient.T)
        n = np.matmul(n, Rorient.T)
    
    istart = len(nodes)
    Nnodes = 0
    for i in range(Nsides):
        l = radius*(e[(i+1)%Nsides]-e[i])
        Nseg = int(np.ceil(np.linalg.norm(l)/maxseg)) if maxseg > 0 else 1
        for j in range(Nseg):
            p = radius*e[i]+1.0*j/Nseg*l+center
            nodes.append(np.concatenate((p, [NodeConstraints.UNCONSTRAINED])))
            n1 = istart+Nnodes
            n2 = istart if (i == Nsides-1 and j == Nseg-1) else n1+1
            segs.append(np.concatenate(([n1, n2], b, n[i])))
            Nnodes += 1
            
    return nodes, segs


def generate_line_config(crystal, Lbox, num_lines, theta=None, maxseg=-1, Rorient=None, seed=-1, verbose=True):
    """Generate a configuration made of straight, infinite dislocation lines
    """
    if verbose: print('generate_line_config()')
    
    if crystal in ['BCC', 'bcc']:
        # Define the 12 <111>{110} slip systems
        b = np.array([
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.],
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.],
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.]
        ])
        n = np.array([
            [0.,-1.,1.], [0.,-1.,1.], [0.,1.,1.], [0.,1.,1.],
            [1.,0.,1.], [-1.,0.,1.], [1.,0.,1.], [-1.,0.,1.],
            [1.,1.,0.], [-1.,1.,0.], [-1.,1.,0.], [1.,1.,0.]
        ])
        
    elif crystal in ['FCC', 'fcc']:
        # Define the 12 <110>{111} slip systems
        b = np.array([
            [0.,1.,-1.], [1.,0.,-1.], [1.,-1.,0.],
            [0.,1.,-1.], [1.,0.,1.], [1.,1.,0.],
            [0.,1.,1.], [1.,0.,-1.], [1.,1.,0.],
            [0.,1.,1.], [1.,0.,1.], [1.,-1.,0.]
        ])
        n = np.array([
            [1.,1.,1.], [1.,1.,1.], [1.,1.,1.],
            [-1.,1.,1.], [-1.,1.,1.], [-1.,1.,1.],
            [1.,-1.,1.], [1.,-1.,1.], [1.,-1.,1.],
            [1.,1.,-1.], [1.,1.,-1.], [1.,1.,-1.]
        ])
        
    else:
        raise ValueError('Error: unsupported crystal type = %s in generate_line_config()' % crystal)
    
    nsys = b.shape[0]
    b = b / np.linalg.norm(b, axis=1)[:,None]
    n = n / np.linalg.norm(n, axis=1)[:,None]
    if Rorient is not None:
        Rorient = np.array(Rorient)
        Rorient = Rorient / np.linalg.norm(Rorient, axis=1)[:,None]
        b = np.matmul(b, Rorient.T)
        n = np.matmul(n, Rorient.T)
    
    cell = pyexadis.Cell(Lbox)
    Lmax = np.max(np.linalg.norm(cell.h, axis=0))
    
    if theta is None:
        ntheta = 19
        theta = 90.0/(ntheta-1)*np.arange(ntheta)
        theta_minlength = np.zeros((nsys, ntheta))
        for isys in range(nsys):
            burg, plane = b[isys], n[isys]
            c = np.array(cell.center())
            minlength = 1e20
            for t in range(ntheta):
                nodes, segs = [], []
                length = insert_infinite_line(cell, nodes, segs, burg, plane, c,
                                              theta=theta[t], maxseg=maxseg, trial=True)
                theta_minlength[isys,t] = length
        
        theta_minlength = np.ma.masked_less(theta_minlength, 0.0)
        minlength = theta_minlength.min(axis=1).filled(-1.0)
        maxlength = np.max(minlength)
        if maxlength > 10*Lmax or np.min(minlength) < 0.0:
            raise ValueError('Error: cannot find appropriate line to insert')
        
        theta_sys = np.argmin(np.abs(theta_minlength-maxlength), axis=1)
        theta_sys = theta[theta_sys][:,None]
    else:
        theta_sys = np.tile(np.array(theta), (nsys, 1))
    
    if seed > 0: np.random.seed(seed)
    pos = np.random.rand(num_lines, 3)
    pos = np.array(cell.origin) + np.matmul(pos, np.array(cell.h).T)
    ithe = np.random.randint(0, theta_sys.shape[1], num_lines)
    nodes, segs = [], []
    
    for i in range(num_lines):
        isys = i % nsys
        burg, plane = b[isys], n[isys]
        
        idip = np.floor(i/nsys).astype(int) % 2
        lsign = 1-2*idip
        
        edir = np.cross(plane, burg)
        edir = edir / np.linalg.norm(edir)
        theta = theta_sys[isys,ithe[i-idip*nsys]]
        ldir = np.cos(theta*np.pi/180.0)*burg + np.sin(theta*np.pi/180.0)*edir
        
        if verbose: print(' insert dislocation: b = %.3f %.3f %.3f, n = %.3f %.3f %.3f, theta = %.1f deg' % (*burg, *plane, theta))
        nodes, segs = insert_infinite_line(cell, nodes, segs, burg, plane, pos[i],
                                           linedir=lsign*ldir, maxseg=maxseg)
    
    G = ExaDisNet(cell, nodes, segs)
    return G


def generate_prismatic_config(crystal, Lbox, num_loops, radius, maxseg=-1, Rorient=None, seed=-1, uniform=False):
    """Generate a configuration made of prismatic dislocation loops
    """
    if crystal in ['BCC', 'bcc']:
        b = np.array([[1.,1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,-1.]])
    elif crystal in ['FCC', 'fcc']:
        b = np.array([[1.,1.,0.],[-1.,1.,0.],[1.,0.,1.],[-1.,0.,1.],[0.,1.,1.],[0.,-1.,1.]])
    else:
        raise ValueError('Error: unsupported crystal type = %s in generate_prismatic_config()' % crystal)
    
    nburg = b.shape[0]
    b = b / np.linalg.norm(b, axis=1)[:,None]
    
    # Insert the loops
    cell = pyexadis.Cell(Lbox)
    if seed > 0: np.random.seed(seed)
    if uniform:
        # random uniform positions
        ngrid = np.ceil((1.0*num_loops)**(1.0/3.0))
        H = 1.0/ngrid
        x = 0.5*H + H*np.arange(ngrid)
        x, y, z = np.meshgrid(x, x, x)
        p = np.random.permutation(len(x.flatten()))
        x, y, z = x.flatten()[p], y.flatten()[p], z.flatten()[p]
        pos = np.vstack((x, y, z)).T + 0.5*H*(np.random.rand(len(x), 3)-0.5)
    else:
        pos = np.random.rand(num_loops, 3)
    pos = np.array(cell.origin) + np.matmul(pos, np.array(cell.h).T)
    if isinstance(radius, list):
        R = np.random.uniform(radius[0], radius[1], size=(num_loops,))
    else:
        R = radius*np.ones(num_loops)
    
    nodes, segs = [], []
    for i in range(num_loops):
        iburg = i % nburg
        burg = b[iburg]
        nodes, segs = insert_prismatic_loop(crystal, cell, nodes, segs, burg,
                                            R[i], pos[i], maxseg, Rorient)
    
    G = ExaDisNet(cell, nodes, segs)
    return G


def _compute_slip_system_ids(bsegs, planes, crystal='BCC', tol=1e-3):
    """Compute slip system IDs for segments.
    For BCC: 
        - 1-12: <111>{110} 滑移系
        - 13-24: <111>{112} 滑移系  
        - 25: 其他/未识别
    For FCC: 
        - 1-12: <110>{111} 滑移系
        - 13: 其他/未识别
    """
    bsegs = np.asarray(bsegs)
    planes = np.asarray(planes)
    nsegs = bsegs.shape[0]
    
    if crystal is not None and crystal.lower() in ['fcc']:
        ids = 13 * np.ones(nsegs, dtype=int)
    else:
        ids = 25 * np.ones(nsegs, dtype=int)

    if crystal is None:
        return ids

    c = crystal.lower()
    if c in ['bcc']:
        b_110 = np.array([
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.],
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.],
            [-1.,1.,1.], [1.,1.,1.], [-1.,-1.,1.], [1.,-1.,1.]
        ])
        n_110 = np.array([
            [0.,-1.,1.], [0.,-1.,1.], [0.,1.,1.], [0.,1.,1.],
            [1.,0.,1.], [-1.,0.,1.], [1.,0.,1.], [-1.,0.,1.],
            [1.,1.,0.], [-1.,1.,0.], [-1.,1.,0.], [1.,1.,0.]
        ])
        
        b_112 = np.array([
            [1.,1.,1.], [1.,1.,1.], [1.,1.,1.],
            [-1.,1.,1.], [-1.,1.,1.], [-1.,1.,1.],
            [1.,-1.,1.], [1.,-1.,1.], [1.,-1.,1.],
            [1.,1.,-1.], [1.,1.,-1.], [1.,1.,-1.]
        ])
        n_112 = np.array([
            [ 2.,-1.,-1.], [-1., 2.,-1.], [-1.,-1., 2.],   # b=[1,1,1]
            [-2.,-1.,-1.], [ 1., 2.,-1.], [ 1.,-1., 2.],   # b=[-1,1,1]
            [ 2., 1.,-1.], [-1.,-2.,-1.], [-1., 1., 2.],   # b=[1,-1,1]
            [ 2.,-1., 1.], [-1., 2., 1.], [-1.,-1.,-2.],   # b=[1,1,-1]
        ])
        
        
        b_sys = np.vstack([b_110, b_112])
        n_sys = np.vstack([n_110, n_112])
        b_sys = b_sys / np.linalg.norm(b_sys, axis=1)[:, None]
        n_sys = n_sys / np.linalg.norm(n_sys, axis=1)[:, None]
        nsys = b_sys.shape[0]

        for i in range(nsegs):
            bi = bsegs[i]
            pi = planes[i]
            if np.linalg.norm(bi) == 0 or np.linalg.norm(pi) == 0:
                continue
            bi = bi / np.linalg.norm(bi)
            pi = pi / np.linalg.norm(pi)
            
            best = 1e9
            best_j = -1
            for j in range(nsys):
                sb = abs(abs(np.dot(bi, b_sys[j])) - 1.0)
                sp = abs(abs(np.dot(pi, n_sys[j])) - 1.0)
                score = sb + sp
                if score < best:
                    best = score
                    best_j = j
                    
            if best_j >= 0 and best < tol:
                ids[i] = best_j + 1

    elif c in ['fcc']:
        b_sys = np.array([
            [0., 1., -1.], [-1., 0., 1.], [1., -1., 0.],
            [0., 1., -1.], [1., 0., 1.], [1., 1., 0.],
            [0., 1., 1.], [1., 0., -1.], [-1., -1., 0.],
            [0., 1., 1.], [-1., 0., -1.], [1., -1., 0.]
        ])
        n_sys = np.array([
            [1., 1., 1.], [1., 1., 1.], [1., 1., 1.],
            [-1., 1., 1.], [-1., 1., 1.], [-1., 1., 1.],
            [1., -1., 1.], [1., -1., 1.], [1., -1., 1.],
            [1., 1., -1.], [1., 1., -1.], [1., 1., -1.]
        ])
        b_sys = b_sys / np.linalg.norm(b_sys, axis=1)[:, None]
        n_sys = n_sys / np.linalg.norm(n_sys, axis=1)[:, None]
        nsys = b_sys.shape[0]

        for i in range(nsegs):
            bi = bsegs[i]
            pi = planes[i]
            if np.linalg.norm(bi) == 0 or np.linalg.norm(pi) == 0:
                continue
            bi = bi / np.linalg.norm(bi)
            pi = pi / np.linalg.norm(pi)
            best = 1e9
            best_j = -1
            for j in range(nsys):
                sb = abs(abs(np.dot(bi, b_sys[j])) - 1.0)
                sp = abs(abs(np.dot(pi, n_sys[j])) - 1.0)
                score = sb + sp
                if score < best:
                    best = score
                    best_j = j
            if best_j >= 0 and best < tol:
                ids[i] = best_j + 1

    return ids


def _compute_segment_character(bsegs, r1, r2, screw_tol=15.0, edge_tol=75.0):
    """计算位错段的特征类型和特征角
    
    Args:
        bsegs: (N,3) Burgers矢量数组
        r1: (N,3) 段起点坐标
        r2: (N,3) 段终点坐标
        screw_tol: 螺位错判定阈值(度)
        edge_tol: 刃位错判定阈值(度)
    
    Returns:
        char_ids: (N,) 位错类型 (1=螺位错, 2=混合位错, 3=刃位错)
        char_angles: (N,) 特征角(度)
    """
    bsegs = np.asarray(bsegs)
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    nsegs = bsegs.shape[0]
    
    t = r2 - r1
    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    t_norm = np.where(t_norm > 1e-10, t_norm, 1.0)
    t = t / t_norm
    
    b_norm = np.linalg.norm(bsegs, axis=1, keepdims=True)
    b_norm = np.where(b_norm > 1e-10, b_norm, 1.0)
    b_unit = bsegs / b_norm
    
    cos_angle = np.abs(np.sum(b_unit * t, axis=1))
    cos_angle = np.clip(cos_angle, 0.0, 1.0)
    char_angles = np.degrees(np.arccos(cos_angle))
    
    char_ids = 2 * np.ones(nsegs, dtype=int)
    char_ids[char_angles < screw_tol] = 1
    char_ids[char_angles > edge_tol] = 3
    
    return char_ids, char_angles


def get_segments_end_points(N: DisNetManager) -> Tuple[np.ndarray, np.ndarray]:
    """Get the end points of all segments
    """
    data = N.export_data()
    cell = pyexadis.Cell(**data["cell"])
    nodes = data.get("nodes")
    rn = nodes.get("positions")
    segs = data.get("segs")
    segsnid = segs.get("nodeids")
    
    r1 = np.array(cell.closest_image(Rref=np.array(cell.center()), R=rn[segsnid[:,0]]))
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:,1]]))
    
    return r1, r2


def get_segments_length(N: DisNetManager) -> np.ndarray:
    """Get the length of all segments
    """
    r1, r2 = get_segments_end_points(N)
    Lseg = np.linalg.norm(r2-r1, axis=1)
    return Lseg


def dislocation_density(N: DisNetManager, burgmag: float) -> float:
    """Returns the dislocation density of the network
    """
    Lseg = get_segments_length(N)
    data = N.export_data()
    cell = pyexadis.Cell(**data["cell"])
    V = cell.volume()
    rho = np.sum(Lseg) / V / (burgmag**2)
    return rho


def dislocation_charge(N: DisNetManager) -> np.ndarray:
    """Returns the total Burgers vector charge of the network
    """
    data = N.export_data()
    segs = data.get("segs")
    b = segs.get("burgers")
    btot = np.sum(b, axis=0)
    return btot


def read_paradis(datafile: str) -> DisNetManager:
    """Read a ParaDiS data file and return a DisNetManager object
    """
    G = ExaDisNet()  # 先创建空对象
    G.read_paradis(datafile, verbose=False)  # 再读取文件
    return DisNetManager(G)


def replicate_network(N: DisNetManager, Nrep) -> DisNetManager:
    """Periodically replicate a dislocation network along the three dimensions
    """
    import copy
    
    if np.isscalar(Nrep): 
        Nrep = Nrep * np.ones(3)
    Nrep = np.array(Nrep).astype(int)
    if np.any(Nrep < 1):
        raise ValueError('replicate_network(): periodic replica (%d,%d,%d) must be at least 1 in each direction' % tuple(Nrep))
    if np.all(Nrep == 1):
        return N
    
    data = N.export_data()
    cell0 = pyexadis.Cell(**data["cell"])
    h0 = np.array(cell0.h)
    origin0 = np.array(cell0.origin)
    
    h = h0 * Nrep[:, None]
    cell = pyexadis.Cell(h=h, origin=origin0)
    
    nodes0 = data.get("nodes")
    rn0 = nodes0.get("positions")
    constraint0 = nodes0.get("constraint")
    tags0 = nodes0.get("tags")
    
    segs0 = data.get("segs")
    segsnid0 = segs0.get("nodeids")
    b0 = segs0.get("burgers")
    p0 = segs0.get("planes")
    
    Ntot = np.prod(Nrep)
    nnodes0 = rn0.shape[0]
    nsegs0 = segsnid0.shape[0]
    
    rn = np.zeros((Ntot * nnodes0, 3))
    constraint = np.zeros(Ntot * nnodes0, dtype=int)
    tags = np.zeros((Ntot * nnodes0, 2), dtype=int)
    segsnid = np.zeros((Ntot * nsegs0, 2), dtype=int)
    b = np.zeros((Ntot * nsegs0, 3))
    p = np.zeros((Ntot * nsegs0, 3))
    
    irep = 0
    for i in range(Nrep[0]):
        for j in range(Nrep[1]):
            for k in range(Nrep[2]):
                shift = i * h0[0] + j * h0[1] + k * h0[2]
                istart_nodes = irep * nnodes0
                istart_segs = irep * nsegs0
                rn[istart_nodes:istart_nodes + nnodes0] = rn0 + shift
                constraint[istart_nodes:istart_nodes + nnodes0] = constraint0
                tags[istart_nodes:istart_nodes + nnodes0] = tags0
                segsnid[istart_segs:istart_segs + nsegs0] = segsnid0 + istart_nodes
                b[istart_segs:istart_segs + nsegs0] = b0
                p[istart_segs:istart_segs + nsegs0] = p0
                irep += 1
    
    tags = np.stack((np.zeros(Ntot * nnodes0), np.arange(Ntot * nnodes0))).T
    
    nodes = {"positions": rn, "constraint": constraint, "tags": tags}
    segs = {"nodeids": segsnid, "burgers": b, "planes": p}
    data = {"cell": {"h": h, "origin": origin0, "is_periodic": [1, 1, 1]}, "nodes": nodes, "segs": segs}
    G = ExaDisNet().import_data(data)
    return DisNetManager(G)


def combine_networks(Nlist) -> DisNetManager:
    """Combine several DisNetManager into a single network
    """
    if not isinstance(Nlist, list) or len(Nlist) == 0:
        raise ValueError('combine_networks() argument must be a list of DisNetManager')
    
    for i, Ni in enumerate(Nlist):
        if i == 0:
            data = Ni.export_data()
            nodes, segs = data["nodes"], data["segs"]
            num_nodes = Ni.num_nodes()
        else:
            datai = Ni.export_data()
            if not np.all(datai["cell"]["h"] == data["cell"]["h"]) or \
               not np.all(datai["cell"]["origin"] == data["cell"]["origin"]):
                raise ValueError('combine_networks() networks must use the same cell')
            for k, v in nodes.items():
                nodes[k] = np.vstack((nodes[k], datai["nodes"][k]))
            for k, v in segs.items():
                if k == 'nodeids':
                    segs[k] = np.vstack((segs[k], datai["segs"][k] + num_nodes))
                else:
                    segs[k] = np.vstack((segs[k], datai["segs"][k]))
            num_nodes += Ni.num_nodes()
            
    nodes["tags"] = np.stack((np.zeros(num_nodes), np.arange(num_nodes))).T
    N = DisNetManager(ExaDisNet().import_data(data))
    return N


def extract_segments(N: DisNetManager, seglist) -> DisNetManager:
    """Return a new network that contains a subset of segments
    """
    data = N.export_data()
    segs = data["segs"]
    for k, v in segs.items():
        segs[k] = v[seglist]
    nodelist, nind = np.unique(segs["nodeids"].ravel(), return_inverse=True)
    nodes = data["nodes"]
    for k, v in nodes.items():
        nodes[k] = v[nodelist]
    segs["nodeids"] = nind.reshape(-1, 2)
    G = ExaDisNet().import_data(data)
    return DisNetManager(G)


def delete_segments(N: DisNetManager, seglist) -> DisNetManager:
    """Return a new network in which segments have been deleted
    """
    keeplist = np.setxor1d(seglist, np.arange(N.num_segments()))
    return extract_segments(N, keeplist)


def write_data(N: DisNetManager, datafile: str):
    """Write dislocation network in ParaDiS format
    """
    N.get_disnet(ExaDisNet).write_data(datafile)


def write_vtk(N: DisNetManager, vtkfile: str, segprops={}, pbc_wrap=True, 
              crystal=None, add_slipsystems=True, verbose=False, precipitates=None):
    """Write dislocation network to VTK format with automatic crystal type detection
    
    Args:
        N: DisNetManager object containing the dislocation network
        vtkfile: output VTK file path
        segprops: additional segment properties (dict)
        pbc_wrap: apply periodic boundary conditions (bool)
        crystal: crystal type ('FCC', 'BCC', or None for auto-detection)
        add_slipsystems: add slip system information (bool)
        verbose: print detailed information (bool)
        precipitates: SphericalPrecipitates object (optional)
                     if provided, will add OutsideSphere field to POINT_DATA
    """
    data = N.export_data()
    
    # ========== 自动检测晶体类型 ==========
    if crystal is None:
        segs = data.get("segs")
        b = segs.get("burgers")
        
        if b.shape[0] == 0:
            crystal = 'BCC'
            if verbose:
                print("  警告: 没有位错段，默认使用BCC")
        else:
            b_norm_values = np.linalg.norm(b, axis=1, keepdims=True)
            b_norm_values = np.where(b_norm_values > 1e-10, b_norm_values, 1.0)
            b_norm = b / b_norm_values
            
            fcc_burgers = np.array([
                [0., 1., -1.], [0., -1., 1.], [1., 0., -1.], [-1., 0., 1.],
                [1., -1., 0.], [-1., 1., 0.], [0., 1., 1.], [0., -1., -1.],
                [1., 0., 1.], [-1., 0., -1.], [1., 1., 0.], [-1., -1., 0.]
            ])
            fcc_burgers = fcc_burgers / np.linalg.norm(fcc_burgers, axis=1, keepdims=True)
            
            bcc_burgers = np.array([
                [1., 1., 1.], [-1., -1., -1.], [-1., 1., 1.], [1., -1., -1.],
                [1., -1., 1.], [-1., 1., -1.], [1., 1., -1.], [-1., -1., 1.]
            ])
            bcc_burgers = bcc_burgers / np.linalg.norm(bcc_burgers, axis=1, keepdims=True)
            
            tol = 0.1
            fcc_match = 0
            bcc_match = 0
            
            for i in range(b_norm.shape[0]):
                bi = b_norm[i]
                for fb in fcc_burgers:
                    if abs(abs(np.dot(bi, fb)) - 1.0) < tol:
                        fcc_match += 1
                        break
                for bb in bcc_burgers:
                    if abs(abs(np.dot(bi, bb)) - 1.0) < tol:
                        bcc_match += 1
                        break
            
            if fcc_match == 0 and bcc_match == 0:
                crystal = 'BCC'
                if verbose:
                    print(f"  警告: 无法识别晶体类型，默认使用BCC")
            elif fcc_match > bcc_match:
                crystal = 'FCC'
            else:
                crystal = 'BCC'
            
            if verbose:
                print(f"  自动检测晶体类型: {crystal} (FCC匹配:{fcc_match}, BCC匹配:{bcc_match})")
    
    # ========== 开始写入VTK ==========
    if verbose:
        print(f"开始写入VTK文件: {vtkfile}")
        start_time = time.time()
    
    cell = pyexadis.Cell(**data["cell"])
    cell_origin, cell_center, h = np.array(cell.origin), np.array(cell.center()), np.array(cell.h)
    c = cell_origin + np.array([np.zeros(3), h[0], h[1], h[2], h[0]+h[1],
                                h[0]+h[2], h[1]+h[2], h[0]+h[1]+h[2]])
    
    nodes = data.get("nodes")
    rn = nodes.get("positions")
    segs = data.get("segs")
    segsnid = segs.get("nodeids")
    
    if verbose:
        print(f"  网络信息: {segsnid.shape[0]}个段, {rn.shape[0]}个节点")
    
    r1 = np.array(cell.closest_image(Rref=np.array(cell.center()), R=rn[segsnid[:,0]]))
    r2 = np.array(cell.closest_image(Rref=r1, R=rn[segsnid[:,1]]))
    b = segs.get("burgers")
    p = segs.get("planes")
    
    slip_ids = _compute_slip_system_ids(b, p, crystal=crystal, tol=1e-3)
    char_ids, char_angles = _compute_segment_character(b, r1, r2)
    
    nsegs = segsnid.shape[0]
    rsegs = np.hstack((r1, r2)).reshape(-1,3)

    if verbose:
        if crystal.upper() == 'BCC':
            n_110 = np.sum((slip_ids >= 1) & (slip_ids <= 12))
            n_112 = np.sum((slip_ids >= 13) & (slip_ids <= 24))
            n_other = np.sum(slip_ids == 25)
            print(f"  滑移系统计: {n_110}个{{110}}系, {n_112}个{{112}}系, {n_other}个其他")
        else:
            n_fcc = np.sum((slip_ids >= 1) & (slip_ids <= 12))
            n_other = np.sum(slip_ids == 13)
            print(f"  滑移系统计: {n_fcc}个{{111}}系, {n_other}个其他")
        print(f"  位错类型统计: {np.sum(char_ids==1)}个螺位错, {np.sum(char_ids==2)}个混合位错, {np.sum(char_ids==3)}个刃位错")
    
    f = open(vtkfile, 'w')
    f.write("# vtk DataFile Version 3.0\n")
    f.write(f"Dislocation network ({crystal} crystal) exported from OpenDiS\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    
    total_points = c.shape[0] + 2 * nsegs
    if verbose:
        print(f"  写入{total_points}个点...")
    
    f.write("POINTS %d FLOAT\n" % total_points)
    np.savetxt(f, c, fmt='%.8e')
    np.savetxt(f, rsegs, fmt='%.8e')
    
    if verbose:
        print(f"  写入{1+nsegs}个单元...")
    
    f.write("CELLS %d %d\n" % (1+nsegs, 9+3*nsegs))
    f.write("8 0 1 4 2 3 5 7 6\n")
    nid = np.hstack((2*np.ones((nsegs,1)), np.arange(2*nsegs).reshape(-1,2)+8))
    np.savetxt(f, nid, fmt='%d')
    
    f.write("CELL_TYPES %d\n" % (1+nsegs))
    f.write("12\n")
    np.savetxt(f, 4*np.ones(nsegs), fmt='%d')
    
    f.write("CELL_DATA %d\n" % (1+nsegs))
    
    f.write("SCALARS SlipSystemID int 1\n")
    f.write("LOOKUP_TABLE default\n")
    f.write("0\n")
    np.savetxt(f, slip_ids, fmt='%d')
    
    f.write("SCALARS DislocationType int 1\n")
    f.write("LOOKUP_TABLE default\n")
    f.write("0\n")
    np.savetxt(f, char_ids, fmt='%d')
    
    f.write("SCALARS CharacterAngle float 1\n")
    f.write("LOOKUP_TABLE default\n")
    f.write("0.0\n")
    np.savetxt(f, char_angles, fmt='%.6f')
    
    seg_lengths = np.linalg.norm(r2 - r1, axis=1)
    f.write("SCALARS SegmentLength float 1\n")
    f.write("LOOKUP_TABLE default\n")
    f.write("0.0\n")
    np.savetxt(f, seg_lengths, fmt='%.6e')
    
    f.write("VECTORS Burgers FLOAT\n")
    f.write("0.0 0.0 0.0\n")
    np.savetxt(f, b, fmt='%.8e')
    
    f.write("VECTORS Planes FLOAT\n")
    f.write("0.0 0.0 0.0\n")
    np.savetxt(f, p, fmt='%.8e')
    
    line_dir = r2 - r1
    line_dir_norm = np.linalg.norm(line_dir, axis=1, keepdims=True)
    line_dir_norm = np.where(line_dir_norm > 1e-10, line_dir_norm, 1.0)
    line_dir = line_dir / line_dir_norm
    f.write("VECTORS LineDirection FLOAT\n")
    f.write("0.0 0.0 0.0\n")
    np.savetxt(f, line_dir, fmt='%.8e')
    
    for prop_name, prop_values in segprops.items():
        if isinstance(prop_values, np.ndarray) and len(prop_values) == nsegs:
            if prop_values.ndim == 1:
                f.write(f"SCALARS {prop_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                f.write("0.0\n")
                np.savetxt(f, prop_values, fmt='%.6e')
            elif prop_values.ndim == 2 and prop_values.shape[1] == 3:
                f.write(f"VECTORS {prop_name} FLOAT\n")
                f.write("0.0 0.0 0.0\n")
                np.savetxt(f, prop_values, fmt='%.8e')
    
    # ========== POINT_DATA 部分（节点数据）==========
    # ⚠️ 重要：POINT_DATA的数据个数必须等于VTK文件中的总点数
    # 总点数 = 8个晶胞顶点 + 2*段数
    
    nnodes_vtk = total_points  # VTK中的总点数
    nnodes_network = rn.shape[0]  # 网络中的原始节点数
    
    if verbose:
        print(f"  VTK总点数: {nnodes_vtk}")
        print(f"  网络节点数: {nnodes_network}")
    
    # 计算节点度数（每个节点连接的段数）
    node_degree = np.zeros(nnodes_network, dtype=int)
    for i in range(nsegs):
        n1, n2 = segsnid[i]
        node_degree[n1] += 1
        node_degree[n2] += 1
    
    # 获取节点约束信息
    constraint = nodes.get("constraints")
    if constraint is None:
        constraint = nodes.get("constraint")
     # 展平为 (N,) 一维数组
    if constraint is not None:
        constraint = np.asarray(constraint).ravel()
    
    # 计算OutsideSphere字段（如果提供了precipitates）
    outside_sphere = None
    if precipitates is not None and len(precipitates.centers) > 0:
        if verbose:
            print(f"  计算节点与球形杂质的关系...")
        
        # 为每个节点应用PBC，获得最近镜像
        rn_pbc = np.array([
            cell.closest_image(Rref=cell_center, R=rn[i]) 
            for i in range(nnodes_network)
        ])
        
        # 判断每个节点是否在球内
        inside_sphere = precipitates.is_inside_any_sphere(rn_pbc)
        outside_sphere = (~inside_sphere).astype(int)
        
        n_inside = np.sum(inside_sphere)
        n_outside = np.sum(~inside_sphere)
        if verbose:
            print(f"  球内节点: {n_inside}, 球外节点: {n_outside}")
    
    # 写入POINT_DATA部分
    f.write(f"\nPOINT_DATA {nnodes_vtk}\n")
    
    # ========== 方案1：为所有VTK点写入数据 ==========
    # 前8个点是晶胞顶点（设为默认值）
    # 后面的点对应段的端点
    
    # 写入节点度数
    f.write("SCALARS NodeDegree int 1\n")
    f.write("LOOKUP_TABLE default\n")
    
    # 晶胞顶点的度数（设为0）
    for i in range(8):
        f.write("0\n")
    
    # 段端点的度数
    for i in range(nsegs):
        n1, n2 = segsnid[i]
        f.write(f"{node_degree[n1]}\n")
        f.write(f"{node_degree[n2]}\n")
    
    # 写入节点约束
    f.write("\nSCALARS Constraint int 1\n")
    f.write("LOOKUP_TABLE default\n")
    
    # 晶胞顶点的约束（设为0）
    for i in range(8):
        f.write("0\n")
    
    # 段端点的约束
    if constraint is not None:
        try:
            constraint_array = np.asarray(constraint)
            for i in range(nsegs):
                n1, n2 = segsnid[i]
                c1 = constraint_array[n1] if n1 < len(constraint_array) else 0
                c2 = constraint_array[n2] if n2 < len(constraint_array) else 0
                if c1 is None:
                    c1 = 0
                if c2 is None:
                    c2 = 0
                f.write(f"{int(c1)}\n")
                f.write(f"{int(c2)}\n")
        except (TypeError, IndexError):
            for i in range(nsegs):
                f.write("0\n")
                f.write("0\n")
    else:
        for i in range(2 * nsegs):
            f.write("0\n")
    
    # 写入节点 tag：domain 和 index（与超算日志输出的节点编号一致）
    node_tags = nodes.get("tags")  # shape (N, 2)，每行为 [domain, index]

    f.write("\nSCALARS NodeTag_Domain int 1\n")
    f.write("LOOKUP_TABLE default\n")
    for i in range(8):
        f.write("-1\n")
    for i in range(nsegs):
        n1, n2 = segsnid[i]
        f.write(f"{node_tags[n1][0]}\n")
        f.write(f"{node_tags[n2][0]}\n")

    f.write("\nSCALARS NodeTag_Index int 1\n")
    f.write("LOOKUP_TABLE default\n")
    for i in range(8):
        f.write("-1\n")
    for i in range(nsegs):
        n1, n2 = segsnid[i]
        f.write(f"{node_tags[n1][1]}\n")
        f.write(f"{node_tags[n2][1]}\n")

    # 写入OutsideSphere字段（如果计算了）
    if outside_sphere is not None:
        f.write("\nSCALARS OutsideSphere int 1\n")
        f.write("LOOKUP_TABLE default\n")
        
        # 晶胞顶点的OutsideSphere（设为1，表示在球外）
        for i in range(8):
            f.write("1\n")
        
        # 段端点的OutsideSphere
        for i in range(nsegs):
            n1, n2 = segsnid[i]
            f.write(f"{outside_sphere[n1]}\n")
            f.write(f"{outside_sphere[n2]}\n")
    
    f.close()
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"  VTK写入完成, 耗时: {elapsed:.2f}秒")
        print(f"  文件保存至: {vtkfile}")