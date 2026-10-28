"""Generate exact paired FR backgrounds; add loops only to with_loops."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from runtime import setup


def make_layout(seed=12345):
    rng = np.random.default_rng(seed)
    burgers = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]], dtype=float)
    burgers /= np.sqrt(3)
    normals = []
    for xyz in itertools.product((-1,0,1), repeat=3):
        v = np.array(xyz)
        nz = v[v != 0]
        if np.count_nonzero(v) == 2 and nz[0] > 0:
            normals.append(v / np.sqrt(2))
    systems = [(b,n) for b in burgers for n in normals if abs(np.dot(b,n)) < 1e-10]
    ids = np.arange(250) % 12
    rng.shuffle(ids)
    centers = rng.uniform(0, 1, (250,3))
    angles = rng.uniform(0,180,250)
    # Separate RNG preserves the FR background when loop settings change.
    loops = np.random.default_rng(seed + 1)
    types = np.arange(1000) % 4
    loops.shuffle(types)
    return dict(centers=centers, systems=ids, angles=angles,
                burgers=np.array([x[0] for x in systems]),
                planes=np.array([x[1] for x in systems]),
                loop_centers=loops.uniform(0,1,(1000,3)), loop_types=types,
                loop_burgers=burgers)


def build_arrays(layout, cell, insert_fr, insert_loop, with_loops):
    print('[generate] building FR nodes and segments', flush=True)
    nodes, segs = [], []
    origin = np.asarray(cell.origin)
    box = np.asarray(cell.h)
    # PHYS-APPROX: artificial open pinned FR sources; ceiling = fixed-end effects;
    # upgrade = physically closed network/source representation with recalculated density.
    for p, sid, theta in zip(layout['centers'], layout['systems'], layout['angles']):
        insert_fr(cell, nodes, segs, layout['burgers'][sid], layout['planes'][sid],
                  0.4e-6/0.248e-9, origin + box @ p, theta=theta,
                  numnodes=int(np.ceil((0.4e-6/0.248e-9)/20.0))+1)
    fr_nodes, fr_segs = len(nodes), len(segs)
    print(f'[generate] FR complete: nodes={fr_nodes}, segments={fr_segs}', flush=True)
    if with_loops:
        print('[generate] adding irradiation loops', flush=True)
        # PHYS-APPROX: monodisperse hexagons, diameter is vertex-to-vertex;
        # ceiling = not measured size distribution; upgrade = measured sampling.
        for index, (p, kind) in enumerate(zip(layout['loop_centers'], layout['loop_types']), 1):
            insert_loop('bcc', cell, nodes, segs, layout['loop_burgers'][kind],
                        10e-9/0.248e-9, origin + box @ p, maxseg=20.0)
            if index == 1 or index % 100 == 0:
                print(f'[generate] loops completed: {index}/{len(layout["loop_types"])}', flush=True)
    print(f'[generate] converting arrays: nodes={len(nodes)}, segments={len(segs)}', flush=True)
    return np.asarray(nodes), np.asarray(segs), fr_nodes, fr_segs


def main(default_case='both'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['both','no_loops','with_loops'], default=default_case)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=Path, help='Initial directory for a single case; relative to that case')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    cases = ['no_loops','with_loops'] if args.case == 'both' else [args.case]
    if args.out is not None and len(cases) != 1:
        parser.error('--out requires a single --case')
    folders = {case: base/case/(args.out if args.out is not None else Path('init')) for case in cases}
    for case in cases:
        if (folders[case]/'init_config.data').exists():
            raise FileExistsError(f'{case}/init already exists; reuse or archive it explicitly')
    print('[generate] locating OpenDiS', flush=True)
    root = setup()
    print(f'[generate] OpenDiS root={root}; importing pyexadis', flush=True)
    import pyexadis
    print(f'[generate] pyexadis imported from {pyexadis.__file__}', flush=True)
    print('[generate] importing Python bindings and geometry helpers', flush=True)
    from pyexadis_base import ExaDisNet, DisNetManager
    from pyexadis_utils import insert_frank_read_src, insert_prismatic_loop, write_vtk
    print(f'[generate] imports complete; sampling layout, seed={args.seed}', flush=True)
    layout = make_layout(args.seed)
    print('[generate] calling pyexadis.initialize()', flush=True)
    pyexadis.initialize()
    print('[generate] initialization complete', flush=True)
    try:
        print('[generate] constructing periodic cell', flush=True)
        cell = pyexadis.Cell(h=(1e-6/0.248e-9)*np.eye(3), is_periodic=[True]*3)
        print('[generate] cell complete', flush=True)
        reference = None
        for case in cases:
            print(f'[generate] case={case}; constructing geometry', flush=True)
            nodes,segs,nn,ns = build_arrays(layout,cell,insert_frank_read_src,
                                          insert_prismatic_loop,case=='with_loops')
            print('[generate] arrays complete; checking geometry and density', flush=True)
            prefix = nodes[:nn].tobytes()+segs[:ns].tobytes()
            if reference is not None and prefix != reference:
                raise AssertionError('FR backgrounds differ')
            reference = prefix
            ends = segs[:,:2].astype(int)
            delta = nodes[ends[:,1],:3]-nodes[ends[:,0],:3]
            box_b = 1e-6/0.248e-9
            delta -= box_b*np.rint(delta/box_b)
            lengths = np.linalg.norm(delta,axis=1)*0.248e-9
            if not np.isclose(lengths[:ns].sum()/1e-18,1e14,rtol=1e-10):
                raise AssertionError('Wrong FR density')
            if not np.allclose(np.sum(segs[:,2:5]*segs[:,5:8],axis=1),0,atol=1e-10):
                raise AssertionError('Burgers vector not in glide plane')
            folder = folders[case]
            print(f'[generate] checks passed; creating output directory {folder}', flush=True)
            folder.mkdir(parents=True,exist_ok=True)
            print('[generate] constructing native ExaDisNet', flush=True)
            network = ExaDisNet(cell,nodes.tolist(),segs.tolist())
            print('[generate] native network complete; writing init_config.data', flush=True)
            network.write_data(str(folder/'init_config.data'))
            print('[generate] data written; writing VTK', flush=True)
            write_vtk(DisNetManager(network),str(folder/'init_config_labeled.vtk'),
                      crystal='BCC',verbose=False)
            print('[generate] VTK written; writing metadata and layout', flush=True)
            metadata = dict(case=case,seed=args.seed,box_m=1e-6,burgmag_m=0.248e-9,
                fr_length_m=0.4e-6,fr_sources=250,fr_nodes=nn,fr_segments=ns,
                fr_density_m2=float(lengths[:ns].sum()/1e-18),
                total_density_m2=float(lengths.sum()/1e-18),
                fr_sha256=hashlib.sha256(prefix).hexdigest(),
                slip_system_counts=np.bincount(layout['systems']).tolist(),
                loops=1000 if case=='with_loops' else 0,
                loop_circumdiameter_m=20e-9,loop_segments=int(len(segs)-ns))
            (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            np.savez(folder/'layout.npz',**layout)
            import csv
            with (folder/'source_layout.csv').open('w',newline='',encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['source_id','fx','fy','fz','slip_system_id','theta_deg'])
                for i,(p,sid,theta) in enumerate(zip(layout['centers'],layout['systems'],layout['angles'])):
                    writer.writerow([i,*p,int(sid),theta])
            print(json.dumps(metadata,indent=2))
    finally:
        print('[generate] calling pyexadis.finalize()', flush=True)
        pyexadis.finalize()
        print('[generate] finalization complete', flush=True)

if __name__ == '__main__':
    main()
