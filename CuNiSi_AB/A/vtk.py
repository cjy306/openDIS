"""Convert dislocations, particles and twin planes to VTK."""
import argparse
import csv
import os
from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("OPENDIS_ROOT", BASE_DIR.parent.parent))
for part in ("python", "lib", "core/pydis/python", "core/exadis/python"):
    sys.path.append(str(REPO_ROOT / part))
BOX_M, BURG_M = 0.3e-6, 0.2556e-9
Rorient = np.array([[1., -1., 0.], [1., 1., -2.], [1., 1., 1.]])
Rorient /= np.linalg.norm(Rorient, axis=1)[:,None]


def write_particles(path,particles):
    n=len(particles)
    with path.open('w',encoding='ascii') as stream:
        stream.write('# vtk DataFile Version 3.0\nParticle centers and physical radii (b units)\nASCII\nDATASET POLYDATA\n')
        stream.write(f'POINTS {n} double\n')
        np.savetxt(stream,particles[:,:3],fmt='%.12e')
        stream.write(f'VERTICES {n} {2*n}\n')
        for i in range(n):
            stream.write(f'1 {i}\n')
        stream.write(f'POINT_DATA {n}\nSCALARS Radius double 1\nLOOKUP_TABLE default\n')
        np.savetxt(stream,particles[:,3],fmt='%.12e')
        stream.write('SCALARS ParticleType int 1\nLOOKUP_TABLE default\n')
        np.savetxt(stream,particles[:,4],fmt='%d')


def write_twins(path,twins):
    if not len(twins):
        return
    box=BOX_M/BURG_M
    with path.open('w',encoding='ascii') as stream:
        stream.write('# vtk DataFile Version 3.0\nParallel twin blocking planes (b units)\nASCII\nDATASET POLYDATA\n')
        stream.write(f'POINTS {4*len(twins)} double\n')
        for row in twins:
            z=row[2]
            for x,y in ((0,0),(box,0),(box,box),(0,box)):
                stream.write(f'{x:.12e} {y:.12e} {z:.12e}\n')
        stream.write(f'POLYGONS {len(twins)} {5*len(twins)}\n')
        for i in range(len(twins)):
            stream.write('4 '+' '.join(str(4*i+j) for j in range(4))+'\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    folder = BASE_DIR / args.input
    paths = sorted(folder.glob("config.*.data"), key=lambda p:int(p.stem.split(".")[-1]))[::args.stride]
    paths += [folder/name for name in ("init_config.data","loading_start.data","relaxed_config.data") if (folder/name).is_file()]
    if not paths:
        raise FileNotFoundError(f"No configurations in {folder}")
    target = BASE_DIR / ("vtk_"+folder.name)
    target.mkdir(exist_ok=True)
    with np.load(BASE_DIR/"init/obstacles.npz") as data:
        write_particles(target/"particles.vtk",data["particles"])
        write_twins(target/"twin_planes.vtk",data["twins"])
    import pyexadis
    from pyexadis_utils import read_paradis, write_vtk
    pyexadis.initialize()
    try:
        with (target/"slab_check.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file","nodes_below_60nm","nodes_above_240nm"])
            for path in paths:
                net = read_paradis(str(path))
                write_vtk(net,str(target/(path.stem+".vtk")),crystal="FCC",Rorient=Rorient,verbose=False)
                z = np.asarray(net.export_data()["nodes"]["positions"])[:,2]*BURG_M/1e-9
                writer.writerow([path.name,int(np.count_nonzero(z<60-1e-6)),int(np.count_nonzero(z>240+1e-6))])
                print(target/(path.stem+".vtk"),flush=True)
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
