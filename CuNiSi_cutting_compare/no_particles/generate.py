"""Generate the initial CuNiSi FR network and 50 breakaway point obstacles (or the particle-free control)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("OPENDIS_ROOT", BASE_DIR.parent.parent))
for part in ("python", "lib", "core/pydis/python", "core/exadis/python"):
    sys.path.append(str(REPO_ROOT / part))

CASE = "no_particles"
NUM_PARTICLES = 0
LBOX, BURGMAG, FR_LENGTH, MAXSEG = 0.3e-6, 0.2556e-9, 125e-9, 20.
RHO, NUM_FR = 1e15, 216
Rorient = np.array([[1., -1., 0.], [1., 1., -2.], [1., 1., 1.]])
Rorient /= np.linalg.norm(Rorient, axis=1)[:, None]
# diameter(m), volume fraction, type: 0=Orowan, 1=90-degree breakaway
PARTICLES = [(4.9e-9, 0.0, 1)]  # Diameter and mechanism; count is set explicitly.


def make_arrays(insert_fr, cell, seed):
    rng = np.random.default_rng(seed)
    planes = np.array([[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]], float)
    burgers = np.array([[0,1,1], [0,-1,1], [1,0,1], [-1,0,1], [1,1,0], [-1,1,0]], float)
    systems = [(Rorient @ (b/np.linalg.norm(b)), Rorient @ (n/np.linalg.norm(n)))
               for n in planes for b in burgers if abs(b@n) < 1e-10]
    ids = np.tile(np.arange(12), NUM_FR//12)
    rng.shuffle(ids)
    box, length = LBOX/BURGMAG, FR_LENGTH/BURGMAG
    nodes, segs, layout = [], [], []
    for sid in ids:
        b, n = systems[sid]
        theta = rng.uniform(0., 180.)
        tangent = np.cos(np.deg2rad(theta))*b + np.sin(np.deg2rad(theta))*np.cross(n,b)
        margin = .5*length*np.abs(tangent) + 1e-6*box
        low, high = margin.copy(), box-margin
        low[2] += .2*box
        high[2] -= .2*box
        center = np.asarray(cell.origin) + rng.uniform(low, high)
        # PHYS-APPROX: pinned FR seeds; ceiling = artificial endpoints;
        # upgrade = relaxed, predeformed connected dislocation networks.
        insert_fr(cell, nodes, segs, b, n, length, center, theta=theta,
                  numnodes=int(np.ceil(length/MAXSEG))+1)
        layout.append([sid, theta, *center])
    nodes, segs = np.asarray(nodes), np.asarray(segs)
    rng = np.random.default_rng(seed+101)
    obstacles = []
    for diameter, fraction, kind in reversed(PARTICLES):
        radius = diameter/(2*BURGMAG)
        count = NUM_PARTICLES
        for _ in range(count):
            for attempt in range(10000):
                # PHYS-APPROX: spheres stay inside the primary box;
                # ceiling = boundary depletion; upgrade = full PBC sphere enforcement.
                center = rng.uniform(1.05*radius, box-1.05*radius, 3)
                old = np.asarray(obstacles)
                if len(old):
                    delta = old[:,:3]-center
                    delta -= box*np.rint(delta/box)
                    if np.any(np.linalg.norm(delta,axis=1) < old[:,3]+radius):
                        continue
                delta = nodes[nodes[:,3]==7,:3]-center
                delta -= box*np.rint(delta/box)
                if np.any(np.linalg.norm(delta,axis=1) < radius):
                    continue
                obstacles.append([*center, radius, kind, np.pi/2])
                break
            else:
                raise RuntimeError("Particle placement failed; count was not reduced")
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1,6)
    obstacles = np.vstack([obstacles[obstacles[:,4]==1], obstacles[obstacles[:,4]==0]])
    return nodes, segs, obstacles, np.asarray(layout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager
    from pyexadis_utils import insert_frank_read_src, write_vtk
    folder = BASE_DIR / "init"
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"{folder} already contains inputs")
    pyexadis.initialize()
    try:
        cell = pyexadis.Cell(h=(LBOX/BURGMAG)*np.eye(3), is_periodic=[True]*3)
        nodes, segs, particles, layout = make_arrays(insert_frank_read_src, cell, args.seed)
        twins = np.empty((0,6))
        folder.mkdir(parents=True, exist_ok=True)
        network = ExaDisNet(cell, nodes.tolist(), segs.tolist())
        network.write_data(str(folder / "init_config.data"))
        write_vtk(DisNetManager(network), str(folder / "init_config.vtk"), crystal="FCC", Rorient=Rorient, verbose=False)
        np.save(folder / "particles_small.npy", particles[particles[:,4]==1])
        np.save(folder / "particles_large.npy", particles[particles[:,4]==0])
        np.save(folder / "twin_planes.npy", twins)
        np.savetxt(folder / "source_layout.csv", layout, delimiter=",", header="slip_id,theta_deg,x_b,y_b,z_b", comments="")
        meta = dict(particle_count=NUM_PARTICLES, particle_diameter_m=4.9e-9,
            actual_volume_fraction=NUM_PARTICLES*(np.pi*(4.9e-9)**3/6)/LBOX**3, case=CASE, seed=args.seed, burgmag_m=BURGMAG, box_m=LBOX, density_m2=RHO,
            fr_length_m=FR_LENGTH, fr_sources=NUM_FR, local_density_m2=RHO/.6,
            fr_sha256=hashlib.sha256(nodes.tobytes()+segs.tobytes()).hexdigest(),
            particles_sha256=hashlib.sha256(particles.tobytes()).hexdigest())
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(CASE, "FR:", NUM_FR, "particles:", len(particles))
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
