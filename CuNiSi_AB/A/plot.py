"""Plot stress-strain and density-time curves for this case."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def offset_yield(data,E):
    difference = data[:,2]-E*(data[:,1]-.002)
    crossing = np.flatnonzero((difference[:-1]>0)&(difference[1:]<=0))
    if not len(crossing):
        return None
    i = crossing[0]
    f = difference[i]/(difference[i]-difference[i+1])
    return dict(strain=float(data[i,1]+f*(data[i+1,1]-data[i,1])),
                stress_MPa=float((data[i,2]+f*(data[i+1,2]-data[i,2]))/1e6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = BASE_DIR/args.out
    relax = out.with_name(out.name+"_relax")
    target = BASE_DIR/"plots"
    target.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2,2,figsize=(11,8),layout="constrained")
    summary = {}
    if (relax/"relax.csv").is_file():
        with (relax/"relax.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        t = np.array([float(r["time_s"]) for r in rows])
        rho = np.array([float(r["density_m-2"]) for r in rows])
        axes[1,0].plot(t*1e9,rho)
        summary.update(initial_density_m2=float(rho[0]),relaxed_density_m2=float(rho[-1]))
    if (out/"stress_strain_dens.dat").is_file():
        data = np.loadtxt(out/"stress_strain_dens.dat",comments="#",ndmin=2)
        axes[0,0].plot(data[:,1]*100,data[:,2]/1e6)
        axes[0,1].plot(data[:,1]*100,data[:,3])
        axes[1,1].plot(data[:,4]*1e6,data[:,3])
        state = json.loads((out/"run_settings.json").read_text(encoding="utf-8"))["state"]
        summary.update(final_strain=float(data[-1,1]),yield_02=offset_yield(data,2*state["mu"]*(1+state["nu"])))
    labels = [("Total strain (%)","Axial stress (MPa)"),("Total strain (%)","Density (m$^{-2}$)"),
              ("Relaxation time (ns)","Density (m$^{-2}$)"),("Loading time (us)","Density (m$^{-2}$)")]
    if not any(ax.lines for ax in axes.flat):
        raise FileNotFoundError("No relaxation/loading output")
    for ax,(x,y) in zip(axes.flat,labels):
        ax.set(xlabel=x,ylabel=y)
        ax.grid(alpha=.25)
    fig.savefig(target/"curves.png",dpi=220)
    plt.close(fig)
    (target/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(target/"curves.png")


if __name__ == "__main__":
    main()
