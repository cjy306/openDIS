"""Compare A and B using their saved stress-strain and density-time data."""
import argparse
import csv
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,8),layout="constrained")
    for case,color in (("A","black"),("B","tab:blue")):
        out = BASE_DIR/case/args.out
        data = np.loadtxt(out/"stress_strain_dens.dat",comments="#",ndmin=2)
        axes[0,0].plot(data[:,1]*100,data[:,2]/1e6,label=case,color=color)
        axes[0,1].plot(data[:,1]*100,data[:,3],label=case,color=color)
        axes[1,1].plot(data[:,4]*1e6,data[:,3],label=case,color=color)
        with (out.with_name(out.name+"_relax")/"relax.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        axes[1,0].plot([float(r["time_s"])*1e9 for r in rows],
                       [float(r["density_m-2"]) for r in rows],label=case,color=color)
    labels = [("Total strain (%)","Axial stress (MPa)"),("Total strain (%)","Density (m$^{-2}$)"),
              ("Relaxation time (ns)","Density (m$^{-2}$)"),("Loading time (us)","Density (m$^{-2}$)")]
    for ax,(x,y) in zip(axes.flat,labels):
        ax.set(xlabel=x,ylabel=y)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(BASE_DIR/"compare.png",dpi=220)
    plt.close(fig)
    print(BASE_DIR/"compare.png")


if __name__ == "__main__":
    main()
