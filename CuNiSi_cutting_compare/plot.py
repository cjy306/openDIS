"""Compare particle-free and 50-point-obstacle cutting cases."""
import argparse
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
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for case,label,color in (("no_particles","No particles","black"),
                             ("with_cutting","50 cutting obstacles","tab:blue")):
        out = BASE_DIR/case/args.out
        data = np.loadtxt(out/"stress_strain_dens.dat",comments="#",ndmin=2)
        axes[0].plot(data[:,1]*100,data[:,2]/1e6,label=label,color=color)
        axes[1].plot(data[:,1]*100,data[:,3],label=label,color=color)
    labels = [("Total strain (%)","Stress (MPa)"),("Total strain (%)","Density (m$^{-2}$)")]
    for ax,(x,y) in zip(axes.flat,labels):
        ax.set(xlabel=x,ylabel=y)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(BASE_DIR/"compare.png",dpi=220)
    plt.close(fig)
    print(BASE_DIR/"compare.png")


if __name__ == "__main__":
    main()
