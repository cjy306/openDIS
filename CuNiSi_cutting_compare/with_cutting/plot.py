"""Plot stress-strain and density-time curves for this case."""
import argparse
import json
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
    out = BASE_DIR/args.out
    target = BASE_DIR/"plots"
    target.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    summary = {}
    if (out/"stress_strain_dens.dat").is_file():
        data = np.loadtxt(out/"stress_strain_dens.dat",comments="#",ndmin=2)
        axes[0].plot(data[:,1]*100,data[:,2]/1e6)
        axes[1].plot(data[:,1]*100,data[:,3])
        summary.update(final_strain=float(data[-1,1]))
    labels = [("Total strain (%)","Stress (MPa)"),("Total strain (%)","Density (m$^{-2}$)")]
    if not any(ax.lines for ax in axes.flat):
        raise FileNotFoundError("No loading output")
    for ax,(x,y) in zip(axes.flat,labels):
        ax.set(xlabel=x,ylabel=y)
        ax.grid(alpha=.25)
    fig.savefig(target/"curves.png",dpi=220)
    plt.close(fig)
    (target/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(target/"curves.png")


if __name__ == "__main__":
    main()
