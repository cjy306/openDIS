"""Compare both initial dislocation densities (PNG only)."""
import argparse
from pathlib import Path
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def response(path):
    text = path.read_text(encoding='utf-8')
    if text and not text.endswith('\n'):
        text = text.rsplit('\n', 1)[0]+'\n' if '\n' in text else ''
    data = np.loadtxt(io.StringIO(text), comments='#', ndmin=2)
    if data.shape[0] == 0 or data.shape[1] < 4 or not np.isfinite(data[:,1:4]).all():
        raise ValueError(f'Invalid loading data: {path}')
    if np.any(np.diff(data[:,1]) < 0):
        raise ValueError(f'Non-monotone strain; inspect restart overlap: {path}')
    return data[:,1]*100, data[:,2]/1e6, data[:,3]

def loading_plot(cases, output):
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for label,path in cases:
        strain,stress,rho = response(path)
        axes[0].plot(strain,stress,label=label,lw=1.2)
        axes[1].plot(strain,rho,label=label,lw=1.2)
    axes[0].set_ylabel('Axial stress (MPa)')
    axes[1].set_ylabel('Dislocation density (m$^{-2}$)')
    for ax in axes:
        ax.set_xlabel('Loading strain (%)');ax.grid(alpha=.25);ax.legend();ax.set_xlim(left=0)
    fig.suptitle('FCC Cu | 5 um PBC box | 1 um FR | no relaxation | seed 12345')
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=250);plt.close(fig)
    print(output)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='output')
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    cases = [
        (r'$\rho_0=10^{12}$ m$^{-2}$', base/'rho1e12'/args.input/'stress_strain_dens.dat'),
        (r'$\rho_0=10^{13}$ m$^{-2}$', base/'rho1e13'/args.input/'stress_strain_dens.dat'),
    ]
    missing = [str(path) for _, path in cases if not path.is_file()]
    if missing:
        raise FileNotFoundError('Both groups are required; missing: ' + ', '.join(missing))
    loading_plot(cases, base/'plots'/'density_compare.png')
