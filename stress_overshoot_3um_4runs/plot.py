"""Plot five 3 um FR repeats; seed 12345 is read from its existing sibling folder."""
from pathlib import Path
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SEEDS = (12345,23456,34567,45678,56789)
COLORS = ('#0072B2','#D55E00','#009E73','#CC79A7','#E69F00')

def load_response(path):
    text = path.read_text(encoding='utf-8')
    if text and not text.endswith('\n'):
        text = text.rsplit('\n',1)[0]+'\n' if '\n' in text else ''
        print(f'[ignored unfinished final row] {path}')
    data = np.loadtxt(io.StringIO(text),comments='#',ndmin=2)
    if data.shape[0] == 0 or data.shape[1] < 4:
        raise ValueError(f'No complete data or fewer than four columns: {path}')
    values = data[:,1:4]
    if not np.isfinite(values).all() or np.any(np.diff(values[:,0]) < 0):
        raise ValueError(f'Nonfinite values or decreasing strain: {path}')
    return values[:,0]*100, values[:,1]/1e6, values[:,2]

def main(only_seed=None):
    responses = []
    for index,(seed,color) in enumerate(zip(SEEDS,COLORS),1):
        if only_seed is not None and seed != only_seed:
            continue
        directory = (BASE_DIR.parent/'stress_overshoot_3um_seed12345' if seed==12345
                     else BASE_DIR/f'run_{index:02d}')
        path = directory/f'output_A75_3um_seed{seed}'/'stress_strain_dens.dat'
        if not path.is_file():
            print(f'[missing] {path}')
            continue
        strain,stress,density = load_response(path)
        responses.append((seed,color,strain,stress,density))
        print(f'seed={seed}: points={len(strain)}, latest strain={strain[-1]:.5g}%')
    if not responses:
        raise FileNotFoundError('No simulation responses found')
    fig,axes = plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
    for seed,color,strain,stress,density in responses:
        axes[0].plot(strain,stress,color=color,lw=1.2,label=f'seed {seed}')
        axes[1].plot(strain,density,color=color,lw=1.2,label=f'seed {seed}')
    axes[0].set(title='Stress-Strain',ylabel='Stress (MPa)')
    axes[1].set(title='Dislocation Density-Strain',ylabel=r'Dislocation density (m$^{-2}$)')
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    for ax in axes:
        ax.set_xlabel('Strain (%)')
        ax.set_xlim(left=0)
        ax.grid(True,linestyle='--',alpha=0.25)
        ax.legend(fontsize=9)
    expected = 5 if only_seed is None else 1
    fig.suptitle(f'FCC Cu | 3 um FR | {len(responses)}/{expected} runs available')
    output = BASE_DIR/'plots'
    output.mkdir(exist_ok=True)
    name = 'five_3um_runs_comparison' if only_seed is None else f'3um_seed{only_seed}'
    for ext in ('png',):
        target = output/f'{name}.{ext}'
        fig.savefig(target,dpi=300,facecolor='white')
        print(f'[saved] {target}')
    plt.close(fig)

if __name__ == '__main__':
    main()
