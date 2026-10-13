"""Compare original seed34567 with the two 1 um parameter variants."""
from pathlib import Path
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BASE_DIR = Path(__file__).resolve().parent

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


def main():
    baseline = BASE_DIR.parent/'stress_overshoot_5runs'/'run_03'
    cases = (
        ('Baseline: a4, rtol1, rann2, seg200/40', '#0072B2', baseline),
        ('a6: rtol1.5, rann3, seg200/40', '#D55E00', BASE_DIR/'a6'),
        ('seg400/100: a4, rtol1, rann2', '#009E73', BASE_DIR/'seg400'),
    )
    responses = []
    for label, color, directory in cases:
        path = directory/'output_A75_seed34567'/'stress_strain_dens.dat'
        if not path.is_file():
            print(f'[missing] {path}')
            continue
        try:
            strain, stress, density = load_response(path)
        except (ValueError, OSError) as error:
            print(f'[skip] {path}: {error}')
            continue
        if strain[-1] < 0.5 - 1e-6:
            label += f' | incomplete {strain[-1]:.4f}%'
        responses.append((label, color, strain, stress, density))
    if not responses:
        raise SystemExit('No valid response files found.')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for label, color, strain, stress, density in responses:
        axes[0].plot(strain, stress, color=color, lw=1.2, label=label)
        axes[1].plot(strain, density, color=color, lw=1.2, label=label)
    axes[0].set(title='Stress-Strain', ylabel='Stress (MPa)')
    axes[1].set(title='Dislocation Density-Strain', ylabel=r'Dislocation density (m$^{-2}$)')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    for ax in axes:
        ax.set_xlabel('Strain (%)')
        ax.set_xlim(left=0)
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'FCC Cu | 1 um FR | seed 34567 | {len(responses)}/3 cases available')
    output = BASE_DIR/'plots'
    output.mkdir(exist_ok=True)
    for suffix in ('png',):
        target = output/f'parameter_comparison_seed34567.{suffix}'
        fig.savefig(target, dpi=300, facecolor='white')
        print(f'[saved] {target}')
    plt.close(fig)

if __name__ == '__main__':
    main()
