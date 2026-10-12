"""Compare FR lengths separately for each seed; run with optional --seed."""
import argparse
from pathlib import Path

from plot import BASE_DIR, SEEDS, load_response, plt


def response_paths(seed):
    index = SEEDS.index(seed) + 1
    root = BASE_DIR.parent
    one = root/'stress_overshoot_5runs'/f'run_{index:02d}'
    three = (root/'stress_overshoot_3um_seed12345' if seed == 12345
             else BASE_DIR/f'run_{index:02d}')
    return (one/f'output_A75_seed{seed}'/'stress_strain_dens.dat',
            three/f'output_A75_3um_seed{seed}'/'stress_strain_dens.dat')


def plot_seed(seed):
    paths = response_paths(seed)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f'[skip seed {seed}: missing] {path}')
        return False
    try:
        responses = [load_response(path) for path in paths]
    except (ValueError, OSError) as error:
        print(f'[skip seed {seed}: unreadable data] {error}')
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    cases = (('1 um | 128 sources | A75', '#0072B2'),
             ('3 um | 43 sources | high-Schmid 74.42%', '#D55E00'))
    for (label, color), (strain, stress, density) in zip(cases, responses):
        # Strain returned by load_response is in percent; target is 0.5%.
        if strain[-1] < 0.5 - 1e-6:
            label += f' | incomplete: {strain[-1]:.4f}%'
        print(f'[seed {seed}] {label}; {len(strain)} points')
        axes[0].plot(strain, stress, color=color, lw=1.2, label=label)
        axes[1].plot(strain, density, color=color, lw=1.2, label=label)
    axes[0].set(title='Stress-Strain', ylabel='Stress (MPa)')
    axes[1].set(title='Dislocation Density-Strain',
                ylabel=r'Dislocation density (m$^{-2}$)')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    for ax in axes:
        ax.set_xlabel('Strain (%)')
        ax.set_xlim(left=0)
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'FCC Cu | seed {seed} | 1 um vs 3 um FR')
    output = BASE_DIR/'plots'/'1um_vs_3um'
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        target = output/f'fr_1um_vs_3um_seed{seed}.{suffix}'
        fig.savefig(target, dpi=300, facecolor='white')
        print(f'[saved] {target}')
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS,
                        help='Compare only this seed (default: all five)')
    args = parser.parse_args()
    seeds = SEEDS if args.seed is None else (args.seed,)
    count = sum(plot_seed(seed) for seed in seeds)
    print(f'Generated {count}/{len(seeds)} seed comparisons in this invocation.')
    if not count:
        raise SystemExit('No comparable pairs found; check simulation data paths above.')


if __name__ == '__main__':
    main()
