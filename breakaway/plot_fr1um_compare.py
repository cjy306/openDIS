#!/usr/bin/env python3
"""绘制无障碍/点障碍的应力-应变和位错密度-应变对比图。"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURVES = [
    (os.path.join(BASE_DIR, 'output_fr1um_no_obstacles', 'stress_strain_dens.dat'),
     'No obstacles', '#202020'),
    (os.path.join(BASE_DIR, 'output_fr1um_point_obstacles', 'stress_strain_dens.dat'),
     'Type-1 point obstacles', '#D62728'),
]
OUTPUT_DIR = os.path.join(BASE_DIR, 'post_fr1um_compare')
OUTPUT_NAME = 'fr1um_compare.png'


def load_data(filepath):
    data = np.loadtxt(filepath, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    strain, stress, density = data[:, 1], data[:, 2] / 1e6, data[:, 3]
    mask = np.concatenate(([True], np.diff(strain) > 0))
    return strain[mask], stress[mask], density[mask]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    summaries = []

    for filepath, label, color in CURVES:
        strain, stress, density = load_data(filepath)
        strain_pct = 100.0 * strain
        peak = int(np.argmax(stress))
        ax1.plot(strain_pct, stress, color=color, lw=1.3, label=label)
        ax1.scatter(strain_pct[peak], stress[peak], color=color, s=24)
        ax2.plot(strain_pct, density, color=color, lw=1.3, label=label)
        summaries.append((label, len(strain), strain_pct[-1], stress[peak],
                          strain_pct[peak], stress[-1], density[0], density[-1]))

    ax1.set_xlabel('Total strain (%)')
    ax1.set_ylabel('Axial stress (MPa)')
    ax1.set_title('Stress-strain response')
    ax2.set_xlabel('Total strain (%)')
    ax2.set_ylabel(r'Dislocation density (m$^{-2}$)')
    ax2.set_title('Dislocation-density evolution')
    ax2.set_yscale('log')
    for axis in (ax1, ax2):
        axis.grid(True, linestyle='--', alpha=0.3)
        axis.legend(frameon=False)
        axis.set_xlim(left=0)

    plt.tight_layout()
    figure_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    summary_path = os.path.join(OUTPUT_DIR, 'fr1um_compare_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as stream:
        for item in summaries:
            label, points, maxstrain, peakstress, peakstrain, finalstress, rho0, rhof = item
            stream.write('[%s]\n' % label)
            stream.write('points = %d\nmax_strain = %.6f %%\n' % (points, maxstrain))
            stream.write('peak_stress = %.6f MPa @ %.6f %%\n' % (peakstress, peakstrain))
            stream.write('final_stress = %.6f MPa\n' % finalstress)
            stream.write('density: %.8e -> %.8e m^-2\n\n' % (rho0, rhof))
        stream.write('point_minus_baseline_peak = %.6f MPa\n' %
                     (summaries[1][3] - summaries[0][3]))
    print('图像已保存:', figure_path)
    print('摘要已保存:', summary_path)


if __name__ == '__main__':
    main()
